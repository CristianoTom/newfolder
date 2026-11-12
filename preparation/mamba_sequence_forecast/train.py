"""生成单变量时间序列，训练 Mamba，评估并输出未来多步预测。"""

import argparse
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import MambaForecaster


def make_series(length: int, seed: int) -> torch.Tensor:
    """生成含两种周期、缓慢变化的振幅以及少量噪声的合成数据。"""
    time = torch.arange(length, dtype=torch.float32)
    amplitude = 1.0 + 0.15 * torch.sin(2 * torch.pi * time / 240)
    main_wave = amplitude * torch.sin(2 * torch.pi * time / 48)
    short_wave = 0.35 * torch.sin(2 * torch.pi * time / 13 + 0.4)
    generator = torch.Generator().manual_seed(seed)
    noise = 0.04 * torch.randn(length, generator=generator)
    return main_wave + short_wave + noise


@dataclass(frozen=True)
class Standardizer:
    """保存训练段的统计量，供验证、测试和未来预测共同使用。"""

    mean: float
    std: float

    @classmethod
    def fit(cls, training_values: torch.Tensor) -> "Standardizer":
        return cls(
            mean=training_values.mean().item(),
            std=training_values.std(unbiased=False).clamp_min(1e-6).item(),
        )

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean


class ForecastWindows(Dataset):
    """按目标的时间范围构造样本，历史窗口可使用此前已知的观测。"""

    def __init__(
        self,
        values: torch.Tensor,
        history_length: int,
        horizon: int,
        target_start: int,
        target_end: int,
        stride: int = 1,
    ):
        self.values = values
        self.history_length = history_length
        self.horizon = horizon

        # origin 是未来目标的第一个位置。末尾目标也必须落在本数据段内。
        first_origin = max(target_start, history_length)
        self.origins = range(first_origin, target_end - horizon + 1, stride)
        if len(self.origins) == 0:
            raise ValueError("当前数据段无法构造样本，请缩短历史窗口或预测步数。")

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        origin = self.origins[index]
        history = self.values[origin - self.history_length : origin]
        target = self.values[origin : origin + self.horizon]
        # 单个样本：[历史长度, 1] 和 [预测步数]。DataLoader 会增加批量维。
        return history.unsqueeze(-1), target


def train_one_epoch(model, batches, optimizer, loss_fn, device) -> float:
    """返回整个训练集上按样本数加权的平均损失。"""
    model.train()
    total_loss = 0.0
    sample_count = 0
    for history, target in batches:
        history, target = history.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(history)
        loss = loss_fn(prediction, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * target.size(0)
        sample_count += target.size(0)
    return total_loss / sample_count


@torch.inference_mode()
def evaluate(model, batches, device, scale: Standardizer) -> dict[str, float]:
    """以原始量纲计算 MAE、RMSE，以及持久性基线的 MAE。"""
    model.eval()
    absolute_error = 0.0
    squared_error = 0.0
    baseline_error = 0.0
    point_count = 0
    for history, target in batches:
        history, target = history.to(device), target.to(device)
        # 标准化值相减时均值抵消，误差乘以标准差即可恢复原始量纲。
        error = (model(history) - target) * scale.std
        # 持久性基线把最后一个已知值用作每个未来步的预测。
        baseline = history[:, -1, 0:1]
        absolute_error += error.abs().sum().item()
        squared_error += error.square().sum().item()
        baseline_error += ((baseline - target).abs() * scale.std).sum().item()
        point_count += target.numel()
    return {
        "mae": absolute_error / point_count,
        "rmse": (squared_error / point_count) ** 0.5,
        "baseline_mae": baseline_error / point_count,
    }


@torch.inference_mode()
def predict_future(
    model: MambaForecaster,
    observed: torch.Tensor,
    history_length: int,
    scale: Standardizer,
    device: torch.device,
) -> torch.Tensor:
    """接收原始量纲的一维观测，返回原始量纲的未来预测。"""
    if observed.ndim != 1 or observed.numel() < history_length:
        raise ValueError("observed 必须是一维序列，且至少包含 history_length 个值。")

    model.eval()
    # 只取末尾的历史窗口：[L] → [1, L, 1]。
    history = observed[-history_length:].to(device=device, dtype=torch.float32)
    history = scale.transform(history).reshape(1, history_length, 1)
    forecast = model(history).squeeze(0)
    return scale.inverse_transform(forecast).cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15, help="训练轮数")
    parser.add_argument("--history", type=int, default=32, help="历史窗口长度")
    parser.add_argument("--horizon", type=int, default=8, help="一次预测的未来步数")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if min(args.epochs, args.history, args.horizon) < 1:
        parser.error("--epochs、--history 和 --horizon 必须为正整数")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("当前 PyTorch 环境无法使用 CUDA")

    device = torch.device(args.device)
    if device.type == "cpu":
        # 小张量的逐步扫描容易受线程调度开销影响。
        torch.set_num_threads(min(2, torch.get_num_threads()))
    seed = 42
    torch.manual_seed(seed)
    observed = make_series(length=720, seed=seed)

    # 按时间顺序划分 60% / 20% / 20%，仅用训练段拟合标准化统计量。
    train_end = int(len(observed) * 0.6)
    valid_end = int(len(observed) * 0.8)
    if args.history + args.horizon > train_end or args.horizon > valid_end - train_end:
        parser.error("历史窗口或预测步数过长，无法在当前数据段构造样本")

    scale = Standardizer.fit(observed[:train_end])
    normalized = scale.transform(observed)
    train_set = ForecastWindows(normalized, args.history, args.horizon, 0, train_end, stride=2)
    valid_set = ForecastWindows(normalized, args.history, args.horizon, train_end, valid_end)
    test_set = ForecastWindows(normalized, args.history, args.horizon, valid_end, len(observed))

    # 打乱的是训练窗口的顺序，每个窗口内部仍按时间排列。
    train_batches = DataLoader(train_set, batch_size=32, shuffle=True)
    valid_batches = DataLoader(valid_set, batch_size=32)
    test_batches = DataLoader(test_set, batch_size=32)

    model = MambaForecaster(horizon=args.horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.MSELoss()
    best_valid_mae = float("inf")
    best_weights = deepcopy(model.state_dict())
    best_epoch = 0

    print(f"Device: {device} | windows: train={len(train_set)}, "
          f"validation={len(valid_set)}, test={len(test_set)}")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_batches, optimizer, loss_fn, device)
        valid = evaluate(model, valid_batches, device, scale)
        if valid["mae"] < best_valid_mae:
            best_valid_mae = valid["mae"]
            best_weights = deepcopy(model.state_dict())
            best_epoch = epoch
        print(f"Epoch {epoch:02d} | train MSE={train_loss:.4f} | val MAE={valid['mae']:.4f}")

    # 先按验证集选择权重，再用测试集评估一次。
    model.load_state_dict(best_weights)
    test = evaluate(model, test_batches, device, scale)
    print(f"Best epoch: {best_epoch} | test MAE={test['mae']:.4f} "
          f"| test RMSE={test['rmse']:.4f} | persistence MAE={test['baseline_mae']:.4f}")

    # 模拟部署：在已观测序列末尾生成未来值。
    future = predict_future(model, observed, args.history, scale, device)
    print("Step  Forecast")
    for step, value in enumerate(future.tolist(), start=1):
        print(f"+{step:<4d} {value:8.4f}")


if __name__ == "__main__":
    main()
