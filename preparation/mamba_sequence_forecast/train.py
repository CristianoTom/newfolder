"""生成单变量时间序列，训练 Mamba，评估并输出未来多步预测。"""

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import MambaForecaster


def make_series(length: int, seed: int) -> torch.Tensor:
    """生成含两种周期、缓慢变化的振幅以及少量噪声的合成数据。"""
    # time 是从 0 开始的时间索引。长度为 720 时，最后一个索引是 719。
    time = torch.arange(length, dtype=torch.float32)
    # 主波每 48 步重复一次。振幅本身也会缓慢变化。
    amplitude = 1.0 + 0.15 * torch.sin(2 * torch.pi * time / 240)
    main_wave = amplitude * torch.sin(2 * torch.pi * time / 48)
    # 叠加周期更短的小波，以及用固定随机种子生成的噪声。
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
        mean = training_values.mean().item()
        # 常数序列的标准差为 0，设置下限以避免除以 0。
        std = max(training_values.std(unbiased=False).item(), 1e-6)
        return cls(mean=mean, std=std)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        """标准化：例如均值为 10、标准差为 2，原始值 14 会变成 2。"""
        return (values - self.mean) / self.std

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        """反标准化：把模型输出恢复成原始数据的尺度。"""
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

        # forecast_start 是“第一个要预测的点”的索引。
        # 例如目标段是 [10, 16)，预测 3 步，则起点可以是 10、11、12、13。
        # 还要确保起点左侧有足够的历史，所以起点至少是 history_length。
        first_start = max(target_start, history_length)
        last_start = target_end - horizon
        self.forecast_starts = range(first_start, last_start + 1, stride)
        if len(self.forecast_starts) == 0:
            raise ValueError("当前数据段无法构造样本，请缩短历史窗口或预测步数。")

    def __len__(self) -> int:
        # len(dataset) 会调用这个方法，返回可用窗口的数量。
        return len(self.forecast_starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # dataset[index] 会调用这个方法。Python 切片包含左端，不包含右端。
        forecast_start = self.forecast_starts[index]
        history_start = forecast_start - self.history_length
        forecast_end = forecast_start + self.horizon
        history = self.values[history_start:forecast_start]
        target = self.values[forecast_start:forecast_end]
        # 单个样本：[历史长度, 1] 和 [预测步数]。DataLoader 会增加批量维。
        return history.unsqueeze(-1), target


def train_one_epoch(
    model: MambaForecaster,
    batches: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """返回整个训练集上按样本数加权的平均损失。"""
    model.train()
    total_loss = 0.0
    sample_count = 0
    for history, target in batches:
        # 默认批量大小为 32。history 通常是 [32, 32, 1]，target 是 [32, 8]。
        # 最后一批可能小于 32，后面用 target.size(0) 得到实际大小。
        history = history.to(device)
        target = target.to(device)

        # 清空上一批的梯度，得到这一批的预测和误差。
        optimizer.zero_grad(set_to_none=True)
        prediction = model(history)
        loss = loss_fn(prediction, target)

        # backward 计算梯度，裁剪限制梯度大小，step 根据梯度更新参数。
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * target.size(0)
        sample_count += target.size(0)
    return total_loss / sample_count


@torch.inference_mode()
def evaluate(
    model: MambaForecaster,
    batches: DataLoader,
    device: torch.device,
    scale: Standardizer,
) -> dict[str, float]:
    """以原始量纲计算 MAE、RMSE，以及持久性基线的 MAE。"""
    model.eval()
    absolute_error = 0.0
    squared_error = 0.0
    baseline_error = 0.0
    point_count = 0
    for history, target in batches:
        history = history.to(device)
        target = target.to(device)
        prediction = scale.inverse_transform(model(history))
        actual = scale.inverse_transform(target)
        error = prediction - actual
        # [:, -1, :] 表示取所有窗口的最后一个历史点，形状为 [批量, 1]。
        # 与 [批量, 预测步数] 的真实值相减时，每行的最后一个历史值会被重复使用。
        last_observation = scale.inverse_transform(history[:, -1, :])
        absolute_error += error.abs().sum().item()
        squared_error += error.square().sum().item()
        baseline_error += (last_observation - actual).abs().sum().item()
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
    # 只取末尾的历史窗口，标准化后再加两个维度。
    # 第一个 1 表示只有一个样本，最后一个 1 表示每个时间点只有一个数值。
    history = observed[-history_length:].to(device=device, dtype=torch.float32)
    normalized_history = scale.transform(history)
    model_input = normalized_history.reshape(1, history_length, 1)
    forecast = model(model_input)[0]  # 从 [1, 预测步数] 中取出唯一一行。
    return scale.inverse_transform(forecast).cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15, help="训练轮数")
    parser.add_argument("--history", type=int, default=32, help="历史窗口长度")
    parser.add_argument("--horizon", type=int, default=8, help="一次预测的未来步数")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="图表和绘图数据的保存位置（默认是脚本旁的 outputs 目录）",
    )
    args = parser.parse_args()
    if min(args.epochs, args.history, args.horizon) < 1:
        parser.error("--epochs、--history 和 --horizon 必须为正整数")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("当前 PyTorch 环境无法使用 CUDA")

    # 第 1 步：生成数据，确定训练、验证和测试的时间边界。
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

    # 第 2 步：把连续序列变成一组 (历史窗口, 未来真值) 样本。
    scale = Standardizer.fit(observed[:train_end])
    normalized = scale.transform(observed)
    train_set = ForecastWindows(normalized, args.history, args.horizon, 0, train_end, stride=2)
    valid_set = ForecastWindows(normalized, args.history, args.horizon, train_end, valid_end)
    test_set = ForecastWindows(normalized, args.history, args.horizon, valid_end, len(observed))

    # 打乱的是训练窗口的顺序，每个窗口内部仍按时间排列。
    train_batches = DataLoader(train_set, batch_size=32, shuffle=True)
    valid_batches = DataLoader(valid_set, batch_size=32)
    test_batches = DataLoader(test_set, batch_size=32)

    # 第 3 步：重复训练，每轮用验证集判断是否保留这一轮的模型。
    model = MambaForecaster(horizon=args.horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.MSELoss()
    best_valid_mae = float("inf")
    best_weights = deepcopy(model.state_dict())
    best_epoch = 0
    training_losses = []
    validation_errors = []

    print(
        f"Device: {device} | windows: train={len(train_set)}, "
        f"validation={len(valid_set)}, test={len(test_set)}"
    )
    example_history, example_target = train_set[0]
    print(f"One window: history={tuple(example_history.shape)}, target={tuple(example_target.shape)}")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_batches, optimizer, loss_fn, device)
        valid = evaluate(model, valid_batches, device, scale)
        training_losses.append(train_loss)
        validation_errors.append(valid["mae"])
        if valid["mae"] < best_valid_mae:
            best_valid_mae = valid["mae"]
            # 复制参数张量，避免接下来的训练改动已经保留的权重。
            best_weights = deepcopy(model.state_dict())
            best_epoch = epoch
        print(f"Epoch {epoch:02d} | train MSE={train_loss:.4f} | val MAE={valid['mae']:.4f}")

    # 第 4 步：加载验证集表现最佳的权重，汇报测试集误差。
    model.load_state_dict(best_weights)
    test = evaluate(model, test_batches, device, scale)
    print(f"Best epoch: {best_epoch} | test MAE={test['mae']:.4f} "
          f"| test RMSE={test['rmse']:.4f} | persistence MAE={test['baseline_mae']:.4f}")

    # 第 5 步：演示两种预测。第一个测试窗口有真实值，可用来画对比图。
    # 将序列截断在预测起点，保证 predict_future 只看到起点之前的历史。
    test_start = test_set.forecast_starts[0]
    test_forecast = predict_future(model, observed[:test_start], args.history, scale, device)
    # 序列末尾之后的预测模拟实际使用，此时未来真值还没有观测到。
    future = predict_future(model, observed, args.history, scale, device)
    print("Step  Forecast")
    for step, value in enumerate(future.tolist(), start=1):
        print(f"+{step:<4d} {value:8.4f}")

    # 第 6 步：把张量转成普通列表，保存绘图所需的数据。
    # 将数据保存在 JSON 中，可以只重画图表，不必每次都重新训练。
    plot_data = {
        "observed": observed.tolist(),
        "train_end": train_end,
        "validation_end": valid_end,
        "history_length": args.history,
        "test_start": test_start,
        "test_forecast": test_forecast.tolist(),
        "future_forecast": future.tolist(),
        "training_losses": training_losses,
        "validation_errors": validation_errors,
        "best_epoch": best_epoch,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_file = args.output_dir / "plot_data.json"
    data_file.write_text(json.dumps(plot_data, indent=2), encoding="utf-8")

    # 用同一个 Python 解释器单独启动绘图脚本。
    # 这样 PyTorch 和 Matplotlib 各自在自己的进程中加载依赖。
    plot_script = Path(__file__).with_name("plots.py")
    subprocess.run([sys.executable, str(plot_script), str(data_file)], check=True)


if __name__ == "__main__":
    main()
