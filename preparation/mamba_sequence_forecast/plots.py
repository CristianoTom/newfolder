"""用 Matplotlib 把原始序列、训练误差和预测结果保存为 PNG。"""

import argparse
import json
from pathlib import Path

import matplotlib

# Agg 可以直接写 PNG，在服务器或没有图形界面的终端也能运行。
matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def save_series_plot(
    observed: list[float], train_end: int, validation_end: int, output_dir: Path
) -> None:
    """用背景色和分界线标出训练、验证、测试各自的时间段。"""
    figure, axis = plt.subplots(figsize=(11, 3.4), layout="constrained")
    total_points = len(observed)
    axis.plot(range(total_points), observed, color="#27364a", linewidth=1)

    # 点落在整数时间索引上。边界画在相邻两个点之间，如 431 和 432 之间。
    axis.axvspan(-0.5, train_end - 0.5, color="#c4e5ee", alpha=0.55, label="Training")
    axis.axvspan(
        train_end - 0.5, validation_end - 0.5,
        color="#f6dfb5", alpha=0.65, label="Validation",
    )
    axis.axvspan(
        validation_end - 0.5, total_points - 0.5,
        color="#dcd9f2", alpha=0.65, label="Test",
    )
    axis.set_xlim(-0.5, total_points - 0.5)
    axis.set(title="Synthetic time series and data splits", xlabel="Time index", ylabel="Value")
    axis.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.10), frameon=False)
    figure.savefig(output_dir / "01_data_splits.png", dpi=150)
    plt.close(figure)


def save_learning_curves(
    training_losses: list[float],
    validation_errors: list[float],
    best_epoch: int,
    output_dir: Path,
) -> None:
    """MSE 与 MAE 的量纲不同，分别绘图便于阅读各自的变化趋势。"""
    epochs = range(1, len(training_losses) + 1)
    figure, (training_axis, validation_axis) = plt.subplots(
        1, 2, figsize=(11, 3.6), layout="constrained"
    )
    training_axis.plot(epochs, training_losses, "o-", color="#185b82")
    training_axis.set(title="Training loss", xlabel="Epoch", ylabel="MSE (normalized data)")
    validation_axis.plot(epochs, validation_errors, "o-", color="#177561")
    validation_axis.set(title="Validation error", xlabel="Epoch", ylabel="MAE (original units)")
    if best_epoch > 0:
        validation_axis.axvline(
            best_epoch, color="#515461", linestyle="--", label=f"Best epoch: {best_epoch}"
        )
        validation_axis.legend()

    for axis in (training_axis, validation_axis):
        # 防止刻度显示 1.5 之类不存在的轮次。
        axis.set_xticks(list(epochs) if len(epochs) <= 20 else range(5, len(epochs) + 1, 5))
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_dir / "02_learning_curves.png", dpi=150)
    plt.close(figure)


def draw_forecast(
    axis: Axes,
    observed: list[float],
    history_length: int,
    forecast_start: int,
    forecast: list[float],
) -> None:
    """在同一横轴上对齐输入窗口与预测点，便于比较它们的位置。"""
    history_start = forecast_start - history_length
    history_times = range(history_start, forecast_start)
    forecast_times = range(forecast_start, forecast_start + len(forecast))
    # 历史曲线截止于预测起点之前；橙线从第一个预测点开始。
    axis.plot(history_times, observed[history_start:forecast_start],
              color="#27364a", label="Input history")
    axis.plot(forecast_times, forecast, "o--", color="#a94f14",
              markersize=4, label="Mamba forecast")
    axis.axvline(forecast_start - 0.5, color="#777b85", linestyle=":", label="Forecast start")
    axis.set(xlabel="Time index", ylabel="Value")
    axis.grid(axis="y", alpha=0.25)


def save_forecast_plot(
    observed: list[float],
    history_length: int,
    test_start: int,
    test_forecast: list[float],
    future_forecast: list[float],
    output_dir: Path,
) -> None:
    """上图对比测试集真值，下图展示序列末尾之后的预测。"""
    figure, (test_axis, future_axis) = plt.subplots(
        2, 1, figsize=(10, 6.6), layout="constrained"
    )
    draw_forecast(test_axis, observed, history_length, test_start, test_forecast)
    test_end = test_start + len(test_forecast)
    test_axis.plot(
        range(test_start, test_end), observed[test_start:test_end],
        "s-", color="#177561", markersize=4, label="Actual",
    )
    test_axis.set_title("First test window: forecast vs. actual")
    test_axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=4, frameon=False)

    draw_forecast(future_axis, observed, history_length, len(observed), future_forecast)
    future_axis.set_title("Forecast beyond the observed series")
    future_axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False)
    figure.savefig(output_dir / "03_forecast.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_file", type=Path, help="train.py 生成的 plot_data.json 路径")
    args = parser.parse_args()

    # 从 JSON 读出的字典里取出各张图所需的数据，图片保存到 JSON 所在目录。
    data = json.loads(args.data_file.read_text(encoding="utf-8"))
    output_dir = args.data_file.parent
    save_series_plot(data["observed"], data["train_end"], data["validation_end"], output_dir)
    save_learning_curves(
        data["training_losses"], data["validation_errors"], data["best_epoch"], output_dir
    )
    save_forecast_plot(
        data["observed"], data["history_length"], data["test_start"],
        data["test_forecast"], data["future_forecast"], output_dir,
    )
    print(f"Plots saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
