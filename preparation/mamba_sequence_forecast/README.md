# 用 Mamba 做时间序列预测

本示例用一段带噪声的合成序列训练 Mamba-1。默认给模型连续 **32 个历史观测值**，让它一次输出**未来 8 个预测值**。Mamba 的选择性扫描由 PyTorch 张量运算和时间循环实现，安装 PyTorch 后即可在 CPU 上运行。

## 运行

在 `newfolder` 根目录执行。已有 PyTorch 时可跳过安装命令。

```powershell
python -m pip install torch
python preparation/mamba_sequence_forecast/train.py
```

也可以调整历史窗口、预测步数和训练轮数。

```powershell
python preparation/mamba_sequence_forecast/train.py --epochs 20 --history 48 --horizon 12
```

脚本逐轮打印训练 MSE 和验证 MAE。训练 MSE 在标准化后的量纲上计算。验证 MAE、测试 MAE 和 RMSE 使用原始量纲，越低表示误差越小。测试结果同时给出持久性基线的 MAE，该基线把最后一个已知值作为所有未来步的预测。最后的 `+1` 到 `+8` 是序列结束后的未来预测。

## 阅读代码

1. [train.py](train.py) 的 `ForecastWindows` 用滑动窗口构造样本。例如，从索引 `0–31` 取历史，预测 `32–39`。训练、验证和测试按目标时间分段，标准化统计量只从训练段计算。评估时，历史窗口可以使用预测起点之前已观测到的数据，目标全部位于各自的数据段。
2. [model.py](model.py) 的 `MambaMixer` 先用因果卷积提取局部特征。每个时间步再根据当前特征计算步长 `Δ`、写入向量 `B_t` 和读取向量 `C_t`，用它们更新和读取状态。代码在每步旁边标注了张量形状。
3. `MambaForecaster` 将最后一个历史位置的特征交给线性预测头。模型输入形状为 `[批量大小, 历史长度, 1]`，输出形状为 `[批量大小, 预测步数]`。训练结束后，`predict_future` 负责截取最新历史、标准化输入和反标准化输出。

如需使用自己的单变量数据，将 `main()` 中的 `make_series(...)` 换成按时间升序排列、等间隔采样的一维 `torch.float32` 张量，随后运行同一套训练和预测流程。

选择性状态空间模型和状态递推公式参考 [Mamba 论文](https://arxiv.org/abs/2312.00752) 与[官方选择性扫描实现](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py)。
