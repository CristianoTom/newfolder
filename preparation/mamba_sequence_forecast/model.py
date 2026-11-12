"""用 PyTorch 实现教学版 Mamba-1，并将它用于多步时间序列预测。

第一次阅读建议先看文件末尾的 MambaForecaster.forward。
它串起了整个预测流程，再回来看 MambaMixer 如何处理一个历史窗口。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class MambaMixer(nn.Module):
    """因果局部卷积、选择性状态空间模型（SSM）和输出门控。

    形状记号：B 是批量大小，L 是序列长度，D 是模型宽度，
    E 是扩展后的通道数，N 是每个通道的状态大小。
    输入与输出均为 [B, L, D]，方便在外层做残差连接。
    """

    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.d_conv = d_conv

        # 一次线性投影得到内容分支 u 和门控分支 z。
        self.input_projection = nn.Linear(d_model, 2 * d_inner, bias=False)
        # groups 等于通道数，每个通道独立提取短期时间特征。
        self.local_conv = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv, groups=d_inner
        )
        # 从当前输入生成 Δ 的低秩表示，以及状态写入和读取向量 B_t、C_t。
        self.selective_projection = nn.Linear(
            d_inner, self.dt_rank + 2 * d_state, bias=False
        )
        self.step_projection = nn.Linear(self.dt_rank, d_inner)

        # 初始化时让 softplus(bias) 约落在 [0.001, 0.1]。
        # 在对数空间采样，可使不同通道从不同的时间尺度开始学习。
        scale = self.dt_rank ** -0.5
        nn.init.uniform_(self.step_projection.weight, -scale, scale)
        with torch.no_grad():
            initial_step = torch.empty(d_inner).uniform_(math.log(0.001), math.log(0.1)).exp()
            inverse_softplus = initial_step + torch.log(-torch.expm1(-initial_step))
            self.step_projection.bias.copy_(inverse_softplus)

        # A = -exp(log_decay)，让每个状态的连续时间衰减率保持为负。
        initial_rates = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.log_decay = nn.Parameter(initial_rates.log().repeat(d_inner, 1))
        # 公式中的 D_skip 是输入直连权重，形状为 [E]。
        self.skip_weight = nn.Parameter(torch.ones(d_inner))
        self.output_projection = nn.Linear(d_inner, d_model, bias=False)

    def selective_scan(self, features: torch.Tensor) -> torch.Tensor:
        """逐时间步更新记忆，输入和输出均为 [B, L, E]。

        可以先把每个通道的 N 个状态想成 N 个记忆槽。每一步先保留一部分
        旧状态，再写入当前特征，然后用读取权重把 N 个状态合成一个输出。
        """
        batch_size, sequence_length, channels = features.shape
        # Linear 只作用在最后一维，因此每个样本和时间步都会得到一组参数。
        parameters = self.selective_projection(features)
        parameter_sizes = [self.dt_rank, self.d_state, self.d_state]
        step_features, write_weights, read_weights = parameters.split(parameter_sizes, dim=-1)

        # softplus 把步长 Δ 变为正数：[B, L, E]。
        # 写入权重 B_t 和读取权重 C_t 都是 [B, L, N]。
        step_sizes = F.softplus(self.step_projection(step_features))
        decay_rates = -self.log_decay.exp()  # [E, N]，所有衰减率都为负数。

        # 状态 h_t 为 [B, E, N]。每个输入窗口都从零状态开始。
        state = features.new_zeros((batch_size, channels, self.d_state))
        step_outputs = []
        for time_index in range(sequence_length):
            # 冒号 : 表示取这一维的所有元素。unsqueeze 增加一个长度为 1 的维度。
            step_size = step_sizes[:, time_index, :].unsqueeze(-1)       # [B, E, 1]
            current_input = features[:, time_index, :].unsqueeze(-1)    # [B, E, 1]
            write_weight = write_weights[:, time_index, :].unsqueeze(1) # [B, 1, N]
            read_weight = read_weights[:, time_index, :].unsqueeze(1)   # [B, 1, N]

            # 形状为 1 的维度会自动对齐（广播），计算结果为 [B, E, N]。
            # 因为 Δ > 0 且 A < 0，保留比例 exp(Δ × A) 介于 0 和 1 之间。
            retention = torch.exp(step_size * decay_rates)
            retained_state = retention * state
            new_information = step_size * write_weight * current_input
            state = retained_state + new_information

            # 将 N 个状态乘以对应读取权重，再沿状态维求和，得到 [B, E]。
            current_output = (state * read_weight).sum(dim=-1)
            step_outputs.append(current_output)

        # 恢复时间维，并加上从 u 到输出的逐通道直连。
        sequence_output = torch.stack(step_outputs, dim=1)  # [B, L, E]
        return sequence_output + self.skip_weight * features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(inputs)  # [B, L, 2E]
        features, gate = projected.chunk(2, dim=-1)  # 各为 [B, L, E]
        # Conv1d 接受 [B, E, L]。只填充左侧，所以 t 时刻的卷积
        # 最多看到 t 时刻，输出长度保持为 L。
        channels_first = features.transpose(1, 2)
        padded = F.pad(channels_first, (self.d_conv - 1, 0))
        local_features = self.local_conv(padded).transpose(1, 2)
        mixed = self.selective_scan(F.silu(local_features))

        # 门控分支按时间、按通道调节 SSM 输出的大小。
        gated_output = mixed * F.silu(gate)
        return self.output_projection(gated_output)


class MambaForecaster(nn.Module):
    """用一个历史窗口直接预测未来 horizon 个数值。"""

    def __init__(self, horizon: int, d_model: int = 24, num_layers: int = 2):
        super().__init__()
        self.input_embedding = nn.Linear(1, d_model)
        self.layers = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(d_model), MambaMixer(d_model))
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.forecast_head = nn.Linear(d_model, horizon)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # 每个时间点原本只有一个数，把它映射成 D 个特征：[B, L, 1] → [B, L, D]。
        hidden = self.input_embedding(history)
        for layer in self.layers:
            # layer 由 LayerNorm 和 MambaMixer 组成，处理前后的形状相同。
            # 把处理结果加回原特征，这一步叫残差连接。
            processed = layer(hidden)
            hidden = hidden + processed

        # 因果结构让末位置汇集整个历史窗口的信息。
        # 预测头一次输出 [B, horizon]，每一列对应一个未来步长。
        last_position = hidden[:, -1, :]  # [B, D]，-1 表示序列的最后一个位置。
        last_hidden = self.output_norm(last_position)
        return self.forecast_head(last_hidden)
