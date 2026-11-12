"""用 PyTorch 实现教学版 Mamba-1，并将它用于多步时间序列预测。"""

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

    def selective_scan(self, u: torch.Tensor) -> torch.Tensor:
        """按时间顺序更新状态，输入和输出的形状均为 [B, L, E]。"""
        batch_size, length, channels = u.shape
        step_features, write, read = self.selective_projection(u).split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        # Δ 为 [B, L, E]，B_t 和 C_t 各为 [B, L, N]。
        step = F.softplus(self.step_projection(step_features))
        decay = -self.log_decay.exp()  # [E, N]

        # 状态 h_t 为 [B, E, N]。每个输入窗口都从零状态开始。
        state = u.new_zeros(batch_size, channels, self.d_state)
        outputs = []
        for t in range(length):
            delta_t = step[:, t, :, None]  # [B, E, 1]
            u_t = u[:, t, :, None]        # [B, E, 1]
            b_t = write[:, t, None, :]    # [B, 1, N]
            c_t = read[:, t, None, :]     # [B, 1, N]

            # Mamba-1 参考扫描：保留 exp(Δ_t A) 倍旧状态，
            # 再写入 Δ_t B_t u_t。广播后两项都为 [B, E, N]。
            retention = torch.exp(delta_t * decay)
            state = retention * state + delta_t * b_t * u_t

            # 用 C_t 读取状态，在 N 维求和，得到 [B, E]。
            outputs.append((state * c_t).sum(dim=-1))

        # 恢复时间维，并加上从 u 到输出的逐通道直连。
        return torch.stack(outputs, dim=1) + self.skip_weight * u

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        u, gate = self.input_projection(inputs).chunk(2, dim=-1)
        # Conv1d 接受 [B, E, L]。只填充左侧，所以 t 时刻的卷积
        # 最多看到 t 时刻，输出长度保持为 L。
        u = F.pad(u.transpose(1, 2), (self.d_conv - 1, 0))
        u = F.silu(self.local_conv(u).transpose(1, 2))
        mixed = self.selective_scan(u)
        return self.output_projection(mixed * F.silu(gate))


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
        # [B, L, 1] → [B, L, D]。每层先归一化，再用残差保留原有特征。
        hidden = self.input_embedding(history)
        for layer in self.layers:
            hidden = hidden + layer(hidden)

        # 因果结构让末位置汇集整个历史窗口的信息。
        # 预测头一次输出 [B, horizon]，每一列对应一个未来步长。
        last_hidden = self.output_norm(hidden[:, -1, :])
        return self.forecast_head(last_hidden)
