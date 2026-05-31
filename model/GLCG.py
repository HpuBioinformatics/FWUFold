import torch
import torch.nn as nn
import torch.nn.functional as F

class GLCG(nn.Module):

    def __init__(self, dim: int, kernel_size: int = 7,
                 reduction_ratio: int = 4):
        super().__init__()
        self.dim = dim
        reduced_dim = max(dim // reduction_ratio, 4)

        # ─── 0. Pre-Norm & 降维 ─────────────────────────────────────────
        self.pre_norm = nn.GroupNorm(num_groups=min(8, dim), num_channels=dim)
        self.to_reduced = nn.Conv2d(dim, reduced_dim, 1)

        # ─── 1. 共享轴向聚合 (保持竞争逻辑) ─────────────────────────────
        self.axis_mlp = nn.Sequential(
            nn.Conv1d(reduced_dim, reduced_dim, 1),
            nn.GroupNorm(num_groups=min(4, reduced_dim), num_channels=reduced_dim),
            nn.SiLU(inplace=True),
            nn.Conv1d(reduced_dim, reduced_dim, 1)
        )

        # ─── 2. 局部纹理提取 (Depthwise + Pointwise) ────────────────────
        padding = (kernel_size - 1) // 2
        self.local_dw = nn.Sequential(
            nn.Conv2d(reduced_dim, reduced_dim, kernel_size,
                      padding=padding, groups=reduced_dim),
            nn.GroupNorm(num_groups=min(8, reduced_dim), num_channels=reduced_dim),
            nn.SiLU(inplace=True)
        )
        self.local_pw = nn.Conv2d(reduced_dim, reduced_dim, 1)

        # ─── 3. 门控机制 (🌟核心修改点：通道数翻倍🌟) ───────────────────
        # 接收 [local_feat, global_context] 的拼接特征，因此输入维度是 reduced_dim * 2
        self.gate_fn = nn.Sequential(
            nn.Conv2d(reduced_dim * 2, reduced_dim, 1),
            nn.Sigmoid()
        )

        # ─── 4. 相对距离偏置 ─────────────────────────────────────────────
        # self.dist_bias = DynamicDistanceBias(dim=reduced_dim, max_seq_len=max_seq_len)

        # ─── 5. 输出投影 (Zero-init 保证残差稳定性) ─────────────────────
        self.final_expand = nn.Conv2d(reduced_dim, dim, 1)
        nn.init.zeros_(self.final_expand.weight)
        nn.init.zeros_(self.final_expand.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  [B, C, L, L]
        Output: [B, C, L, L] (Residual Delta)
        """
        B, C, L, _ = x.shape

        # Step 0: Pre-Norm & 通道降维
        h = self.pre_norm(x)
        x_red = self.to_reduced(h)  # [B, C', L, L]

        # Step 1: 竞争性池化 (结合 Mean 和 Max)
        # 获取行信号
        r_max, _ = x_red.max(dim=3)  # [B, C', L]
        r_mean = x_red.mean(dim=3)  # [B, C', L]
        row_agg = self.axis_mlp(r_max + r_mean).unsqueeze(3)  # [B, C', L, 1]

        # 获取列信号
        c_max, _ = x_red.max(dim=2)  # [B, C', L]
        c_mean = x_red.mean(dim=2)  # [B, C', L]
        col_agg = self.axis_mlp(c_max + c_mean).unsqueeze(2)  # [B, C', 1, L]

        # Step 2: 局部纹理提取
        local_feat = self.local_pw(self.local_dw(x_red))  # [B, C', L, L]

        # Step 3: 🌟 对比压制门控融合 (核心逻辑重构) 🌟
        # 3.1 提取全局上下文 (通过广播机制天然构成矩阵)
        global_context = row_agg + col_agg  # [B, C', L, L]

        # 3.2 局部特征与全局信号发生物理碰撞 (Concat)
        gate_input = torch.cat([local_feat, global_context], dim=1)  # [B, 2*C', L, L]

        # 3.3 网络自行决定惩罚力度：真配对给绿灯，假阳性(多余对角线)给红灯
        gate = self.gate_fn(gate_input)  # [B, C', L, L]

        # 3.4 施加软压制
        fused = local_feat * gate  # [B, C', L, L]

        # Step 4: 注入相对距离偏置
        # fused = fused + self.dist_bias(L, x.device)  # [B, C', L, L]

        # Step 5: 映射回原通道
        delta = self.final_expand(fused)  # [B, C, L, L]

        # 注：为了保持梯度的顺畅下降，不强制执行硬对称约束，
        # 交给网络自己通过对比压制门控去消除非对称噪音。
        return delta