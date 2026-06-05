import torch
import torch.nn as nn
import torch.nn.functional as F

class GLCG(nn.Module):

    def __init__(self, dim: int, kernel_size: int = 7,
                 reduction_ratio: int = 4):
        super().__init__()
        self.dim = dim
        reduced_dim = max(dim // reduction_ratio, 4)

        self.pre_norm = nn.GroupNorm(num_groups=min(8, dim), num_channels=dim)
        self.to_reduced = nn.Conv2d(dim, reduced_dim, 1)

        self.axis_mlp = nn.Sequential(
            nn.Conv1d(reduced_dim, reduced_dim, 1),
            nn.GroupNorm(num_groups=min(4, reduced_dim), num_channels=reduced_dim),
            nn.SiLU(inplace=True),
            nn.Conv1d(reduced_dim, reduced_dim, 1)
        )

        padding = (kernel_size - 1) // 2
        self.local_dw = nn.Sequential(
            nn.Conv2d(reduced_dim, reduced_dim, kernel_size,
                      padding=padding, groups=reduced_dim),
            nn.GroupNorm(num_groups=min(8, reduced_dim), num_channels=reduced_dim),
            nn.SiLU(inplace=True)
        )
        self.local_pw = nn.Conv2d(reduced_dim, reduced_dim, 1)

        self.gate_fn = nn.Sequential(
            nn.Conv2d(reduced_dim * 2, reduced_dim, 1),
            nn.Sigmoid()
        )

        self.final_expand = nn.Conv2d(reduced_dim, dim, 1)
        nn.init.zeros_(self.final_expand.weight)
        nn.init.zeros_(self.final_expand.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  [B, C, L, L]
        Output: [B, C, L, L] (Residual Delta)
        """
        B, C, L, _ = x.shape

        h = self.pre_norm(x)
        x_red = self.to_reduced(h)  # [B, C', L, L]

        r_max, _ = x_red.max(dim=3)  # [B, C', L]
        r_mean = x_red.mean(dim=3)  # [B, C', L]
        row_agg = self.axis_mlp(r_max + r_mean).unsqueeze(3)  # [B, C', L, 1]

        c_max, _ = x_red.max(dim=2)  # [B, C', L]
        c_mean = x_red.mean(dim=2)  # [B, C', L]
        col_agg = self.axis_mlp(c_max + c_mean).unsqueeze(2)  # [B, C', 1, L]

        local_feat = self.local_pw(self.local_dw(x_red))  # [B, C', L, L]

        global_context = row_agg + col_agg  # [B, C', L, L]

        gate_input = torch.cat([local_feat, global_context], dim=1)  # [B, 2*C', L, L]

        gate = self.gate_fn(gate_input)  # [B, C', L, L]

        fused = local_feat * gate  # [B, C', L, L]

        delta = self.final_expand(fused)  # [B, C, L, L]

        return delta
