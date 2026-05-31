import torch
import torch.nn as nn
import torch.nn.functional as F
from .DWT import IDWT


# ==========================================
# 1. ConvBlock
# ==========================================
class ConvBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, ch_out), ch_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, ch_out), ch_out),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# ==========================================
# 2. FusionInteract（通道 + 空间 + Conv融合）
# ==========================================
class FusionInteract(nn.Module):
    def __init__(self, dim):
        super().__init__()

        hidden = max(dim // 8, 4)

        # -------- Channel Attention --------
        self.channel_attn = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, dim * 2, 1, bias=False),
            nn.Sigmoid()
        )

        # -------- Spatial Attention --------
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, feat_pix, feat_detail):
        # concat
        x = torch.cat([feat_pix, feat_detail], dim=1)

        # -------- Channel Attention --------
        ca = self.channel_attn(
            F.adaptive_avg_pool2d(x, 1) +
            F.adaptive_max_pool2d(x, 1)
        )
        x = x * ca

        # -------- Spatial Attention --------
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attn(torch.cat([avg_map, max_map], dim=1))
        x = x * sa

        # -------- Learnable Fusion --------
        pix, detail = x.chunk(2, dim=1)
        return pix+detail


# ==========================================
# 3. WaveletAware_FusionUpsampler（可学习参数跳跃版）
# ==========================================
class HFWU(nn.Module):
    def __init__(self, in_ch, out_ch, wave='haar'):
        super().__init__()
        mid_ch = in_ch // 2

        # -------------------------
        # Pixel 分支（语义路径）
        # -------------------------
        self.pixel_upsample = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch * 4, 1, bias=False),
            nn.PixelShuffle(2),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(inplace=True)
        )

        # -------------------------
        # IDWT 分支（物理路径）
        # -------------------------
        self.idwt = IDWT(wave=wave)

        # -------------------------
        # 内部融合模块（处理 pix 和 detail）
        # -------------------------
        self.fusion = FusionInteract(out_ch)

        # -------------------------
        # 🌟 可学习参数跳跃连接
        # -------------------------
        # 初始化为 0.0，经过 sigmoid 后刚好是 0.5（表示 1:1 的公平初始比例）
        self.skip_w = nn.Parameter(torch.FloatTensor([0.0]), requires_grad=True)

        # -------------------------
        # 输出精修
        # -------------------------
        self.post_conv = ConvBlock(out_ch, out_ch)

    def forward(self, x, LH, HL, HH, skip):
        # 1. split
        feat_pix, feat_detail = x.chunk(2, dim=1)

        # 2. Pixel 分支放大
        pix_out = self.pixel_upsample(feat_pix)

        # 3. IDWT 分支重构
        detail_out = self.idwt(feat_detail, LH, HL, HH)

        # 4. 主特征融合
        fused_feat = self.fusion(pix_out, detail_out)

        # 5. 🌟 可学习跳跃连接 (Learnable Skip Connection)
        # 将无界的参数 w 映射到 [0, 1] 区间
        alpha = torch.sigmoid(self.skip_w)
        # 按比例混合：主干特征 + 跳跃特征
        combined = (alpha * fused_feat) + ((1.0 - alpha) * skip)

        # 6. 精修与残差输出
        out = self.post_conv(combined)

        return out