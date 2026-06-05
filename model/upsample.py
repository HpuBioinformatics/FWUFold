import torch
import torch.nn as nn
import torch.nn.functional as F
from .DWT import IDWT


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


class HFWU(nn.Module):
    def __init__(self, in_ch, out_ch, wave='haar'):
        super().__init__()
        mid_ch = in_ch // 2

        self.pixel_upsample = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch * 4, 1, bias=False),
            nn.PixelShuffle(2),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(inplace=True)
        )

        self.idwt = IDWT(wave=wave)

        self.fusion = FusionInteract(out_ch)

        self.skip_w = nn.Parameter(torch.FloatTensor([0.0]), requires_grad=True)

        self.post_conv = ConvBlock(out_ch, out_ch)

    def forward(self, x, LH, HL, HH, skip):
        feat_pix, feat_detail = x.chunk(2, dim=1)

        pix_out = self.pixel_upsample(feat_pix)

        detail_out = self.idwt(feat_detail, LH, HL, HH)

        fused_feat = self.fusion(pix_out, detail_out)

        alpha = torch.sigmoid(self.skip_w)
        
        combined = (alpha * fused_feat) + ((1.0 - alpha) * skip)

        out = self.post_conv(combined)

        return out
