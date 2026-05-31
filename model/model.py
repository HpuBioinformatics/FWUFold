import torch
import torch.nn as nn
import torch.nn.functional as F

from .DWT import WaveletDWT, IDWT
from .GLCG import GLCG
from .upsample import HFWU

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

class FWUNet(nn.Module):
    CHANNELS = [32, 64, 128, 256, 512]

    def __init__(self, img_ch=17, output_ch=1, wave='haar'):
        super().__init__()
        self.dwt = WaveletDWT(wave=wave)
        C = self.CHANNELS

        self.Conv1 = ConvBlock(img_ch, C[0])
        self.hta1 = GLCG(dim=C[0])

        self.Conv2 = ConvBlock(C[0], C[1])
        self.hta2 = GLCG(dim=C[1])

        self.Conv3 = ConvBlock(C[1], C[2])
        self.hta3 = GLCG(dim=C[2])

        self.Conv4 = ConvBlock(C[2], C[3])
        self.hta4 = GLCG(dim=C[3])

        # ── Bottleneck ──
        self.convb = ConvBlock(C[3], C[4])
        self.hta_bottle = GLCG(dim=C[4])

        # ── Decoder (依靠 FusionUpsampler 的内部注意力精修) ──
        self.Up5 = HFWU(C[4], C[3], wave=wave)
        self.Up4 = HFWU(C[3], C[2], wave=wave)
        self.Up3 = HFWU(C[2], C[1], wave=wave)
        self.Up2 = HFWU(C[1], C[0], wave=wave)

        # ── Output Head ──
        self.out_head = nn.Conv2d(C[0], output_ch, 1)

        # 稀疏背景偏置初始化
        nn.init.constant_(self.out_head.bias, -4.6)

    def forward(self, x):
        # ──────── Encoder 路径 (加入残差) ────────
        c1 = self.Conv1(x)
        x1 = c1 + self.hta1(c1)  # 🌟 卷积输出 + 注意力输出
        yl1, (LH1, HL1, HH1) = self.dwt(x1)

        c2 = self.Conv2(yl1)
        x2 = c2 + self.hta2(c2)  # 🌟
        yl2, (LH2, HL2, HH2) = self.dwt(x2)

        c3 = self.Conv3(yl2)
        x3 = c3 + self.hta3(c3)  # 🌟
        yl3, (LH3, HL3, HH3) = self.dwt(x3)

        c4 = self.Conv4(yl3)
        x4 = c4 + self.hta4(c4)  # 🌟
        yl4, (LH4, HL4, HH4) = self.dwt(x4)

        # ──────── Bottleneck (加入残差) ────────
        cb = self.convb(yl4)
        d5 = cb + self.hta_bottle(cb) # 🌟 瓶颈层也保持一致的残差结构

        # ──────── Decoder 路径 ────────
        d4 = self.Up5(d5, LH4, HL4, HH4, x4)
        d3 = self.Up4(d4, LH3, HL3, HH3, x3)
        d2 = self.Up3(d3, LH2, HL2, HH2, x2)
        d1 = self.Up2(d2, LH1, HL1, HH1, x1)

        # ──────── 输出层与物理对称性约束 ────────
        out = self.out_head(d1).squeeze(1)
        out = (out + out.transpose(-1, -2)) / 2

        return out