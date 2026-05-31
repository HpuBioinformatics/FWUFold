import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse

class WaveletDWT(nn.Module):
    def __init__(self, wave='haar', J=1):
        super().__init__()
        self.dwt = DWTForward(J=J, wave=wave)

    def forward(self, x):
        # 进行小波分解
        yl, yh = self.dwt(x)  # yl: 低频, yh: 高频(列表)
        yh = yh[0]             # 取第1层高频 (B, C, 3, H/2, W/2)

        LL = yl                # 低频部分
        LH = yh[:, :, 0]       # 高频方向1
        HL = yh[:, :, 1]       # 高频方向2
        HH = yh[:, :, 2]       # 高频方向3

        # 返回低频和高频 (三方向)
        return LL, (LH, HL, HH)


class IDWT(nn.Module):
    """小波逆变换模块：输入低频(LL)和高频(LH, HL, HH)，重建原图像"""

    def __init__(self, wave='haar'):
        super().__init__()
        self.idwt = DWTInverse(wave=wave)

    def forward(self, LL, LH, HL, HH):
        """
        参数:
            LL: 低频分量 (B, C, H/2, W/2)
            LH, HL, HH: 高频分量 (B, C, H/2, W/2)
        返回:
            x_rec: 重建后的图像 (B, C, H, W)
        """
        # 按 pytorch_wavelets 的格式组装高频张量
        yh = torch.stack([LH, HL, HH], dim=2)  # -> (B, C, 3, H/2, W/2)
        yl = LL  # 低频部分

        # 调用官方逆变换
        x_rec = self.idwt((yl, [yh]))
        return x_rec

if __name__ == "__main__":
    # model = WaveletDWT()
    # x = torch.randn(1, 3, 128, 128)
    # LL, (LH, HL, HH) = model(x)

    dwt = DWTForward(J=1, wave='haar')
    idwt = IDWT(wave='haar')

    x = torch.randn(1, 3, 128, 128)
    yl, yh = dwt(x)
    yh = yh[0]                # (B, C, 3, H/2, W/2)
    LL = yl
    LH, HL, HH = yh[:, :, 0], yh[:, :, 1], yh[:, :, 2]

    # 重建
    x_rec = idwt(LL, LH, HL, HH)

    print("Original shape:", x.shape)
    print("Reconstructed shape:", x_rec.shape)
    print("Reconstruction error:", torch.mean((x - x_rec)**2).item())
