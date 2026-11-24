"""球场线语义分割网络：轻量级 Encoder-Decoder。

检测 6 类球场线：
  0 - 背景
  1 - 边线（ sideline + baseline）
  2 - 三分线
  3 - 罚球区/两分线
  4 - 中线
  5 - 中圈

架构：
  Encoder: MobileNetV3-Small（ImageNet 预训练）
  Decoder: 逐级上采样 + 卷积融合
  输入: [B, 3, 384, 384]
  输出: [B, 6, 384, 384] 语义分割热图

为什么用线分割而不是关键点：
  广播视角下球场角点经常被裁掉或遮挡，但线段的任意一小段都能提供
  强几何约束。6 类线 × 每类若干段 = 远多于 8 个关键点的约束量，
  而且对遮挡和视角变化的鲁棒性高得多。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small

NUM_CLASSES = 6
CLASS_NAMES = ["background", "boundary", "three_point", "paint", "center_line", "center_circle"]
INPUT_SIZE = 384


class DecoderBlock(nn.Module):
    """上采样 + skip connection 融合。"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.up(x)))
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class CourtLineSegNet(nn.Module):
    """球场线语义分割网络。"""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        backbone = mobilenet_v3_small(
            weights="DEFAULT" if pretrained else None
        )
        features = list(backbone.features.children())

        # MobileNetV3-Small 实际 stride/通道分布（输入 384x384）：
        # Layer  0: stride  2, ch  16   (192x192)
        # Layer  1: stride  4, ch  16   ( 96x 96)
        # Layer 2-3: stride 8, ch  24   ( 48x 48)
        # Layer 4-8: stride 16, ch 48   ( 24x 24)
        # Layer 9-12: stride 32, ch 576 ( 12x 12)

        self.enc1 = nn.Sequential(*features[:2])   # stride 4, ch 16
        self.enc2 = nn.Sequential(*features[2:4])  # stride 8, ch 24
        self.enc3 = nn.Sequential(*features[4:9])  # stride 16, ch 48
        self.enc4 = nn.Sequential(*features[9:13]) # stride 32, ch 576

        # Decoder: 逐级上采样 + skip fusion
        self.dec3 = DecoderBlock(576, 48, 64)
        self.dec2 = DecoderBlock(64, 24, 48)
        self.dec1 = DecoderBlock(48, 16, 32)

        # 最终分类头：上采样到原始分辨率
        self.cls = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] 归一化图像
        Returns:
            logits: [B, num_classes, H, W]
        """
        input_size = x.shape[2:]

        s1 = self.enc1(x)    # [B, 16, H/2, W/2]
        s2 = self.enc2(s1)   # [B, 24, H/4, W/4]
        s3 = self.enc3(s2)   # [B, 48, H/8, W/8]
        s4 = self.enc4(s3)   # [B, 96, H/16, W/16]

        d3 = self.dec3(s4, s3)  # [B, 64, H/8, W/8]
        d2 = self.dec2(d3, s2)  # [B, 48, H/4, W/4]
        d1 = self.dec1(d2, s1)  # [B, 32, H/2, W/2]

        logits = self.cls(d1)   # [B, 16, H, W] -> [B, num_classes, H, W]

        if logits.shape[2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

        return logits


if __name__ == "__main__":
    model = CourtLineSegNet(num_classes=NUM_CLASSES, pretrained=False)
    n = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n / 1e6:.2f}M")
    print(f"Size (FP32): {n * 4 / 1e6:.2f}MB")

    x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
