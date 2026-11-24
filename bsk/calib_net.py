"""轻量级球场标定网络：MobileNetV3 + Keypoint Head。

设计目标：
- 4-8GB GPU 实时推理（30+ FPS）
- 模型大小 < 15MB
- 重投影误差 < 5px

架构：
- Backbone: MobileNetV3-Small（预训练 ImageNet）
- Head: 关键点热图回归（高斯热图）
- 输出: N 个球场关键点的像素坐标
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small


class KeypointHead(nn.Module):
    """关键点热图头：从特征图回归关键点位置。"""

    def __init__(self, in_channels: int, num_keypoints: int, heatmap_size: int = 64):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size

        # 上采样 + 卷积生成热图
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.heatmap_head = nn.Conv2d(64, num_keypoints, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 特征图 [B, C, H, W]
        Returns:
            heatmaps: [B, num_keypoints, heatmap_size, heatmap_size]
        """
        x = self.upsample(x)
        heatmaps = self.heatmap_head(x)
        # 确保输出尺寸
        if heatmaps.shape[-1] != self.heatmap_size:
            heatmaps = F.interpolate(
                heatmaps, size=(self.heatmap_size, self.heatmap_size), mode="bilinear", align_corners=False
            )
        return heatmaps


class CourtCalibNet(nn.Module):
    """轻量级球场标定网络。"""

    def __init__(self, num_keypoints: int = 8, pretrained: bool = True):
        """
        Args:
            num_keypoints: 检测的关键点数量（默认 8：4 个角点 + 4 个罚球区角点）
            pretrained: 是否使用 ImageNet 预训练权重
        """
        super().__init__()
        self.num_keypoints = num_keypoints

        # Backbone: MobileNetV3-Small
        backbone = mobilenet_v3_small(pretrained=pretrained)
        self.features = backbone.features

        # 获取最后一个特征图的通道数（MobileNetV3-Small 是 576）
        # 通过一次前向传播确定
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            feat = self.features(dummy)
            feat_channels = feat.shape[1]

        # 关键点热图头
        self.head = KeypointHead(feat_channels, num_keypoints, heatmap_size=64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 [B, 3, H, W]，归一化到 [0, 1]
        Returns:
            heatmaps: [B, num_keypoints, 64, 64]
        """
        features = self.features(x)
        heatmaps = self.head(features)
        return heatmaps


def heatmaps_to_keypoints(heatmaps: torch.Tensor, input_size: tuple[int, int]) -> torch.Tensor:
    """从热图中提取关键点坐标。

    Args:
        heatmaps: [B, num_keypoints, H, W]
        input_size: 原始输入图像尺寸 (height, width)

    Returns:
        keypoints: [B, num_keypoints, 2]，像素坐标 (x, y)
    """
    B, K, H, W = heatmaps.shape
    # 找到每个热图的最大值位置
    heatmaps_flat = heatmaps.view(B, K, -1)
    indices = torch.argmax(heatmaps_flat, dim=2)  # [B, K]

    # 转换为 (x, y) 坐标
    y = indices // W
    x = indices % W

    # 缩放到原始输入尺寸
    keypoints = torch.stack([x, y], dim=2).float()  # [B, K, 2]
    keypoints[:, :, 0] *= input_size[1] / W
    keypoints[:, :, 1] *= input_size[0] / H

    return keypoints


def compute_homography_from_keypoints(
    keypoints: torch.Tensor, court_keypoints: torch.Tensor
) -> torch.Tensor:
    """从关键点计算单应矩阵。

    Args:
        keypoints: [B, K, 2] 图像中的关键点像素坐标
        court_keypoints: [K, 2] 球场关键点的米制坐标

    Returns:
        H: [B, 3, 3] 单应矩阵
    """
    # 这里需要实现 DLT 算法
    # 简化版本：使用 cv2.findHomography 的批量版本
    # 实际部署时需要用 PyTorch 实现以保证可微性
    raise NotImplementedError("Use cv2.findHomography for inference")


# 训练工具函数
def generate_gaussian_heatmap(
    keypoints: torch.Tensor, heatmap_size: int = 64, sigma: float = 2.0
) -> torch.Tensor:
    """生成高斯热图作为训练目标。

    Args:
        keypoints: [B, K, 2] 关键点坐标（归一化到 [0, heatmap_size]）
        heatmap_size: 热图尺寸
        sigma: 高斯标准差

    Returns:
        heatmaps: [B, K, heatmap_size, heatmap_size]
    """
    B, K, _ = keypoints.shape
    device = keypoints.device

    # 创建坐标网格
    y_grid, x_grid = torch.meshgrid(
        torch.arange(heatmap_size, device=device),
        torch.arange(heatmap_size, device=device),
        indexing="ij",
    )

    heatmaps = []
    for k in range(K):
        kp = keypoints[:, k, :]  # [B, 2]
        # 计算高斯热图
        dist = (x_grid - kp[:, 0:1, None]) ** 2 + (y_grid - kp[:, 1:2, None]) ** 2
        heatmap = torch.exp(-dist / (2 * sigma ** 2))
        heatmaps.append(heatmap)

    return torch.stack(heatmaps, dim=1)  # [B, K, H, W]


if __name__ == "__main__":
    # 测试模型
    model = CourtCalibNet(num_keypoints=8, pretrained=False)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 测试前向传播
    x = torch.randn(2, 3, 256, 256)
    heatmaps = model(x)
    print(f"Input: {x.shape}")
    print(f"Output heatmaps: {heatmaps.shape}")

    # 测试关键点提取
    keypoints = heatmaps_to_keypoints(heatmaps, input_size=(256, 256))
    print(f"Keypoints: {keypoints.shape}")
