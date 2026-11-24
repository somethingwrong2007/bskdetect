"""训练脚本：用合成数据训练 CourtCalibNet。

训练流程：
1. 加载合成数据集
2. 用热图损失训练关键点检测
3. 保存模型权重

损失函数：MSE between predicted and ground truth heatmaps
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time

from bsk.calib_net import CourtCalibNet, generate_gaussian_heatmap
from bsk.calib_data_gen import generate_dataset


class SyntheticCourtDataset(Dataset):
    """合成球场数据集。"""

    def __init__(self, data_path: Path, num_keypoints: int = 8, heatmap_size: int = 64):
        self.data = np.load(data_path, allow_pickle=True).item()
        self.images = self.data["images"]  # [N, H, W, 3]
        self.keypoints = self.data["keypoints"]  # [N, K, 2]
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # [H, W, 3] BGR
        keypoints = self.keypoints[idx]  # [K, 2]

        # BGR -> RGB, HWC -> CHW, 归一化到 [0, 1]
        image = image[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0

        # 关键点坐标归一化到 [0, heatmap_size]
        H_img, W_img = image.shape[1:]
        keypoints_norm = keypoints.copy()
        keypoints_norm[:, 0] *= self.heatmap_size / W_img
        keypoints_norm[:, 1] *= self.heatmap_size / H_img

        # 生成目标热图
        keypoints_tensor = torch.from_numpy(keypoints_norm).float()
        target_heatmap = generate_gaussian_heatmap(
            keypoints_tensor.unsqueeze(0),
            heatmap_size=self.heatmap_size,
            sigma=2.0
        ).squeeze(0)  # [K, H, W]

        return (
            torch.from_numpy(image).float(),
            target_heatmap,
            torch.from_numpy(keypoints).float(),  # 原始像素坐标（用于评估）
        )


def train_model(
    dataset_path: Path,
    output_dir: Path,
    num_epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """训练 CourtCalibNet。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training on {device}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")

    # 加载数据集
    dataset = SyntheticCourtDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print(f"Dataset size: {len(dataset)} samples")

    # 创建模型
    model = CourtCalibNet(num_keypoints=8, pretrained=True).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # 训练循环
    best_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (images, target_heatmaps, _) in enumerate(dataloader):
            images = images.to(device)
            target_heatmaps = target_heatmaps.to(device)

            # 前向传播
            optimizer.zero_grad()
            pred_heatmaps = model(images)

            # 计算损失
            loss = criterion(pred_heatmaps, target_heatmaps)

            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} Batch {batch_idx+1}/{len(dataloader)} Loss: {loss.item():.6f}")

        # 计算平均损失
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.6f} - Time: {elapsed:.1f}s - LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            print(f"  Saved best model (loss: {best_loss:.6f})")

        # 每 10 个 epoch 保存一次
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), output_dir / f"model_epoch_{epoch+1}.pth")

    # 保存最终模型
    torch.save(model.state_dict(), output_dir / "final_model.pth")
    print(f"Training complete. Best loss: {best_loss:.6f}")
    print(f"Models saved to {output_dir}")


def evaluate_model(model_path: Path, dataset_path: Path, device: str = "cuda"):
    """评估模型：计算关键点检测误差。"""
    # 加载模型
    model = CourtCalibNet(num_keypoints=8, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 加载数据集
    dataset = SyntheticCourtDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    total_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, _, gt_keypoints in dataloader:
            images = images.to(device)
            gt_keypoints = gt_keypoints.to(device)

            # 预测热图
            pred_heatmaps = model(images)

            # 提取关键点
            from bsk.calib_net import heatmaps_to_keypoints
            H_img, W_img = images.shape[2:]
            pred_keypoints = heatmaps_to_keypoints(pred_heatmaps, input_size=(H_img, W_img))
            pred_keypoints = pred_keypoints.to(device)

            # 计算误差（像素）
            error = torch.norm(pred_keypoints - gt_keypoints, dim=2).mean()
            total_error += error.item() * len(images)
            total_samples += len(images)

    avg_error = total_error / total_samples
    print(f"Average keypoint error: {avg_error:.2f} pixels")
    return avg_error


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/synthetic/synthetic_dataset.npy")
    parser.add_argument("--output", type=str, default="models/court_calib")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic data first")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of synthetic samples")

    args = parser.parse_args()

    if args.generate_data:
        print(f"Generating {args.num_samples} synthetic samples...")
        generate_dataset(args.num_samples, Path("data/synthetic"))

    train_model(
        dataset_path=Path(args.data),
        output_dir=Path(args.output),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
