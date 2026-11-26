"""球场线分割模型训练脚本。

用法：
  # 先生成合成数据
  python -m bsk.calib_seg_data  # 生成 data/synthetic_seg/

  # 训练
  python -m bsk.calib_seg_train --data data/synthetic_seg --epochs 50 --batch 8

损失函数：加权 CrossEntropy（背景像素远多于线像素，必须平衡）
优化器：AdamW + cosine annealing
评估：mIoU per class
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from bsk.calib_seg_net import CourtLineSegNet, NUM_CLASSES, INPUT_SIZE


def _read_label(path: Path) -> np.ndarray:
    """读取单通道索引图或调色板 PNG，均返回 0..N 的 uint8 类别索引。"""
    with Image.open(path) as im:
        if im.mode in ("P", "L"):
            return np.array(im, dtype=np.uint8)
        return np.array(im.convert("L"), dtype=np.uint8)


class SyntheticSegDataset(Dataset):
    """从 generate_dataset 输出的目录加载样本。"""

    def __init__(self, data_dir: Path, augment: bool = True):
        self.data_dir = data_dir
        self.augment = augment
        img_dir = data_dir / "images"
        msk_dir = data_dir / "masks"

        meta_path = data_dir / "meta.json"
        if meta_path.exists():
            self.meta = json.loads(meta_path.read_text())
        else:
            self.meta = [{"idx": i} for i in range(len(list(img_dir.glob("*.png"))))]

        self.img_paths = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
        self.msk_paths = sorted(msk_dir.glob("*.png"))
        assert len(self.img_paths) == len(self.msk_paths), "images/masks count mismatch"

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        import cv2
        img = cv2.imread(str(self.img_paths[idx]))
        msk = _read_label(self.msk_paths[idx])

        if img is None:
            raise FileNotFoundError(f"Failed to load idx={idx}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.augment:
            img, msk = self._augment(img, msk)

        img_t = self.normalize(self.to_tensor(img))
        msk_t = torch.from_numpy(msk.astype(np.int64))

        return img_t, msk_t

    def _augment(self, img: np.ndarray, msk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import cv2
        h, w = img.shape[:2]

        # 水平翻转：等于从球场另一侧看，几何仍然合法
        if np.random.rand() < 0.5:
            img = img[:, ::-1].copy()
            msk = msk[:, ::-1].copy()

        if np.random.rand() < 0.3:
            angle = np.random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderValue=(30, 30, 30))
            msk = cv2.warpAffine(msk, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

        if np.random.rand() < 0.2:
            scale = np.random.uniform(0.9, 1.1)
            nw, nh = int(w * scale), int(h * scale)
            img = cv2.resize(img, (nw, nh))
            msk = cv2.resize(msk, (nw, nh), interpolation=cv2.INTER_NEAREST)
            if nw > w:
                img = img[(nh - h) // 2:(nh - h) // 2 + h, (nw - w) // 2:(nw - w) // 2 + w]
                msk = msk[(nh - h) // 2:(nh - h) // 2 + h, (nw - w) // 2:(nw - w) // 2 + w]
            else:
                pad_h = (h - nh) // 2
                pad_w = (w - nw) // 2
                img = np.pad(img, ((pad_h, h - nh - pad_h), (pad_w, w - nw - pad_w), (0, 0)),
                             constant_values=30)
                msk = np.pad(msk, ((pad_h, h - nh - pad_h), (pad_w, w - nw - pad_w)),
                             constant_values=0)

        return img, msk


def compute_class_weights(data_dir: Path, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """从训练集统计类别频率，计算平衡权重。"""
    msk_dir = data_dir / "masks"
    counts = np.zeros(num_classes, dtype=np.int64)

    msk_paths = sorted(msk_dir.glob("*.png"))
    sample_paths = msk_paths[:min(500, len(msk_paths))]

    for p in sample_paths:
        msk = _read_label(p)
        for c in range(num_classes):
            counts[c] += (msk == c).sum()

    total = counts.sum()
    freq = counts / max(total, 1)
    weights = 1.0 / np.sqrt(freq + 1e-8)
    weights = weights / weights.mean()

    print(f"Class frequencies: {freq}")
    print(f"Class weights: {weights}")

    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        preds = logits.argmax(dim=1)
        acc = (preds == masks).float().mean().item()

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

        if (batch_idx + 1) % 20 == 0:
            print(f"  Epoch {epoch} [{batch_idx + 1}/{len(loader)}] "
                  f"loss={total_loss / n_batches:.4f} acc={total_acc / n_batches:.4f}")

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=NUM_CLASSES):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        preds = logits.argmax(dim=1)

        for p, t in zip(preds.cpu().numpy(), masks.cpu().numpy()):
            valid = t != 255  # ignore 区域不参与指标
            pv = p[valid].astype(np.int64)
            tv = t[valid].astype(np.int64)
            flat = pv * num_classes + tv
            confusion += np.bincount(flat, minlength=num_classes * num_classes).reshape(
                num_classes, num_classes)

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)

    ious = np.zeros(num_classes)
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[c, :].sum() - tp
        fn = confusion[:, c].sum() - tp
        denom = tp + fp + fn
        ious[c] = tp / denom if denom > 0 else 0.0

    miou = ious.mean()
    return avg_loss, miou, ious


def main():
    parser = argparse.ArgumentParser(description="Train court line segmentation model")
    parser.add_argument("--data", type=str, default="data/synthetic_seg",
                        help="Path to synthetic dataset directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--val-data", type=str, default=None, help="Use a separate validation directory")
    parser.add_argument("--save-dir", type=str, default="checkpoints/calib_seg")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--init-weights", type=str, default=None,
                        help="Load only model weights for incremental fine-tuning")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"Data directory {data_dir} not found. Generating synthetic data first...")
        from bsk.calib_seg_data import generate_dataset
        from bsk.court import FIBA
        generate_dataset(5000, data_dir, FIBA)

    print("Computing class weights...")
    class_weights = compute_class_weights(data_dir).to(device)

    full_dataset = SyntheticSegDataset(data_dir, augment=True)
    if args.val_data:
        train_ds = full_dataset
        val_ds = SyntheticSegDataset(Path(args.val_data), augment=False)
        n_train, n_val = len(train_ds), len(val_ds)
    else:
        n_val = int(len(full_dataset) * args.val_split)
        n_train = len(full_dataset) - n_val
        rng = np.random.default_rng(1234)
        perm = rng.permutation(len(full_dataset))
        val_idx = perm[:n_val].tolist()
        train_idx = perm[n_val:].tolist()
        train_ds = torch.utils.data.Subset(full_dataset, train_idx)
        val_ds = torch.utils.data.Subset(SyntheticSegDataset(data_dir, augment=False), val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"Train: {n_train}, Val: {n_val}")

    model = CourtLineSegNet(num_classes=NUM_CLASSES, pretrained=not args.no_pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")
    if args.init_weights:
        ckpt = torch.load(args.init_weights, map_location=device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
        print(f"Initialized weights from {args.init_weights} (incremental fine-tune)")
    train_meta = data_dir / "meta.json"
    court_model = json.loads(train_meta.read_text(encoding="utf-8"))[0]["court_model"] if train_meta.exists() else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

    start_epoch = 0
    best_miou = 0.0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt.get("best_miou", 0.0)
        # 旧 checkpoint 的余弦调度已把 lr 衰减到 0；续训必须从新的学习率重新起算
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs - start_epoch, 1))
        print(f"Resumed from epoch {start_epoch}, best_miou={best_miou:.4f}, lr={args.lr:g}")

    print(f"\nStarting training for {args.epochs} epochs...")
    history = []

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_miou, val_ious = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}/{args.epochs - 1} ({elapsed:.1f}s) | "
              f"lr={lr_now:.6f} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_miou={val_miou:.4f}")

        for c in range(NUM_CLASSES):
            from bsk.calib_seg_net import CLASS_NAMES
            print(f"  {CLASS_NAMES[c]}: IoU={val_ious[c]:.4f}", end="")
        print()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_miou": val_miou,
            "val_ious": val_ious.tolist(),
            "lr": lr_now,
        })

        is_best = val_miou > best_miou
        if is_best:
            best_miou = val_miou

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_miou": best_miou,
            "val_miou": val_miou,
            "court_model": court_model,
        }, save_dir / "last.pt")

        if is_best:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "best_miou": best_miou,
                "val_miou": val_miou,
                "val_ious": val_ious.tolist(),
                "court_model": court_model,
            }, save_dir / "best.pt")
            print(f"  ** New best model saved (mIoU={val_miou:.4f})")

    (save_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nTraining complete. Best mIoU: {best_miou:.4f}")
    print(f"Checkpoints saved to {save_dir}")


if __name__ == "__main__":
    main()
