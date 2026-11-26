"""训练球场 16 关键点热图头，复用现有线分割骨干。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bsk.calib_keypoint import KEYPOINT_NAMES, CourtKeypointDataset, decode_heatmaps
from bsk.calib_seg_net import CourtLineSegNet
from bsk.calib_seg_train import compute_class_weights


def heatmap_targets(points: torch.Tensor, visible: torch.Tensor, h: int, w: int,
                    sigma: float = 2.5) -> torch.Tensor:
    b, k, _ = points.shape
    points = torch.nan_to_num(points, nan=0.0, posinf=float(w - 1), neginf=0.0)
    yy, xx = torch.meshgrid(
        torch.arange(h, device=points.device, dtype=points.dtype),
        torch.arange(w, device=points.device, dtype=points.dtype),
        indexing="ij",
    )
    x = points[:, :, 0].clamp(0, w - 1)[:, :, None, None]
    y = points[:, :, 1].clamp(0, h - 1)[:, :, None, None]
    target = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return target * visible[:, :, None, None].to(target.dtype)


def run_epoch(model, loader, criterion, optimizer, device, train: bool, sigma: float = 2.5):
    model.train(train)
    total = pixels = hits = visible_n = detected_n = 0
    t0 = time.time()
    for images, masks, points, visible in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        points = points.to(device, non_blocking=True)
        visible = visible.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            seg_logits, kpt_logits, coords, vis_logits = model(images)
            target = heatmap_targets(points, visible, images.shape[2], images.shape[3], sigma)
            normalized = points.clone()
            normalized[..., 0] = 2.0 * normalized[..., 0] / max(images.shape[3] - 1, 1) - 1.0
            normalized[..., 1] = 2.0 * normalized[..., 1] / max(images.shape[2] - 1, 1) - 1.0
            normalized = torch.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)
            coord_loss = F.smooth_l1_loss(coords, normalized, reduction="none")
            coord_loss = (coord_loss * visible[..., None]).sum() / (visible.sum() * 2).clamp_min(1)
            vis_loss = F.binary_cross_entropy_with_logits(vis_logits, visible.float())
            target_flat = target.flatten(2)
            target_flat = target_flat / target_flat.sum(-1, keepdim=True).clamp_min(1e-6)
            log_probs = F.log_softmax(kpt_logits.flatten(2), dim=-1)
            kpt_loss = -(target_flat * log_probs).sum(-1)
            kpt_loss = (kpt_loss * visible).sum() / visible.sum().clamp_min(1)
            seg_loss = criterion(seg_logits, masks)
            loss = seg_loss + kpt_loss + 0.2 * coord_loss + 0.5 * vis_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        total += float(loss.detach()) * len(images)
        with torch.no_grad():
            pred_coords, _heat_visible, _heat_response = decode_heatmaps(kpt_logits, threshold=-1e9)
            pred_visible = torch.sigmoid(vis_logits.detach()).cpu().numpy() >= 0.5
            gt = points.cpu().numpy()
            gt_visible = visible.cpu().numpy()
            error = np.linalg.norm(pred_coords - gt, axis=2)
            mask = gt_visible & pred_visible
            visible_n += int(gt_visible.sum())
            detected_n += int(mask.sum())
            pixels += float(error[mask].sum()) if mask.any() else 0.0
            hits += int((error[mask] <= 5.0).sum()) if mask.any() else 0
    return {
        "loss": total / max(len(loader.dataset), 1),
        "pixel_error_px": pixels / max(visible_n, 1),
        "pck5": hits / max(visible_n, 1),
        "visibility_recall": detected_n / max(visible_n, 1),
        "seconds": time.time() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--size", default="540x960")
    parser.add_argument("--init-weights", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    h, w = (int(v) for v in args.size.lower().split("x"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = CourtKeypointDataset(args.data, (h, w), augment=True)
    val_ds = CourtKeypointDataset(args.val_data, (h, w), augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4,
                            pin_memory=True)
    model = CourtLineSegNet(num_keypoints=len(KEYPOINT_NAMES), pretrained=not args.no_pretrained).to(device)
    if args.init_weights:
        checkpoint = torch.load(args.init_weights, map_location=device, weights_only=False)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"init weights: missing={len(missing)} unexpected={len(unexpected)}")
    class_weights = compute_class_weights(args.data).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    backbone_params = [p for name, p in model.named_parameters()
                       if not name.startswith(("kpt.", "coord_", "vis_"))]
    keypoint_params = [p for name, p in model.named_parameters()
                       if name.startswith(("kpt.", "coord_", "vis_"))]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr},
        {"params": keypoint_params, "lr": args.lr * 5.0},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    meta = json.loads((args.data / "meta.json").read_text(encoding="utf-8"))
    court_model = meta[0]["court_model"]
    args.save_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.save_dir / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            scheduler.last_epoch = start_epoch - 1
        print(f"resumed {args.resume} at epoch {start_epoch}")
    best_score = min((row["val"]["pixel_error_px"] + 200.0 * (1.0 - row["val"]["pck5"])
                      + 50.0 * (1.0 - row["val"]["visibility_recall"]) for row in history),
                     default=float("inf"))
    print(f"device={device} train={len(train_ds)} val={len(val_ds)} court={court_model}")
    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step()
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        score = (val_metrics["pixel_error_px"] + 200.0 * (1.0 - val_metrics["pck5"])
                 + 50.0 * (1.0 - val_metrics["visibility_recall"]))
        if score < best_score:
            best_score = score
            torch.save({
                "model": model.state_dict(), "epoch": epoch, "court_model": court_model,
                "keypoint_names": KEYPOINT_NAMES, "input_size": [h, w],
                "val": val_metrics,
            }, args.save_dir / "best.pt")
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "court_model": court_model, "keypoint_names": KEYPOINT_NAMES, "input_size": [h, w],
        }, args.save_dir / "last.pt")
        (args.save_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"saved {args.save_dir}")


if __name__ == "__main__":
    main()
