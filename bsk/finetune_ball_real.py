"""用 Roboflow 真实球数据 + WebGPU 3D 合成球数据混合微调球检测模型（YOLOv8n，单类）。

不是从零训练：以 `data/models/ball_yolov8n.pt`（单类球权重）为起点继续训练。
默认数据集由 `bsk.merge_ball_dataset` 生成：
    data/ball_mix3d/{train,valid,test}
其中包含真实转播帧和 `bsk.synth3d` 生成的 3D 球馆样本。

用法：
    python -m bsk.finetune_ball_real                 # 训练 + valid/test 评估
    python -m bsk.finetune_ball_real --epochs 20
    python -m bsk.finetune_ball_real --eval-only     # 只评估现有权重

产物（都不入 git）：
    runs/detect/<name>/weights/best.pt
    data/models/ball_yolov8n_real.pt
    runs/detect/<name>/metrics.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DEFAULT_DATA = Path("data/ball_mix3d/data.yaml")
DEFAULT_BASE = Path("data/models/ball_yolov8n.pt")
DEFAULT_OUT = Path("data/models/ball_yolov8n_real.pt")


def _val_metrics(model, data: str, split: str, imgsz: int, device: str, batch: int) -> dict:
    """在指定 split 上跑 ultralytics 验证，返回 mAP/P/R 摘要。"""
    res = model.val(data=data, split=split, imgsz=imgsz, batch=batch, device=device,
                    conf=0.001, iou=0.7, plots=False, verbose=False)
    box = res.box
    return {
        "split": split,
        "instances": int(sum(int(v) for v in (getattr(res, "nt_per_class", None) or []))),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "speed_ms": {k: round(float(v), 2) for k, v in (res.speed or {}).items()},
    }


def evaluate(weights: str | Path, data: str | Path, imgsz: int, device: str,
             batch: int, splits: tuple[str, ...] = ("val", "test")) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out = {"weights": str(weights), "data": str(data), "imgsz": imgsz}
    for split in splits:
        out[split] = _val_metrics(model, str(data), split, imgsz, device, batch)
    return out


def train(data: str | Path = DEFAULT_DATA, base: str | Path = DEFAULT_BASE,
          epochs: int = 30, batch: int = 24, imgsz: int = 640,
          device: str = "0", workers: int = 4, name: str = "ball_real",
          out: str | Path = DEFAULT_OUT, lr0: float = 0.002,
          eval_after: bool = True) -> dict:
    from ultralytics import YOLO

    data, base, out = Path(data), Path(base), Path(out)
    if not data.exists():
        raise FileNotFoundError(f"数据集不存在: {data}（先跑 bsk.prep_ball_real、bsk.synth3d、bsk.merge_ball_dataset）")
    if not base.exists():
        raise FileNotFoundError(f"起点权重不存在: {base}（微调必须从已有权重继续）")

    # 已有 1 类球权重继续训练：不传 pretrained，避免退回 COCO 头导致类别错位
    model = YOLO(str(base))
    assert model.model.nc == 1, f"起点权重类别数不是 1: {model.model.nc}"

    model.train(
        data=str(data.resolve()),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=workers,
        project="runs/detect",
        name=name,
        exist_ok=True,
        # 微调：起点已收敛到"橙色球"，学习率比从零训练低一个量级
        optimizer="AdamW",
        lr0=lr0,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=1.0,
        cos_lr=True,
        # 真实转播帧的增强：颜色/尺度扰动为主，不做上下翻转（重力方向固定）
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.35,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.05,
        close_mosaic=5,
        # 验证与早停
        val=True,
        patience=10,
        save=True,
        save_period=-1,
        plots=False,
        seed=0,
        deterministic=True,
        cache=False,
    )

    candidates = sorted(Path("runs").rglob(f"{name}/weights/best.pt"),
                        key=lambda p: p.stat().st_mtime_ns)
    if not candidates:
        raise FileNotFoundError(f"训练结束但没有 best.pt: runs/**/{name}/weights/best.pt")
    best = candidates[-1]
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    report = {"best": str(best), "copied_to": str(out),
              "epochs": epochs, "batch": batch, "imgsz": imgsz, "base": str(base)}
    if eval_after:
        report["eval"] = evaluate(out, data, imgsz, device, batch)
    (best.parent.parent / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="真实球数据微调（YOLOv8n 单类）")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="ball_real")
    ap.add_argument("--lr0", type=float, default=0.002)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args(argv)

    if args.eval_only:
        report = {"eval": evaluate(args.out, args.data, args.imgsz, args.device, args.batch,
                                   splits=("val", "test"))}
    else:
        report = train(args.data, args.base, args.epochs, args.batch, args.imgsz,
                       args.device, args.workers, args.name, args.out, args.lr0)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())