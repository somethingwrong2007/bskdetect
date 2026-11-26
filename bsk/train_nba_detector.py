"""在本机训练并导出 RF-DETR-S 10 类 NBA 检测模型。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

DEFAULT_DATA = Path("data/nba_10class_v1")
DEFAULT_OUTPUT = Path("output/rfdetr_small_nba_10class_v1")


def train(data: Path = DEFAULT_DATA, output: Path = DEFAULT_OUTPUT,
          epochs: int = 100, batch_size: int = 4, resolution: int = 512,
          device: str = "cuda", workers: int = 2,
          pretrain_weights: str = "rf-detr-small.pth",
          resume: Path | None = None) -> dict:
    from rfdetr import RFDETRSmall

    metadata = yaml.safe_load((data / "data.yaml").read_text(encoding="utf-8")) or {}
    names = metadata.get("names", [])
    if len(names) != 10:
        raise ValueError(f"需要 10 类数据，实际为 {len(names)} 类: {data}")
    for split in ("train", "valid", "test"):
        if not (data / split / "images").exists() or not (data / split / "labels").exists():
            raise FileNotFoundError(f"缺少 {split}/images 或 {split}/labels")

    output.mkdir(parents=True, exist_ok=True)
    model = RFDETRSmall(pretrain_weights=pretrain_weights, resolution=resolution,
                        num_classes=len(names), device=device)
    notes = {
        "dataset": str(data.resolve()),
        "classes": names,
        "source": "Roboflow basketball-player-detection-3-ycjdo v4 (CC BY 4.0)",
        "architecture": "RF-DETR-S (Apache-2.0)",
    }
    model.train(
        dataset_dir=str(data.resolve()),
        output_dir=str(output.resolve()),
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        num_workers=workers,
        tensorboard=False,
        wandb=False,
        run_test=True,
        notes=notes,
        seed=0,
        resume=str(resume.resolve()) if resume else None,
    )
    best = output / "checkpoint_best_total.pth"
    if not best.exists():
        candidates = sorted(output.glob("checkpoint_best_*.pth"))
        if not candidates:
            raise FileNotFoundError(f"训练结束但没有最佳权重: {output}")
        best = candidates[0]
    best_model = RFDETRSmall.from_checkpoint(str(best))
    onnx = best_model.export(output_dir=str(output.resolve()), format="onnx",
                             batch_size=1, dynamic_batch=False, fp16=False, notes=notes)
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    copied = model_dir / "rfdetr_small_nba_10class.pth"
    shutil.copy2(best, copied)
    onnx_copy = model_dir / "rfdetr_small_nba_10class.onnx"
    shutil.copy2(onnx, onnx_copy)
    (model_dir / "rfdetr_small_nba_10class.json").write_text(json.dumps({
        "input_size": resolution,
        "class_names": names,
        "label_offset": 0,
        "mean": model.means,
        "std": model.stds,
        "architecture": "RF-DETR-S",
        "license": "Apache-2.0",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "data": str(data.resolve()), "output": str(output.resolve()),
        "best_checkpoint": str(best), "copied_to": str(copied),
        "onnx": str(onnx), "onnx_copy": str(onnx_copy),
        "epochs": epochs, "batch_size": batch_size,
        "resolution": resolution, "classes": names,
        "resume": str(resume.resolve()) if resume else None,
    }
    (output / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--pretrain-weights", default="rf-detr-small.pth")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args(argv)
    print(json.dumps(train(args.data, args.output, args.epochs, args.batch_size,
                           args.resolution, args.device, args.workers,
                           args.pretrain_weights, args.resume), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())