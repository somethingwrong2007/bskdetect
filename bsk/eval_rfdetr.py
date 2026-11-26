"""在 YOLO 格式数据集上评估 RF-DETR ONNX 的逐类 AP50/召回。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from bsk.perceive.rfdetr import RfDetrDetector


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _ap50(rows: list[tuple[float, bool]], gt_count: int) -> tuple[float, float, int, int]:
    if gt_count == 0:
        return 0.0, 0.0, 0, len(rows)
    rows.sort(key=lambda row: row[0], reverse=True)
    tp = fp = 0
    precision, recall = [], []
    for _, matched in rows:
        if matched:
            tp += 1
        else:
            fp += 1
        precision.append(tp / (tp + fp))
        recall.append(tp / gt_count)
    mrec = [0.0, *recall, 1.0]
    mpre = [0.0, *precision, 0.0]
    for index in range(len(mpre) - 2, 0, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    changes = [index for index in range(len(mrec) - 1) if mrec[index + 1] != mrec[index]]
    ap = sum((mrec[index + 1] - mrec[index]) * mpre[index + 1] for index in changes)
    return float(ap), float(tp / (tp + fp) if tp + fp else 0.0), tp, fp


def evaluate(data: Path, weights: Path, split: str = "test", conf: float = 0.05,
             iou_thr: float = 0.5, limit: int | None = None,
             device: str = "auto") -> dict:
    metadata = yaml.safe_load((data / "data.yaml").read_text(encoding="utf-8")) or {}
    names = list(metadata["names"])
    detector = RfDetrDetector(weights=weights, conf_thr=conf, device=device)
    if not detector.ready:
        raise RuntimeError(detector.error)
    images, labels = data / split / "images", data / split / "labels"
    stems = sorted(label.stem for label in labels.glob("*.txt"))
    if limit:
        stems = stems[:limit]
    gt_by_frame: dict[tuple[str, int], list[tuple]] = defaultdict(list)
    pred_by_frame: dict[tuple[str, int], list[tuple[float, tuple]]] = defaultdict(list)
    gt_counts: dict[int, int] = defaultdict(int)
    source_gt_counts: dict[tuple[int, str], int] = defaultdict(int)
    source_rows: dict[tuple[int, str], list[tuple[float, bool]]] = defaultdict(list)
    frame_hits: dict[int, list[bool]] = defaultdict(list)

    for stem in stems:
        image = next((images / f"{stem}{ext}" for ext in (".jpg", ".jpeg", ".png")
                      if (images / f"{stem}{ext}").exists()), None)
        if image is None:
            continue
        rgb = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
        if rgb is None:
            continue
        h, w = rgb.shape[:2]
        for line in (labels / f"{stem}.txt").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:])
            gt_by_frame[(stem, class_id)].append(((cx - bw / 2) * w, (cy - bh / 2) * h,
                                                  (cx + bw / 2) * w, (cy + bh / 2) * h))
            gt_counts[class_id] += 1
            source_gt_counts[(class_id, stem.split("__", 1)[0])] += 1
        for det in detector.predict(rgb, conf=conf, max_det=300):
            pred_by_frame[(stem, det.class_id)].append((det.conf, det.bbox))

    per_class = {}
    aps = []
    for class_id, name in enumerate(names):
        rows: list[tuple[float, bool]] = []
        for stem in stems:
            boxes = gt_by_frame.get((stem, class_id), [])
            class_predictions = sorted(pred_by_frame.get((stem, class_id), []),
                                       key=lambda row: row[0], reverse=True)
            used: set[int] = set()
            hit = False
            for score, box in class_predictions:
                best_iou, best_index = iou_thr, -1
                for index, gt_box in enumerate(boxes):
                    if index in used:
                        continue
                    value = _iou(box, gt_box)
                    if value > best_iou:
                        best_iou, best_index = value, index
                matched = best_index >= 0
                if matched:
                    used.add(best_index)
                    hit = True
                rows.append((score, matched))
                source_rows[(class_id, stem.split("__", 1)[0])].append((score, matched))
            if boxes:
                frame_hits[class_id].append(hit)
        ap, precision, tp, fp = _ap50(rows, gt_counts[class_id])
        if gt_counts[class_id]:
            aps.append(ap)
        by_source = {}
        sources = sorted({key[1] for key in source_gt_counts if key[0] == class_id})
        for source in sources:
            source_ap, source_precision, source_tp, source_fp = _ap50(
                source_rows[(class_id, source)], source_gt_counts[(class_id, source)])
            by_source[source] = {
                "gt": source_gt_counts[(class_id, source)],
                "ap50": round(source_ap, 4), "precision": round(source_precision, 4),
                "tp": source_tp, "fp": source_fp,
            }
        per_class[name] = {
            "gt": gt_counts[class_id], "predictions": len(rows),
            "tp": tp, "fp": fp, "fn": gt_counts[class_id] - tp,
            "recall": round(tp / gt_counts[class_id], 4) if gt_counts[class_id] else None,
            "precision": round(precision, 4), "ap50": round(ap, 4),
            "frame_recall": round(float(np.mean(frame_hits[class_id])), 4)
            if frame_hits[class_id] else None,
            "by_source": by_source,
        }
    return {
        "weights": str(weights), "data": str(data), "split": split,
        "images": len(stems), "conf": conf, "iou": iou_thr, "device": detector.ep,
        "mean_ap50": round(float(np.mean(aps)), 4) if aps else None,
        "classes": per_class, "mean_inference_ms": round(detector.mean_ms, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/rfdetr_pro_v1"))
    parser.add_argument("--weights", type=Path,
                        default=Path("data/models/rfdetr_small_pro_10class.onnx"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("temp/rfdetr_pro_eval.json"))
    args = parser.parse_args(argv)
    report = evaluate(args.data, args.weights, args.split, args.conf, args.iou, args.limit,
                      args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
