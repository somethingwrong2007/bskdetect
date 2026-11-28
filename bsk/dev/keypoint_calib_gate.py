"""在自建 WebGPU test 上验收关键点模型与 RANSAC 单应。"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from bsk.calib_keypoint import project_keypoints
from bsk.calib_keypoint_infer import KeypointCalibrator
from bsk.court import COURTS, project_court_polygon
from bsk.dev.synth_calib_gate import _error_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--size", default="540x960")
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    h, w = (int(v) for v in args.size.lower().split("x"))
    records = json.loads((args.data / args.split / "meta.json").read_text(encoding="utf-8"))
    records = [r for r in records if r.get("calibratable_gt")]
    if len(records) > args.n:
        indices = np.linspace(0, len(records) - 1, args.n).astype(int)
        records = [records[i] for i in indices]
    court_model = records[0]["court_model"]
    calibrator = KeypointCalibrator(args.weights, court_model, (h, w), device=args.device, conf_thr=args.conf)
    results = []
    overlays = []
    started = time.time()
    pck_hits = pck_visible = 0
    for i, record in enumerate(records):
        idx = record["idx"]
        rgb = np.asarray(Image.open(args.data / args.split / "images" / f"{idx:06d}.jpg").convert("RGB"))
        calibration, diag = calibrator.predict(rgb)
        metrics = _error_metrics(record, calibration, (h, w)) if calibration else None
        gt_px, gt_visible = project_keypoints(record, (h, w))
        pred_px = np.asarray(diag.get("pixel", []), dtype=np.float64)
        pred_visible = np.asarray(diag.get("visible_mask", []), dtype=bool)
        if pred_px.shape == gt_px.shape:
            error = np.linalg.norm(pred_px - gt_px, axis=1)
            mask = gt_visible & pred_visible
            pck_visible += int(gt_visible.sum())
            pck_hits += int((error[mask] <= 5.0).sum()) if mask.any() else 0
        result = {"idx": idx, "camera_mode": record["camera_mode"], "fit_ok": calibration is not None,
                  "error": metrics, "diag": diag}
        results.append(result)
        if metrics:
            gt_poly = project_court_polygon(np.array(record["H_world_to_px"]), COURTS[court_model])
            pred_poly = project_court_polygon(np.array(calibration.H_inv), COURTS[court_model])
            overlay = rgb.copy()
            cv2.polylines(overlay, [np.rint(gt_poly).astype(np.int32)], True, (0, 255, 0), 2)
            cv2.polylines(overlay, [np.rint(pred_poly).astype(np.int32)], True, (255, 0, 255), 2)
            cv2.putText(overlay, f"{idx} symRMSE={metrics['sym_rmse_m'] * 100:.1f}cm", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            overlays.append((metrics["sym_rmse_m"], overlay))
        print(f"[{i + 1}/{len(records)}] idx={idx} fit={calibration is not None} "
              f"sym={metrics['sym_rmse_m'] * 100 if metrics else float('nan'):.1f}cm", flush=True)
    valid = [r for r in results if r["fit_ok"] and r["error"]]
    by_mode = defaultdict(list)
    for r in valid:
        by_mode[r["camera_mode"]].append(r["error"]["sym_rmse_m"])
    summary = {
        "samples": len(results), "fit_ok": len(valid), "coverage": len(valid) / max(len(results), 1),
        "pck5": pck_hits / max(pck_visible, 1),
        "median_sym_rmse_m": float(np.median([r["error"]["sym_rmse_m"] for r in valid])) if valid else None,
        "p90_sym_rmse_m": float(np.percentile([r["error"]["sym_rmse_m"] for r in valid], 90)) if valid else None,
        "max_sym_rmse_m": max((r["error"]["sym_max_m"] for r in valid), default=None),
        "mode_rmse_m": {k: float(np.mean(v)) for k, v in by_mode.items()},
        "elapsed_s": time.time() - started, "target_m": 0.10,
    }
    summary["target_met"] = bool(valid and summary["coverage"] >= 0.90 and summary["p90_sym_rmse_m"] <= 0.10)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")
    if overlays:
        cells = [cv2.resize(im, (480, 270)) for _, im in sorted(overlays, key=lambda x: x[0], reverse=True)[:12]]
        sheet = np.full((810, 1920, 3), 255, np.uint8)
        for i, cell in enumerate(cells):
            sheet[(i // 4) * 270:(i // 4 + 1) * 270, (i % 4) * 480:(i % 4 + 1) * 480] = cell
        cv2.imwrite(str(args.out.with_suffix(".jpg")), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
