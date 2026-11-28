"""在自建 WebGPU 数据上按球场米制坐标验收标定误差。"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from bsk.court import COURTS, project_court_polygon
from bsk.line_calib import DEFAULT_WEIGHTS, LineCalibrator


_D4 = [
    np.eye(2), -np.eye(2),
    np.array([[-1.0, 0.0], [0.0, 1.0]]),
    np.array([[1.0, 0.0], [0.0, -1.0]]),
]


def _project(homography: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ph = np.column_stack([points, np.ones(len(points))]) @ homography.T
    depth = ph[:, 2]
    valid = np.abs(depth) > 1e-9
    uv = ph[:, :2] / np.where(valid, depth, 1.0)[:, None]
    return uv, valid & np.all(np.isfinite(uv), axis=1)


def _error_metrics(record: dict, calibration, shape: tuple[int, int]) -> dict | None:
    court = COURTS[record["court_model"]]
    h, w = shape
    x = np.arange(-court.half_length, court.half_length + 0.01, 1.0)
    y = np.arange(-court.half_width, court.half_width + 0.01, 1.0)
    world = np.array([(xx, yy) for xx in x for yy in y], dtype=np.float64)
    gt_h = np.array(record["H_world_to_px"], dtype=np.float64)
    pixel, front = _project(gt_h, world)
    inside = front & (pixel[:, 0] >= 4) & (pixel[:, 0] < w - 4) & (pixel[:, 1] >= 4) & (pixel[:, 1] < h - 4)
    if inside.sum() < 8:
        return None
    pred_h = np.array(calibration.H, dtype=np.float64)
    pred_world, valid = _project(pred_h, pixel[inside])
    valid &= np.all(np.abs(pred_world) < 1000, axis=1)
    if valid.sum() < 8:
        return None
    truth = world[inside][valid]
    error = np.linalg.norm(pred_world[valid] - truth, axis=1)
    sym_error = np.min(np.stack([
        np.linalg.norm(pred_world[valid] - (truth @ transform.T), axis=1)
        for transform in _D4
    ], axis=0), axis=0)
    return {
        "n_points": int(len(error)),
        "median_m": float(np.median(error)),
        "rmse_m": float(np.sqrt(np.mean(error ** 2))),
        "p90_m": float(np.percentile(error, 90)),
        "max_m": float(np.max(error)),
        "sym_median_m": float(np.median(sym_error)),
        "sym_rmse_m": float(np.sqrt(np.mean(sym_error ** 2))),
        "sym_p90_m": float(np.percentile(sym_error, 90)),
        "sym_max_m": float(np.max(sym_error)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/seg_webgpu_v1/FIBA"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--size", default="540x960")
    parser.add_argument("--out", type=Path, default=Path("temp/synth_calib_gate.json"))
    args = parser.parse_args()

    records = json.loads((args.data / args.split / "meta.json").read_text(encoding="utf-8"))
    records = [r for r in records if r.get("calibratable_gt")]
    if len(records) > args.n:
        indices = np.linspace(0, len(records) - 1, args.n).astype(int)
        records = [records[i] for i in indices]
    h, w = (int(v) for v in args.size.lower().split("x"))
    court_model = records[0]["court_model"] if records else "FIBA"
    calibrator = LineCalibrator(court=COURTS[court_model], size=(h, w), weights=args.weights,
                                device=args.device, conf_thr=args.conf, auto_na=False)
    results = []
    overlays = []
    started = time.time()
    for i, record in enumerate(records):
        idx = record["idx"]
        rgb = np.asarray(Image.open(args.data / args.split / "images" / f"{idx:06d}.jpg").convert("RGB"))
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            scale = np.array([[w / record["image_size"][1], 0, 0], [0, h / record["image_size"][0], 0], [0, 0, 1]])
            record = {**record, "H_world_to_px": (scale @ np.array(record["H_world_to_px"])).tolist()}
        t0 = time.time()
        pred = calibrator.predict(rgb)
        fit = calibrator.fit(pred, (h, w), rgb=rgb)
        elapsed = time.time() - t0
        ok = fit is not None and fit.calib.is_plausible(w, h)
        metrics = _error_metrics(record, fit.calib, (h, w)) if ok else None
        result = {
            "idx": idx, "camera_mode": record["camera_mode"], "court_model": record["court_model"],
            "bench_occlusion_suspected": record["bench_occlusion_suspected"],
            "fit_ok": bool(ok), "error": metrics, "elapsed_s": elapsed,
        }
        results.append(result)
        if metrics:
            court = COURTS[record["court_model"]]
            gt_poly = project_court_polygon(np.array(record["H_world_to_px"]), court)
            pred_poly = project_court_polygon(np.array(fit.calib.H_inv), court)
            overlay = rgb.copy()
            cv2.polylines(overlay, [np.rint(gt_poly).astype(np.int32)], True, (0, 255, 0), 2)
            cv2.polylines(overlay, [np.rint(pred_poly).astype(np.int32)], True, (255, 0, 255), 2)
            cv2.putText(overlay, f"{idx} symRMSE={metrics['sym_rmse_m'] * 100:.1f}cm", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            overlays.append((metrics["rmse_m"], overlay))
        print(f"[{i + 1}/{len(records)}] idx={idx} mode={record['camera_mode']} fit={ok} "
              f"rmse={metrics['sym_rmse_m'] * 100 if metrics else float('nan'):.1f}cm time={elapsed:.2f}s", flush=True)

    valid = [r for r in results if r["fit_ok"] and r["error"]]
    by_mode = defaultdict(list)
    for result in valid:
        by_mode[result["camera_mode"]].append(result["error"]["sym_rmse_m"])
    raw_errors = [r["error"]["rmse_m"] for r in valid]
    sym_errors = [r["error"]["sym_rmse_m"] for r in valid]
    sym_p90 = [r["error"]["sym_p90_m"] for r in valid]
    sym_max = [r["error"]["sym_max_m"] for r in valid]
    summary = {
        "samples": len(results), "fit_ok": sum(r["fit_ok"] for r in results),
        "measured": len(valid), "coverage": len(valid) / max(len(results), 1),
        "median_rmse_m": float(np.median(raw_errors)) if valid else None,
        "p90_rmse_m": float(np.percentile(raw_errors, 90)) if valid else None,
        "max_rmse_m": max(raw_errors, default=None),
        "median_sym_rmse_m": float(np.median(sym_errors)) if valid else None,
        "p90_sym_rmse_m": float(np.percentile(sym_errors, 90)) if valid else None,
        "max_sym_rmse_m": max(sym_max, default=None),
        "p90_point_sym_error_m": float(np.percentile(sym_p90, 90)) if valid else None,
        "orientation_ambiguous": sum(r["error"]["rmse_m"] > 0.10 and r["error"]["sym_rmse_m"] <= 0.10 for r in valid),
        "mode_rmse_m": {mode: float(np.mean(values)) for mode, values in by_mode.items()},
        "elapsed_s": time.time() - started, "target_m": 0.10,
    }
    summary["target_met"] = bool(valid and summary["coverage"] >= 0.90 and summary["p90_sym_rmse_m"] <= 0.10)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")
    if overlays:
        cells = []
        for _, overlay in sorted(overlays, key=lambda item: item[0], reverse=True)[:12]:
            cells.append(cv2.resize(overlay, (480, 270)))
        sheet = np.full((270 * 3, 480 * 4, 3), 255, np.uint8)
        for i, cell in enumerate(cells):
            x, y = (i % 4) * 480, (i // 4) * 270
            sheet[y:y + 270, x:x + 480] = cell
        cv2.imwrite(str(args.out.with_suffix(".jpg")), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
