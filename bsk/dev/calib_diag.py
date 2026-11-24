"""球场线分割诊断：真实转播帧上的每类线像素量 + 可视化拼图。

用于判断"标定失败"是模型弱（分割不出线）还是后处理/拟合错：
在若干真实帧上跑 CourtLineSegNet，统计各类别的预测像素数/占比，
并把预测叠到原图上存 .tmp_calib_diag.jpg 供人眼检查。

用法：
  .venv/Scripts/python.exe -m bsk.dev.calib_diag --weights checkpoints/calib_seg_mix2/best.pt \
      --n 12 --size 540x960 --out .tmp_calib_diag.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from bsk.calib_seg_net import CLASS_NAMES, NUM_CLASSES
from bsk.court import COURTS
from bsk.store import Library


def _uniform_frames(vid: str, n: int, total: int = 8577) -> list[int]:
    return [int(v) for v in np.linspace(60, total - 60, n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid", default="harden-55")
    ap.add_argument("--weights", default="checkpoints/calib_seg_mix2/best.pt")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--frames", default="", help="逗号分隔帧号，覆盖 --n")
    ap.add_argument("--size", default="540x960", help="HxW 模型输入")
    ap.add_argument("--court", default="NBA")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--width", type=int, default=1280, help="解码宽度")
    ap.add_argument("--out", default=".tmp_calib_diag.jpg")
    ap.add_argument("--calibrate", action="store_true", help="同时跑 LineCalibrator 拟合")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    hh, ww = (int(v) for v in args.size.lower().split("x"))
    frames = ([int(x) for x in args.frames.split(",")] if args.frames
              else _uniform_frames(args.vid, args.n))

    lib = Library()
    table = lib.table(args.vid)
    got = lib.picker(args.vid).frames_at(frames, size=(args.width, int(args.width * 9 / 16)))
    picked = {f: rgb for f, _pts, rgb in got}

    from bsk.calib_seg_net import CourtLineSegNet
    import torch
    from torchvision import transforms

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CourtLineSegNet(num_classes=NUM_CLASSES, pretrained=False).to(dev)
    ckpt = torch.load(args.weights, map_location=dev, weights_only=False)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    calibrator = None
    if args.calibrate:
        from bsk.line_calib import LineCalibrator
        calibrator = LineCalibrator(weights=args.weights, court=COURTS[args.court],
                                    size=(hh, ww), conf_thr=args.conf, device=dev)

    colors = np.array([
        [0, 0, 0], [0, 255, 0], [255, 0, 255], [0, 128, 255], [255, 255, 0], [0, 255, 255],
    ], np.uint8)

    cells = []
    print(f"frames={frames} size={ww}x{hh} weights={args.weights}")
    for f in frames:
        rgb = picked.get(f)
        if rgb is None:
            print(f"  f{f}: decode failed")
            continue
        small = cv2.resize(rgb, (ww, hh), interpolation=cv2.INTER_AREA)
        x = norm(torch.from_numpy(small).permute(2, 0, 1).float() / 255.0)
        with torch.no_grad():
            probs = model(x.unsqueeze(0).to(dev)).softmax(1)[0]
        conf = probs.max(0).values.cpu().numpy()
        pred = probs.argmax(0).cpu().numpy().astype(np.uint8)
        pred[conf < args.conf] = 0
        counts = {CLASS_NAMES[c]: int((pred == c).sum()) for c in range(1, NUM_CLASSES)}
        total = sum(counts.values())
        print(f"  f{f}: total={total:6d} px  " +
              "  ".join(f"{k}={v}" for k, v in counts.items()))

        vis = colors[np.clip(pred, 0, 5)]
        m = pred > 0
        # 把线放大 1.5px 便于人眼检查
        vis = cv2.dilate(vis, np.ones((3, 3), np.uint8)) if m.any() else vis
        over = rgb.copy()
        up = cv2.resize(vis, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        mup = cv2.resize(m.astype(np.uint8), (rgb.shape[1], rgb.shape[0]),
                         interpolation=cv2.INTER_NEAREST) > 0
        over[mup] = (0.2 * rgb[mup] + 0.8 * up[mup]).astype(np.uint8)
        cv2.putText(over, f"f{f} lines={total}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)

        if calibrator is not None:
            cal = calibrator.calibrate(rgb)
            print(f"  f{f}: calibrate -> {'OK err=%.2f' % cal.reproj_error_px if cal else 'None'}")
            if cal is not None:
                from bsk.court import project_court_polygon
                poly = project_court_polygon(np.array(cal.H_inv), COURTS[args.court])
                cv2.polylines(over, [poly.astype(np.int32)], True, (0, 255, 0), 3)

        cell = np.hstack([cv2.resize(rgb, (640, 360)), cv2.resize(over, (640, 360))])
        cells.append(cell)

    if cells:
        grid = np.vstack(cells)
        cv2.imwrite(args.out, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        print("saved ->", Path(args.out).resolve())


if __name__ == "__main__":
    main()