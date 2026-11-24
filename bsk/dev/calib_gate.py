"""标定验收门：真实帧上的 H 拟合通过率 + 线重合叠加图 + 观众剔除模拟。

做的事：
1. 在指定视频上均匀抽 N 帧（默认 12），逐帧跑 LineCalibrator 拟合；
2. 统计通过率（fit 成功 + is_plausible + 代价阈值）；
3. 把投影出的球场线画到原图上，拼成 .tmp_calib_overlay.jpg 供人眼检查；
4. 用 RTMO 检测人，按"脚点（bbox 底边中点）在球场多边形外且超出 margin"
   剔除，画成 before/after 对比图 .tmp_calib_filter.jpg，检验是否误删场上球员。

用法：
  .venv/Scripts/python.exe -m bsk.dev.calib_gate --frames 12 --weights checkpoints/calib_seg_hd/best.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def _court_lines(court):
    """投影用的球场线（世界坐标折线列表），带颜色。"""
    import math

    hl, hw = court.half_length, court.half_width
    lines = []
    rect = [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw), (-hl, -hw)]
    lines.append((np.array(rect), (0, 255, 0), 3))                       # 外框
    lines.append((np.array([(0, -hw), (0, hw)]), (255, 200, 0), 2))      # 中线
    n = 72
    cc = np.array([(1.8 * math.cos(2 * math.pi * i / n), 1.8 * math.sin(2 * math.pi * i / n))
                   for i in range(n + 1)])
    lines.append((cc, (0, 255, 255), 2))                                 # 中圈
    for side in ("A", "B"):
        bl = court.baseline_x(side)
        sign = 1 if side == "A" else -1
        ft_x = bl - court.free_throw_dist * sign
        fthw = court.free_throw_width / 2
        lane = np.array([(bl, -fthw), (bl, fthw), (ft_x, fthw), (ft_x, -fthw), (bl, -fthw)])
        lines.append((lane, (255, 0, 255), 2))                           # 罚球区

        rim_x, _ = court.rim_pos(side)
        r3 = court.three_pt_radius
        y_corner = hw - court.three_pt_corner_dist
        theta_c = math.asin(min(1.0, y_corner / r3))
        arc = [(bl, -y_corner)]
        for i in range(49):
            th = -theta_c + 2 * theta_c * i / 48
            arc.append((rim_x - sign * r3 * math.cos(th), r3 * math.sin(th)))
        arc.append((bl, y_corner))
        lines.append((np.array(arc), (0, 128, 255), 2))                  # 三分线
    return lines


def _draw_court(img: np.ndarray, H_inv: np.ndarray, court, poly_col=(0, 255, 0)) -> np.ndarray:
    out = img.copy()
    for pts, col, th in _court_lines(court):
        uv, ok = _project(H_inv, pts)
        if not ok.all():
            continue
        p = uv.astype(np.int32)
        # 远离画布的点会让 cv2.line 画出乱线：按段过滤
        for i in range(len(p) - 1):
            if (-4000 < p[i][0] < 8000 and -4000 < p[i][1] < 8000 and
                    -4000 < p[i + 1][0] < 8000 and -4000 < p[i + 1][1] < 8000):
                cv2.line(out, tuple(p[i]), tuple(p[i + 1]), col, th, cv2.LINE_AA)
    return out


def _project(H: np.ndarray, pts: np.ndarray):
    ph = np.column_stack([pts, np.ones(len(pts))])
    pr = ph @ H.T
    w = pr[:, 2]
    ok = np.abs(w) > 1e-9
    uv = pr[:, :2] / np.where(ok, w, 1.0)[:, None]
    return uv, ok & np.all(np.isfinite(uv), axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid", default="harden-55")
    ap.add_argument("--weights", default="checkpoints/calib_seg_hd/best.pt")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--frames", default="")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--size", default="540x960")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--max-score", type=float, default=9.0)
    ap.add_argument("--out-overlay", default=".tmp_calib_overlay.jpg")
    ap.add_argument("--out-filter", default=".tmp_calib_filter.jpg")
    ap.add_argument("--no-detect", action="store_true")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from bsk.court import COURTS, project_court_polygon
    from bsk.line_calib import LineCalibrator
    from bsk.store import Library

    hh, ww = (int(v) for v in args.size.lower().split("x"))
    cal = LineCalibrator(weights=args.weights, size=(hh, ww), conf_thr=args.conf,
                         max_score=args.max_score)

    lib = Library()
    total = lib.table(args.vid).nframes or 8577
    frames = ([int(x) for x in args.frames.split(",")] if args.frames
              else [int(v) for v in np.linspace(60, total - 60, args.n)])
    got = lib.picker(args.vid).frames_at(frames, size=(args.width, int(args.width * 9 / 16)))
    picked = {f: rgb for f, _p, rgb in got}
    print(f"frames={frames}")

    det = None
    if not args.no_detect:
        from bsk.server_rt import _get_rtmo
        det = _get_rtmo()
        print("rtmo loaded")

    overlay_cells, filter_cells = [], []
    passed = 0
    t_pred = t_fit = 0.0
    records = []
    for f in frames:
        rgb = picked.get(f)
        if rgb is None:
            print(f"  f{f}: decode failed")
            continue
        t0 = time.time()
        pred = cal.predict(rgb)
        t1 = time.time()
        res = cal.fit(pred, (hh, ww))
        t2 = time.time()
        t_pred += t1 - t0
        t_fit += t2 - t1
        ok = res is not None and res.calib.is_plausible(rgb.shape[1], rgb.shape[0])
        if ok:
            passed += 1
        n_line = int((pred > 0).sum())
        print(f"  f{f}: fit={'OK' if res else 'none'} plausible={bool(ok)} "
              f"src={(res.source if res else '-')} score={(res.score if res else float('nan')):.2f} "
              f"linepx={n_line} pred={(t1-t0)*1000:.0f}ms fit={(t2-t1)*1000:.0f}ms")

        # ── 叠加图 ──
        over = rgb.copy()
        cols = np.array([[0, 0, 0], [0, 255, 0], [255, 0, 255], [0, 128, 255], [255, 255, 0], [0, 255, 255]], np.uint8)
        up = cv2.resize(cols[np.clip(pred, 0, 5)], (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = cv2.resize((pred > 0).astype(np.uint8), (rgb.shape[1], rgb.shape[0]),
                       interpolation=cv2.INTER_NEAREST) > 0
        if m.any():
            over[m] = (0.35 * over[m] + 0.65 * up[m]).astype(np.uint8)
        if ok:
            court = COURTS.get(res.calib.court_model, COURTS["NBA"])
            over = _draw_court(over, np.array(res.calib.H_inv), court)
            poly = project_court_polygon(np.array(res.calib.H_inv), court)
            cv2.polylines(over, [poly.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.putText(over, f"f{f} {'PASS' if ok else 'FAIL'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if ok else (0, 0, 255), 2)
        overlay_cells.append(cv2.resize(over, (640, 360)))

        # ── 观众剔除模拟 ──
        if det is not None:
            ppl = det(rgb, thr=0.30)
            hgt = rgb.shape[0]
            humans = []
            for p in ppl:
                bw, bh = p.bbox[2] - p.bbox[0], p.bbox[3] - p.bbox[1]
                if p.score < 0.30 or bh < 36 or bw < 18:
                    continue
                if not (0.6 <= bh / max(bw, 1) <= 5.0):
                    continue
                humans.append(p)
            before = rgb.copy()
            after = rgb.copy()
            kept = removed = 0
            margin = rgb.shape[1] * 0.025
            poly = None
            if ok:
                court = COURTS.get(res.calib.court_model, COURTS["NBA"])
                poly = project_court_polygon(np.array(res.calib.H_inv), court)
            for p in humans:
                x0, y0, x1, y1 = [int(v) for v in p.bbox]
                cv2.rectangle(before, (x0, y0), (x1, y1), (0, 255, 0), 2)
                if poly is None:
                    cv2.rectangle(after, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    kept += 1
                    continue
                fx, fy = (p.bbox[0] + p.bbox[2]) / 2, p.bbox[3]
                inside = cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(fx), float(fy)), True) >= -margin
                if inside:
                    cv2.rectangle(after, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    kept += 1
                else:
                    cv2.rectangle(after, (x0, y0), (x1, y1), (0, 0, 255), 2)
                    removed += 1
            if poly is not None:
                cv2.polylines(after, [poly.astype(np.int32)], True, (0, 255, 0), 2)
            cv2.putText(before, f"f{f} before={len(humans)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(after, f"after kept={kept} removed={removed}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cell = np.hstack([cv2.resize(before, (800, 450)), cv2.resize(after, (800, 450))])
            filter_cells.append(cell)
            records.append((f, ok, len(humans), kept, removed))
            print(f"      people={len(humans)} kept={kept} removed={removed}")

    n_ok = passed
    print(f"\n通过 {n_ok}/{len(frames)} = {100.0 * n_ok / max(len(frames), 1):.0f}%")
    print(f"平均 pred={1000*t_pred/max(len(frames),1):.0f}ms fit={1000*t_fit/max(len(frames),1):.0f}ms")

    if overlay_cells:
        while len(overlay_cells) % 3:
            overlay_cells.append(np.zeros_like(overlay_cells[0]))
        grid = np.vstack([np.hstack(overlay_cells[i:i + 3]) for i in range(0, len(overlay_cells), 3)])
        cv2.imwrite(args.out_overlay, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        print("overlay ->", Path(args.out_overlay).resolve())
    if filter_cells:
        while len(filter_cells) % 2:
            filter_cells.append(np.zeros_like(filter_cells[0]))
        grid = np.vstack([np.hstack(filter_cells[i:i + 2]) for i in range(0, len(filter_cells), 2)])
        cv2.imwrite(args.out_filter, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        print("filter ->", Path(args.out_filter).resolve())
    if records:
        tot_b = sum(r[2] for r in records)
        tot_k = sum(r[3] for r in records)
        tot_r = sum(r[4] for r in records)
        print(f"总计 people={tot_b} kept={tot_k} removed={tot_r}")


if __name__ == "__main__":
    main()