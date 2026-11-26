"""标定验收门：真实帧上的 H 拟合通过率 + 线重合叠加图 + 观众剔除模拟。

做的事：
1. 在指定视频上均匀抽 N 帧（默认 12），逐帧跑 LineCalibrator 拟合；
2. 逐帧打印失败分支诊断（证据像素量 / Hough 线族 / 初值数 / 各候选代价分解 /
   被哪一道关卡拒掉），对应 `LineCalibrator.last_diag`；
3. 统计通过率（fit 成功 + is_plausible + 代价阈值）。近景/球场大半出画的帧可用
   `--na` 标为 NOT_APPLICABLE，必须写明理由，且不计入通过率分母；
4. 把投影出的球场线画到原图上，逐帧存图 + 拼成 .tmp_calib_overlay.jpg 供人眼检查；
5. 用 RTMO 检测人，按"脚点（bbox 底边中点）在球场多边形外且超出 margin"
   剔除，画成 before/after 对比图 .tmp_calib_filter.jpg，检验是否误删场上球员。

用法：
  .venv/Scripts/python.exe -m bsk.dev.calib_gate --n 12 --weights checkpoints/calib_seg_hd/best.pt
  .venv/Scripts/python.exe -m bsk.dev.calib_gate --n 12 --seq --na "5441:近景,6210:近景"
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
    rc = court.center_circle_center_radius
    cc = np.array([(rc * math.cos(2 * math.pi * i / n), rc * math.sin(2 * math.pi * i / n))
                   for i in range(n + 1)])
    lines.append((cc, (0, 255, 255), 2))                                 # 中圈
    for side in ("A", "B"):
        bl = court.baseline_x(side)
        sign = 1 if side == "A" else -1
        ft_x = bl - court.free_throw_center_dist * sign
        fthw = court.paint_width / 2 + court.line_half
        lane = np.array([(bl, -fthw), (bl, fthw), (ft_x, fthw), (ft_x, -fthw), (bl, -fthw)])
        lines.append((lane, (255, 0, 255), 2))                           # 罚球区

        rim_x, _ = court.rim_pos(side)
        r3 = court.three_pt_center_radius
        y_corner = hw - court.three_pt_corner_center_dist
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
    ap.add_argument("--weights", default="")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--frames", default="")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--size", default="540x960")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--max-score", type=float, default=9.0)
    ap.add_argument("--max-inits", type=int, default=24)
    ap.add_argument("--seq", action="store_true",
                    help="顺序传播上一帧外框作为先验（复现 CalibrationTracker 的行为）")
    ap.add_argument("--no-auto-na", dest="auto_na", action="store_false",
                    help="关闭自动 NOT_APPLICABLE 判定")
    ap.set_defaults(auto_na=True)
    ap.add_argument("--na", default="",
                    help='手动 NOT_APPLICABLE 帧与理由，如 "5441:近景人物占满,6210:近景"')
    ap.add_argument("--out-overlay", default=".tmp_calib_overlay.jpg")
    ap.add_argument("--out-filter", default=".tmp_calib_filter.jpg")
    ap.add_argument("--out-dir", default="temp/gate_overlays")
    ap.add_argument("--no-detect", action="store_true")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from bsk.court import COURTS, project_court_polygon
    from bsk.line_calib import DEFAULT_WEIGHTS, LineCalibrator
    from bsk.store import Library

    na: dict[int, str] = {}
    for item in filter(None, (s.strip() for s in args.na.split(","))):
        f_s, _, why = item.partition(":")
        try:
            na[int(f_s)] = why.strip() or "未给理由"
        except ValueError:
            print(f"[warn] --na 解析失败: {item}")

    hh, ww = (int(v) for v in args.size.lower().split("x"))
    cal = LineCalibrator(weights=args.weights or DEFAULT_WEIGHTS, size=(hh, ww),
                         conf_thr=args.conf, max_score=args.max_score,
                         max_inits=args.max_inits, auto_na=args.auto_na)
    cal.diagnose = True

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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    passed = 0
    applicable: list[int] = []
    t_pred = t_fit = 0.0
    fit_ms: list[float] = []
    records = []
    prior = None
    for f in frames:
        rgb = picked.get(f)
        if rgb is None:
            print(f"  f{f}: decode failed")
            continue
        t0 = time.time()
        pred = cal.predict(rgb)
        t1 = time.time()
        res = cal.fit(pred, (hh, ww), rgb=rgb, prior=prior,
                      prior_trusted=bool(args.seq and prior is not None))
        t2 = time.time()
        t_pred += t1 - t0
        t_fit += t2 - t1
        fit_ms.append(1000 * (t2 - t1))
        na_info = (getattr(cal, "last_diag", {}) or {}).get("na") or {}
        if args.auto_na and na_info and not na_info.get("ok", True) and f not in na:
            na[f] = f"auto:{na_info.get('reason', 'not_applicable')}"
        ok = res is not None and res.calib.is_plausible(rgb.shape[1], rgb.shape[0])
        if f in na:
            # 不可标定帧：不参与通过率，且绝不能算通过
            ok = False
        else:
            applicable.append(f)
            if ok:
                passed += 1
        if res is not None and args.seq and ok and f not in na:
            court = COURS["NBA"]
            ct = next((c for c in cal.courts if c.model == res.calib.court_model), None) or court
            hl, hw = ct.half_length, ct.half_width
            rect = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]], np.float64)
            q, okq = _project(np.array(res.calib.H_inv, np.float64), rect)
            prior = (q, res.calib.court_model) if okq.all() else None
        n_line = int((pred > 0).sum())
        status = "NOT_APPLICABLE" if f in na else ("PASS" if ok else "FAIL")
        print(f"  f{f}: {status} fit={'OK' if res else 'none'} "
              f"plausible={bool(res.calib.is_plausible(rgb.shape[1], rgb.shape[0])) if res else False} "
              f"src={(res.source if res else '-')} score={(res.score if res else float('nan')):.2f} "
              f"linepx={n_line} pred={(t1-t0)*1000:.0f}ms fit={(t2-t1)*1000:.0f}ms"
              + (f"  NA理由={na[f]}" if f in na else ""))
        d = getattr(cal, "last_diag", {}) or {}
        if d:
            hu = d.get("hough") or {}
            stg = d.get("strong") or {}
            na_d = d.get("na") or {}
            print(f"      ev={d.get('ev', 0)}px long={stg.get('n_long', 0)} "
                  f"max={stg.get('max_len', 0)}px init={d.get('n_init', 0)} "
                  f"quick_ok={d.get('n_quick', 0)} cands={len(d.get('cands', []))} "
                  f"prior_fast={bool(d.get('prior_fast'))} "
                  f"na={na_d.get('reason', '-') or 'ok'} reject='{d.get('reject', '') or '-'}'")
            for c in d.get("cands", [])[:3]:
                print(f"        cand {c['src']:<18} coarse={c.get('score_coarse', c['score']):6.2f} "
                      f"full={c['score']:6.2f} recall={c['recall']:5.2f} "
                      f"prec={(c['precision'] if c['precision'] is not None else float('nan')):5.2f} "
                      f"rev={(c['rev'] if c['rev'] is not None else float('nan')):5.2f} "
                      f"inb={c['inb']:.2f}")

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
        cv2.putText(over, f"f{f} {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if status == "PASS" else (0, 0, 255), 2)
        cv2.imwrite(str(out_dir / f"f{f:05d}.jpg"), cv2.cvtColor(over, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
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

    n_all = len(frames)
    n_app = len(applicable)
    n_na = n_all - n_app
    print(f"\n通过 {passed}/{n_app} = {100.0 * passed / max(n_app, 1):.0f}%  "
          f"(NOT_APPLICABLE {n_na}/{n_all}，不计入分母)")
    if na:
        for f, why in sorted(na.items()):
            print(f"  NA f{f}: {why}")
    if fit_ms:
        ms = np.array(fit_ms)
        print(f"平均 pred={1000*t_pred/max(n_all,1):.0f}ms fit={ms.mean():.0f}ms "
              f"p90={np.percentile(ms, 90):.0f}ms max={ms.max():.0f}ms "
              f"(30fps 预算 33ms/帧；后台槽目标 ≤80ms)")
        print(f"超 80ms 帧数={int((ms > 80).sum())}/{len(ms)}")

    # ── 生产路径耗时（诊断关闭，另跑一遍，反映实时链路的真实开销）──
    cal.diagnose = False
    tm = []
    for f in frames[:3]:
        rgb = picked.get(f)
        if rgb is None:
            continue
        pred = cal.predict(rgb)
        t0 = time.time()
        cal.fit(pred, (hh, ww), rgb=rgb)
        tm.append(1000 * (time.time() - t0))
    if tm:
        print(f"生产路径 fit（诊断关闭，前 {len(tm)} 帧）：mean={np.mean(tm):.0f}ms "
              f"max={np.max(tm):.0f}ms")

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