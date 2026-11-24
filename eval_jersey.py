"""号码识别评估：裁片级精度 / 混淆矩阵 / 轨迹级融合指标 / 延迟。

用法：
    .venv/Scripts/python.exe eval_jersey.py                      # 合成留出集（默认）
    .venv/Scripts/python.exe eval_jersey.py --mode manifest --manifest jersey_samples/labels.csv
    .venv/Scripts/python.exe eval_jersey.py --mode fusion        # 用真实模型输出仿真轨迹
    .venv/Scripts/python.exe eval_jersey.py --mode video --vid harden-55 --gt jersey_gt.json

口径说明（对齐 docs/jersey_number_detection.md §6）：
- 单帧 top-1：可读裁片上的 argmax 命中率；空白（读不出）样本单独算 precision/recall，
  不混进 top-1，否则"什么都不猜"会被算成满分。
- 轨迹级：把模型对同一号码的多张真实裁片按时间序喂进 JerseyRecognizer，看发布后的
  最终号码是否正确、切换次数是否 ≤1、发布需要多少帧。这里用的是真实推理输出而不是
  理想化噪声模型，所以它衡量的是融合层在真实观测分布下的表现。
- 延迟：分别报单张裁片前向与整帧新增开销（按 every_n 采样与人数折算到每帧）。

注意：--mode synth 只证明"管线正确 + 字形可学"，真实素材的迁移精度必须靠 --mode manifest
（人工标注的真实裁片）或 --mode video（整段轨迹）来出，两者都需要 data/raw 与标注，
不在仓库里。报告写到 .tmp_jersey_report.json（不入库）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from bsk import schema
from bsk.jersey_synth import SynthConfig, find_fonts, make_crop
from bsk.perceive.jersey import (INPUT_H, INPUT_W, JerseyConfig, JerseyRecognizer,
                                 crop_torso, logits_to_dist, preprocess, torso_bbox)

MAX_NUM = 99


def held_out_synth(n_per_number: int, seed: int, cfg: SynthConfig):
    """生成与训练 seed 不同的合成集，避免同图同增广带来的乐观偏差。"""
    rng = np.random.default_rng(seed)
    fonts = find_fonts(cfg)
    imgs, labs = [], []
    for num in range(0, MAX_NUM + 1):
        for _ in range(n_per_number):
            im, _ = make_crop(num, rng, cfg, fonts)
            imgs.append(im)
            labs.append(num)
    for _ in range(n_per_number * 100):
        im, _ = make_crop(None, rng, cfg, fonts)
        imgs.append(im)
        labs.append(-1)
    return imgs, np.array(labs, np.int64)


def load_labeled_crops(manifest: str | Path, root: str | Path | None = None):
    """读真实标注裁片（csv/jsonl），label 为 0–99；空/非数字视为未标注跳过。"""
    path = Path(manifest)
    root = Path(root) if root else path.parent
    rows = []
    if path.suffix == ".jsonl":
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        import csv
        with open(path, encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
    imgs, labs = [], []
    for r in rows:
        raw = str(r.get("label", "")).strip()
        if not raw.lstrip("-").isdigit():
            continue
        p = Path(r.get("crop") or r.get("image") or r.get("path"))
        if not p.is_absolute():
            p = root / p
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            continue
        imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        labs.append(int(raw))
    return imgs, np.array(labs, np.int64)


# ── 裁片级 ──────────────────────────────────────────────────

def eval_crops(rec: JerseyRecognizer, imgs: list, labs: np.ndarray, batch: int = 32) -> dict:
    """逐批前向，统计 top-1、位级准确率、混淆矩阵、空白指标与延迟。"""
    model = rec.model
    preds, confs, lat = [], [], []
    for i in range(0, len(imgs), batch):
        chunk = imgs[i:i + batch]
        t0 = time.perf_counter()
        logits = model.predict(preprocess(chunk))
        lat.append((time.perf_counter() - t0) * 1000.0 / len(chunk))
        pnum, pblank = logits_to_dist(logits)
        best = pnum.argmax(1)
        conf = pnum[np.arange(len(pnum)), best]
        # 空白优先：判成空白就不给号码，与 recognize() 的发布口径保持一致
        take_blank = pblank > 0.5
        best = np.where(take_blank, -1, best)
        conf = np.where(take_blank, pblank, conf)
        preds.append(best)
        confs.append(conf)
    pred = np.concatenate(preds)
    conf = np.concatenate(confs)

    readable = labs >= 0
    top1 = float((pred[readable] == labs[readable]).mean()) if readable.any() else 0.0
    hi = readable & (conf > 0.6)
    hi_err = float((pred[hi] != labs[hi]).mean()) if hi.any() else 0.0
    blank_tp = int(((~readable) & (pred < 0)).sum())
    blank_fp = int((readable & (pred < 0)).sum())
    # 混淆矩阵只统计"确实吐出了号码"的样本：判成空白的可读裁片已经进 blank_fp，
    # 若把它们留在这里，t=-1 会绕回数组末尾，把误判成空白的次数偷偷记进 9 行/9 列。
    sel = readable & (pred >= 0)
    tens_c, units_c = np.zeros((10, 10), int), np.zeros((10, 10), int)
    for t, g in zip(pred[sel], labs[sel]):
        gt_tens, gt_units = (0, int(g)) if g < 10 else (int(g) // 10, int(g) % 10)
        pt_tens, pt_units = (0, int(t)) if t < 10 else (int(t) // 10, int(t) % 10)
        tens_c[gt_tens, pt_tens] += 1
        units_c[gt_units, pt_units] += 1
    return {
        "n": int(len(labs)),
        "n_readable": int(readable.sum()),
        "top1_readable": round(top1, 4),
        "tens_acc": round(float(np.trace(tens_c) / max(tens_c.sum(), 1)), 4),
        "units_acc": round(float(np.trace(units_c) / max(units_c.sum(), 1)), 4),
        "high_conf_error_rate": round(hi_err, 4),
        "blank_recall": round(blank_tp / max(int((~readable).sum()), 1), 4),
        "blank_false_positive": blank_fp,
        "ms_per_crop_mean": round(statistics.fmean(lat), 3),
        "ms_per_crop_p95": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 3),
        "confusion_tens": tens_c.tolist(),
        "confusion_units": units_c.tolist(),
    }


# ── 轨迹级（用真实模型输出驱动融合层）──────────────────────────

def _bbox_for_torso(tx0: int, ty0: int, tw: int, th: int, cfg: JerseyConfig):
    """给定躯干窗口反推人体框，让 recognize() 走真实的 crop_torso 路径。"""
    w, h = tw / (1.0 - 2 * cfg.x_inset), th / (cfg.y_bot - cfg.y_top)
    x0, y0 = tx0 - cfg.x_inset * w, ty0 - cfg.y_top * h
    return (x0, y0, x0 + w, y0 + h)


def eval_fusion(rec: JerseyRecognizer, imgs: list, labs: np.ndarray, n_tracks: int = 40,
                frames_per_track: int = 90, seed: int = 5, obs_rate: float = 0.55) -> dict:
    """把同一号码的多张真实裁片按时间序喂进融合层，看发布质量。

    obs_rate 模拟"这一帧能不能读到号码"（采样节拍、遮挡、侧身都会压低它），真实素材上
    这个值明显小于 1，所以轨迹级指标不能只看单帧精度。
    """
    rng = np.random.default_rng(seed)
    cfg = rec.cfg
    by_num: dict[int, list] = {}
    for im, lab in zip(imgs, labs):
        if lab >= 0:
            by_num.setdefault(int(lab), []).append(im)
    # 每类至少 2 张才进轨迹仿真：只有一张时同一帧反复喂同图，融合指标会好看得没意义
    numbers = [n for n, v in by_num.items() if len(v) >= 2]
    if len(numbers) < 4:
        return {"skipped": "可用号码类别不足"}

    rec.reset()
    published, correct, switches, frames_to_pub, never = 0, 0, [], [], 0
    per_frame_ms: list[float] = []
    canvas_h, canvas_w = 480, 640
    for k in range(n_tracks):
        num = int(numbers[int(rng.integers(0, len(numbers)))])
        pool = by_num[num]
        gt = num
        objs = [schema.Tracked(track_id=k, category="player", frame=1, bbox_px=None)]
        bbox = _bbox_for_torso(64, 96, pool[0].shape[1], pool[0].shape[0], cfg)
        objs[0].bbox_px = bbox
        first = None
        for f in range(1, frames_per_track + 1):
            rgb = np.zeros((canvas_h, canvas_w, 3), np.uint8)
            objs[0].frame = f
            objs[0].jersey, objs[0].jersey_conf = None, 0.0
            if rng.random() < obs_rate:
                im = pool[int(rng.integers(0, len(pool)))]
                x0, y0 = int(bbox[0]), int(bbox[1])
                h = min(im.shape[0], canvas_h - y0)
                w = min(im.shape[1], canvas_w - x0)
                rgb[y0:y0 + h, x0:x0 + w] = im[:h, :w]
            t0 = time.perf_counter()
            rec.recognize("eval", f, rgb, objs)
            per_frame_ms.append((time.perf_counter() - t0) * 1000.0)
            if objs[0].jersey is not None and first is None:
                first = f
        final = objs[0].jersey
        ev = [e for e in rec.events if e["track_id"] == k]
        sw = sum(1 for e in ev if e["kind"] == "switch")
        switches.append(sw)
        if final is None:
            never += 1
        else:
            published += 1
            correct += int(final == gt)
            frames_to_pub.append(first if first is not None else -1)
    return {
        "tracks": n_tracks,
        "published": published,
        "publish_rate": round(published / n_tracks, 4),
        "track_accuracy_among_published": round(correct / max(published, 1), 4),
        "never_published": never,
        "switches_max": int(max(switches) if switches else 0),
        "switches_mean": round(float(np.mean(switches)) if switches else 0.0, 3),
        "frames_to_publish_median": int(np.median(frames_to_pub)) if frames_to_pub else -1,
        "ms_per_frame_mean": round(statistics.fmean(per_frame_ms), 3),
        "ms_per_frame_p95": round(sorted(per_frame_ms)[int(0.95 * (len(per_frame_ms) - 1))], 3),
    }


# ── 门控自检（不依赖权重）────────────────────────────────────

def eval_gating(imgs: list, labs: np.ndarray, seed: int = 3) -> dict:
    """侧身/背影与过小裁片必须被丢弃：用几何构造验证，不需要模型。"""
    cfg = JerseyConfig()
    ok = side = small = 0
    for im in imgs[:200]:
        h, w = im.shape[:2]
        # 正面：肩宽占框宽 0.5；侧身：0.05；框很小：躯干不足 min_crop
        k_front = np.zeros((17, 3)); k_front[5] = [w * 0.25, h * 0.2, 0.9]; k_front[6] = [w * 0.75, h * 0.2, 0.9]
        k_side = np.zeros((17, 3)); k_side[5] = [w * 0.48, h * 0.2, 0.9]; k_side[6] = [w * 0.52, h * 0.2, 0.9]
        bbox = (0.0, 0.0, float(w), float(h * 2.2))
        if crop_torso(im, bbox, k_front, cfg).ok:
            ok += 1
        if not crop_torso(im, bbox, k_side, cfg).ok:
            side += 1
        tiny = (0.0, 0.0, 14.0, 30.0)
        if torso_bbox(tiny, cfg) is None:
            small += 1
    return {"front_kept": ok, "side_rejected": side, "tiny_rejected": small, "n": min(200, len(imgs))}


# ── 真实视频（需要 data/raw 与标注）────────────────────────────

def eval_video(rec: JerseyRecognizer, vid: str, segments: list[int], gt_path: str | None,
               width: int = 960) -> dict:
    from bsk import ingest
    from bsk.analyze import Tracker
    from bsk.perceive.persons import Rtmo
    from bsk.store import Library

    gt = {}
    if gt_path and Path(gt_path).exists():
        gt = {int(k): int(v) for k, v in json.loads(Path(gt_path).read_text(encoding="utf-8")).items()}
    lib = Library()
    v = lib.get(vid)
    table = lib.table(vid)
    rtmo = Rtmo()
    tracker = Tracker()
    rec.reset(vid)
    per_frame_ms, n_frames = [], 0
    for start in segments:
        anchor = table.key_before(start)
        size = (width, int(round(width * v.height / max(v.width, 1))))
        for idx, _pts, rgb in ingest.frames(v.path, start=anchor, size=size, stride=1,
                                            seek_us=table.pts_us[anchor - 1], limit=120):
            ppl = rtmo(rgb, thr=0.3)
            ids = tracker.assign([p.bbox for p in ppl])
            objs = [schema.Tracked(track_id=t, category="player", frame=idx,
                                   bbox_px=p.bbox, conf=p.score, visible=True)
                    for p, t in zip(ppl, ids)]
            poses = [schema.Pose2D(video_id=vid, frame=idx, track_id=t,
                                   kpts=p.kpts.tolist(), height_px=p.bbox[3] - p.bbox[1])
                     for p, t in zip(ppl, ids) if p.kpts is not None]
            t0 = time.perf_counter()
            rec.recognize(vid, idx, rgb, objs, poses)
            per_frame_ms.append((time.perf_counter() - t0) * 1000.0)
            n_frames += 1
    final = rec.published_numbers(vid)
    acc = None
    if gt:
        hit = [tid for tid, j in final.items() if gt.get(tid) == j]
        acc = round(len(hit) / max(len(gt), 1), 4)
    return {"vid": vid, "frames": n_frames, "tracks_with_number": len(final),
            "ms_per_frame_mean": round(statistics.fmean(per_frame_ms), 3) if per_frame_ms else 0,
            "track_accuracy_vs_gt": acc,
            "note": None if gt else "没有 gt 标注文件，只报覆盖率与延迟"}


def main():
    ap = argparse.ArgumentParser(description="号码识别评估")
    ap.add_argument("--mode", default="synth", choices=["synth", "manifest", "fusion", "video", "all"])
    ap.add_argument("--weights", default="data/models/jersey_small.onnx")
    ap.add_argument("--manifest", default="jersey_samples/labels.csv")
    ap.add_argument("--per-number", type=int, default=3, help="合成留出集每类张数")
    ap.add_argument("--seed", type=int, default=991, help="留出集种子，必须与训练种子不同")
    ap.add_argument("--tracks", type=int, default=40)
    ap.add_argument("--vid", default="harden-55")
    ap.add_argument("--segments", default="400,2000,5000")
    ap.add_argument("--gt", default=None, help="track_id → 号码 的 json，用于轨迹级准确率")
    ap.add_argument("--no-trt", action="store_true")
    args = ap.parse_args()

    cfg = JerseyConfig(weights=args.weights, use_trt=not args.no_trt)
    rec = JerseyRecognizer(cfg=cfg)
    if not rec.ready:
        raise SystemExit(f"模型不可用：{rec.unavailable_reason}；先跑 bsk.train_jersey 导出 ONNX")
    print(f"[model] ep={rec.model.ep} 输入 {INPUT_H}x{INPUT_W} 权重 {cfg.weights}")

    scfg = SynthConfig()
    report: dict = {"weights": str(cfg.weights), "ep": rec.model.ep}
    imgs, labs = None, None
    if args.mode in ("synth", "all", "fusion"):
        print(f"[data] 合成留出集 per-number={args.per_number} seed={args.seed} …")
        imgs, labs = held_out_synth(args.per_number, args.seed, scfg)
    elif args.mode == "manifest":
        imgs, labs = load_labeled_crops(args.manifest)
        if len(imgs) == 0:
            raise SystemExit(f"{args.manifest} 里没有已标注裁片：先跑采集再填 label 列")

    if imgs is not None:                      # --mode video 不生成裁片集，门控自检跳过
        report["gating"] = eval_gating(imgs, labs)
    if args.mode in ("synth", "manifest", "all"):
        print("[eval] 裁片级 …")
        report["crops"] = eval_crops(rec, imgs, labs)
        print(json.dumps({k: v for k, v in report["crops"].items()
                          if not k.startswith("confusion")}, ensure_ascii=False, indent=2))
    if args.mode in ("fusion", "all"):
        print("[eval] 轨迹级融合仿真 …")
        report["fusion"] = eval_fusion(rec, imgs, labs, n_tracks=args.tracks)
        print(json.dumps(report["fusion"], ensure_ascii=False, indent=2))
    if args.mode in ("video", "all"):
        try:
            report["video"] = eval_video(rec, args.vid, [int(x) for x in args.segments.split(",")],
                                         args.gt)
            print(json.dumps(report["video"], ensure_ascii=False, indent=2))
        except Exception as exc:
            report["video"] = {"skipped": f"{type(exc).__name__}: {exc}"}
            print(f"[video] 跳过：{report['video']['skipped']}"
                  "（需要 data/raw 素材与 Rtmo 权重，均不入库）")

    Path(".tmp_jersey_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[done] 报告 → .tmp_jersey_report.json")


if __name__ == "__main__":
    main()
