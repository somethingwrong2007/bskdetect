"""区间分析：解码 -> 人体框+掩膜 -> 关键点 -> 跨帧 ID -> 写三份 JSONL。

两个模型各出一种框，所以关键点用 IoU 关联到掩膜上，而不是把两套 ID 各自编号：
调试台里同一个人在框表、骨架表、分割表里必须是同一个 track_id，否则人工纠错时
根本对不上是谁。

球场过滤：每帧用增量标定拿单应矩阵，把脚点（bbox 底边中点）落在球场外框之外
的人剔掉——观众席和替补席都在框外。过滤放在跟踪之前，跟踪器和掩膜匹配都不再
为观众付算力。
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from bsk import ingest, schema
from bsk.court import COURTS, FIBA, project_court_polygon
from bsk.perceive.court_map import apply_court_coords, filter_confident_off_court
from bsk.perceive.persons import RtmdetIns, Rtmo
from bsk.realtime_calibrate import calibrate_frame


@dataclass
class Tracker:
    """恒速预测 + Hungarian 全局关联的轻量球员跟踪器。

    距离均按轨迹框对角线归一化，代价融合预测/观测双向 IoU、中心距离和面积比；
    私有 dummy 列允许轨迹或检测不匹配，避免为了凑一对一强行跳 ID。丢失超过
    ``reacquire_lost`` 后采用更严格的重现门控。所有超参数在 TrackID3x3 三段
    缓存上做过小网格验证：``max_lost=16`` 约覆盖 30fps 的 0.53s，且比 45f 显著
    减少陈旧轨迹抢 ID；``vel_ema=0.8`` 在 0.80--0.82 附近结果稳定；IoU/距离/
    尺寸权重 0.65/0.55/0.18 抑制框抖动与遮挡截断。轨迹命中先验最多 0.08，仅用于
    交叉时的平局裁决。默认 ``min_hits=1`` 保持 assign 立即返回有效 ID 的兼容性；
    设为 2 可抑制单帧误检，但未确认框会返回 -1，调用方必须跳过。
    ``assign(boxes)`` 的输入输出长度和检测顺序保持不变；不修改检测框本身。
    """

    iou_thr: float = 0.20
    max_lost: int = 16
    min_hits: int = 1
    match_thr: float = 1.05
    vel_ema: float = 0.80
    dist_weight: float = 0.55
    size_weight: float = 0.18
    gate: float = 0.85
    lost_gate: float = 0.12
    reacquire_lost: int = 10
    reacquire_dist: float = 0.38
    max_step: float = 0.40
    hit_prior: float = 0.01
    hit_cap: float = 0.08
    _live: dict[int, np.ndarray] = field(default_factory=dict)
    _miss: dict[int, int] = field(default_factory=dict)
    _vel: dict[int, np.ndarray] = field(default_factory=dict)
    _hits: dict[int, int] = field(default_factory=dict)
    _next: int = 1
    _frame: int = 0

    @staticmethod
    def _iou(a, b) -> float:
        x0, y0 = max(a[0], b[0]), max(a[1], b[1])
        x1, y1 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        return inter / max(area_a + area_b - inter, 1e-9)

    def reset(self) -> None:
        self._live.clear()
        self._miss.clear()
        self._vel.clear()
        self._hits.clear()
        self._next = 1
        self._frame = 0

    def assign(self, boxes: list[tuple[float, float, float, float]]) -> list[int]:
        self._frame += 1
        detections = [np.asarray(b, dtype=np.float64) for b in boxes]
        tids = list(self._live)
        ids = [-1] * len(detections)
        matched: set[int] = set()

        if tids and detections:
            n_tracks, n_dets = len(tids), len(detections)
            # 1.0 是 dummy 不匹配成本；每条轨迹有私有 dummy，检测无需额外 dummy。
            cost = np.ones((n_tracks, n_dets + n_tracks), dtype=np.float64)
            valid = np.zeros((n_tracks, n_dets), dtype=bool)
            for ti, tid in enumerate(tids):
                old = self._live[tid]
                lost = self._miss.get(tid, 0)
                dt = lost + 1
                pred = old.copy()
                vel = self._vel.get(tid, np.zeros(2)) * dt
                pred[[0, 2]] += vel[0]
                pred[[1, 3]] += vel[1]
                pc = 0.5 * (pred[:2] + pred[2:])
                oc = 0.5 * (old[:2] + old[2:])
                pw, ph = max(pred[2] - pred[0], 1.0), max(pred[3] - pred[1], 1.0)
                diag = float(np.hypot(pw, ph))
                for di, det in enumerate(detections):
                    dc = 0.5 * (det[:2] + det[2:])
                    dw, dh = max(det[2] - det[0], 1.0), max(det[3] - det[1], 1.0)
                    dist = float(np.linalg.norm(dc - pc) / diag)
                    old_dist = float(np.linalg.norm(dc - oc) / diag)
                    motion_dist = min(dist, 0.65 * old_dist)
                    ratio = (dw * dh) / (pw * ph)
                    iou_pred = self._iou(det, pred)
                    iou_old = self._iou(det, old)
                    gate = self.gate + min(0.35, self.lost_gate * lost)
                    near = (
                        motion_dist <= gate
                        and 0.40 <= ratio <= 2.50
                        and (iou_pred >= self.iou_thr or iou_old >= self.iou_thr or motion_dist <= 0.45)
                    )
                    if lost > self.reacquire_lost:
                        near = near and min(iou_pred, iou_old) >= 0.10
                        near = near and motion_dist <= self.reacquire_dist + min(0.20, 0.01 * lost)
                    if not near:
                        continue
                    valid[ti, di] = True
                    pair_cost = (
                        1.0 - 0.65 * iou_pred - 0.35 * iou_old
                        + self.dist_weight * motion_dist
                        + self.size_weight * abs(np.log(max(ratio, 1e-6)))
                        - min(self.hit_cap, self.hit_prior * self._hits.get(tid, 1))
                    )
                    if pair_cost < self.match_thr:
                        cost[ti, di] = pair_cost

            rows, cols = linear_sum_assignment(cost)
            for ti, di in zip(rows, cols):
                if di < n_dets and valid[ti, di] and cost[ti, di] < self.match_thr:
                    ids[di] = tids[ti]
                    matched.add(tids[ti])

        for di, det in enumerate(detections):
            if ids[di] != -1:
                continue
            tid = self._next
            self._next += 1
            self._live[tid] = det.copy()
            self._vel[tid] = np.zeros(2, dtype=np.float64)
            self._hits[tid] = 1
            self._miss[tid] = 0
            matched.add(tid)
            ids[di] = tid if self.min_hits <= 1 else -1

        for di, tid in enumerate(ids):
            if tid == -1:
                continue
            det = detections[di]
            old = self._live[tid]
            old_c = 0.5 * (old[:2] + old[2:])
            new_c = 0.5 * (det[:2] + det[2:])
            diag = float(np.hypot(max(old[2] - old[0], 1.0), max(old[3] - old[1], 1.0)))
            dt = self._miss.get(tid, 0) + 1
            step = np.clip((new_c - old_c) / dt, -self.max_step * diag, self.max_step * diag)
            old_vel = self._vel.get(tid, step)
            self._vel[tid] = self.vel_ema * old_vel + (1.0 - self.vel_ema) * step
            self._live[tid] = det.copy()
            self._miss[tid] = 0
            self._hits[tid] = self._hits.get(tid, 1) + 1

        for tid in list(self._live):
            if tid in matched:
                continue
            self._miss[tid] = self._miss.get(tid, 0) + 1
            confirmed = self._hits.get(tid, 1) >= max(1, self.min_hits)
            limit = self.max_lost if confirmed else min(3, self.max_lost)
            if self._miss[tid] > limit:
                self._live.pop(tid, None)
                self._miss.pop(tid, None)
                self._vel.pop(tid, None)
                self._hits.pop(tid, None)
        return ids


@dataclass(slots=True)
class RangeResult:
    frames: int = 0
    poses: int = 0
    segments: int = 0
    ms_per_frame: float = 0.0
    ep: str = ""


def _feet_in_court(poly: np.ndarray, bbox, margin: float) -> bool:
    """脚点（bbox 底边中点）到球场多边形的有符号距离 >= -margin 即算场内。"""
    x0, y0, x1, y1 = bbox
    return cv2.pointPolygonTest(
        poly.reshape(-1, 1, 2), (float((x0 + x1) / 2), float(y1)), True
    ) >= -margin


def run(
    video: str,
    table: "object",
    shots: list[dict],
    out_dir,
    frm: int = 1,
    to: int | None = None,
    stride: int = 1,
    size: tuple[int, int] | None = None,
    det_thr: float = 0.45,  # Optimized for basketball (reduces false positives)
    with_masks: bool = True,
    video_id: str = "",
    progress=None,
    filter_court: bool = True,
) -> RangeResult:
    t0 = time.perf_counter()
    res = RangeResult()
    to = to or table.nframes
    anchor = table.key_before(frm)
    rtmo = Rtmo()
    ins = RtmdetIns() if with_masks else None
    tracker = Tracker()
    res.ep = rtmo.ep
    cal_key = video_id or video

    recs, poses, segs = [], [], []
    scale_note = None
    n = 0
    for idx, pts_us, rgb in ingest.frames(video, start=anchor, seek_us=table.pts_us[anchor - 1], size=size):
        if idx < frm:
            continue
        if idx > to:
            break
        if (idx - frm) % stride:
            continue
        h, w = rgb.shape[:2]
        ppl = rtmo(rgb, thr=det_thr)

        # 球场范围剔除：观众/替补席脚点在框外，先滤人再跟踪，省跟踪和掩膜匹配的算力
        cal = None
        if filter_court:
            cal = calibrate_frame(cal_key, rgb)
            if cal is not None:
                poly = project_court_polygon(
                    np.array(cal.H_inv, dtype=np.float64),
                    COURTS.get(cal.court_model, FIBA),
                )
                margin = max(24.0, w * 0.025)
                ppl = [p for p in ppl if _feet_in_court(poly, p.bbox, margin)]

        boxes = [p.bbox for p in ppl]
        ids = tracker.assign(boxes)
        masked = {}
        if ins is not None:
            for d in ins(rgb, thr=det_thr):
                masked[d.bbox] = d
        shot_idx = next((i for i, s in enumerate(shots) if s["start_frame"] <= idx <= s["end_frame"]), -1)
        objects = []
        frame_poses = []
        for p, tid in zip(ppl, ids):
            objects.append(
                schema.Tracked(
                    track_id=tid,
                    category="player",
                    frame=idx,
                    bbox_px=p.bbox,
                    conf=p.score,
                    visible=True,
                )
            )
            if p.kpts is not None:
                pose = schema.Pose2D(
                    video_id=video_id, frame=idx, track_id=tid,
                    kpts=np.asarray(p.kpts).tolist(), model="rtmo_s_coco",
                    height_px=float(p.bbox[3] - p.bbox[1]),
                )
                poses.append(pose)
                frame_poses.append(pose)
                res.poses += 1
            if masked:
                m = _match_mask(p.bbox, masked)
                if m is not None and m.polygon:
                    segs.append(
                        schema.Segment2D(
                            video_id=video_id, frame=idx, track_id=tid, category="player",
                            polygon=m.polygon, area_px=m.area_px, model="rtmdet_ins_s",
                        )
                    )
                    res.segments += 1
        if cal is not None:
            court_status = apply_court_coords(objects, frame_poses, cal)
            objects, off_court = filter_confident_off_court(objects, court_status, cal.court_model)
            if off_court:
                dropped_ids = {o.track_id for o in off_court}
                frame_poses = [p for p in frame_poses if p.track_id not in dropped_ids]
        recs.append(
            schema.FrameRecord(
                video_id=video_id, frame=idx, pts_us=pts_us,
                shot_id=shot_idx,
                width=w, height=h, objects=objects,
            )
        )
        res.frames += 1
        if progress and res.frames % 10 == 0:
            progress(idx, frm, to)
    if recs:
        # 与已有结果合并：区间外的旧记录保留，可断点续跑（任务被打断也不丢已完成部分）
        _merge_jsonl(Path(out_dir) / "frames.jsonl", recs, frm, to)
        _merge_jsonl(Path(out_dir) / "poses.jsonl", poses, frm, to)
        _merge_jsonl(Path(out_dir) / "segments.jsonl", segs, frm, to)
    res.ms_per_frame = (time.perf_counter() - t0) * 1000 / max(res.frames, 1)
    return res


def _merge_jsonl(path: Path, new_records, frm: int, to: int) -> int:
    """把 [frm, to] 内的新记录写入 JSONL，区间外旧记录原样保留。"""
    old: list[tuple[int, dict]] = []
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if frm <= int(obj.get("frame", 0)) <= to:
                    continue
                old.append((int(obj.get("frame", 0)), obj))
    old += [(r.frame, dataclasses.asdict(r)) for r in new_records]
    old.sort(key=lambda t: t[0])
    with open(path, "w", encoding="utf-8") as fh:
        for _, obj in old:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return len(old)


def _match_mask(bbox, masked):
    """把 RTMO 的关键点挂到 RTMDet-Ins 的同一个人框上。"""
    from bsk.perceive.persons import _iou

    best, bi = 0.3, None
    for b, d in masked.items():
        v = _iou(bbox, b)
        if v > best:
            best, bi = v, b
    return masked.get(bi) if bi is not None else None


if __name__ == "__main__":
    import argparse

    from bsk.store import Library

    ap = argparse.ArgumentParser(description="批量分析（可与已有结果合并、断点续跑）")
    ap.add_argument("vid")
    ap.add_argument("--frm", type=int, default=1)
    ap.add_argument("--to", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--no-masks", action="store_true")
    ap.add_argument("--no-court-filter", action="store_true",
                    help="跳过球场范围剔除（标定不可用时更快）")
    args = ap.parse_args()

    lib = Library()
    v = lib.get(args.vid)
    table = lib.table(args.vid)
    shots = lib.shots(args.vid).get("shots", [])

    def prog(cur, start, end):
        if cur % 100 == 0 or cur >= end:
            print(f"[analyze] {cur}/{end}", flush=True)

    res = run(
        v.path, table, shots, v.dir,
        frm=args.frm, to=args.to or table.nframes, stride=args.stride,
        det_thr=args.thr, with_masks=not args.no_masks,
        video_id=args.vid, progress=prog,
        filter_court=not args.no_court_filter,
    )
    lib.reload(args.vid)
    print(f"[analyze] done: frames={res.frames} poses={res.poses} "
          f"segs={res.segments} ms/frame={res.ms_per_frame:.0f}", flush=True)


__all__ = ["RangeResult", "Tracker", "run"]
