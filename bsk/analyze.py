"""区间分析：解码 -> 人体框+掩膜 -> 关键点 -> 跨帧 ID -> 写三份 JSONL。

两个模型各出一种框，所以关键点用 IoU 关联到掩膜上，而不是把两套 ID 各自编号：
调试台里同一个人在框表、骨架表、分割表里必须是同一个 track_id，否则人工纠错时
根本对不上是谁。

球场过滤：每帧用增量标定拿单应矩阵，把脚点（bbox 底边中点）落在球场外框之外
的人剔掉——观众席和替补席都在框外。过滤放在跟踪之前，跟踪器和掩膜匹配都不再
为观众付算力。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from bsk import ingest, schema
from bsk.court import COURTS, FIBA, project_court_polygon
from bsk.perceive.persons import RtmdetIns, Rtmo
from bsk.realtime_calibrate import calibrate_frame


@dataclass
class Tracker:
    """贪心 IoU 关联：够用的最小跟踪器，之后换 ReID 也不改对外契约。"""

    iou_thr: float = 0.35
    max_lost: int = 12
    _live: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)
    _miss: dict[int, int] = field(default_factory=dict)
    _next: int = 1

    def assign(self, boxes: list[tuple[float, float, float, float]]) -> list[int]:
        from bsk.perceive.persons import _iou

        ids = []
        taken = set()
        for b in boxes:
            best, bi = self.iou_thr, None
            for tid, tb in self._live.items():
                if tid in taken:
                    continue
                v = _iou(b, tb)
                if v > best:
                    best, bi = v, tid
            if bi is None:
                bi = self._next
                self._next += 1
            taken.add(bi)
            self._live[bi] = b
            self._miss[bi] = 0
            ids.append(bi)
        for tid in list(self._miss):
            self._miss[tid] += 1
            if self._miss[tid] > self.max_lost:
                self._live.pop(tid, None)
                self._miss.pop(tid, None)
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
                poses.append(
                    schema.Pose2D(
                        video_id=video_id, frame=idx, track_id=tid,
                        kpts=np.asarray(p.kpts).tolist(), model="rtmo_s_coco",
                        height_px=float(p.bbox[3] - p.bbox[1]),
                    )
                )
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
        schema.write_obj_jsonl(f"{out_dir}/frames.jsonl", recs)
        schema.write_obj_jsonl(f"{out_dir}/poses.jsonl", poses)
        schema.write_obj_jsonl(f"{out_dir}/segments.jsonl", segs)
    res.ms_per_frame = (time.perf_counter() - t0) * 1000 / max(res.frames, 1)
    return res


def _match_mask(bbox, masked):
    """把 RTMO 的关键点挂到 RTMDet-Ins 的同一个人框上。"""
    from bsk.perceive.persons import _iou

    best, bi = 0.3, None
    for b, d in masked.items():
        v = _iou(bbox, b)
        if v > best:
            best, bi = v, b
    return masked.get(bi) if bi is not None else None


__all__ = ["RangeResult", "Tracker", "run"]
