"""把球员脚点从像素反投影到球场平面（米制 2D 坐标）。

标定来源是线识别模型输出的 Calibration：H 为 pixel->court，H_inv 为
court->pixel。这里只使用地平面假设，不依赖 3D 骨架，因此 4GB profile
也能拿到稳定的俯视坐标。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from bsk import court, schema


COCO_LEFT_ANKLE = 15
COCO_RIGHT_ANKLE = 16
ANKLE_CONF = 0.5
BOUND_MARGIN_M = 4.0


def foot_pixel(bbox: Iterable[float],
               kpts: Iterable[Iterable[float]] | None = None) -> tuple[float, float, str]:
    """脚点像素：底边中点；双踝都可信时用踝点横向中点修正左右站位。"""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    u = 0.5 * (x0 + x1)
    v = y1
    source = "bbox"
    if kpts is None:
        return u, v, source
    k = np.asarray(kpts, dtype=np.float64)
    if k.ndim != 2 or len(k) <= COCO_RIGHT_ANKLE:
        return u, v, source
    left, right = k[COCO_LEFT_ANKLE], k[COCO_RIGHT_ANKLE]
    if left[2] < ANKLE_CONF or right[2] < ANKLE_CONF:
        return u, v, source
    ankle_u = 0.5 * (float(left[0]) + float(right[0]))
    width = max(x1 - x0, 1.0)
    if not (x0 - 0.2 * width <= ankle_u <= x1 + 0.2 * width):
        return u, v, source
    return ankle_u, v, "ankles"


def pixel_to_court(H: np.ndarray, u: float, v: float) -> tuple[float, float] | None:
    """单点反投影；分母过小或结果非有限值时返回 None。"""
    h = np.asarray(H, dtype=np.float64)
    if h.shape != (3, 3) or not np.all(np.isfinite(h)):
        return None
    w = h[2, 0] * u + h[2, 1] * v + h[2, 2]
    if abs(w) < 1e-9:
        return None
    x = (h[0, 0] * u + h[0, 1] * v + h[0, 2]) / w
    y = (h[1, 0] * u + h[1, 1] * v + h[1, 2]) / w
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return float(x), float(y)


def apply_court_coords(objects: list[schema.Tracked],
                       poses: list[schema.Pose2D],
                       cal,
                       margin_m: float = BOUND_MARGIN_M) -> dict:
    """原地写入 Tracked.x/y。cal.H 必须是当前帧尺度的 pixel->court。"""
    st: dict = {"ok": False, "n": 0, "rejected": 0, "reason": "no_calib"}
    if cal is None:
        return st
    try:
        H = np.asarray(cal.H, dtype=np.float64)
        court_model = getattr(cal, "court_model", "FIBA")
        court_obj = court.COURTS.get(court_model, court.FIBA)
    except Exception as exc:
        st.update(reason="calib_invalid", detail=f"{type(exc).__name__}: {exc}")
        return st
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        st["reason"] = "h_invalid"
        return st

    kpts_by_id = {p.track_id: p.kpts for p in poses}
    half_l = 0.5 * float(court_obj.length)
    half_w = 0.5 * float(court_obj.width)
    mapped = rejected = 0
    for obj in objects:
        if obj.category != "player" or obj.bbox_px is None:
            continue
        u, v, _src = foot_pixel(obj.bbox_px, kpts_by_id.get(obj.track_id))
        xy = pixel_to_court(H, u, v)
        if xy is None:
            rejected += 1
            continue
        x, y = xy
        if abs(x) > half_l + margin_m or abs(y) > half_w + margin_m:
            rejected += 1
            continue
        obj.x = round(x, 4)
        obj.y = round(y, 4)
        mapped += 1

    st.update(ok=True, n=mapped, rejected=rejected, court_model=court_model)
    if mapped == 0:
        st["reason"] = "no_players" if not objects else "all_rejected"
    else:
        st["reason"] = "ok"
    return st


__all__ = ["ANKLE_CONF", "BOUND_MARGIN_M", "apply_court_coords", "foot_pixel",
           "pixel_to_court"]