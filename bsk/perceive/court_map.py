"""把球员脚点反投影到球场平面，并按 track_id 做在线世界坐标滤波。

标定来源是线识别模型输出的 Calibration：H 为 pixel->court，H_inv 为
court->pixel。只用当前与历史状态，不预取未来帧。``CourtMapState`` 应按
视频/后端分别持有；无状态调用仍兼容旧 analyze 流程，但不会跨帧平滑。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable

import numpy as np

from bsk import court, schema


COCO_LEFT_ANKLE = 15
COCO_RIGHT_ANKLE = 16
ANKLE_CONF = 0.5
SINGLE_ANKLE_CONF = 0.65
BOUND_MARGIN_M = 4.0
MAX_CALIB_AGE_S = 2.5
MAX_REPROJ_PX = 25.0
MAX_PREDICTION_AGE_S = 0.6
MAX_SPEED_M_S = 12.0
MAX_LOCAL_M_PER_PX = 0.35


def _finite_point(point) -> bool:
    return len(point) >= 2 and bool(np.all(np.isfinite(np.asarray(point[:2], dtype=float))))


def foot_pixel(bbox: Iterable[float],
               kpts: Iterable[Iterable[float]] | None = None,
               image_shape: tuple[int, int] | None = None,
               allow_ankles: bool = True) -> tuple[float, float, str]:
    """选择可信脚点。

    双踝取两点中点，单踝也可作为地面锚；踝点不在框内/出画时回退 bbox。
    bbox 底边触画或横向截断时拒绝，避免把画框边缘当成脚点。纵向使用踝点
    本身，避免继续把 bbox padding 的系统误差混入横向修正。
    """
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        raise ValueError("invalid bbox") from None
    if not np.all(np.isfinite([x0, y0, x1, y1])) or x1 <= x0 or y1 <= y0:
        raise ValueError("invalid bbox")
    width, height = x1 - x0, y1 - y0
    u, v, source = 0.5 * (x0 + x1), y1, "bbox"

    if allow_ankles and kpts is not None:
        k = np.asarray(kpts, dtype=np.float64)
        if k.ndim == 2 and k.shape[1] >= 3 and len(k) > COCO_RIGHT_ANKLE:
            ankles = []
            for idx, threshold in ((COCO_LEFT_ANKLE, ANKLE_CONF),
                                   (COCO_RIGHT_ANKLE, ANKLE_CONF)):
                p = k[idx]
                if _finite_point(p) and float(p[2]) >= threshold:
                    px, py = float(p[0]), float(p[1])
                    # 允许少量检测框 padding，但排除肩/膝等误识别踝点。
                    if (x0 - 0.1 * width <= px <= x1 + 0.1 * width and
                            y0 <= py <= y1 + 0.06 * height):
                        ankles.append((px, py, float(p[2])))
            if len(ankles) == 2:
                left, right = ankles
                spread = abs(left[0] - right[0])
                if spread <= 0.8 * width:
                    # 置信度加权横向中点；纵向中点避免偏好前后脚。
                    weights = np.maximum([left[2], right[2]], 1e-6)
                    au = float(np.average([left[0], right[0]], weights=weights))
                    av = float(np.average([left[1], right[1]], weights=weights))
                    if _point_visible(au, av, image_shape):
                        return au, av, "ankles"
            elif len(ankles) == 1 and ankles[0][2] >= SINGLE_ANKLE_CONF:
                au, av, _ = ankles[0]
                if _point_visible(au, av, image_shape):
                    return au, av, "ankle_single"

    if image_shape is not None:
        h, w = int(image_shape[0]), int(image_shape[1])
        # 头部出画不影响脚点；脚点所在框的底边/左右出画则不可观测。
        if (x0 < 0.0 or x1 > w - 1.0 or y1 > h - 1.0 or y1 >= h - 1.0):
            raise ValueError("foot_bbox_truncated")
        if not (0.0 <= u <= w - 1.0):
            raise ValueError("foot_outside_frame")
    return u, v, source


def _point_visible(u: float, v: float, shape: tuple[int, int] | None) -> bool:
    if shape is None:
        return True
    h, w = int(shape[0]), int(shape[1])
    return 0.0 <= u <= w - 1.0 and 0.0 <= v <= h - 1.0


def _denominator_ok(h: np.ndarray, u: float, v: float) -> bool:
    w = h[2, 0] * u + h[2, 1] * v + h[2, 2]
    scale = max(float(np.linalg.norm(h[2, :2])) * math.hypot(u, v),
                abs(float(h[2, 2])), 1e-12)
    return bool(np.isfinite(w) and abs(w) > 1e-9 * scale)


def pixel_to_court(H: np.ndarray, u: float, v: float) -> tuple[float, float] | None:
    """单点反投影；接近地平线、分母过小或结果非有限值时返回 None。"""
    h = np.asarray(H, dtype=np.float64)
    if h.shape != (3, 3) or not np.all(np.isfinite(h)) or not np.isfinite([u, v]).all():
        return None
    if not _denominator_ok(h, float(u), float(v)):
        return None
    w = h[2, 0] * u + h[2, 1] * v + h[2, 2]
    x = (h[0, 0] * u + h[0, 1] * v + h[0, 2]) / w
    y = (h[1, 0] * u + h[1, 1] * v + h[1, 2]) / w
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return float(x), float(y)


def _project(H_inv: np.ndarray, x: float, y: float) -> tuple[float, float] | None:
    p = H_inv @ np.array([x, y, 1.0], dtype=np.float64)
    if not np.isfinite(p).all() or abs(p[2]) < 1e-12:
        return None
    return float(p[0] / p[2]), float(p[1] / p[2])


def _local_metric(H: np.ndarray, u: float, v: float) -> float | None:
    """脚点附近 1 像素对应的最大世界位移，用于拒绝近地平线退化。"""
    neighbours = []
    for du, dv in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        uv = pixel_to_court(H, u + du, v + dv)
        if uv is None:
            return None
        neighbours.append(uv)
    centre = pixel_to_court(H, u, v)
    if centre is None:
        return None
    return max(math.hypot(x - centre[0], y - centre[1]) for x, y in neighbours)


def _gate(cal, image_shape, now_s, max_age_s, max_reproj_px,
          trusted_calibration: bool = False) -> tuple[dict, np.ndarray | None, object | None]:
    st: dict = {"ok": False, "calib_ok": False, "reason": "no_calib"}
    if cal is None:
        return st, None, None
    source = str(getattr(cal, "source", "line_calib"))
    st["source"] = source
    try:
        H = np.asarray(cal.H, dtype=np.float64)
        H_inv = np.asarray(cal.H_inv, dtype=np.float64)
        model = str(getattr(cal, "court_model", "FIBA"))
        court_obj = court.COURTS.get(model, court.FIBA)
        timestamp = float(cal.timestamp)
        reproj = float(cal.reproj_error_px)
        keypoints = int(cal.keypoints_used)
        age = max(0.0, float(now_s) - timestamp)
    except Exception as exc:
        st.update(reason="calib_invalid", detail=f"{type(exc).__name__}: {exc}")
        return st, None, None
    st.update(court_model=model, reproj_px=round(reproj, 3), age_s=round(age, 3))
    if H.shape != (3, 3) or H_inv.shape != (3, 3):
        st["reason"] = "h_invalid"
        return st, None, None
    if not np.isfinite(H).all() or not np.isfinite(H_inv).all():
        st["reason"] = "h_invalid"
        return st, None, None
    if not np.isfinite([timestamp, reproj, age]).all() or timestamp <= 0:
        st["reason"] = "calib_invalid"
        return st, None, None
    if not trusted_calibration and age > max_age_s:
        st["reason"] = "stale"
        return st, None, None
    if not trusted_calibration and source != "override" and (reproj > max_reproj_px or keypoints < 4):
        st["reason"] = "high_reproj" if reproj > max_reproj_px else "too_few_keypoints"
        return st, None, None
    norm = float(np.linalg.norm(H))
    if norm <= 1e-12:
        st["reason"] = "h_degenerate"
        return st, None, None
    try:
        cond = float(np.linalg.cond(H / norm))
        computed_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        st["reason"] = "h_degenerate"
        return st, None, None
    st["h_condition"] = round(cond, 2)
    if not np.isfinite(cond) or cond > 1e6:
        st["reason"] = "h_degenerate"
        return st, None, None
    # H_inv 必须真是当前 H 的逆；允许单应任意非零缩放。
    corners = np.array([[-court_obj.half_length, -court_obj.half_width],
                        [court_obj.half_length, -court_obj.half_width],
                        [court_obj.half_length, court_obj.half_width],
                        [-court_obj.half_length, court_obj.half_width],
                        [0.0, 0.0]], dtype=np.float64)
    scale = float(np.sum(computed_inv * H_inv) / max(np.sum(H_inv * H_inv), 1e-30))
    inv_error = float(np.linalg.norm(computed_inv - scale * H_inv) /
                      max(np.linalg.norm(scale * H_inv), 1e-12))
    expected_px = np.array([_project(computed_inv, *p) or (np.nan, np.nan) for p in corners])
    actual_px = np.array([_project(H_inv, *p) or (np.nan, np.nan) for p in corners])
    roundtrip_px = (float(np.max(np.linalg.norm(expected_px - actual_px, axis=1)))
                    if np.isfinite(expected_px).all() and np.isfinite(actual_px).all() else math.inf)
    st.update(inverse_error=round(inv_error, 6), inverse_roundtrip_px=round(roundtrip_px, 4))
    if not np.isfinite(inv_error) or inv_error > 1e-3 or roundtrip_px > 0.5:
        st["reason"] = "h_inverse_mismatch"
        return st, None, None
    projected = actual_px[:4]
    if not np.isfinite(projected).all():
        st["reason"] = "court_projection_invalid"
        return st, None, None
    if image_shape is not None:
        h, w = int(image_shape[0]), int(image_shape[1])
        if not trusted_calibration:
            try:
                if hasattr(cal, "is_plausible") and not bool(cal.is_plausible(w, h)):
                    st["reason"] = "implausible"
                    return st, None, None
            except Exception:
                st["reason"] = "implausible"
                return st, None, None
        x, y = projected[:, 0], projected[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
        if (area < 0.03 * w * h or x.max() - x.min() < 0.2 * w or
                y.max() - y.min() < 0.06 * h):
            st["reason"] = "court_projection_invalid"
            return st, None, None
    st.update(ok=True, calib_ok=True, reason="ok")
    return st, H, court_obj


@dataclass(slots=True)
class _WorldTrack:
    x: float
    y: float
    vx: float
    vy: float
    covariance: np.ndarray
    frame: int
    observed_frame: int


class CourtMapState:
    """逐 (视频, 后端) 的轻量世界坐标状态；不修改检测 Tracker。"""

    def __init__(self, fps: float = 25.0, max_prediction_age_s: float = MAX_PREDICTION_AGE_S):
        self.fps = float(fps)
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive")
        self.max_prediction_age_s = float(max_prediction_age_s)
        self.tracks: dict[int, _WorldTrack] = {}
        self._H_ref: np.ndarray | None = None
        self._shape_ref: tuple[int, int] | None = None

    def reset(self) -> None:
        self.tracks.clear()
        self._H_ref = None
        self._shape_ref = None

    def _reset_for_camera_change(self, H: np.ndarray, shape) -> bool:
        normalized = H / np.linalg.norm(H)
        changed = self._shape_ref is not None and shape != self._shape_ref
        if self._H_ref is not None and not changed and not np.allclose(normalized, self._H_ref, atol=1e-7):
            # 标定轻微抖动不清状态；球场角点世界坐标明显变化才视为切机位。
            try:
                old_inv = np.linalg.inv(self._H_ref)
                pts = np.array([[-14, -7.5], [14, -7.5], [14, 7.5], [-14, 7.5], [0, 0]], float)
                old_list = [pixel_to_court(self._H_ref, *_project(old_inv, *p)) for p in pts]
                new_list = [pixel_to_court(H, *_project(old_inv, *p)) for p in pts]
                if any(p is None for p in old_list + new_list):
                    changed = True
                else:
                    old, new = np.asarray(old_list), np.asarray(new_list)
                    drift = float(np.sqrt(np.mean(np.sum((old - new) ** 2, axis=1))))
                    changed = not np.isfinite(drift) or drift > 0.4
            except np.linalg.LinAlgError:
                changed = True
        self._H_ref = normalized
        self._shape_ref = shape
        if changed:
            self.tracks.clear()
        return changed

    def apply(self, objects: list[schema.Tracked], poses: list[schema.Pose2D], cal,
              frame: int | None = None, image_shape: tuple[int, int] | None = None,
              now_s: float | None = None, calibration_status: dict | None = None,
              margin_m: float = BOUND_MARGIN_M, max_age_s: float = MAX_CALIB_AGE_S,
              max_reproj_px: float = MAX_REPROJ_PX) -> dict:
        t0 = time.perf_counter()
        for obj in objects:
            if obj.category == "player":
                obj.x = obj.y = None
        if calibration_status is not None and not calibration_status.get("ok", False):
            self.reset()
            st = dict(calibration_status)
            st.update(ok=False, calib_ok=False, n=0, observed=0, predicted=0, rejected=0,
                      tracks=[], reason_counts={})
            return self._finish(st, t0)
        now_s = time.time() if now_s is None else float(now_s)
        trusted = bool(calibration_status and calibration_status.get("ok", False))
        gate, H, court_obj = _gate(cal, image_shape, now_s, max_age_s, max_reproj_px,
                                   trusted_calibration=trusted)
        if not gate["ok"] or H is None or court_obj is None:
            self.reset()
            gate.update(n=0, observed=0, predicted=0, rejected=0)
            return self._finish(gate, t0)
        frame = int(frame if frame is not None else max((o.frame for o in objects), default=0))
        if self.tracks and frame < max(track.frame for track in self.tracks.values()):
            self.reset()
        shape = tuple(int(v) for v in image_shape) if image_shape is not None else None
        self._reset_for_camera_change(H, shape)
        kpts_by_id = {p.track_id: p for p in poses}
        half_l, half_w = court_obj.half_length, court_obj.half_width
        reasons: dict[str, int] = {}
        audits: list[dict] = []
        seen: set[int] = set()
        observed_n = predicted_n = rejected_n = outlier_n = 0

        for obj in objects:
            if obj.category != "player":
                continue
            tid = int(obj.track_id)
            seen.add(tid)
            pose = kpts_by_id.get(tid)
            occluded = bool(pose is not None and pose.occluded)
            state = self.tracks.get(tid)
            measurement = None
            foot_source = None
            foot_quality = 0.0
            metric = None
            observe_allowed = bool(obj.visible) and obj.bbox_px is not None
            if observe_allowed and float(obj.conf) < 0.25:
                observe_allowed = False
                reasons["low_confidence"] = reasons.get("low_confidence", 0) + 1
            if observe_allowed:
                try:
                    u, v, foot_source = foot_pixel(
                        obj.bbox_px, pose.kpts if pose is not None else None,
                        image_shape=shape, allow_ankles=not occluded)
                    measurement = pixel_to_court(H, u, v)
                    if measurement is None:
                        reasons["projection_failed"] = reasons.get("projection_failed", 0) + 1
                    else:
                        metric = _local_metric(H, u, v)
                        if metric is None or not np.isfinite(metric) or metric > MAX_LOCAL_M_PER_PX:
                            measurement = None
                            reasons["projection_degenerate"] = reasons.get("projection_degenerate", 0) + 1
                        elif (abs(measurement[0]) > half_l + margin_m or
                              abs(measurement[1]) > half_w + margin_m):
                            measurement = None
                            reasons["outside_court"] = reasons.get("outside_court", 0) + 1
                        else:
                            foot_quality = {"ankles": 0.95, "ankle_single": 0.72,
                                            "bbox": 0.58}[foot_source]
                            foot_quality *= max(0.0, min(1.0, float(obj.conf)))
                            if occluded:
                                foot_quality *= 0.6
                except (TypeError, ValueError, IndexError):
                    measurement = None
                    reasons["foot_invalid_or_truncated"] = reasons.get("foot_invalid_or_truncated", 0) + 1
            elif not obj.visible:
                reasons["not_visible"] = reasons.get("not_visible", 0) + 1
            else:
                reasons["bbox_missing"] = reasons.get("bbox_missing", 0) + 1

            dt = 0.0 if state is None else max(0.0, (frame - state.frame) / self.fps)
            predicted_xy = None
            if state is not None:
                F = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                              [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
                q = 3.0
                Q = np.diag([q * dt ** 4 / 4, q * dt ** 4 / 4,
                             q * dt ** 2, q * dt ** 2])
                P = F @ state.covariance @ F.T + Q
                prior = F @ np.array([state.x, state.y, state.vx, state.vy])
                if (math.hypot(float(prior[2]), float(prior[3])) <= MAX_SPEED_M_S and
                        abs(float(prior[0])) <= half_l + margin_m and
                        abs(float(prior[1])) <= half_w + margin_m):
                    predicted_xy = (float(prior[0]), float(prior[1]))
                state.x, state.y, state.vx, state.vy = map(float, prior)
                state.covariance = P
                state.frame = frame

            innovation = None
            mahal = None
            accepted = False
            if measurement is not None and predicted_xy is not None:
                z = np.asarray(measurement, dtype=np.float64)
                sigma = max(0.10, metric * max(1.0, 0.008 * float(obj.bbox_px[3] - obj.bbox_px[1])) *
                            (1.0 if foot_source == "ankles" else 1.5 if foot_source == "ankle_single" else 2.0))
                sigma /= max(0.25, math.sqrt(foot_quality))
                R = np.eye(2) * sigma ** 2
                S = P[:2, :2] + R
                d = z - np.asarray(predicted_xy)
                innovation = float(np.linalg.norm(d))
                try:
                    mahal = float(d @ np.linalg.solve(S, d))
                except np.linalg.LinAlgError:
                    mahal = math.inf
                if (np.isfinite(mahal) and mahal <= 16.0 and innovation <= 2.0 and
                        math.hypot(float(prior[2]), float(prior[3])) <= MAX_SPEED_M_S):
                    K = np.linalg.solve(S, P[:2, :]).T
                    updated = prior + K @ d
                    if math.hypot(float(updated[2]), float(updated[3])) <= MAX_SPEED_M_S:
                        P = P - K @ P[:2, :]
                        state.x, state.y, state.vx, state.vy = map(float, updated)
                        state.covariance = 0.5 * (P + P.T)
                        state.observed_frame = frame
                        accepted = True
                    else:
                        outlier_n += 1
                        reasons["outlier_rejected"] = reasons.get("outlier_rejected", 0) + 1
                else:
                    outlier_n += 1
                    reasons["outlier_rejected"] = reasons.get("outlier_rejected", 0) + 1
            elif measurement is not None:
                state = _WorldTrack(float(measurement[0]), float(measurement[1]), 0.0, 0.0,
                                    np.diag([0.25, 0.25, 1.0, 1.0]), frame, frame)
                self.tracks[tid] = state
                accepted = True

            observation_age = 0.0 if state is None else (frame - state.observed_frame) / self.fps
            is_prediction = (not accepted and predicted_xy is not None and
                             observation_age <= self.max_prediction_age_s)
            if state is None or (not accepted and not is_prediction):
                obj.x = obj.y = None
                rejected_n += 1
                reason = "stale_track" if state is not None else "no_observation"
                reasons[reason] = reasons.get(reason, 0) + 1
                audits.append(self._audit(obj, None, False, occluded, foot_source, reason,
                                          innovation, mahal, metric, observation_age))
                continue

            x, y = state.x, state.y
            if abs(x) > half_l + margin_m or abs(y) > half_w + margin_m:
                self.tracks.pop(tid, None)
                obj.x = obj.y = None
                rejected_n += 1
                reasons["filtered_outside_court"] = reasons.get("filtered_outside_court", 0) + 1
                audits.append(self._audit(obj, None, False, occluded, foot_source,
                                          "filtered_outside_court", innovation, mahal,
                                          metric, observation_age))
                continue
            obj.x, obj.y = round(x, 4), round(y, 4)
            if is_prediction:
                obj.visible = False
                obj.conf = min(float(obj.conf), 0.35)
                quality = 0.30
                predicted_n += 1
            else:
                quality = foot_quality * (0.75 if mahal is not None else 1.0)
                quality = max(0.15, min(0.95, quality))
                observed_n += 1
            audits.append(self._audit(obj, (x, y), not is_prediction, occluded, foot_source,
                                      "predicted" if is_prediction else "observed", innovation,
                                      mahal, metric, observation_age, quality,
                                      math.hypot(state.vx, state.vy)))

        # 丢失轨迹只保留短时状态；阻止 ID 复用继承旧位置。
        for tid in list(self.tracks):
            if tid not in seen and (frame - self.tracks[tid].observed_frame) / self.fps > 1.0:
                self.tracks.pop(tid, None)
        n = observed_n + predicted_n
        st = dict(gate)
        st.update(ok=True, n=n, observed=observed_n, predicted=predicted_n,
                  rejected=rejected_n, outlier_rejected=outlier_n,
                  reason="ok" if n else "all_rejected", reason_counts=reasons, tracks=audits,
                  fps=self.fps, max_prediction_age_s=self.max_prediction_age_s)
        return self._finish(st, t0)

    @staticmethod
    def _audit(obj, xy, observed: bool, occluded: bool, source, reason: str,
               innovation, mahal, metric, age_s: float, quality: float = 0.0,
               speed: float | None = None) -> dict:
        return {"track_id": int(obj.track_id), "frame": int(obj.frame),
                "x": None if xy is None else round(float(xy[0]), 4),
                "y": None if xy is None else round(float(xy[1]), 4),
                "observed": bool(observed), "predicted": bool(xy is not None and not observed),
                "visible": bool(obj.visible), "occluded": bool(occluded),
                "quality": round(float(quality), 3), "foot_source": source,
                "reason": reason,
                "innovation_m": None if innovation is None else round(float(innovation), 4),
                "mahalanobis": None if mahal is None or not np.isfinite(mahal) else round(float(mahal), 3),
                "local_m_per_px": None if metric is None else round(float(metric), 4),
                "observation_age_s": round(float(age_s), 4),
                "speed_m_s": None if speed is None else round(float(speed), 4)}

    @staticmethod
    def _finish(st: dict, t0: float) -> dict:
        st["ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        st.setdefault("tracks", [])
        st.setdefault("reason_counts", {})
        return st


def apply_court_coords(objects: list[schema.Tracked],
                       poses: list[schema.Pose2D],
                       cal,
                       margin_m: float = BOUND_MARGIN_M, *,
                       state: CourtMapState | None = None,
                       frame: int | None = None,
                       image_shape: tuple[int, int] | None = None,
                       now_s: float | None = None,
                       calibration_status: dict | None = None,
                       fps: float = 25.0) -> dict:
    """原地写入 Tracked.x/y；传入持久 state 才启用跨帧平滑/预测。

    预测坐标强制 ``visible=False`` 且 ``conf<=0.35``；``tracks`` 返回逐轨迹
    审计信息。标定无效时清除本帧球员坐标，不沿用旧值。
    """
    owner = state if state is not None else CourtMapState(fps=fps)
    return owner.apply(objects, poses, cal, frame=frame, image_shape=image_shape,
                       now_s=now_s, calibration_status=calibration_status,
                       margin_m=margin_m)


__all__ = ["ANKLE_CONF", "BOUND_MARGIN_M", "CourtMapState", "apply_court_coords",
           "foot_pixel", "pixel_to_court"]
