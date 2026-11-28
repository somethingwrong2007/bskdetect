"""实时球员候选门控：形状、关键点质量和球场落点共用同一组因果规则。"""

from __future__ import annotations

import math

import cv2
import numpy as np


# COCO：左右踝。骨架绑定使用脚踝而不是框底边，跑动/起跳时框底边会明显偏离落点。
_ANKLES = (15, 16)


def ground_point(bbox, kpts=None) -> tuple[float, float]:
    """返回脚踝加权落点；脚踝不可信时退回框底边中点。"""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    fallback = ((x0 + x1) * 0.5, y1)
    if kpts is None:
        return fallback
    kp = np.asarray(kpts)
    if kp.ndim != 2 or kp.shape[1] < 3 or kp.shape[0] <= max(_ANKLES):
        return fallback
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    weighted = []
    for i, threshold in ((15, 0.45), (16, 0.45)):
        x, y, conf = (float(v) for v in kp[i, :3])
        if not all(math.isfinite(v) for v in (x, y, conf)) or conf < threshold:
            continue
        # 拒绝飘到身体上方/框外的错误踝点；允许起跳和检测框轻微误差。
        if y < y0 + bh * 0.45 or y > y1 + bh * 0.35:
            continue
        if x < x0 - bw * 0.5 or x > x1 + bw * 0.5:
            continue
        weighted.append((x, y, conf))
    if not weighted:
        return fallback
    total = sum(c for _, _, c in weighted)
    return (sum(x * c for x, _, c in weighted) / total,
            sum(y * c for x, y, c in weighted) / total)


def player_quality(person, frame_shape, backend: str = "fast") -> bool:
    """在跟踪前剔除无效框和缺少身体关键点的假阳性。

    球衣号码是下游信息，不能作为骨架门控；这里只使用当前帧可解释的身体几何。
    """
    bbox = getattr(person, "bbox", None)
    if bbox is None or len(bbox) != 4:
        return False
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox)
        score = float(person.score)
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1, score)):
        return False
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0 or not 0.45 <= bh / bw <= 5.0:
        return False
    h, w = frame_shape[:2]
    if backend == "accurate":
        det_thr, min_h, min_w, min_kpts = 0.30, max(30.0, h * 0.025), max(14.0, w * 0.012), 3
    else:
        det_thr, min_h, min_w, min_kpts = 0.45, max(42.0, h * 0.035), max(18.0, w * 0.018), 4
    if score < det_thr or bh < min_h or bw < min_w:
        return False
    kp = getattr(person, "kpts", None)
    if kp is None:
        return False
    kp = np.asarray(kp)
    if kp.ndim != 2 or kp.shape[0] < 12 or kp.shape[1] < 3:
        return False
    # 眼/耳容易噪声大，只统计身体；坐标也有限才认为该关节真的可用。
    body = kp[5:17, :3]
    usable = np.isfinite(body).all(axis=1) & (body[:, 2] > 0.35)
    if int(np.sum(usable)) < min_kpts:
        return False
    # 只有肩/腕的看台或摄影师半身像不能进入球员轨迹；至少要有髋/膝/踝中的两点，
    # 但不用求踝点置信度高（远端球员的踝点经常偏弱）。
    lower = kp[11:17, :3]
    lower_usable = np.isfinite(lower).all(axis=1) & (lower[:, 2] > 0.35)
    return int(np.sum(lower_usable)) >= 2


def rfdetr_player_gate(people, detections, known_boxes=()) -> list:
    """用当前帧 RF-DETR 的球员类别语义筛选 RTMO 人物框。

    RF-DETR 只负责“这个人是不是球员”，关键点仍来自同一个 RTMO Person，避免把
    A 的骨架挂到 B 的框上。新候选至少 0.40；已经确认的球员轨迹允许当前帧降到
    0.30，以吸收运动模糊造成的置信度波动。known_boxes 只含过去帧，不预取未来。
    """
    player_classes = {
        "player", "player-in-possession", "player-jump-shot",
        "player-layup-dunk", "player-shot-block",
    }
    # 真实权重会把少数裁判误报成低分 player（例如 0.38），因此新候选不能降到
    # 0.30；但已确认轨迹在当前帧短暂降到 0.30 时仍应保留，避免真实玩家掉骨架。
    person_classes = player_classes | {"referee"}
    known_boxes = tuple(known_boxes or ())

    def _iou(a, b) -> float:
        ax0, ay0, ax1, ay1 = (float(v) for v in a)
        bx0, by0, bx1, by1 = (float(v) for v in b)
        inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * \
            max(0.0, min(ay1, by1) - max(ay0, by0))
        return inter / max(1.0, (ax1 - ax0) * (ay1 - ay0) +
                           (bx1 - bx0) * (by1 - by0) - inter)

    def _supported(det) -> bool:
        conf = float(getattr(det, "conf", 0.0))
        return conf >= 0.40 or (
            conf >= 0.30 and any(_iou(det.bbox, box) >= 0.45 for box in known_boxes)
        )

    candidates = [
        d for d in detections
        if getattr(d, "class_name", "") in person_classes and _supported(d)
    ]
    if not candidates:
        return []
    # 动作类别之间、以及 player/referee 之间都可能重叠。必须先在“人类类别”
    # 中做一次类无关 NMS，让更高分的裁判框压掉同位置的球员误报，再做球员筛选；
    # 若先过滤掉 referee，低分 player 误报就没有竞争者了。
    boxes = np.asarray([
        [d.bbox[0], d.bbox[1], d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]]
        for d in candidates
    ], np.float32)
    scores = np.asarray([float(d.conf) for d in candidates], np.float32)
    keep = np.asarray(cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.30, 0.50)).reshape(-1)
    ranked = sorted((candidates[int(i)] for i in keep), key=lambda d: -float(d.conf))
    matched: set[int] = set()
    for det in ranked:
        if det.class_name not in player_classes:
            continue
        bx0, by0, bx1, by1 = (float(v) for v in det.bbox)
        area_b = max(1.0, (bx1 - bx0) * (by1 - by0))
        best_index, best_iou = -1, 0.0
        for index, person in enumerate(people):
            if index in matched:
                continue
            ax0, ay0, ax1, ay1 = (float(v) for v in person.bbox)
            area_a = max(1.0, (ax1 - ax0) * (ay1 - ay0))
            ix0, iy0 = max(ax0, bx0), max(ay0, by0)
            ix1, iy1 = min(ax1, bx1), min(ay1, by1)
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            iou = inter / max(1.0, area_a + area_b - inter)
            if iou > best_iou:
                best_index, best_iou = index, iou
        if best_index >= 0 and best_iou >= 0.12:
            matched.add(best_index)
    return [person for index, person in enumerate(people) if index in matched]


def conservative_court_polygon(polygon, frame_shape) -> np.ndarray | None:
    """给过小的部分球场 ROI 向近端留出保守边，只用于球员筛选。

    古典无标签标定偶尔把场内线拟合成过小的“全场”，重投影误差仍很小，却会把
    近端球员全部判成场外。这里不修改/发布 H，也不用于 3D；仅当四边形明显过小时，
    以图像下方为近端扩展筛选范围。正常全场标定完全原样返回。
    """
    if polygon is None:
        return None
    poly = np.asarray(polygon, np.float32)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3 or not np.isfinite(poly).all():
        return None
    poly = poly.astype(np.float64)
    h, w = frame_shape[:2]
    area = abs(float(cv2.contourArea(poly.astype(np.float32))))
    if area >= 0.12 * w * h or len(poly) != 4:
        return poly.astype(np.float32)
    # 近端基线：四条边中中点更靠图像下方的一条；摄像机方向不明时不猜。
    mid_y = (poly[:, 1] + np.roll(poly[:, 1], -1)) * 0.5
    near_i = int(np.argmax(mid_y))
    far_i = (near_i + 2) % 4
    if mid_y[near_i] - mid_y[far_i] < h * 0.03:
        return poly.astype(np.float32)
    near = np.array([poly[near_i], poly[(near_i + 1) % 4]])
    near_mid = near.mean(axis=0)
    extra = near.copy()
    extra[:, 0] += np.where(near[:, 0] >= near_mid[0], 1.0, -1.0) * w * 0.12
    extra[:, 1] += h * 0.35
    hull = cv2.convexHull(np.concatenate([poly, extra]).astype(np.float32)).reshape(-1, 2)
    hull_area = abs(float(cv2.contourArea(hull)))
    return hull if hull_area <= 1.6 * w * h else poly.astype(np.float32)


def safe_keypoints(kpts):
    """把非有限关节置为不可见，避免单点 NaN 污染整条时序轨迹。"""
    kp = np.asarray(kpts, dtype=np.float64)
    if np.isfinite(kp).all():
        return kp
    clean = kp.copy()
    bad = ~np.isfinite(clean).all(axis=1)
    clean[bad, :2] = 0.0
    clean[bad, 2] = 0.0
    return clean


__all__ = ["ground_point", "player_quality", "safe_keypoints", "rfdetr_player_gate",
           "conservative_court_polygon"]
