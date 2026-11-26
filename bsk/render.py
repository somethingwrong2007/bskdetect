"""叠加渲染：把测试数据烧进帧图，前端只负责显示服务端画好的图。

为什么烧录而不是前端 canvas 画：帧精确。前端拿 <video> 时钟对齐数据帧号要踩
mediaTime / presentedFrames 的坑，而调试台判断的恰恰是"这一帧的数据对不对"，
错一帧就会把合格的算法看成不合格的，所以宁可牺牲交互灵活度。

所有绘制函数对缺失数据静默跳过：算法阶段是一个个补上来的，渲染层不该因为
姿态还没跑就报错。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from bsk.schema import COCO17_NAMES, Pose2D, Segment2D, Tracked, pose_edges

CAT_COLOR = {
    "player": (60, 220, 60),
    "ball": (40, 120, 255),
    "referee": (200, 200, 200),
    "coach": (180, 120, 60),
    "bench": (120, 120, 120),
    "rim": (0, 200, 220),
    "backboard": (0, 140, 200),
}
TEAM_COLOR = {"A": (255, 90, 60), "B": (70, 160, 255)}
EDGE_COLOR = (255, 255, 255)
JOINT_CONF = 0.15
# 关节按"腿-躯干-手臂-头"分色，一眼能看出是哪一段崩了
LIMB_GROUP = {
    (11, 13): "leg", (13, 15): "leg", (12, 14): "leg", (14, 16): "leg",
    (5, 6): "torso", (11, 12): "torso", (5, 11): "torso", (6, 12): "torso",
    (5, 7): "arm", (7, 9): "arm", (6, 8): "arm", (8, 10): "arm",
    (0, 5): "head", (0, 6): "head",
}
GROUP_COLOR = {"leg": (80, 255, 180), "torso": (255, 255, 255), "arm": (255, 180, 80), "head": (255, 90, 230)}


@dataclass
class Layers:
    hud: bool = True
    boxes: bool = True
    pose: bool = True
    jersey: bool = True
    seg: bool = False
    floor: bool = False
    motion: bool = False
    trail: bool = True
    court: bool = True
    conf: float = JOINT_CONF


def _text(img, s, org, color=(255, 255, 255), scale=0.55, thick=1, box=True) -> None:
    """视频上叠字必须有底衬，否则亮地板和白色球衣会把字吃掉。"""
    if box:
        (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x, y = org
        cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + 4), (12, 12, 12), cv2.FILLED)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def timecode(pts_us: int) -> str:
    total_s = pts_us / 1e6
    m, s = divmod(total_s, 60.0)
    return f"{int(m):02d}:{s:06.3f}"


def draw_boxes(img, objects: list[Tracked], show_jersey: bool = True) -> None:
    # 同色框攒成一次 polylines 调用（每帧几十个框时显著减少 cv2 调用开销）
    groups: dict[tuple, list[np.ndarray]] = {}
    labels = []
    for o in objects:
        if not o.bbox_px:
            continue
        x0, y0, x1, y1 = (int(v) for v in o.bbox_px)
        color = TEAM_COLOR.get(o.team or "", CAT_COLOR.get(o.category, (160, 160, 160)))
        groups.setdefault(color, []).append(np.array(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32))
        bits = [f"#{o.track_id}", o.category[:4]]
        if o.team:
            bits.append(o.team)
        if show_jersey and o.jersey is not None:
            bits.append(str(o.jersey))
        if not o.visible:
            bits.append("OCCLUDED")
        if o.provenance == "human":
            bits.append("HUMAN")
        labels.append((" ".join(bits), (x0, max(y0 - 6, 12)), color))
    for color, polys in groups.items():
        cv2.polylines(img, polys, True, color, 2, cv2.LINE_AA)
    for text, org, color in labels:
        _text(img, text, org, color, 0.45)


def _joint_squares(points: list[np.ndarray], half: int) -> list[np.ndarray]:
    out = []
    for arr in points:
        for x, y in arr:
            out.append(np.array([[x - half, y - half], [x + half, y - half],
                                 [x + half, y + half], [x - half, y + half]], dtype=np.int32))
    return out


def draw_poses(img, poses: list[Pose2D], conf: float = JOINT_CONF) -> None:
    """多人骨架批量绘制：按颜色分组攒 polylines/fillPoly，减少逐点逐线调用。"""
    seg_groups: dict[tuple, list[np.ndarray]] = {}
    good_pts: list[np.ndarray] = []
    low_pts: list[np.ndarray] = []
    for pose in poses:
        if not pose.kpts:
            continue
        pts = np.asarray(pose.kpts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        edges = pose_edges(pose.schema)
        n = pts.shape[0]
        for a, b in edges:
            if a >= n or b >= n:
                continue
            pa, pb = pts[a], pts[b]
            if min(pa[2], pb[2]) < conf:
                continue
            color = GROUP_COLOR.get(LIMB_GROUP.get((a, b), ""), EDGE_COLOR)
            seg_groups.setdefault(color, []).append(
                np.array([[pa[0], pa[1]], [pb[0], pb[1]]], dtype=np.int32))
        vis = pts[pts[:, 2] >= conf][:, :2].astype(np.int32)
        inv = pts[pts[:, 2] < conf][:, :2].astype(np.int32)
        if len(vis):
            good_pts.append(vis)
        if len(inv):
            low_pts.append(inv)

    for color, segs in seg_groups.items():
        cv2.polylines(img, segs, False, color, 2, cv2.LINE_AA)
    if good_pts:
        cv2.fillPoly(img, _joint_squares(good_pts, 2), (0, 0, 0))
        cv2.fillPoly(img, _joint_squares(good_pts, 1), (255, 255, 255))
    if low_pts:
        cv2.fillPoly(img, _joint_squares(low_pts, 1), (90, 90, 90))


def draw_pose(img, pose: Pose2D, conf: float = JOINT_CONF, show_names: bool = False) -> None:
    draw_poses(img, [pose], conf)


def draw_segments(img, segments: list[Segment2D], alpha: float = 0.35) -> None:
    if not segments:
        return
    layer = img.copy()
    for s in segments:
        if len(s.polygon) < 3:
            continue
        cnt = np.array(s.polygon, np.int32).reshape(-1, 1, 2)
        color = CAT_COLOR.get(s.category, (160, 160, 160))
        cv2.fillPoly(layer, [cnt], color)
        cv2.polylines(img, [cnt], True, color, 1, cv2.LINE_AA)
    cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, dst=img)


def draw_mask(img, mask: np.ndarray, color=(80, 200, 255), alpha: float = 0.35, outline: bool = True) -> None:
    """掩膜淡染 + 轮廓：只淡染在球场上几乎看不出来，轮廓才判得出边界对不对。"""
    if mask is None or not mask.any():
        return
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
    tint = img.copy()
    tint[mask] = color
    cv2.addWeighted(tint, alpha, img, 1 - alpha, 0, dst=img)
    if outline:
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, color, 1, cv2.LINE_AA)


def draw_trail(img, objects: list[Tracked], history: list[tuple[int, float, float]]) -> None:
    """球的像素轨迹拖影：history 是 [(frame, u, v), ...]，按时间由远及近。"""
    ball = next((o for o in objects if o.category == "ball"), None)
    if ball and ball.bbox_px:
        x0, y0, x1, y1 = (int(v) for v in ball.bbox_px)
        cv2.rectangle(img, (x0, y0), (x1, y1), CAT_COLOR["ball"], 2, cv2.LINE_AA)
    if len(history) >= 2:
        pts = np.array([(u, v) for _f, u, v in history], np.int32)
        for i in range(1, len(pts)):
            w = 1 + (2 * i) // max(len(pts) - 1, 1)
            cv2.line(img, tuple(pts[i - 1]), tuple(pts[i]), (40, 120, 255), w, cv2.LINE_AA)
        cv2.circle(img, tuple(pts[-1]), 6, (255, 255, 255), 1, cv2.LINE_AA)


def draw_court_lines(img, H_inv: np.ndarray, court: "Court", color=(0, 255, 200), thick: int = 1) -> None:
    """把球场线通过 H_inv 投影到像素空间。"""
    from bsk.court import FIBA, Court
    import math

    c = court if isinstance(court, Court) else FIBA
    h, w = img.shape[:2]

    def project(x, y):
        pt = np.array([x, y, 1.0])
        p = H_inv @ pt
        if abs(p[2]) < 1e-8:
            return None
        return (int(p[0] / p[2]), int(p[1] / p[2]))

    def draw_line_seg(x1, y1, x2, y2):
        p1, p2 = project(x1, y1), project(x2, y2)
        if p1 and p2:
            cv2.line(img, p1, p2, color, thick, cv2.LINE_AA)

    def draw_arc(cx, cy, r, start_deg, end_deg, n=36):
        pts = []
        for i in range(n + 1):
            theta = math.radians(start_deg + (end_deg - start_deg) * i / n)
            px, py = cx + r * math.cos(theta), cy + r * math.sin(theta)
            p = project(px, py)
            if p:
                pts.append(p)
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], color, thick, cv2.LINE_AA)

    def draw_dashed_arc(cx, cy, r, start_deg, end_deg, dash=0.38, gap=0.28):
        total = math.radians(abs(end_deg - start_deg)) * r
        direction = 1.0 if end_deg >= start_deg else -1.0
        pos = 0.0
        while pos < total:
            end = min(pos + dash, total)
            start_angle = start_deg + direction * math.degrees(pos / r)
            end_angle = start_deg + direction * math.degrees(end / r)
            draw_arc(cx, cy, r, start_angle, end_angle, max(3, math.ceil(dash / 0.025)))
            pos += dash + gap

    hl, hw = c.half_length, c.half_width
    edge = c.line_half

    draw_line_seg(-hl - edge, -hw - edge, hl + edge, -hw - edge)
    draw_line_seg(hl + edge, -hw - edge, hl + edge, hw + edge)
    draw_line_seg(hl + edge, hw + edge, -hl - edge, hw + edge)
    draw_line_seg(-hl - edge, hw + edge, -hl - edge, -hw - edge)
    # 中线（FIBA 向外延长 0.15 m）
    ext = c.line_half + c.center_line_extension
    draw_line_seg(0, -(hw + ext), 0, hw + ext)

    # 两个半场的三分线 + 罚球区
    for side_sign in (1, -1):
        bl = hl * side_sign
        rim_x = bl - c.rim_offset_from_baseline * side_sign
        ft_x = bl - c.free_throw_center_dist * side_sign
        ft_hw = c.paint_width / 2 + c.line_half

        # 罚球区
        draw_line_seg(bl, -ft_hw, ft_x, -ft_hw)
        draw_line_seg(ft_x, -ft_hw, ft_x, ft_hw)
        draw_line_seg(ft_x, ft_hw, bl, ft_hw)
        if c.free_throw_dashed_inside:
            if side_sign > 0:
                draw_arc(ft_x, 0, c.free_throw_circle_center_radius, 90, 270, 36)
                draw_dashed_arc(ft_x, 0, c.free_throw_circle_center_radius, -90, 90)
            else:
                draw_arc(ft_x, 0, c.free_throw_circle_center_radius, -90, 90, 36)
                draw_dashed_arc(ft_x, 0, c.free_throw_circle_center_radius, 90, 270)
        else:
            draw_arc(ft_x, 0, c.free_throw_circle_center_radius, 0, 360, 56)

        # 三分线：底角直线 + 朝中场的弧
        R = c.three_pt_center_radius
        corner_y = hw - c.three_pt_corner_center_dist
        dx = math.sqrt(max(R * R - corner_y * corner_y, 0.0))
        arc_x = rim_x - side_sign * dx
        draw_line_seg(bl, -corner_y, arc_x, -corner_y)
        draw_line_seg(bl, corner_y, arc_x, corner_y)
        half = math.degrees(math.asin(min(1.0, corner_y / R)))
        if side_sign > 0:
            draw_arc(rim_x, 0, R, 180 - half, 180 + half, 80)
        else:
            draw_arc(rim_x, 0, R, -half, half, 80)

        # 合理冲撞区
        rr = c.restricted_center_radius
        back_x = bl - side_sign * c.backboard_from_baseline
        if side_sign > 0:
            draw_arc(rim_x, 0, rr, 90, 270, 36)
        else:
            draw_arc(rim_x, 0, rr, -90, 90, 36)
        draw_line_seg(back_x, -rr, rim_x, -rr)
        draw_line_seg(back_x, rr, rim_x, rr)

    # 中圈
    draw_arc(0, 0, c.center_circle_center_radius, 0, 360, 72)


def draw_hud(
    img,
    *,
    frame: int,
    pts_us: int,
    nframes: int,
    shot_id: int | None,
    court_visible: bool | None,
    layers: Layers,
    sources: dict[str, str],
    notes: list[str] | None = None,
    scale: float = 1.0,
) -> None:
    h, w = img.shape[:2]
    fs = max(0.42, 0.62 * scale)
    _text(img, f"f{frame}/{nframes}  {timecode(pts_us)}  shot {shot_id if shot_id is not None else '-'}",
          (10, int(24 * scale) + 4), (255, 255, 255), fs, 1)
    gate = "UNKNOWN" if court_visible is None else ("COURT-OK" if court_visible else "GATED-OUT")
    gc = (90, 220, 90) if court_visible else (80, 80, 240)
    _text(img, gate, (10, int(52 * scale) + 4), gc, fs, 1)

    legend = " | ".join(f"{k}:{sources.get(k, 'none')}" for k in ("boxes", "pose", "seg", "ball", "camera"))
    _text(img, legend, (10, h - int(14 * scale)), (190, 190, 190), max(0.36, fs * 0.8), 1)
    for i, note in enumerate(notes[:4]):
        _text(img, note, (w - 420, int(28 * scale) + i * int(24 * scale)), (80, 200, 255), max(0.36, fs * 0.8), 1)


def render_frame(
    rgb: np.ndarray,
    *,
    frame: int,
    pts_us: int,
    nframes: int,
    shot_id: int | None = None,
    court_visible: bool | None = None,
    objects: list[Tracked] | None = None,
    poses: list[Pose2D] | None = None,
    segments: list[Segment2D] | None = None,
    floor: np.ndarray | None = None,
    motion: np.ndarray | None = None,
    ball_history: list[tuple[int, float, float]] | None = None,
    layers: Layers | None = None,
    sources: dict[str, str] | None = None,
    notes: list[str] | None = None,
    calib_H_inv: np.ndarray | None = None,
    court_model: "Court | None" = None,
    in_place: bool = False,
) -> np.ndarray:
    layers = layers or Layers()
    out = rgb if in_place else rgb.copy()
    scale = min(out.shape[1] / 1280.0, out.shape[0] / 720.0)

    if layers.court and calib_H_inv is not None and court_model is not None:
        draw_court_lines(out, calib_H_inv, court_model)
    if layers.floor and floor is not None:
        draw_mask(out, floor, (80, 200, 255), 0.30)
    if layers.motion and motion is not None:
        draw_mask(out, motion, (255, 90, 220), 0.40)
    if layers.seg and segments:
        draw_segments(out, segments)
    if layers.boxes and objects:
        draw_boxes(out, objects, layers.jersey)
    if layers.pose and poses:
        draw_poses(out, poses, layers.conf)
    if layers.trail:
        draw_trail(out, objects or [], ball_history or [])
    if layers.hud:
        draw_hud(out, frame=frame, pts_us=pts_us, nframes=nframes, shot_id=shot_id,
                 court_visible=court_visible, layers=layers, sources=sources or {},
                 notes=notes or [], scale=scale)
    return out


def to_jpeg(rgb: np.ndarray, quality: int = 82) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


__all__ = ["Layers", "draw_boxes", "draw_pose", "draw_segments", "draw_trail", "render_frame", "timecode", "to_jpeg"]
