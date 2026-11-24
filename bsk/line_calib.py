"""从球场线分割结果拟合单应矩阵（快速实时标定）。

流程
1. 小球场线分割网络（bsk.calib_seg_net）逐帧输出 5 类线掩码；
2. 两种初始化（都参数化为"球场外框四角的像素位置"）：
   a. 边线：对 boundary 掩码做 Hough 线段聚类得到 2~4 条主线，把"图像线 ↔
      世界矩形的一条边"的对应关系全枚举（4! 种；边线不足 4 条时用中线补，
      即 3 条边线 + 中线），每条对应用线对应 DLT 解 H；
   b. 罚球区：漆区轮廓 approxPolyDP 四边形 + minAreaRect 兜底，左右两侧和
      4 种旋转都试，4 点对应求 H。
   候选按（粗采样）chamfer 代价排序；
3. 取前几个候选，以四角像素坐标为自由变量做 Powell 精化。代价 = 召回 + 精确率：
   召回：模板线（边线/三分/罚球区/中线/中圈）采样点经 H 投影后，落到对应类别
         掩码距离变换上的截断均值（线被遮挡是常态，所以截断而非平方）；
   精确率：boundary 掩码点到投影外框的距离中值，防止"把模板缩到一小块"的退化解；
4. 几何校验：Calibration.is_plausible + 代价阈值；
5. 时序：CalibrationTracker 每 K 帧拟合一次，最近若干次有效结果的角点取中值；
   单帧失败沿用上一帧（≤2s），并随陈旧时间放大 ROI margin，绝不误删球员。

矩阵约定（与 bsk.calibrate 一致，别搞反）：
  Calibration.H     = pixel -> court（米）
  Calibration.H_inv = court -> pixel
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from bsk.calib_seg_net import NUM_CLASSES, CourtLineSegNet
from bsk.calibrate import Calibration
from bsk.court import COURTS, FIBA, Court, project_court_polygon

CL_BOUNDARY = 1
CL_THREE_POINT = 2
CL_PAINT = 3
CL_CENTER_LINE = 4
CL_CENTER_CIRCLE = 5

DEFAULT_SIZE = (540, 960)  # (H, W)：HD 训练/推理分辨率
DEFAULT_WEIGHTS = "checkpoints/calib_seg_hd/best.pt"
FALLBACK_WEIGHTS = "checkpoints/calib_seg_mix2/best.pt"

# 各类线在代价里的权重：边线最结构、中圈最容易误检
CLASS_W = {CL_BOUNDARY: 1.0, CL_THREE_POINT: 0.9, CL_PAINT: 0.8,
           CL_CENTER_LINE: 0.7, CL_CENTER_CIRCLE: 0.5}


def _seg(a, b, step: float) -> list[tuple[float, float]]:
    n = max(int(math.hypot(b[0] - a[0], b[1] - a[1]) / step), 1)
    return [(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(n + 1)]


def court_line_samples(court: Court, step: float = 0.5) -> dict[int, np.ndarray]:
    """各类线在世界坐标下的采样点（米）。"""
    hl, hw = court.half_length, court.half_width
    out: dict[int, list[tuple[float, float]]] = {c: [] for c in range(1, NUM_CLASSES)}

    rect = [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]
    for i in range(4):
        out[CL_BOUNDARY] += _seg(rect[i], rect[(i + 1) % 4], step)
    out[CL_CENTER_LINE] += _seg((0.0, -hw), (0.0, hw), step)
    r_center = 1.8
    n = 48
    out[CL_CENTER_CIRCLE] += [
        (r_center * math.cos(2 * math.pi * i / n), r_center * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]

    for side in ("A", "B"):
        bl = court.baseline_x(side)
        sign = 1 if side == "A" else -1
        ft_x = bl - court.free_throw_dist * sign
        fthw = court.free_throw_width / 2
        lane = [(bl, -fthw), (bl, fthw), (ft_x, fthw), (ft_x, -fthw)]
        for i in range(4):
            out[CL_PAINT] += _seg(lane[i], lane[(i + 1) % 4], step)

        rim_x, _ = court.rim_pos(side)
        r3 = court.three_pt_radius
        y_corner = hw - court.three_pt_corner_dist
        theta_c = math.asin(min(1.0, y_corner / r3))
        arc_start_lo = (rim_x - sign * r3 * math.cos(-theta_c), -y_corner)
        arc_start_hi = (rim_x - sign * r3 * math.cos(theta_c), y_corner)
        out[CL_THREE_POINT] += _seg((bl, -y_corner), arc_start_lo, step)
        out[CL_THREE_POINT] += _seg((bl, y_corner), arc_start_hi, step)
        n_arc = 48
        for i in range(n_arc + 1):
            th = -theta_c + 2 * theta_c * i / n_arc
            out[CL_THREE_POINT].append((rim_x - sign * r3 * math.cos(th), r3 * math.sin(th)))

    return {c: np.array(v, dtype=np.float64) for c, v in out.items() if v}


# ── 直线工具 ────────────────────────────────────────────────────

def _line_from_pts(p1, p2) -> Optional[np.ndarray]:
    """两点 → 归一化直线 (a, b, c)，满足 ax+by+c=0，符号规范（a>0 或 a==0,b>0）。"""
    a = -(p2[1] - p1[1])
    b = p2[0] - p1[0]
    n = math.hypot(a, b)
    if n < 1e-9:
        return None
    a, b = a / n, b / n
    c = -(a * p1[0] + b * p1[1])
    if a < 0 or (abs(a) < 1e-12 and b < 0):
        a, b, c = -a, -b, -c
    return np.array([a, b, c], dtype=np.float64)


def _ang_diff(l1: np.ndarray, l2: np.ndarray) -> float:
    d = abs(math.atan2(l1[1], l1[0]) - math.atan2(l2[1], l2[0]))
    return min(d, math.pi - d)


def _line_dist(l: np.ndarray, x: float, y: float) -> float:
    return abs(l[0] * x + l[1] * y + l[2])


def _poly_area(quad: np.ndarray) -> float:
    x, y = quad[:, 0], quad[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _project(H: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ph = np.column_stack([pts, np.ones(len(pts))])
    pr = ph @ H.T
    w = pr[:, 2]
    ok = np.abs(w) > 1e-9
    uv = pr[:, :2] / np.where(ok, w, 1.0)[:, None]
    return uv, ok & np.all(np.isfinite(uv), axis=1)


def _world_rect(court: Court) -> np.ndarray:
    hl, hw = court.half_length, court.half_width
    return np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]], dtype=np.float64)


def _rect_seg_dist(quad: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """点集到四边形 4 条边的最近距离。"""
    best = np.full(len(pts), 1e9)
    for i in range(4):
        a = quad[i]
        b = quad[(i + 1) % 4]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-9:
            continue
        t = np.clip(((pts - a) @ ab) / L2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        best = np.minimum(best, np.linalg.norm(pts - proj, axis=1))
    return best


def _dlt_from_lines(pairs: Iterable[tuple[np.ndarray, np.ndarray]]) -> Optional[np.ndarray]:
    """线对应 (图像线, 世界线) → 单应（世界→像素，即 H_inv）。

    线传递关系：l_pixel ∝ H_inv^{-T} l_world，两边左乘 H_inv^T 得
    H_inv^T l_pixel ∝ l_world，即 cross(H_inv^T l_pixel, l_world) = 0。
    每条线 2 个独立方程，4 条线 = 8 方程可解 8 自由度（h33 归一化）。
    """
    rows = []
    for lp, lw in pairs:
        p0, p1, p2 = lp
        w0, w1, w2 = lw
        rows.append([0.0, w2 * p0, -w1 * p0,
                     0.0, w2 * p1, -w1 * p1,
                     0.0, w2 * p2, -w1 * p2])
        rows.append([-w2 * p0, 0.0, w0 * p0,
                     -w2 * p1, 0.0, w0 * p1,
                     -w2 * p2, 0.0, w0 * p2])
    A = np.array(rows, dtype=np.float64)
    if A.shape[0] < 8:
        return None
    try:
        _u, _s, vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    h = vt[-1].reshape(3, 3)
    if abs(h[2, 2]) > 1e-9:
        h = h / h[2, 2]
    if not np.all(np.isfinite(h)):
        return None
    return h


@dataclass
class LineFitResult:
    calib: Calibration
    score: float
    source: str = ""
    candidates: int = 0


@dataclass
class _Cand:
    H: np.ndarray  # 世界→像素（模型输入尺度）
    score: float
    source: str


class LineCalibrator:
    """分割线 → 单应。模型懒加载。"""

    def __init__(self, court: Court | None = None, size: tuple[int, int] = DEFAULT_SIZE,
                 weights: str | Path | None = None, device: str | None = None,
                 conf_thr: float = 0.5, step: float = 0.5, max_score: float = 9.0,
                 refine_top: int = 2, powell_iter: int = 40,
                 label: str = ""):
        self.courts: list[Court] = [court] if court is not None else [COURTS["NBA"], FIBA]
        self.court = self.courts[0]
        self.size = size
        self.weights = weights
        self.device = device
        self.conf_thr = conf_thr
        self.step = step
        self.max_score = max_score
        self.refine_top = refine_top
        self.powell_iter = powell_iter
        self.label = label
        self._templates = {c.model: court_line_samples(c, step) for c in self.courts}
        self._templates_mid = {c.model: court_line_samples(c, max(step, 1.0)) for c in self.courts}
        self._templates_coarse = {c.model: court_line_samples(c, max(step, 2.5)) for c in self.courts}
        self._model = None
        self._norm = None
        self._infer_lock = threading.Lock()
        self.last_candidates: list[tuple[str, float, str]] = []

    # ── 模型 ──────────────────────────────────────────────
    def _load(self):
        if self._model is not None:
            return
        import torch
        from torchvision import transforms

        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        model = CourtLineSegNet(num_classes=NUM_CLASSES, pretrained=False).to(dev)
        wpath = self.weights or DEFAULT_WEIGHTS
        if not Path(wpath).exists() and Path(FALLBACK_WEIGHTS).exists():
            wpath = FALLBACK_WEIGHTS
        if Path(wpath).exists():
            try:
                ckpt = torch.load(wpath, map_location=dev, weights_only=False)
            except TypeError:
                ckpt = torch.load(wpath, map_location=dev)
            state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state)
        else:
            print(f"[calib] WARN: 权重不存在 {wpath}，用随机初始化（仅排障用）")
        model.eval()
        self._model = model
        self._norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """返回 [h, w] 类别图（0..5），h/w 为模型输入尺寸。"""
        import torch

        self._load()
        h, w = self.size
        small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        x = self._norm(torch.from_numpy(small).permute(2, 0, 1).float() / 255.0)
        with self._infer_lock, torch.no_grad():
            logits = self._model(x.unsqueeze(0).to(self.device))
            probs = logits.softmax(1)[0]
        pred = probs.argmax(0).cpu().numpy().astype(np.uint8)
        conf = probs.max(0).values.cpu().numpy()
        pred[conf < self.conf_thr] = 0
        return pred

    # ── 掩码索引 ──────────────────────────────────────────
    @staticmethod
    def _mask_index(pred: np.ndarray) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        dts: dict[int, np.ndarray] = {}
        mpts: dict[int, np.ndarray] = {}
        for c in range(1, NUM_CLASSES):
            m = (pred == c).astype(np.uint8)
            n = int(m.sum())
            if n < 25:
                continue
            dts[c] = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
            if c in (CL_BOUNDARY, CL_PAINT, CL_CENTER_LINE) and n >= 30:
                ys, xs = np.nonzero(m)
                if len(xs) > 800:
                    sel = np.linspace(0, len(xs) - 1, 800).astype(int)
                    ys, xs = ys[sel], xs[sel]
                mpts[c] = np.column_stack([xs, ys]).astype(np.float64)
        return dts, mpts

    # ── 代价 ──────────────────────────────────────────────
    def _score(self, H: np.ndarray, dts: dict[int, np.ndarray], mpts: dict[int, np.ndarray] | None,
               court: Court, shape: tuple[int, int], templates=None,
               cap: float = 24.0) -> float:
        h, w = shape
        tpl = templates if templates is not None else self._templates[court.model]
        total, weight = 0.0, 0.0
        any_vis = False
        for c, pts in tpl.items():
            dt = dts.get(c)
            if dt is None:
                continue
            uv, ok = _project(H, pts)
            inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            if int(inside.sum()) < 12:
                continue
            any_vis = True
            xs = np.clip(uv[inside, 0], 0, w - 1).astype(np.int32)
            ys = np.clip(uv[inside, 1], 0, h - 1).astype(np.int32)
            d = dt[ys, xs]
            total += CLASS_W[c] * float(np.mean(np.minimum(d, cap)))
            weight += CLASS_W[c]
        if not any_vis or weight <= 0:
            return 1e9

        recall = total / weight
        if mpts is None:
            return recall
        precision = 0.0
        src = mpts.get(CL_BOUNDARY)
        rect_w = _world_rect(court)
        quad, okq = _project(H, rect_w)
        if not okq.all():
            return 1e9
        if src is not None and len(src) >= 30:
            d = _rect_seg_dist(quad, src)
            precision = float(np.median(np.minimum(d, cap)))
        return 0.7 * recall + 0.3 * precision

    def _quick_ok(self, H: np.ndarray, shape: tuple[int, int], court: Court) -> bool:
        h, w = shape
        quad, ok = _project(H, _world_rect(court))
        if not ok.all():
            return False
        area = _poly_area(quad)
        if not (0.03 * w * h <= area <= 3.0 * w * h):
            return False
        cnt = quad.reshape(-1, 1, 2).astype(np.float32)
        hull = abs(cv2.contourArea(cv2.convexHull(cnt)))
        if hull <= 1.0 or area < 0.86 * hull:
            return False
        if quad[:, 0].max() < 0 or quad[:, 0].min() > w or quad[:, 1].max() < 0 or quad[:, 1].min() > h:
            return False
        return True

    # ── 初始化 A：边线 Hough 线段聚类 + 线对应 DLT ───────────
    @staticmethod
    def _hough_lines(mask: np.ndarray, min_len: int, max_n: int = 4) -> list[np.ndarray]:
        m = cv2.morphologyEx(mask * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        lines = cv2.HoughLinesP(m, 1, np.pi / 180, threshold=18,
                                minLineLength=max(min_len, 8), maxLineGap=8)
        if lines is None:
            return []
        segs = []
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            L = math.hypot(x2 - x1, y2 - y1)
            ln = _line_from_pts((float(x1), float(y1)), (float(x2), float(y2)))
            if ln is not None:
                segs.append((L, ln, (x1, y1, x2, y2)))
        segs.sort(key=lambda t: -t[0])

        clusters: list[dict] = []
        for L, ln, seg in segs:
            cx, cy = (seg[0] + seg[2]) / 2.0, (seg[1] + seg[3]) / 2.0
            placed = False
            for cl in clusters:
                if _ang_diff(ln, cl["line"]) < math.radians(10) and _line_dist(cl["line"], cx, cy) < 16:
                    cl["pts"] += [(seg[0], seg[1]), (seg[2], seg[3])]
                    cl["votes"] += L
                    pts = np.array(cl["pts"], np.float64).astype(np.float32)
                    if len(pts) >= 2:
                        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
                        new = _line_from_pts((x0, y0), (x0 + vx, y0 + vy))
                        if new is not None:
                            cl["line"] = new
                    placed = True
                    break
            if not placed:
                clusters.append({"line": ln, "pts": [(seg[0], seg[1]), (seg[2], seg[3])],
                                 "votes": L})
        clusters.sort(key=lambda c: -c["votes"])
        return [c["line"] for c in clusters[:max_n]]

    @staticmethod
    def _side_lines(court: Court) -> list[np.ndarray]:
        hl, hw = court.half_length, court.half_width
        return [
            np.array([1.0, 0.0, hl]),    # x = -hl（底角侧）
            np.array([1.0, 0.0, -hl]),   # x = +hl
            np.array([0.0, 1.0, hw]),    # y = -hw（边线）
            np.array([0.0, 1.0, -hw]),   # y = +hw
        ]

    def _init_from_lines(self, pred: np.ndarray, court: Court,
                         shape: tuple[int, int]) -> list[tuple[np.ndarray, str]]:
        h, w = shape
        bw = self._hough_lines((pred == CL_BOUNDARY).astype(np.uint8), int(0.05 * w), 5)
        cw = self._hough_lines((pred == CL_CENTER_LINE).astype(np.uint8), int(0.04 * w), 1)
        sides = self._side_lines(court)
        center = np.array([1.0, 0.0, 0.0])  # x = 0
        out: list[tuple[np.ndarray, str]] = []

        # 边线常常被裁掉/遮挡，只剩 3 条 → 用中线补足第 4 条线对应，
        # 但"哪 3 条"和"对应哪 3 个边"都要枚举（Hough 可能把一条边拆成两簇）
        for sub in self._choose(bw, 4):
            for perm in permutations(range(4)):
                pairs = [(sub[i], sides[perm[i]]) for i in range(4)]
                H = _dlt_from_lines(pairs)
                if H is not None:
                    out.append((H, "lines4"))
        if cw:
            for sub in self._choose(bw, 3):
                for perm in permutations(range(4), 3):
                    pairs = [(sub[i], sides[perm[i]]) for i in range(3)]
                    pairs.append((cw[0], center))
                    H = _dlt_from_lines(pairs)
                    if H is not None:
                        out.append((H, "lines3+cl"))
        return out

    @staticmethod
    def _choose(lines: list[np.ndarray], k: int, max_sets: int = 6) -> list[list[np.ndarray]]:
        """从 lines 里挑 k 条的组合（按票数顺序，最多 max_sets 组）。"""
        if len(lines) < k:
            return []
        from itertools import combinations

        out = []
        for combo in combinations(range(len(lines)), k):
            out.append([lines[i] for i in combo])
            if len(out) >= max_sets:
                break
        return out

    # ── 初始化 B：罚球区四边形 ─────────────────────────────
    def _init_from_paint(self, pred: np.ndarray, court: Court,
                         shape: tuple[int, int]) -> list[tuple[np.ndarray, str]]:
        m = (pred == CL_PAINT).astype(np.uint8)
        if m.sum() < 200:
            return []
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]
        quads: list[tuple[np.ndarray, str]] = []
        for c in cnts:
            if cv2.contourArea(c) < 300:
                continue
            peri = cv2.arcLength(c, True)
            for eps in (0.02, 0.04, 0.07):
                ap = cv2.approxPolyDP(c, eps * peri, True)
                if len(ap) == 4:
                    quad = ap.reshape(4, 2).astype(np.float64)
                    ctr = quad.mean(axis=0)
                    ang = np.arctan2(quad[:, 1] - ctr[1], quad[:, 0] - ctr[0])
                    quads.append((quad[np.argsort(ang)], "paint"))
                    break
            # minAreaRect 兜底：漆区被球员切碎时 approxPolyDP 常常给不出四边形
            rect = cv2.minAreaRect(c)
            if min(rect[1]) >= 8:
                box = cv2.boxPoints(rect).astype(np.float64)
                ctr = box.mean(axis=0)
                ang = np.arctan2(box[:, 1] - ctr[1], box[:, 0] - ctr[0])
                quads.append((box[np.argsort(ang)], "paint_rect"))

        out: list[tuple[np.ndarray, str]] = []
        for side in ("A", "B"):
            bl = court.baseline_x(side)
            sign = 1 if side == "A" else -1
            ft_x = bl - court.free_throw_dist * sign
            fthw = court.free_throw_width / 2
            lane = np.array([[bl, -fthw], [bl, fthw], [ft_x, fthw], [ft_x, -fthw]], np.float64)
            for quad, tag in quads:
                if _poly_area(quad) < 200:
                    continue
                # 图像角点按角度排序后环绕方向可能与世界不同（单应行列式符号未知），
                # 反向也要试：只有环绕一致时四边形的内部才映射到内部
                for q in (quad, quad[::-1]):
                    for rot in range(4):
                        src = np.roll(lane, rot, axis=0)
                        H, _ = cv2.findHomography(src, q, 0)
                        if H is not None:
                            out.append((H, tag))
        return out

    # ── 精化 ──────────────────────────────────────────────
    def _refine(self, H0: np.ndarray, dts, mpts, court: Court,
                shape: tuple[int, int]) -> tuple[np.ndarray, float]:
        """以四角像素坐标为自由变量的模式搜索（步长递减）。

        不用 scipy Powell：它在 8 维 chamfer 代价上要几百次评估（>300ms），
        而相机几何是平滑的，从不错初值出发用 1/2/4/8px 的坐标下降
        四五十次评估就够，耗时可控（每 K 帧一次，摊到每帧 <1ms）。
        """
        rect_w = _world_rect(court)
        quad0, ok = _project(H0, rect_w)
        if not ok.all():
            return H0, 1e9
        x = quad0.reshape(-1).copy()
        tpl = self._templates_mid[court.model]

        def obj(xv: np.ndarray) -> float:
            pts = xv.reshape(4, 2)
            if not np.all(np.isfinite(pts)):
                return 1e9
            Hm, _ = cv2.findHomography(rect_w, pts, 0)
            if Hm is None:
                return 1e9
            return self._score(Hm, dts, mpts, court, shape, templates=tpl)

        best = obj(x)
        # 步长按外框尺寸定：初值偏差可能有几十像素（线对应 DLT 是恰定问题，
        # 线条噪声会被放大），固定 8px 起步出不了盆地
        diag = float(np.linalg.norm(quad0[2] - quad0[0]))
        diag = max(diag, 120.0)
        for step in (0.06 * diag, 0.03 * diag, 0.015 * diag):
            for _pass in range(2):
                improved = False
                for i in range(8):
                    for d in (step, -step):
                        x2 = x.copy()
                        x2[i] += d
                        sc = obj(x2)
                        if sc < best - 1e-6:
                            x, best, improved = x2, sc, True
                if not improved:
                    break
            if best >= 1e8:
                return H0, 1e9
        H, _ = cv2.findHomography(rect_w, x.reshape(4, 2), 0)
        if H is None:
            return H0, 1e9
        final = self._score(H, dts, mpts, court, shape)  # 用全采样复核
        return H, float(final)

    # ── 主流程 ────────────────────────────────────────────
    def fit(self, pred: np.ndarray, shape: tuple[int, int]) -> LineFitResult | None:
        if int((pred == CL_BOUNDARY).sum()) < 25 and int((pred == CL_PAINT).sum()) < 25:
            self.last_candidates = []
            return None
        dts, mpts = self._mask_index(pred)
        if not dts or (CL_BOUNDARY not in dts and CL_PAINT not in dts):
            self.last_candidates = []
            return None
        h, w = shape

        cands: list[_Cand] = []
        n_cand = 0
        for court in self.courts:
            inits = self._init_from_lines(pred, court, shape) + \
                self._init_from_paint(pred, court, shape)
            with_scores = []
            for H, src in inits:
                if not self._quick_ok(H, shape, court):
                    continue
                sc = self._score(H, dts, None, court, shape,
                                 templates=self._templates_coarse[court.model])
                with_scores.append(_Cand(H=H, score=sc, source=f"{court.model}:{src}"))
                if len(with_scores) >= 40:
                    break
            with_scores.sort(key=lambda c: c.score)
            cands += with_scores[:2]
            n_cand += len(with_scores)

        if not cands:
            self.last_candidates = []
            return None
        cands.sort(key=lambda c: c.score)
        self.last_candidates = [(c.source, round(c.score, 2), "") for c in cands[:6]]

        best_H, best_s, best_src = None, 1e9, ""
        for c in cands[: max(1, self.refine_top)]:
            court = next(ct for ct in self.courts if c.source.startswith(ct.model))
            H, sc = self._refine(c.H, dts, mpts, court, shape)
            if sc < best_s:
                best_H, best_s, best_src = H, sc, c.source
        if best_H is None or not np.isfinite(best_s) or best_s > self.max_score:
            return None

        court = next(ct for ct in self.courts if best_src.startswith(ct.model))

        # 模型输入尺度 → 原始帧尺度：pixel_frame = S @ pixel_model
        sx, sy = shape[1] / w, shape[0] / h
        S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
        Hinv = S @ best_H  # court -> pixel（帧尺度）
        try:
            H = np.linalg.inv(Hinv)  # pixel -> court
        except np.linalg.LinAlgError:
            return None
        scale = 0.5 * (sx + sy)
        cal = Calibration(
            H=H.tolist(), H_inv=Hinv.tolist(),
            reproj_error_px=float(best_s * scale),
            keypoints_used=4,
            court_model=court.model,
            timestamp=time.time(),
        )
        return LineFitResult(calib=cal, score=float(best_s), source=best_src,
                             candidates=n_cand)

    def calibrate(self, rgb: np.ndarray) -> Calibration | None:
        """整帧标定：返回通过几何校验的 Calibration 或 None。"""
        pred = self.predict(rgb)
        res = self.fit(pred, (self.size[0], self.size[1]))
        if res is None:
            return None
        if not res.calib.is_plausible(rgb.shape[1], rgb.shape[0]):
            return None
        res.calib.timestamp = time.time()
        return res.calib


# ── 时序平滑 ────────────────────────────────────────────────────

@dataclass
class TrackerStats:
    fits: int = 0
    ok: int = 0
    stale_returns: int = 0
    resets: int = 0
    last_score: float = 0.0
    last_fit_frame: int = -10 ** 9
    last_source: str = ""
    corners: Optional[np.ndarray] = None


class CalibrationTracker:
    """每个视频一份：每 K 帧拟合一次 → 近 N 次角点取中值 → 单帧失败沿用上一帧。

    - 只有"通过 plausible + 代价阈值"的拟合才进历史；
    - 相机切镜（新角点离中值过远）时清空历史，避免把两个机位的 H 平均成垃圾；
    - 陈旧回退最多 stale_s 秒，超过就返回 None（宁可不剔除，也不误删球员）。
    """

    def __init__(self, calibrator: LineCalibrator, every: int = 10, window: int = 5,
                 stale_s: float = 2.0, cut_frac: float = 0.22, retry_every: int = 3,
                 async_fit: bool = True):
        self.calibrator = calibrator
        self.every = max(1, int(every))
        self.retry_every = max(1, int(retry_every))
        self.window = max(1, int(window))
        self.stale_s = float(stale_s)
        self.cut_frac = float(cut_frac)
        self._hist: deque[np.ndarray] = deque(maxlen=self.window)
        self._calib: Optional[Calibration] = None
        self._calib_time = 0.0
        self.stats = TrackerStats()
        self._lock = threading.Lock()
        self._slot: Optional[tuple[int, np.ndarray]] = None
        self._wake = threading.Event()
        self._stop = False
        self._worker: Optional[threading.Thread] = None
        if async_fit:
            self._worker = threading.Thread(target=self._loop, daemon=True,
                                            name=f"calib-{calibrator.label or 'x'}")
            self._worker.start()

    # ── 拟合结果落到时序状态 ──────────────────────────────
    def _project_corners(self, cal: Calibration) -> np.ndarray:
        court = COURTS.get(cal.court_model, FIBA)
        quad, _ok = _project(np.array(cal.H_inv, dtype=np.float64), _world_rect(court))
        return quad

    def _build(self, corners: np.ndarray, court_model: str, err: float) -> Optional[Calibration]:
        court = COURTS.get(court_model, FIBA)
        Hinv, _ = cv2.findHomography(_world_rect(court), corners, 0)
        if Hinv is None:
            return None
        try:
            H = np.linalg.inv(Hinv)
        except np.linalg.LinAlgError:
            return None
        return Calibration(H=H.tolist(), H_inv=Hinv.tolist(), reproj_error_px=float(err),
                           keypoints_used=4, court_model=court_model, timestamp=time.time())

    def _apply(self, cal: Optional[Calibration], shape: tuple[int, int], now: float) -> None:
        st = self.stats
        st.fits += 1
        if cal is None:
            with self._lock:
                if self._hist and now - self._calib_time > self.stale_s:
                    self._hist.clear()
                    self._calib = None
            return
        st.ok += 1
        st.last_score = cal.reproj_error_px
        st.last_source = (self.calibrator.last_candidates[0][0]
                          if self.calibrator.last_candidates else "")
        corners = self._project_corners(cal)
        with self._lock:
            if self._hist:
                med = np.median(np.stack(self._hist), axis=0)
                diag = math.hypot(shape[1], shape[0])
                if float(np.mean(np.linalg.norm(corners - med, axis=1))) > self.cut_frac * diag:
                    self._hist.clear()  # 切镜：历史作废，从头累积
                    st.resets += 1
            self._hist.append(corners)
            built = self._build(np.median(np.stack(self._hist), axis=0),
                                cal.court_model, st.last_score)
            if built is not None:
                self._calib = built
                self._calib_time = now
                st.corners = self._project_corners(built)

    # ── 后台拟合线程（不阻塞帧路径；只处理已解码过的帧） ──────
    def _loop(self) -> None:
        while not self._stop:
            if not self._wake.wait(0.2):
                continue
            self._wake.clear()
            with self._lock:
                job = self._slot
                self._slot = None
            if job is None:
                continue
            frame_no, small = job
            try:
                cal = self.calibrator.calibrate(small)
            except Exception as e:  # 后台线程不能把服务搞挂
                print(f"[calib] worker error frame={frame_no}: {e}")
                continue
            self._apply(cal, small.shape[:2], time.time())

    def stop(self) -> None:
        self._stop = True
        self._wake.set()

    # ── 对外：取当前 H，必要时提交一次拟合 ──────────────────
    def update(self, frame_no: int, rgb: np.ndarray, force: bool = False) -> Optional[Calibration]:
        now = time.time()
        st = self.stats
        # 还没有可用标定时更勤快地重试（前几帧就能抓到）
        step = self.every if self._calib is not None else min(self.every, self.retry_every)
        due = (frame_no < st.last_fit_frame) or (frame_no - st.last_fit_frame >= step) or force
        if due:
            st.last_fit_frame = frame_no
            if self._worker is not None:
                h, w = self.calibrator.size
                small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
                with self._lock:
                    self._slot = (frame_no, small)  # 单槽：旧任务直接丢弃，不排队
                self._wake.set()
            else:
                self._apply(self.calibrator.calibrate(rgb), rgb.shape[:2], now)

        with self._lock:
            cal = self._calib
            age = now - self._calib_time
        if cal is not None:
            if age > self.stale_s:
                with self._lock:
                    self._calib = None
                    self._hist.clear()
                return None
            if age > 0.05:
                st.stale_returns += 1
            return cal
        return None

    @property
    def age_s(self) -> float:
        return time.time() - self._calib_time if self._calib_time else 1e9


__all__ = ["LineCalibrator", "LineFitResult", "CalibrationTracker",
           "court_line_samples", "DEFAULT_SIZE"]