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
from scipy.spatial import cKDTree

from bsk.calib_seg_net import NUM_CLASSES, CourtLineSegNet
from bsk.calibrate import Calibration
from bsk.court import COURTS, FIBA, Court, project_court_polygon

CL_BOUNDARY = 1
CL_THREE_POINT = 2
CL_PAINT = 3
CL_CENTER_LINE = 4
CL_CENTER_CIRCLE = 5

DEFAULT_SIZE = (540, 960)  # (H, W)：HD 训练/推理分辨率
# 验收门实测：calib_seg_mix2 真实帧边界召回显著高于 hd2（hd2 训练轮数不足），
# 默认权重必须指到通过验收的模型；hd2 保留在 checkpoints 里备查，不再作为默认。
DEFAULT_WEIGHTS = "checkpoints/calib_seg_mix2/best.pt"
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
    rank: float = 0.0
    structure: float = 0.0


class LineCalibrator:
    """分割线 → 单应。模型懒加载。"""

    def __init__(self, court: Court | None = None, size: tuple[int, int] = DEFAULT_SIZE,
                 weights: str | Path | None = None, device: str | None = None,
                 conf_thr: float = 0.5, step: float = 0.5, max_score: float = 9.0,
                 refine_top: int = 2, powell_iter: int = 40, max_inits: int = 24,
                 label: str = "", auto_na: bool = True, prior_max_score: float = 16.0):
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
        # 粗筛参与打分的初值数上限（真实帧上 Hough 线簇多，初值可达数百个，
        # 无上限时单帧拟合 600~1300ms，超实时预算）
        self.max_inits = max(4, int(max_inits))
        self.auto_na = bool(auto_na)
        self.prior_max_score = float(prior_max_score)
        self.label = label
        self._templates = {c.model: court_line_samples(c, step) for c in self.courts}
        self._templates_mid = {c.model: court_line_samples(c, max(step, 1.0)) for c in self.courts}
        self._templates_coarse = {c.model: court_line_samples(c, max(step, 2.5)) for c in self.courts}
        self._model = None
        self._norm = None
        self._infer_lock = threading.Lock()
        self.last_candidates: list[tuple[str, float, str]] = []
        # 逐帧诊断（供 bsk/dev/calib_gate.py 打印失败分支）：证据像素量、
        # Hough 线族、初值数、各候选代价分解、被哪一道关卡拒掉。
        # diagnose=True 时额外对前 3 名候选做一次完整代价分解（多 ~100ms/帧），
        # 实时链路保持 False。
        self.last_diag: dict = {}
        self.diagnose = False

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
    def _mask_index(pred: np.ndarray, wmask: np.ndarray | None = None
                    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray | None]:
        dts: dict[int, np.ndarray] = {}
        mpts: dict[int, np.ndarray] = {}
        wpts = None
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
        # 白线证据点：既参与外框精确率项，也参与反向 chamfer（防"缩成一条线"）。
        if wmask is not None and int(wmask.sum()) >= 60:
            ys, xs = np.nonzero(wmask)
            if len(xs) > 1200:
                sel = np.linspace(0, len(xs) - 1, 1200).astype(int)
                ys, xs = ys[sel], xs[sel]
            wpts = np.column_stack([xs, ys]).astype(np.float64)
        return dts, mpts, wpts

    # ── 白线证据（真实域兜底通道） ─────────────────────────
    @staticmethod
    def white_evidence(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """真实帧里的白色地板线掩码（与分割并行的一条证据通道）。

        分割模型在真实转播上有明显域差（边界类召回可以掉到 0），但地板线
        本身通常是亮的低饱和结构，且一定落在地板色（木色/漆区色）附近。
        实测踩过的坑（真实转播帧）：
          * 用"连通域面积≥14 且长宽比≥2.5"细线过滤会砍掉 87% 证据
            （线上被压缩噪点切成 5~10px 的短虚线段，中位面积只有 6px），
            代价随即饱和在 cap 上，任何 H 得分都一样 → 拟合必然失败；
          * 只按白阈值取像素又会带进白色球衣、地板广告/logo（面积大、
            实心），它们比线更"白"，且面积是线段的上百倍。
        因此这里改成分两种形态丢弃而非按形状保留：
          1. 白：V>150 且 S<60（真实线被灯光压暗时 S/V 都会掉，不能卡死）；
          2. 地板门控：木色/漆区色最大连通域膨胀 9px 内，排除看台/广告牌；
          3. 丢实心大块：面积≥2600，或面积≥900 且填充率>0.42（球衣/地板 logo）；
          4. 丢粗白内部：距背景 >2.7px 的像素（球衣主体、字母笔画；
             line 宽通常 2~4px，内部距离≤2，不受影响）。
        地板色占比不足（其它球馆配色不同）时退化为"画面下半部"门控。
        """
        h, w = size
        im = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        hh = hsv[:, :, 0].astype(np.int16)
        ss, vv = hsv[:, :, 1], hsv[:, :, 2]
        wood = ((hh >= 8) & (hh <= 26) & (ss > 40) & (vv > 120)).astype(np.uint8)
        paint = ((hh >= 82) & (hh <= 112) & (ss > 80) & (vv > 55)).astype(np.uint8)
        floor = cv2.morphologyEx((wood | paint).astype(np.uint8), cv2.MORPH_CLOSE,
                                 np.ones((13, 13), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(floor, 8)
        near = None
        if n > 1:
            idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            if stats[idx, cv2.CC_STAT_AREA] > 0.02 * w * h:
                fl = (lab == idx).astype(np.uint8)
                # 地板被球员/广告遮断成多块时补洞，避免门控只覆盖半个场地
                ff = fl.copy()
                mk = np.zeros((h + 2, w + 2), np.uint8)
                cv2.floodFill(ff, mk, (0, 0), 1)
                fl = cv2.morphologyEx((fl | (ff == 0).astype(np.uint8)), cv2.MORPH_CLOSE,
                                      np.ones((21, 21), np.uint8))
                near = cv2.dilate(fl, np.ones((9, 9), np.uint8))
        if near is None:  # 退化为下半部带状门控
            near = np.zeros((h, w), np.uint8)
            near[int(0.12 * h):int(0.89 * h), :] = 1
        lines = ((vv > 150) & (ss < 60) & (near > 0)).astype(np.uint8)
        lines[:int(0.05 * h), :] = 0
        lines[int(0.895 * h):, :] = 0
        nl, lb, st2, _ = cv2.connectedComponentsWithStats(lines, 8)
        if nl > 1:
            drop = np.zeros(nl, np.uint8)
            ar = st2[:, cv2.CC_STAT_AREA]
            bw = st2[:, cv2.CC_STAT_WIDTH].astype(np.float64)
            bh = st2[:, cv2.CC_STAT_HEIGHT].astype(np.float64)
            fill = ar / np.maximum(bw * bh, 1.0)
            drop[1:] = ((ar[1:] >= 2600) | ((ar[1:] >= 900) & (fill[1:] > 0.42))).astype(np.uint8)
            if drop.any():
                lines[drop[lb] > 0] = 0
        dist = cv2.distanceTransform(lines, cv2.DIST_L2, 3)
        lines[dist > 2.7] = 0
        return lines

    @staticmethod
    def long_line_evidence(wmask: np.ndarray, shape: tuple[int, int],
                           min_frac: float = 0.07, max_n: int = 48
                           ) -> tuple[np.ndarray, list[dict], float]:
        """从白线证据里保留长直线段；球衣/广告/噪声短段不参与拟合。"""
        h, w = shape
        lines = cv2.HoughLinesP(
            (wmask * 255).astype(np.uint8), 1, np.pi / 180, threshold=18,
            minLineLength=max(36, int(min_frac * w)), maxLineGap=12,
        )
        out = np.zeros((h, w), np.uint8)
        items: list[dict] = []
        if lines is None:
            return out, items, 0.0
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            length = math.hypot(float(x2) - float(x1), float(y2) - float(y1))
            if length < min_frac * w:
                continue
            ln = _line_from_pts((float(x1), float(y1)), (float(x2), float(y2)))
            if ln is None:
                continue
            items.append({"line": ln, "pts": ((x1, y1), (x2, y2)), "len": length,
                          "angle": math.atan2(ln[1], ln[0]) % math.pi})
        items.sort(key=lambda d: -d["len"])
        items = items[:max_n]
        for it in items:
            (x1, y1), (x2, y2) = it["pts"]
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), 1, 2)
        max_len = items[0]["len"] if items else 0.0
        return out, items, float(max_len)

    def frame_metrics(self, rgb: np.ndarray, pred: np.ndarray | None = None) -> dict:
        """帧级可标定性：给 gate 的 NOT_APPLICABLE 与 runtime 快速跳过用。

        只看“确实有球场结构”的证据，不把近景/球场大半出画算作失败：
        长直线数量、最大线长、白线像素、地板占比、分割类像素量。
        """
        h, w = self.size
        im = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        hh = hsv[:, :, 0].astype(np.int16)
        ss, vv = hsv[:, :, 1], hsv[:, :, 2]
        wood = ((hh >= 8) & (hh <= 26) & (ss > 40) & (vv > 120)).astype(np.uint8)
        paint = ((hh >= 82) & (hh <= 112) & (ss > 80) & (vv > 55)).astype(np.uint8)
        floor_ratio = float((wood | paint).mean())
        wmask = self.white_evidence(rgb, (h, w))
        long_mask, lines, max_len = self.long_line_evidence(wmask, (h, w))
        counts = {}
        if pred is not None:
            for c in range(1, NUM_CLASSES):
                counts[c] = int((pred == c).sum())
        n_long = len(lines)
        white_px = int(wmask.sum())
        reason = ""
        if floor_ratio < 0.10:
            reason = "floor_too_small"
        elif white_px < max(250, int(0.0015 * w * h)):
            reason = "white_evidence_too_small"
        elif n_long < 2:
            reason = "no_long_court_lines"
        elif max_len < 0.16 * w:
            # 只有很短的白线：近景/球员占满，不能可靠恢复整场 H
            reason = "court_structure_too_small"
        elif n_long < 4 and max_len < 0.22 * w and counts.get(CL_PAINT, 0) < 120:
            reason = "court_structure_too_small"
        return {
            "ok": not reason, "reason": reason, "floor_ratio": round(floor_ratio, 3),
            "white_px": white_px, "n_long": n_long, "max_len": round(max_len, 1),
            "long_px": int(long_mask.sum()), "class_px": counts,
        }

    @staticmethod
    def structure_evidence(rgb: np.ndarray, shape: tuple[int, int],
                           lines: list[dict] | None = None) -> dict:
        """强结构证据：线族消失点、线交点、中圈、漆区矩形。

        这些量只用于候选重排（不是硬门槛）：真实帧里任何单一线索都可能
        缺一段，硬门槛会把正确 H 一起删掉。
        """
        h, w = shape
        im = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        hh = hsv[:, :, 0].astype(np.int16)
        ss, vv = hsv[:, :, 1], hsv[:, :, 2]
        out = {"vps": [], "inter": [], "circle": None, "lanes": [], "teal": None}
        if lines is None:
            return out

        # 线族：按方向分两簇，簇内两两求交得到消失点
        angs = np.array([[it["angle"]] for it in lines], dtype=np.float32)
        if len(lines) >= 4:
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.05)
            try:
                _compact, labels, _centers = cv2.kmeans(
                    angs, 2, None, crit, 8, cv2.KMEANS_PP_CENTERS)
                fams = [[lines[i]["line"] for i, lab in enumerate(labels.ravel()) if lab == k]
                        for k in (0, 1)]
            except cv2.error:
                fams = []
        else:
            fams = []

        def isect(l1, l2):
            A = np.array([[l1[0], l1[1]], [l2[0], l2[1]]], dtype=np.float64)
            det = float(np.linalg.det(A))
            if abs(det) < 1e-9:
                return None
            p = np.linalg.solve(A, -np.array([l1[2], l2[2]], dtype=np.float64))
            return p if np.all(np.isfinite(p)) else None

        for fam in fams:
            pts = []
            for i in range(len(fam)):
                for j in range(i + 1, len(fam)):
                    p = isect(fam[i], fam[j])
                    if p is not None and -w < p[0] < 2 * w and -h < p[1] < 2 * h:
                        pts.append(p)
            if pts:
                out["vps"].append(np.median(np.asarray(pts), axis=0))
        if len(fams) == 2:
            pts = []
            for a in fams[0][:10]:
                for b in fams[1][:10]:
                    p = isect(a, b)
                    if p is not None and 0 <= p[0] < w and 0 <= p[1] < h:
                        pts.append(p)
            out["inter"] = pts[:80]

        # 中圈：白色掩码上的圆；只取合理半径
        white = ((vv > 150) & (ss < 75)).astype(np.uint8) * 255
        circles = cv2.HoughCircles(white, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80,
                                   param1=100, param2=22, minRadius=15,
                                   maxRadius=min(h, w) // 3)
        if circles is not None:
            cs = np.asarray(circles[0], dtype=np.float64)
            cs = cs[(cs[:, 1] > 0.15 * h) & (cs[:, 1] < 0.9 * h)]
            if len(cs):
                out["circle"] = cs[np.argmax(cs[:, 2])][:3].copy()

        # 漆区色块：只作为弱约束
        teal = ((hh >= 82) & (hh <= 112) & (ss > 80) & (vv > 55)).astype(np.uint8)
        out["teal"] = teal
        n, lab, st, _ = cv2.connectedComponentsWithStats(teal, 8)
        for i in range(1, n):
            if int(st[i, cv2.CC_STAT_AREA]) < 0.008 * w * h:
                continue
            ys, xs = np.nonzero(lab == i)
            rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
            if min(rect[1]) >= 8:
                out["lanes"].append(cv2.boxPoints(rect).astype(np.float64))
        return out

    def _structure_penalty(self, H: np.ndarray, court: Court, shape: tuple[int, int],
                           st: dict) -> float:
        """候选的强结构残差（越小越可信）；只用于排序，不直接当硬门槛。"""
        h, w = shape
        pen = 0.0
        rect = _world_rect(court)
        uv, ok = _project(H, rect)
        if not ok.all():
            return 20.0
        corners = uv
        if st.get("inter"):
            inter = np.asarray(st["inter"], dtype=np.float64)
            if len(inter) >= 3:
                d = []
                for c in corners:
                    d.append(float(np.min(np.linalg.norm(inter - c, axis=1))))
                pen += min(4.0, float(np.median(d)) / (0.035 * w))
        if st.get("vps"):
            # 世界 x/y 方向在 H 下的消失点（齐次坐标归一化）
            vps = []
            for col in (0, 1):
                q = np.asarray(H, dtype=np.float64)[:, col]
                if abs(q[2]) > 1e-9:
                    vps.append(q[:2] / q[2])
            if vps:
                ds = []
                for v in vps:
                    ds.append(min(float(np.linalg.norm(v - vp)) for vp in st["vps"]))
                pen += min(4.0, float(np.mean(ds)) / (0.06 * w))
        if st.get("circle") is not None:
            c0, ok0 = _project(H, np.array([[0.0, 0.0]]))
            if ok0[0]:
                pen += min(3.0, float(np.linalg.norm(c0[0] - st["circle"][:2])) / (0.05 * w))
        if st.get("lanes") and st.get("teal") is not None:
            hit = []
            teal = st["teal"]
            for side in ("A", "B"):
                bl = court.baseline_x(side)
                sign = 1 if side == "A" else -1
                ft = bl - court.free_throw_dist * sign
                xs = np.linspace(bl, ft, 4)
                ys = np.linspace(-court.paint_width / 2, court.paint_width / 2, 4)
                pts = np.array([(x, y) for x in xs for y in ys], dtype=np.float64)
                u, o = _project(H, pts)
                if not o.all():
                    continue
                ui = np.clip(u[:, 0], 0, w - 1).astype(int)
                vi = np.clip(u[:, 1], 0, h - 1).astype(int)
                hit.append(float(teal[vi, ui].mean()))
            if hit:
                pen += min(3.0, 3.0 * (1.0 - float(np.mean(hit))))

        # 在画比例过低说明把球场推到了画外
        pen += 4.0 * max(0.0, 0.35 - float(ok.mean()))
        return float(pen)

    # ── 代价 ──────────────────────────────────────────────
    def _score(self, H: np.ndarray, dts: dict[int, np.ndarray], mpts: dict[int, np.ndarray] | None,
               court: Court, shape: tuple[int, int], templates=None,
               cap: float = 24.0, dt_white: np.ndarray | None = None,
               wpts: np.ndarray | None = None) -> float:
        return self._score_comp(H, dts, mpts, court, shape, templates=templates, cap=cap,
                                dt_white=dt_white, wpts=wpts)["score"]

    def _score_comp(self, H: np.ndarray, dts: dict[int, np.ndarray],
                    mpts: dict[int, np.ndarray] | None, court: Court, shape: tuple[int, int],
                    templates=None, cap: float = 24.0, dt_white: np.ndarray | None = None,
                    wpts: np.ndarray | None = None) -> dict:
        """代价分解：召回（模板→证据）、外框精确率、反向 chamfer、在画比例。

        诊断需要逐项数值，因此返回 dict；`_score` 只是取 score 字段。
        """
        h, w = shape
        tpl = templates if templates is not None else self._templates[court.model]
        total, weight = 0.0, 0.0
        any_vis = False
        for c, pts in tpl.items():
            dt = dts.get(c)
            if dt is None and dt_white is None:
                continue
            uv, ok = _project(H, pts)
            inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            if int(inside.sum()) < 12:
                continue
            any_vis = True
            xs = np.clip(uv[inside, 0], 0, w - 1).astype(np.int32)
            ys = np.clip(uv[inside, 1], 0, h - 1).astype(np.int32)
            d = None
            if dt is not None:
                d = dt[ys, xs]
            if dt_white is not None:
                dw = 1.5 * dt_white[ys, xs]
                d = dw if d is None else np.minimum(d, dw)
            total += CLASS_W[c] * float(np.mean(np.minimum(d, cap)))
            weight += CLASS_W[c]
        if not any_vis or weight <= 0:
            return {"score": 1e9, "recall": 1e9, "precision": None, "rev": None, "inb": 0.0}
        recall = total / weight
        comp = {"score": recall, "recall": recall, "precision": None, "rev": None, "inb": 0.0}
        if mpts is None:
            return comp
        precision = 0.0
        src = mpts.get(CL_BOUNDARY)
        rect_w = _world_rect(court)
        quad, okq = _project(H, rect_w)
        if not okq.all():
            return {"score": 1e9, "recall": recall, "precision": None, "rev": None, "inb": 0.0}
        if src is not None and len(src) >= 30:
            d = _rect_seg_dist(quad, src)
            precision = float(np.median(np.minimum(d, cap)))
        comp["precision"] = precision
        score = 0.7 * recall + 0.3 * precision
        # 反向 chamfer：白线证据点 → 最近的模板投影点。
        # 没有它，"整张球场缩成一条/一小块"的退化解仍能让前向 chamfer 很低
        # （模板所有类都压到同一条白线上），验收门会假通过。
        if wpts is not None and len(wpts) >= 60:
            full = self._templates[court.model]
            full_pts = np.concatenate([v for v in full.values()], 0)
            uv, ok = _project(H, full_pts)
            vis = ok & (uv[:, 0] >= -20) & (uv[:, 0] < w + 20) & \
                (uv[:, 1] >= -20) & (uv[:, 1] < h + 20)
            if int(vis.sum()) >= 40:
                tree = cKDTree(uv[vis])
                d, _idx = tree.query(wpts, k=1, workers=1)
                rev = float(np.mean(np.minimum(d, cap)))
                comp["rev"] = rev
                # 在画比例：真实帧里球场常常只有一部分在画面内，用可见比例
                # 而不是硬阈值来压"把球场推到画外换低代价"的解
                comp["inb"] = float(vis.mean())
                score = 0.55 * recall + 0.25 * precision + 0.45 * rev
        comp["score"] = score
        return comp

    def _quick_ok(self, H: np.ndarray, shape: tuple[int, int], court: Court) -> bool:
        h, w = shape
        quad, ok = _project(H, _world_rect(court))
        if not ok.all():
            return False
        area = _poly_area(quad)
        if not (0.07 * w * h <= area <= 3.0 * w * h):
            return False
        cnt = quad.reshape(-1, 1, 2).astype(np.float32)
        hull = abs(cv2.contourArea(cv2.convexHull(cnt)))
        if hull <= 1.0 or area < 0.86 * hull:
            return False
        # 面积/对角²：细长条退化解（整张球场压到一条字幕/边线上）在这里被挡掉
        d0 = quad[0] - quad[2]
        d1 = quad[1] - quad[3]
        diag2 = 0.5 * (float(d0 @ d0) + float(d1 @ d1))
        if diag2 <= 1e-6 or area / diag2 < 0.05:
            return False
        if quad[:, 0].max() < 0 or quad[:, 0].min() > w or quad[:, 1].max() < 0 or quad[:, 1].min() > h:
            return False
        return True

    # ── 初始化 A：边线 Hough 线段聚类 + 线对应 DLT ───────────
    @staticmethod
    def _hough_lines(mask: np.ndarray, min_len: int, max_n: int = 4,
                     thr: int = 18, gap: int = 8) -> list[np.ndarray]:
        m = cv2.morphologyEx(mask * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        lines = cv2.HoughLinesP(m, 1, np.pi / 180, threshold=thr,
                                minLineLength=max(min_len, 8), maxLineGap=gap)
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
    def _hough_families(pred: np.ndarray, shape: tuple[int, int],
                        wmask: np.ndarray | None = None) -> tuple[list, list]:
        """边线族 / 中线族的 Hough 聚类（与球场模型无关，可跨模型复用）。"""
        _, w = shape
        bm = (pred == CL_BOUNDARY).astype(np.uint8)
        bw = LineCalibrator._hough_lines(bm, int(0.035 * w), 6, thr=14, gap=12)
        if len(bw) < 3 and wmask is not None:
            bw = LineCalibrator._hough_lines(np.maximum(bm, wmask), int(0.035 * w), 6, thr=14, gap=12)
        cm = (pred == CL_CENTER_LINE).astype(np.uint8)
        cw = LineCalibrator._hough_lines(cm, int(0.03 * w), 1, thr=14, gap=12)
        return bw, cw

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
                         shape: tuple[int, int], wmask: np.ndarray | None = None,
                         hcw: tuple[list, list] | None = None
                         ) -> list[tuple[np.ndarray, str]]:
        h, w = shape
        # Hough 线族与球场模型无关：两个模型各算一遍要浪费 ~40ms/帧，缓存在 fit 里
        if hcw is None:
            hcw = self._hough_families(pred, shape, wmask)
        bw, cw = hcw
        sides = self._side_lines(court)
        center = np.array([1.0, 0.0, 0.0])  # x = 0
        out: list[tuple[np.ndarray, str]] = []

        # 边线常常被裁掉/遮挡，只剩 3 条 → 用中线补足第 4 条线对应，
        # 但"哪 3 条"和"对应哪 3 个边"都要枚举（Hough 可能把一条边拆成两簇）
        for sub in self._choose(bw, 4, max_sets=8):
            for perm in permutations(range(4)):
                pairs = [(sub[i], sides[perm[i]]) for i in range(4)]
                H = _dlt_from_lines(pairs)
                if H is not None:
                    out.append((H, "lines4"))
        if cw:
            for sub in self._choose(bw, 3, max_sets=6):
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
        # paint 类是"线"而不是填充区域：直接对细线做 findContours，得到的
        # 每条边都是薄长条，approxPolyDP/minAreaRect 会把一条边当成整个漆区。
        # 先闭运算把同一侧漆区的四条边接起来，再对连通域的全部像素做
        # minAreaRect，才是真正的罚球区矩形。
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        quads: list[tuple[np.ndarray, str]] = []
        for i in range(1, n):
            if int(stats[i, cv2.CC_STAT_AREA]) < 250:
                continue
            ys, xs = np.nonzero(lab == i)
            if len(xs) < 60:
                continue
            contours, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea)
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:
                box = approx.reshape(4, 2).astype(np.float64)
            else:
                rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
                ww, hh = rect[1]
                if min(ww, hh) < 8 or ww * hh < 300:
                    continue
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
                shape: tuple[int, int], dt_white: np.ndarray | None = None,
                wpts: np.ndarray | None = None) -> tuple[np.ndarray, float]:
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
            # 退化解防线：投影外框必须有足够面积、且是凸的
            area = _poly_area(pts)
            if area < 0.07 * shape[1] * shape[0] or area > 3.0 * shape[1] * shape[0]:
                return 1e9
            cnt = pts.reshape(-1, 1, 2).astype(np.float32)
            hull = abs(cv2.contourArea(cv2.convexHull(cnt)))
            if hull <= 1.0 or area < 0.84 * hull:
                return 1e9
            d0 = pts[0] - pts[2]
            d1 = pts[1] - pts[3]
            diag2 = 0.5 * (float(d0 @ d0) + float(d1 @ d1))
            if diag2 <= 1e-6 or area / diag2 < 0.06:
                return 1e9
            Hm, _ = cv2.findHomography(rect_w, pts, 0)
            if Hm is None:
                return 1e9
            return self._score(Hm, dts, mpts, court, shape, templates=tpl, dt_white=dt_white, wpts=wpts)

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
        final = self._score(H, dts, mpts, court, shape, dt_white=dt_white, wpts=wpts)  # 用全采样复核
        return H, float(final)

    # ── 主流程 ────────────────────────────────────────────
    def fit(self, pred: np.ndarray, shape: tuple[int, int], rgb: np.ndarray | None = None,
            prior: tuple[np.ndarray, str] | None = None,
            prior_trusted: bool = False) -> LineFitResult | None:
        """分割掩码（+ 可选真实帧白线证据 / 上一帧角点先验）→ 单应。

        rgb：原始帧。给定时额外计算白线证据通道，作为分割之外的第二证据；
             分割在真实域不可靠时它是唯一能用的线索。
        prior：(上一帧外框角点 [4,2], court_model)，作为额外候选初值。
        prior_trusted：prior 来自已通过验收的 H 时置 True，允许局部优先收敛，
             不跑全量候选（镜头段内减少单帧自由度）。
        """
        h, w = shape
        self.last_diag = {"ev": 0, "hough": {}, "n_init": 0, "n_quick": 0, "cands": [],
                          "best": None, "reject": "", "line_px": int((pred > 0).sum()),
                          "na": None, "strong": {}, "prior_fast": None}
        wmask = None
        evmask = None
        dt_white = None
        structure: dict = {}
        if rgb is not None:
            wmask = self.white_evidence(rgb, (h, w))
            self.last_diag["ev"] = int(wmask.sum())
            # 自动 NOT_APPLICABLE：近景/球场大半出画不进入拟合分母
            if self.auto_na:
                metrics = self.frame_metrics(rgb, pred)
                self.last_diag["na"] = metrics
                if not metrics["ok"]:
                    self.last_candidates = []
                    self.last_diag["reject"] = f"not_applicable:{metrics['reason']}"
                    return None
            # 强结构：只让长直线参与 chamfer/反向项，压掉球衣、广告字、
            # 看台白线；长线不足时退回完整白线证据。
            strong, lines, max_len = self.long_line_evidence(wmask, (h, w))
            self.last_diag["strong"] = {
                "n_long": len(lines), "max_len": round(max_len, 1),
                "strong_px": int(strong.sum()),
            }
            structure = self.structure_evidence(rgb, (h, w), lines)
            self.last_diag["structure"] = {
                "vps": len(structure.get("vps", [])),
                "inter": len(structure.get("inter", [])),
                "lanes": len(structure.get("lanes", [])),
                "circle": bool(structure.get("circle") is not None),
            }
            evmask = strong if (len(lines) >= 2 and int(strong.sum()) >= 120) else wmask
            if int(wmask.sum()) >= 120:
                dt_white = cv2.distanceTransform(1 - wmask, cv2.DIST_L2, 5)
            else:
                wmask = None
                evmask = None
        if int((pred == CL_BOUNDARY).sum()) < 25 and int((pred == CL_PAINT).sum()) < 25 \
                and wmask is None:
            self.last_candidates = []
            self.last_diag["reject"] = "no-evidence"
            return None
        dts, mpts, wpts = self._mask_index(pred, evmask)
        if int((pred > 0).sum()) >= 120:
            ys, xs = np.nonzero(pred > 0)
            if len(xs) > 1200:
                sel = np.linspace(0, len(xs) - 1, 1200).astype(int)
                ys, xs = ys[sel], xs[sel]
            wpts = np.column_stack([xs, ys]).astype(np.float64)
        if not dts and wmask is None:
            self.last_candidates = []
            self.last_diag["reject"] = "no-evidence"
            return None

        cands: list[_Cand] = []
        n_cand = 0
        prior_fast = None
        # 上一帧先验：相机慢速平移/变焦时，上一帧角点就是最好的初值。
        # 可信 prior 先做一次局部精化；分数够好就直接返回，不跑全量候选，
        # 把 8 自由度单帧问题压成“围绕上一帧 H 的局部修正”。
        prior_model = None
        if prior is not None:
            pc, prior_model = prior
            ct = next((c for c in self.courts if c.model == prior_model), None)
            if ct is not None and pc is not None and np.all(np.isfinite(pc)):
                Hp, _ = cv2.findHomography(_world_rect(ct), np.asarray(pc, np.float64), 0)
                if Hp is not None and self._quick_ok(Hp, shape, ct):
                    comp = self._score_comp(Hp, dts, mpts, ct, shape,
                                            templates=self._templates_coarse[ct.model],
                                            dt_white=dt_white, wpts=wpts)
                    sp = self._structure_penalty(Hp, ct, shape, structure) if structure else 0.0
                    comp["structure"] = sp
                    comp["rank"] = comp["score"] + 0.5 * sp
                    self.last_diag["cands"].append({"src": f"{ct.model}:prior", **comp})
                    n_cand += 1
                    if prior_trusted and comp["score"] <= self.prior_max_score:
                        Hr, sr = self._refine(Hp, dts, mpts, ct, shape,
                                              dt_white=dt_white, wpts=wpts)
                        if sr > comp["score"]:
                            Hr, sr = Hp, comp["score"]
                        if sr <= self.prior_max_score:
                            prior_fast = _Cand(H=Hr, score=float(sr),
                                               source=f"{ct.model}:prior")
                            self.last_diag["prior_fast"] = {
                                "score": round(float(sr), 2), "raw": round(float(comp["score"]), 2)}
                    if prior_fast is None:
                        cands.append(_Cand(H=Hp, score=comp["score"],
                                           source=f"{ct.model}:prior",
                                           rank=comp["rank"], structure=sp))

        if prior_fast is not None:
            cands = [prior_fast]
        else:
            for court in self.courts:
                inits = self._init_from_lines(pred, court, shape, evmask) + \
                    self._init_from_paint(pred, court, shape)
                self.last_diag["n_init"] += len(inits)
                with_scores = []
                for H, src in inits:
                    if not self._quick_ok(H, shape, court):
                        continue
                    comp = self._score_comp(H, dts, None, court, shape,
                                            templates=self._templates_coarse[court.model],
                                            dt_white=dt_white, wpts=wpts)
                    sp = self._structure_penalty(H, court, shape, structure) if structure else 0.0
                    comp["structure"] = sp
                    comp["rank"] = comp["score"] + 0.5 * sp
                    self.last_diag["n_quick"] += 1
                    if len(self.last_diag["cands"]) < 24:
                        self.last_diag["cands"].append({"src": f"{court.model}:{src}", **comp,
                                                        "score_coarse": comp["score"],
                                                        "H": H, "court": court})
                    with_scores.append(_Cand(H=H, score=comp["score"],
                                             source=f"{court.model}:{src}",
                                             rank=comp["rank"], structure=sp))
                    if len(with_scores) >= self.max_inits:
                        break
                with_scores.sort(key=lambda c: c.rank or c.score)
                best_by_source: dict[str, _Cand] = {}
                for cand in with_scores:
                    prev = best_by_source.get(cand.source)
                    if prev is None or (cand.rank or cand.score) < (prev.rank or prev.score):
                        best_by_source[cand.source] = cand
                cands += list(best_by_source.values())
                n_cand += len(with_scores)
        self.last_diag["hough"] = dict(getattr(self, "_diag_hough", {}))

        if not cands:
            self.last_candidates = []
            self.last_diag["reject"] = "no-candidate"
            return None
        cands.sort(key=lambda c: c.rank or c.score)
        self.last_diag["cands"].sort(key=lambda d: d.get("rank", d["score"]))
        # 粗筛阶段 mpts=None（没有外框精确率/反向项），诊断要对前 3 名补一次完整分解，
        # 否则面板上 precision/rev/inb 全是 nan，看不出失败原因
        for d in self.last_diag["cands"][:3]:
            Hd = d.pop("H", None)
            ctd = d.pop("court", None)
            if Hd is None or ctd is None:
                continue
            if self.diagnose:
                comp = self._score_comp(Hd, dts, mpts, ctd, shape, dt_white=dt_white, wpts=wpts)
                d.update({k: comp[k] for k in ("recall", "precision", "rev", "inb", "score")})
        self.last_candidates = [(c.source, round(c.score, 2), "") for c in cands[:6]]

        best_H, best_s, best_src = None, 1e9, ""
        for c in cands[: max(1, self.refine_top)]:
            court = next(ct for ct in self.courts if c.source.startswith(ct.model))
            H, sc = self._refine(c.H, dts, mpts, court, shape, dt_white=dt_white, wpts=wpts)
            if sc < best_s:
                best_H, best_s, best_src = H, sc, c.source
        prior_ok = prior_fast is not None and np.isfinite(best_s) and best_s <= self.prior_max_score
        if best_H is None or not np.isfinite(best_s) or (best_s > self.max_score and not prior_ok):
            self.last_diag["reject"] = f"score>{self.max_score:g} (best={best_s:.2f})"
            self.last_diag["best"] = {"src": best_src, "score": float(best_s)}
            return None
        self.last_diag["best"] = {"src": best_src, "score": float(best_s)}

        court = next(ct for ct in self.courts if best_src.startswith(ct.model))

        # 模型输入尺度 → 原始帧尺度：pixel_frame = S @ pixel_model
        # 注意：shape 是网络输入尺度（960x540），不是原始帧；必须用 rgb 的尺寸
        # 把 H 放大回去。否则 H 会在 1280/1920 帧里被当成 960 宽使用，叠加、
        # ROI 和 3D 都会整体缩错。
        ref_w, ref_h = (rgb.shape[1], rgb.shape[0]) if rgb is not None else (w, h)
        sx, sy = ref_w / w, ref_h / h
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

    def calibrate(self, rgb: np.ndarray,
                  prior: tuple[np.ndarray, str] | None = None,
                  prior_trusted: bool = False) -> Calibration | None:
        """整帧标定：返回通过几何校验的 Calibration 或 None。"""
        pred = self.predict(rgb)
        res = self.fit(pred, (self.size[0], self.size[1]), rgb=rgb, prior=prior,
                       prior_trusted=prior_trusted)
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
        self._slot: Optional[tuple[int, np.ndarray, tuple[int, int]]] = None
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
            diag = math.hypot(shape[1], shape[0])
            if self._hist:
                prev = self._hist[-1]
                resid = float(np.mean(np.linalg.norm(corners - prev, axis=1))) / diag
                if resid > self.cut_frac:
                    self._hist.clear()  # 切镜：历史作废，从头累积
                    st.resets += 1
                    ema = corners
                else:
                    # 残差门控 EMA：残差小（静止/慢推）= 重历史更稳；
                    # 残差大（快速变焦/平移）= 更信当前帧，避免中值滞后
                    a = 0.55 if resid < 0.02 else 0.8
                    ema = a * corners + (1.0 - a) * prev
            else:
                ema = corners
            self._hist.append(corners)
            if len(self._hist) > self.window:
                self._hist.popleft()
            built = self._build(ema, cal.court_model, st.last_score)
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
            frame_no, small, full_shape = job
            prior = None
            with self._lock:
                if self._calib is not None:
                    prior = (self._project_corners(self._calib), self._calib.court_model)
            try:
                cal = self.calibrator.calibrate(small, prior=prior, prior_trusted=prior is not None)
            except Exception as e:  # 后台线程不能把服务搞挂
                print(f"[calib] worker error frame={frame_no}: {e}")
                continue
            # 拟合在 960 宽模型尺度上完成；后方帧可能是 1280/1920，
            # 必须把 H 放大到原始帧尺度后再交给 ROI / 3D。
            if cal is not None and tuple(full_shape) != tuple(small.shape[:2]):
                cal = self._rescale(cal, small.shape[:2], full_shape)
            self._apply(cal, full_shape, time.time())

    @staticmethod
    def _rescale(cal: Calibration, src_shape: tuple[int, int],
                 dst_shape: tuple[int, int]) -> Optional[Calibration]:
        sh, sw = int(src_shape[0]), int(src_shape[1])
        dh, dw = int(dst_shape[0]), int(dst_shape[1])
        if (sw, sh) == (dw, dh) or sw <= 0 or sh <= 0:
            return cal
        S = np.array([[dw / sw, 0.0, 0.0], [0.0, dh / sh, 0.0], [0.0, 0.0, 1.0]],
                     dtype=np.float64)
        Hinv = S @ np.asarray(cal.H_inv, dtype=np.float64)
        try:
            H = np.linalg.inv(Hinv)
        except np.linalg.LinAlgError:
            return None
        out = Calibration(
            H=H.tolist(), H_inv=Hinv.tolist(),
            reproj_error_px=float(cal.reproj_error_px) * 0.5 * (dw / sw + dh / sh),
            keypoints_used=cal.keypoints_used, court_model=cal.court_model,
            timestamp=cal.timestamp,
        )
        out.source = getattr(cal, "source", "line_calib")
        return out

    def stop(self) -> None:
        self._stop = True
        self._wake.set()

    # ── 对外：取当前 H，必要时提交一次拟合 ──────────────────
    def update(self, frame_no: int, rgb: np.ndarray, force: bool = False) -> Optional[Calibration]:
        now = time.time()
        st = self.stats
        # 还没有可用标定时前几次快速重试；连续失败后指数退避，
        # 否则每次 3 帧就重跑一次 ~600ms 拟合，会把实时帧率拖到 20fps 以下。
        if self._calib is not None:
            step = self.every
        else:
            step = min(60, max(self.retry_every,
                               self.retry_every + 3 * (self.stats.fits // 2)))
        due = (frame_no < st.last_fit_frame) or (frame_no - st.last_fit_frame >= step) or force
        if due:
            st.last_fit_frame = frame_no
            if self._worker is not None:
                h, w = self.calibrator.size
                small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
                with self._lock:
                    self._slot = (frame_no, small, rgb.shape[:2])  # 单槽：旧任务直接丢弃，不排队
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