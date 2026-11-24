"""从球场线分割结果拟合单应矩阵（快速实时标定）。

思路：
1. 小球场线分割网络（bsk.calib_seg_net）逐帧输出 5 类线掩码；
2. 用罚球区（paint）四边形给出 4 点初始对应（左右两块都试）；
3. 用"模板线采样点到分割掩码距离变换"的 chamfer 代价评估/精化：
   把模板线（边线/三分/罚球区/中线/中圈）在世界坐标采样，经 H 投影到
   像素空间，要求每个点落在对应类别的线像素附近。Powell 搜索球场外框
   四角的像素质心完成精化。

输出 Calibration（H=pixel->court, H_inv=court->pixel），并做几何合理性校验。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from bsk.calibrate import Calibration
from bsk.court import Court, FIBA
from bsk.calib_seg_net import CLASS_NAMES, INPUT_SIZE, CourtLineSegNet, NUM_CLASSES

CL_BOUNDARY = 1
CL_THREE_POINT = 2
CL_PAINT = 3
CL_CENTER_LINE = 4
CL_CENTER_CIRCLE = 5

DEFAULT_SIZE = (360, 640)  # (H, W)，与混合训练数据一致


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
        # 底角直线
        arc_start_lo = (rim_x - sign * r3 * math.cos(-theta_c), -y_corner)
        arc_start_hi = (rim_x - sign * r3 * math.cos(theta_c), y_corner)
        out[CL_THREE_POINT] += _seg((bl, -y_corner), arc_start_lo, step)
        out[CL_THREE_POINT] += _seg((bl, y_corner), arc_start_hi, step)
        n_arc = 48
        for i in range(n_arc + 1):
            th = -theta_c + 2 * theta_c * i / n_arc
            out[CL_THREE_POINT].append((rim_x - sign * r3 * math.cos(th), r3 * math.sin(th)))

    return {c: np.array(v, dtype=np.float64) for c, v in out.items() if v}


@dataclass
class LineFitResult:
    calib: Calibration
    score: float


class LineCalibrator:
    """分割线 → 单应。模型懒加载。"""

    def __init__(self, court: Court = FIBA, size: tuple[int, int] = DEFAULT_SIZE,
                 weights: str | Path | None = None, device: str | None = None,
                 conf_thr: float = 0.5, step: float = 0.5):
        self.court = court
        self.size = size
        self.weights = weights
        self.device = device
        self.conf_thr = conf_thr
        self.step = step
        self._samples = court_line_samples(court, step)
        self._model = None
        self._norm = None

    # ── 模型 ──────────────────────────────────────────────
    def _load(self):
        if self._model is not None:
            return
        import torch
        from torchvision import transforms

        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        model = CourtLineSegNet(num_classes=NUM_CLASSES, pretrained=False).to(dev)
        if self.weights and Path(self.weights).exists():
            try:
                ckpt = torch.load(self.weights, map_location=dev, weights_only=False)
            except TypeError:
                ckpt = torch.load(self.weights, map_location=dev)
            state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state)
        model.eval()
        self._model = model
        self._norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """返回 [H, W] 类别图（0..5）。"""
        import torch

        self._load()
        h, w = self.size
        small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        x = self._norm(torch.from_numpy(small).permute(2, 0, 1).float() / 255.0)
        with torch.no_grad():
            logits = self._model(x.unsqueeze(0).to(self.device))
            probs = logits.softmax(1)[0]
        pred = probs.argmax(0).cpu().numpy().astype(np.uint8)
        conf = probs.max(0).values.cpu().numpy()
        pred[conf < self.conf_thr] = 0
        return pred

    # ── 拟合 ──────────────────────────────────────────────
    def _dts(self, pred: np.ndarray) -> dict[int, np.ndarray]:
        dts = {}
        for c in range(1, NUM_CLASSES):
            m = (pred == c).astype(np.uint8)
            if m.sum() < 20:
                continue
            dts[c] = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
        return dts

    def _cost(self, H: np.ndarray, dts: dict[int, np.ndarray], shape: tuple[int, int]) -> float:
        h, w = shape
        total, weight = 0.0, 0.0
        for c, pts in self._samples.items():
            dt = dts.get(c)
            if dt is None:
                continue
            ph = np.column_stack([pts, np.ones(len(pts))])
            pr = ph @ H.T
            ok = np.abs(pr[:, 2]) > 1e-8
            uv = pr[:, :2] / np.where(ok, pr[:, 2:3], 1.0)
            inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            if inside.sum() < 15:
                continue
            xs = uv[inside, 0].astype(np.int32)
            ys = uv[inside, 1].astype(np.int32)
            d = dt[ys, xs]
            # 截断大误差，避免个别离群点主导（线被遮挡是常态）
            total += float(np.mean(np.minimum(d, 24.0)))
            weight += 1.0
        return total / weight if weight > 0 else 1e9

    def _corner_candidates(self, pred: np.ndarray) -> list[np.ndarray]:
        """从罚球区掩码取候选四边形角点。"""
        m = (pred == CL_PAINT).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
            if cv2.contourArea(c) < 400:
                continue
            peri = cv2.arcLength(c, True)
            for eps in (0.02, 0.04, 0.07):
                ap = cv2.approxPolyDP(c, eps * peri, True)
                if len(ap) == 4:
                    quad = ap.reshape(4, 2).astype(np.float64)
                    # 按角度排序成一致的顺时针顺序
                    ctr = quad.mean(axis=0)
                    ang = np.arctan2(quad[:, 1] - ctr[1], quad[:, 0] - ctr[0])
                    cands.append(quad[np.argsort(ang)])
                    break
        return cands

    def fit(self, pred: np.ndarray, shape: tuple[int, int]) -> LineFitResult | None:
        dts = self._dts(pred)
        if not dts:
            return None
        h, w = shape
        hl, hw = self.court.half_length, self.court.half_width

        # 罚球区四边形的世界角点（左右两侧分开试）
        best: tuple[float, np.ndarray] | None = None
        for side in ("A", "B"):
            bl = self.court.baseline_x(side)
            sign = 1 if side == "A" else -1
            ft_x = bl - self.court.free_throw_dist * sign
            fthw = self.court.free_throw_width / 2
            lane = np.array([
                [bl, -fthw], [bl, fthw], [ft_x, fthw], [ft_x, -fthw],
            ], dtype=np.float64)
            for quad in self._corner_candidates(pred):
                # 图像角点按角度排序后，与 lane 的 4 种旋转对齐
                for rot in range(4):
                    src = np.roll(lane, rot, axis=0)
                    # 只接受不退化/不自交的对应
                    if _poly_area(quad) < 200:
                        continue
                    Hm, _ = cv2.findHomography(src, quad, 0)
                    if Hm is None:
                        continue
                    cst = self._cost(Hm, dts, (h, w))
                    if best is None or cst < best[0]:
                        best = (cst, Hm)
        if best is None:
            return None

        # 以球场外框四角为参数做 Powell 精化
        score, H = best
        corners_world = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]], dtype=np.float64)
        proj = np.column_stack([corners_world, np.ones(4)]) @ H.T
        corners_px = proj[:, :2] / proj[:, 2:3]
        if not np.all(np.isfinite(corners_px)):
            return None

        try:
            from scipy.optimize import minimize
            x0 = corners_px.reshape(-1)

            def obj(x):
                pts = x.reshape(4, 2).astype(np.float64)
                Hm, _ = cv2.findHomography(corners_world, pts, 0)
                if Hm is None:
                    return 1e9
                return self._cost(Hm, dts, (h, w))

            res = minimize(obj, x0, method="Powell",
                           options={"maxiter": 80, "xtol": 0.5, "ftol": 0.05})
            if res.fun < score:
                H, score = cv2.findHomography(corners_world, res.x.reshape(4, 2), 0), float(res.fun)
            if H is None:
                return None
        except ImportError:
            pass

        scale_x, scale_y = shape[1] / w, shape[0] / h
        # 当前在模型输入尺寸空间拟合，换算回原始帧尺寸
        if abs(scale_x - 1.0) > 1e-6 or abs(scale_y - 1.0) > 1e-6:
            S = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float64)
            H = S @ H
        H_inv = np.linalg.inv(H)
        cal = Calibration(
            H=H.tolist(), H_inv=H_inv.tolist(),
            reproj_error_px=float(score),
            keypoints_used=0,
            court_model=self.court.model,
            timestamp=0.0,
        )
        return LineFitResult(calib=cal, score=float(score))

    def calibrate(self, rgb: np.ndarray) -> Calibration | None:
        """整帧标定：返回通过几何校验的 Calibration 或 None。"""
        pred = self.predict(rgb)
        res = self.fit(pred, (self.size[0], self.size[1]))
        if res is None:
            return None
        if not res.calib.is_plausible(rgb.shape[1], rgb.shape[0]):
            return None
        import time
        res.calib.timestamp = time.time()
        return res.calib


def _poly_area(quad: np.ndarray) -> float:
    x, y = quad[:, 0], quad[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


__all__ = ["LineCalibrator", "LineFitResult", "court_line_samples"]