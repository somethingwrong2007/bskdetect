"""2D→3D 骨架提升（单目 + 已知单应）：像素关键点 → 球场世界坐标（米，右手系，z 向上）。

为什么需要这一层
----------------
下游（3D 回放、战术空间统计、球员间距离/速度、垂直起跳高度）只认米制世界坐标。
2D 像素点里没有深度信息，单应 H 又只描述**地面**（z=0）。本模块把"H + 像素射线 +
人体骨长"三者联立，逐关节解出沿射线的深度，得到几何自洽的 17×3 骨架。全部因果
（在线），只用当前帧与历史状态，不预取未来帧。

相机模型（单应分解）
--------------------
H 约定：``H`` 是 court→pixel（即项目里的 ``Calibration.H_inv``），
``H = λ K [r1 r2 t]``（平面 z=0 让第三列消失）。内参只假设焦距未知：主点＝图像
中心、方形像素、无倾斜。于是 ``M = K⁻¹H`` 的两列满足 ``m1·m2 = 0``、``|m1| = |m2|``，
两式对 ``f²`` 都是线性的，最小二乘解出 f（Zhang 的单应标定法）。再由
``λ = 2/(|m1|+|m2|)`` 取回旋转/平移，符号歧义用"相机必须在地面之上（C_z>0）"消掉，
得到 ``P = K[R|t]``。已知球场米制尺寸，所以尺度是真的（单目只欠一个整体相似变换，
这里由球场尺寸补上）。

根节点落地（精度基准）
----------------------
1. 双踝像素射线 ∩ 平面 z=0 → 两个脚点（整副骨架的地面锚）；
2. 髋中点射线与"以双脚中点为球心、半径 sqrt(腿长²−(脚间距/2)²) 的球"求交，得到
   骨盆位置（腿近似伸直时精确，屈膝时偏深）；
3. 身高 S 由鼻/肩/髋三点的"射线 ∩ 过脚点的竖直线"反推（鼻≈0.930S、肩≈0.818S、
   髋≈0.530S），限幅 1.55~2.2 m，逐 track 做 EMA。
脚点/骨盆/身高三者互相校验，3D 误差主要来自这里。

逐关节深度（骨长 + 共视连续性 + 时序）
--------------------------------------
从骨盆按父子树推进：已知父关节 3D 点 X_p、子关节像素射线 (C,d)、骨长 L，解
``|C + s·d − X_p| = L`` 的二次方程。两个根（射线上前后两个交点）用"上一帧深度 +
父关节深度"择优，于是**每根骨长度严格保持**（几何自洽），且点始终在自己的像素
射线上，重投影误差天然≈0；代价是深度对 2D 噪声极敏感——这是单目固有的深度歧义，
只能靠时序 EMA（默认 tau=0.1s）压住。

误差来源（必须写清楚，避免过度承诺）
------------------------------------
- 单目深度歧义：1px 像素噪声在 20 m 距离上约等于 0.3~0.5 m 深度误差，越远越糟；
- 蹲下/腾空/侧向跨步：根节点用"腿伸直 + 双脚触地"假设，骨盆高度会偏高；
- 骨长表是 1.85 m 标准比例，个体差异（±7%）会整体缩放四肢；
- H 自身误差（线拟合 1~5px）直接变成米制平移/旋转误差，标定挂了 3D 就不该信；
- 手臂前后摆动（沿光轴方向）几乎不可观测，只能靠时序连续性，不是精度问题。

与学习法（MotionBERT / VideoPose3D / MeTRAbs 类）的取舍
------------------------------------------------------
学习法在根相对骨长/姿态（尤其上半身前后方向）明显更准，但要 GPU、要训练/推理预算，
且主流权重多为研究/非商用许可，与项目"MIT/Apache/BSD 白名单 + 实时 ≤1.5ms/人/帧"
冲突。这里选择"几何解析 + 时序"的可解释基线：误差可复现、可评估、零额外依赖。
等 3D 展示/战术指标证明深度精度不够时，再把学习法作为**可选增强**替换逐关节深度
解算（对外接口不变，见 lift/lift_result）。

接口
----
    lifter = Pose3DLifter(court=FIBA, image_size=(1920, 1080))
    xyz = lifter.lift(kpts_px(17,3), bbox_px, H_court_to_pixel, track_id, frame)
    res = lifter.lift_result(...)      # 带 conf/visible/身高/根节点/重投影自检
    lifter.lift_batch(poses, boxes, H, frame) -> dict[int, LiftResult]

世界坐标约定与 bsk.court / bsk.schema 一致：原点球场中心，x 沿长边（A 队进攻方向
为正），y 沿短边，z 向上，地面 z=0，单位米。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from bsk.court import FIBA, Court

# ── 人体比例模型（1.85 m 参考身高；Drillis & Contini / NASA STD-3000 骨长比例）──
REF_STATURE = 1.85
TORSO = 0.288          # 髋中点 → 肩中点（本项目里的"颈"虚拟点）
SHOULDER_HALF = 0.105  # 肩中点 → 肩关节
HIP_HALF = 0.055       # 髋中点 → 髋关节
THIGH = 0.245          # 髋 → 膝
SHIN = 0.246           # 膝 → 踝
ANKLE_H = 0.039        # 踝关节离地高（鞋+踝）
UPPER_ARM = 0.186      # 肩 → 肘
FOREARM = 0.146        # 肘 → 腕
HEAD = 0.112           # 肩中点（颈）→ 鼻
NOSE_EYE = 0.032       # 鼻 → 眼
NOSE_EAR = 0.058       # 鼻 → 耳

# 站立时各关节离地高度（身高比例）：身高反推、深度两根择优的高度参考。
# 末尾两个虚拟节点：17=颈（双肩中点）、18=骨盆（双髋中点）
H_RATIO = np.array([
    0.930, 0.945, 0.945, 0.920, 0.920,   # 鼻 眼 眼 耳 耳
    0.818, 0.818,                         # 肩
    0.630, 0.630,                         # 肘
    0.485, 0.485,                         # 腕
    0.530, 0.530,                         # 髋
    0.285, 0.285,                         # 膝
    0.039, 0.039,                         # 踝
    0.818,                                # 颈（虚拟）
    0.530,                                # 骨盆（虚拟）
])

# 父子链：索引＝子关节，值＝父关节（17=颈/肩中点虚拟点，18=骨盆虚拟点）
PELVIS, NECK = 18, 17
PARENT = np.array([NECK, 0, 0, 0, 0, NECK, NECK, 5, 6, 7, 8,
                   PELVIS, PELVIS, 11, 12, 13, 14], dtype=np.intp)
# 每根骨的长度（身高比例），索引同 PARENT
BONE_RATIO = np.array([
    HEAD, NOSE_EYE, NOSE_EYE, NOSE_EAR, NOSE_EAR,
    SHOULDER_HALF, SHOULDER_HALF,
    UPPER_ARM, UPPER_ARM,
    FOREARM, FOREARM,
    HIP_HALF, HIP_HALF,
    THIGH, THIGH,
    SHIN, SHIN,
])

COCO17_NAMES = ("nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle")

VIS_CONF = 0.15        # 与渲染层一致：低于它的点画不出来 → visible=False
FPS_DEFAULT = 30.0


# ======================================================================
#  相机：单应分解 + 射线/投影
# ======================================================================

@dataclass(slots=True)
class CourtCamera:
    """由地面单应 H(court→pixel) 解出的针孔相机（世界坐标以米为单位）。"""

    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    P: np.ndarray
    C: np.ndarray          # 相机中心（世界坐标）
    f: float
    cx: float
    cy: float
    f_estimated: bool = True   # f 是否由单应解出（False = 兜底值）

    @property
    def height(self) -> float:
        return float(self.C[2])

    def rays(self, uv: np.ndarray) -> np.ndarray:
        """像素 (N,2) → 世界单位方向 (N,3)。"""
        uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
        d_cam = np.column_stack([(uv[:, 0] - self.cx) / self.f,
                                 (uv[:, 1] - self.cy) / self.f,
                                 np.ones(len(uv))])
        d = d_cam @ self.R                    # 等价于对每行做 R^T d_cam
        return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """世界点 (N,3) → 像素 (N,2) 与深度 (N,)（深度≤0 表示在相机后方）。"""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        xc = (X - self.C) @ self.R.T
        z = xc[:, 2]
        safe = np.where(np.abs(z) > 1e-6, z, 1e-6)
        u = self.f * xc[:, 0] / safe + self.cx
        v = self.f * xc[:, 1] / safe + self.cy
        return np.column_stack([u, v]), z


def _normalize_H(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        raise ValueError("H 必须是有限的 3x3 矩阵")
    return H / (np.linalg.norm(H) or 1.0)


def _looks_like_pixel_units(M: np.ndarray, W: float, Hh: float) -> bool:
    """判断 M 是不是 court→pixel：把球场四角投出去，跨度必须接近图像尺度。

    误传 pixel→court 时，米制坐标投出来还是米制（跨度 ~10），一眼能分辨——
    这比"原点是否在画面内"可靠得多（球场中心常常就在 0 附近，两种矩阵都满足）。
    """
    hl, hw = 14.0, 7.5
    pts = np.array([[-hl, -hw, 1.0], [hl, hw, 1.0], [hl, -hw, 1.0]], dtype=np.float64)
    p = (M @ pts.T).T
    if not np.all(np.isfinite(p)):
        return False
    if not np.all(np.abs(p[:, 2]) > 1e-9):
        return False
    p = p[:, :2] / p[:, 2:3]
    span = float(np.max(np.linalg.norm(p - p[0], axis=1)))
    return span > 0.25 * min(W, Hh)


def camera_from_homography(H: np.ndarray, image_size: tuple[int, int],
                           force_f: float | None = None) -> CourtCamera | None:
    """H(court→pixel) → CourtCamera；退化时返回 None（调用方应放弃 3D）。"""
    Hm = _normalize_H(H)
    W, Hh = float(image_size[0]), float(image_size[1])
    cx, cy = W * 0.5, Hh * 0.5

    if not _looks_like_pixel_units(Hm, W, Hh):
        try:
            Hm2 = _normalize_H(np.linalg.inv(Hm))
        except np.linalg.LinAlgError:
            return None
        if not _looks_like_pixel_units(Hm2, W, Hh):
            return None
        Hm = Hm2

    h = Hm.reshape(-1)
    # M = K⁻¹ H 的第 j 列 = [(h[0j]−cx·h[2j])/f, (h[1j]−cy·h[2j])/f, h[2j]]
    # （h 按行优先展开：h[0..2]=第0行，h[3..5]=第1行，h[6..8]=第2行）
    x1, y1, z1 = h[0] - cx * h[6], h[3] - cy * h[6], h[6]
    x2, y2, z2 = h[1] - cx * h[7], h[4] - cy * h[7], h[7]
    x3, y3, z3 = h[2] - cx * h[8], h[5] - cy * h[8], h[8]

    f = None
    if force_f is None:
        # m1·m2 = 0，|m1|² = |m2|²，对 g = f² 线性
        A = np.array([[z1 * z2], [z1 * z1 - z2 * z2]], dtype=np.float64)
        b = -np.array([x1 * x2 + y1 * y2, x1 * x1 + y1 * y1 - x2 * x2 - y2 * y2])
        try:
            g = float(np.linalg.lstsq(A, b, rcond=None)[0][0])
        except np.linalg.LinAlgError:
            g = -1.0
        if g > 0.0:
            f = math.sqrt(g)
        lo, hi = 0.45 * W, 4.5 * W      # 广角~长焦的合理域
        if f is not None and not (lo <= f <= hi):
            f = None
    if f is None:
        f = force_f if force_f else 1.15 * W   # 兜底：约 48° 水平视场
        f_est = False
    else:
        f_est = force_f is None

    a1 = np.array([x1 / f, y1 / f, z1])
    a2 = np.array([x2 / f, y2 / f, z2])
    a3 = np.array([x3 / f, y3 / f, z3])
    n1, n2 = float(np.linalg.norm(a1)), float(np.linalg.norm(a2))
    if not (n1 > 1e-12 and n2 > 1e-12):
        return None
    lam = 2.0 / (n1 + n2)
    r1 = lam * a1
    r2 = lam * a2
    r3 = np.cross(r1, r2)
    nr3 = float(np.linalg.norm(r3))
    if nr3 < 1e-9:
        return None
    r3 /= nr3
    t0 = lam * a3

    # H 与 -H 给出两个符号解，取"相机在地面之上"的那个
    for s in (1.0, -1.0):
        Rs = np.column_stack([s * r1, s * r2, r3])
        ts = s * t0
        C = -Rs.T @ ts
        if not (C[2] > 0.0 and np.isfinite(C).all()):
            continue
        # SVD 正交化（H 估计噪声会让 R 轻微不正交）
        U, _S, Vt = np.linalg.svd(Rs)
        Rn = U @ Vt
        if np.linalg.det(Rn) < 0:
            Rn[:, 2] *= -1.0
        tn = -Rn @ C
        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
        return CourtCamera(K=K, R=Rn, t=tn, P=K @ np.column_stack([Rn, tn]),
                           C=C, f=f, cx=cx, cy=cy, f_estimated=f_est)
    return None


# ======================================================================
#  单个人体的提升结果
# ======================================================================

@dataclass(slots=True)
class LiftResult:
    kpts3d: np.ndarray                 # (17,3) 米，世界坐标
    conf: np.ndarray = field(default_factory=lambda: np.zeros(17))
    visible: np.ndarray = field(default_factory=lambda: np.zeros(17, dtype=bool))
    root: np.ndarray = field(default_factory=lambda: np.zeros(3))     # 骨盆
    ground: np.ndarray = field(default_factory=lambda: np.zeros(3))   # 双足中点
    stature: float = REF_STATURE
    reproj_err_px: float = 0.0
    on_court: bool = True
    track_id: int = -1
    frame: int = -1


# ======================================================================
#  提升器
# ======================================================================

class Pose3DLifter:
    """逐帧、因果、纯 numpy 的 2D→3D 提升器（按 track_id 维护深度/身高 EMA）。"""

    def __init__(self, court: Court = FIBA, image_size: tuple[int, int] = (1920, 1080),
                 depth_tau: float = 0.10, height_alpha: float = 0.15,
                 max_track_age: int = 30, fps: float = FPS_DEFAULT):
        self.court = court
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.depth_tau = float(depth_tau)      # 深度 EMA 时间常数（秒）
        self.height_alpha = float(height_alpha)
        self.max_track_age = int(max_track_age)
        self.fps = float(fps)
        self._cam_key: bytes | None = None
        self._cam: CourtCamera | None = None
        self._st: dict[int, dict] = {}

    # ── 生命周期 ────────────────────────────────────────────
    def reset(self) -> None:
        self._st.clear()
        self._cam = None
        self._cam_key = None

    def set_image_size(self, size: tuple[int, int]) -> None:
        self.image_size = (int(size[0]), int(size[1]))
        self._cam = None
        self._cam_key = None

    def camera(self, H: np.ndarray) -> CourtCamera | None:
        """H 不变时走缓存（常见情形零开销）；H 变了才重新分解。"""
        Hm = np.ascontiguousarray(np.asarray(H, dtype=np.float64))
        key = Hm.tobytes()
        if key == self._cam_key:
            return self._cam
        cam = camera_from_homography(Hm, self.image_size)
        self._cam_key = key
        self._cam = cam
        return cam

    # ── 对外接口 ────────────────────────────────────────────
    def lift(self, kpts_px: np.ndarray, bbox_px, H_court_to_pixel: np.ndarray,
             track_id: int, frame: int) -> np.ndarray | None:
        """返回 (17,3) 米制世界坐标；不可解时 None（调用方不要输出 3D）。"""
        res = self.lift_result(kpts_px, bbox_px, H_court_to_pixel, track_id, frame)
        return None if res is None else res.kpts3d

    def lift_result(self, kpts_px, bbox_px, H_court_to_pixel, track_id: int = -1,
                    frame: int = 0) -> LiftResult | None:
        cam = self.camera(H_court_to_pixel) if H_court_to_pixel is not None else None
        if cam is None:
            return None
        k = np.asarray(kpts_px, dtype=np.float64)
        if k.ndim != 2 or k.shape[0] != 17 or k.shape[1] < 2:
            return None
        conf = k[:, 2].copy() if k.shape[1] > 2 else np.ones(17)
        uv = k[:, :2].copy()
        ok = np.isfinite(uv).all(1)
        if ok.sum() < 6:
            return None
        if not ok.all():
            ref = uv[[11, 12]][ok[[11, 12]]].mean(0) if ok[[11, 12]].any() else uv[ok].mean(0)
            uv[~ok] = ref
            conf[~ok] = 0.0
        bbox = np.asarray(bbox_px, dtype=np.float64) if bbox_px is not None else None

        st = self._state(track_id, frame)
        solved = self._solve(cam, uv, bbox, st, frame)
        if solved is None:
            return None
        xyz, extra = solved
        proj, _depth = cam.project(xyz)
        rp = float(np.median(np.linalg.norm(proj - uv, axis=1)))
        if not np.isfinite(rp):
            rp = float("nan")
        return LiftResult(kpts3d=xyz, conf=conf, visible=conf >= VIS_CONF,
                          root=extra["root"], ground=extra["ground"],
                          stature=float(st["h"]), reproj_err_px=rp,
                          on_court=bool(extra["on_court"]),
                          track_id=track_id, frame=int(frame))

    def lift_batch(self, poses, boxes: dict, H_court_to_pixel: np.ndarray, frame: int) -> dict:
        """批量入口：poses 为 schema.Pose2D 列表，boxes 为 {track_id: bbox_px}。"""
        out = {}
        for p in poses:
            res = self.lift_result(p.kpts, boxes.get(p.track_id), H_court_to_pixel,
                                   p.track_id, frame)
            if res is not None:
                out[p.track_id] = res
        return out

    # ── 状态 ────────────────────────────────────────────────
    def _state(self, tid: int, frame: int) -> dict:
        st = self._st.get(tid)
        if st is None or frame < st["frame"] or frame - st["frame"] > self.max_track_age:
            st = {"depth": None, "h": REF_STATURE, "frame": int(frame)}
            self._st[tid] = st
            if len(self._st) > 128:      # 防长视频 track 泄漏
                dead = [k for k, v in self._st.items() if frame - v["frame"] > self.max_track_age]
                for k in dead:
                    self._st.pop(k, None)
        return st

    # ── 核心解算 ────────────────────────────────────────────
    def _solve(self, cam: CourtCamera, uv: np.ndarray, bbox, st: dict,
               frame: int) -> tuple[np.ndarray, dict] | None:
        dirs = cam.rays(uv)                 # (17,3) 单位方向
        # 虚拟节点（颈=双肩中点、骨盆=双髋中点）也要两条射线，追加到 17/18 行
        neck_px = 0.5 * (uv[5] + uv[6])
        pelvis_px = 0.5 * (uv[11] + uv[12])
        vd = cam.rays(np.stack([neck_px, pelvis_px]))
        dirs = np.vstack([dirs, vd])
        C = cam.C
        prev = st["depth"]
        gap = frame - st["frame"]
        dt = float(np.clip(gap / max(self.fps, 1.0), 0.0, 0.2)) if prev is not None else 0.0

        # ---------- 1) 双脚落地 ----------
        feet = np.zeros((2, 3))
        foot_ok = np.zeros(2, dtype=bool)
        for i, j in enumerate((15, 16)):
            dz = dirs[j, 2]
            if dz < -1e-3:
                s = -C[2] / dz
                if 0.5 < s < 400.0:
                    feet[i] = C + dirs[j] * s
                    foot_ok[i] = True
        if foot_ok.any():
            fm = feet[foot_ok].mean(0)
        elif bbox is not None:
            # 踝不可用：用框底中点射线打地面兜底（框底≈鞋底）
            bx = 0.5 * (bbox[0] + bbox[2])
            d0 = cam.rays(np.array([[bx, bbox[3]]]))[0]
            if d0[2] >= -1e-3:
                return None
            feet[:] = C + d0 * (-C[2] / d0[2])
            fm = feet[0]
            foot_ok[:] = False
        else:
            return None
        spread = float(np.linalg.norm(feet[0] - feet[1])) if foot_ok.all() else 0.0
        on_court = (abs(fm[0]) <= self.court.half_length + 4.0
                    and abs(fm[1]) <= self.court.half_width + 4.0)

        # ---------- 2) 身高（鼻/肩/髋 射线 ∩ 过脚点竖直线）----------
        h_est = _stature_from_vertical(cam, dirs, fm)
        if np.isfinite(h_est):
            st["h"] = float(np.clip(st["h"] * (1.0 - self.height_alpha)
                                    + h_est * self.height_alpha, 1.55, 2.20))
        S = st["h"]

        # ---------- 3) 腿：从"踩地的脚"往上反解（脚点是全场最可靠的地面锚）----------
        # 旧的"骨盆=双足球面交点"隐含"腿伸直"，屈膝/跨步时骨盆被推远 0.2~0.4m。
        # 改成：踝(地面点) → 膝(小腿长) → 髋(大腿长) 逐段二次方程反推，得到骨盆
        # 候选；两条腿各给一个，取更高的那个（被抬起的脚若被硬当成着地，链路会算低）。
        pos = np.zeros((19, 3))            # 0..16 关节 + 17 颈 + 18 骨盆
        depth = np.zeros(19)
        d_pel = dirs[PELVIS]
        cands = []
        for i, (j_ank, j_knee, j_hip) in enumerate(((15, 13, 11), (16, 14, 12))):
            if not foot_ok[i]:
                continue
            depth[j_ank] = -C[2] / dirs[j_ank, 2]
            pos[j_ank] = feet[i]
            z_ref = float(pos[j_ank, 2] + SHIN * S)
            pj = float(prev[j_knee]) if (prev is not None and prev[j_knee] > 0.3) else None
            depth[j_knee] = _pick_bone_root(C, dirs[j_knee], pos[j_ank], SHIN * S, pj, z_ref, S)
            pos[j_knee] = C + dirs[j_knee] * depth[j_knee]
            z_ref = float(np.clip(pos[j_knee, 2] + THIGH * S, 0.25, 1.35))
            pj = float(prev[j_hip]) if (prev is not None and prev[j_hip] > 0.3) else None
            depth[j_hip] = _pick_bone_root(C, dirs[j_hip], pos[j_knee], THIGH * S, pj, z_ref, S)
            pos[j_hip] = C + dirs[j_hip] * depth[j_hip]
            cands.append(pos[j_hip].copy())
        if not cands:
            # 双脚都没落地（画面边缘/腾空）：退回"髋中点射线 ∩ 以框底地面点为心的球"
            L_leg = (THIGH + SHIN) * S
            r = math.sqrt(max(L_leg * L_leg - (0.5 * spread) ** 2, 0.05))
            s_pel = _pick_pelvis_root(C, d_pel, fm, r, 0.530 * S)
            if s_pel is None:
                return None
            pos[PELVIS] = C + d_pel * s_pel
        else:
            zc = np.array([c[2] for c in cands])
            sel = cands[int(np.argmax(zc))]        # 最高的那条腿＝最可能的着地腿
            if len(cands) == 2 and abs(zc[0] - zc[1]) < 0.06:
                sel = 0.5 * (cands[0] + cands[1])
            pos[PELVIS] = sel + np.array([0.0, 0.0, 0.0])
            # 骨盆在两条髋关节中点：用观测髋像素的横向位置把偏移补回来
            z_pel = float(np.clip(sel[2], 0.30, 1.20))
            pos[PELVIS][2] = z_pel
        pos[NECK] = pos[PELVIS]
        depth[PELVIS] = float(np.linalg.norm(pos[PELVIS] - C))

        # ---------- 4) 髋：与骨盆同高的平面求交（横向位置来自像素）----------
        z_hip = float(pos[PELVIS][2])
        for j in (11, 12):
            sj = _plane_hit(C, dirs[j], 2, z_hip)
            depth[j] = depth[PELVIS] if sj is None else sj
            pos[j] = C + dirs[j] * depth[j]
        pos[PELVIS] = 0.5 * (pos[11] + pos[12])
        pos[NECK] = pos[PELVIS]
        # 有脚落地的那条腿：膝保持反解结果（更可信）；另一条腿：髋→膝→踝往下解
        for i, (j_ank, j_knee, j_hip) in enumerate(((15, 13, 11), (16, 14, 12))):
            if foot_ok[i]:
                continue
            z_ref = float(np.clip(pos[j_hip, 2] - THIGH * S, 0.02, 1.2))
            pj = float(prev[j_knee]) if (prev is not None and prev[j_knee] > 0.3) else None
            depth[j_knee] = _pick_bone_root(C, dirs[j_knee], pos[j_hip], THIGH * S, pj, z_ref, S)
            pos[j_knee] = C + dirs[j_knee] * depth[j_knee]
            z_ref = float(np.clip(pos[j_knee, 2] - SHIN * S, 0.0, 1.0))
            pj = float(prev[j_ank]) if (prev is not None and prev[j_ank] > 0.3) else None
            depth[j_ank] = _pick_bone_root(C, dirs[j_ank], pos[j_knee], SHIN * S, pj, z_ref, S)
            pos[j_ank] = C + dirs[j_ank] * depth[j_ank]
            if pos[j_ank, 2] < 0.0:
                pos[j_ank, 2] = 0.0

        # ---------- 5) 躯干/上肢/头：骨长二次方程推进 ----------
        order = ((NECK, PELVIS, TORSO),
                 (5, NECK, SHOULDER_HALF), (6, NECK, SHOULDER_HALF),
                 (7, 5, UPPER_ARM), (8, 6, UPPER_ARM),
                 (9, 7, FOREARM), (10, 8, FOREARM),
                 (0, NECK, HEAD),
                 (1, 0, NOSE_EYE), (2, 0, NOSE_EYE),
                 (3, 0, NOSE_EAR), (4, 0, NOSE_EAR))
        for j, par, ratio in order:
            L = ratio * S
            # 高度参考：父关节高度 + 该骨在"标准站姿"里的竖直投影（H_RATIO 差）。
            # 只拿父关节高度当参考会把脖子也拉到骨盆高度（两解里有一个正好在那），
            # 头部/肩部误差因此翻倍——这是本模块最初 0.9m 误差的主因。
            z_par = float(pos[par, 2])
            z_ref = z_par + (H_RATIO[j] - H_RATIO[par]) * S
            z_ref = float(np.clip(z_ref, 0.05, 2.45))
            prev_j = float(prev[j]) if (prev is not None and prev[j] > 0.3) else None
            s = _pick_bone_root(C, dirs[j], pos[par], L, prev_j, z_ref, S)
            depth[j] = s
            pos[j] = C + dirs[j] * s

        # ---------- 6) 触地修正：踝点在解算高度接近地面时压到 z=0 ----------
        for i, j in enumerate((15, 16)):
            if foot_ok[i]:
                zsol = float(pos[j, 2])
                dz = dirs[j, 2]
                # 骨的二次方程在"该脚着地"时应该解到 z≈0；若解出来离地 <0.3m 就按
                # 像素射线的真实落地点重解（z 恰为 0，且仍在同一射线上）。
                if abs(zsol) < 0.30 and dz < -1e-3:
                    depth[j] = -C[2] / dz
                    pos[j] = feet[i]
                elif zsol < 0.0:
                    depth[j] = -C[2] / dz if dz < -1e-3 else depth[j]
                    pos[j] = feet[i]
            elif pos[j, 2] < 0.0:
                pos[j, 2] = 0.0

        # ---------- 7) 时序 EMA（深度域：平滑后点仍在自己的像素射线上）----------
        if prev is not None and dt > 0.0:
            a = 1.0 - math.exp(-dt / max(self.depth_tau, 1e-3))
            valid = depth > 0.3
            smoothed = a * depth + (1.0 - a) * np.where(prev > 0.3, prev, depth)
            depth = np.where(valid, smoothed, depth)
        st["depth"] = depth
        st["frame"] = int(frame)

        xyz_all = C + dirs * depth[:, None]
        xyz = xyz_all[:17]
        root = 0.5 * (xyz[11] + xyz[12])
        ground = 0.5 * (xyz[15] + xyz[16])
        return xyz, {"root": root, "ground": ground, "on_court": on_court}


# ======================================================================
#  几何小工具
# ======================================================================

def _stature_from_vertical(cam: CourtCamera, dirs: np.ndarray,
                           foot_mid: np.ndarray) -> float:
    """身高：鼻/肩/髋 射线 ∩ 过脚点竖直轴 → 高度 / 该关节站位比例 → 取中位数。

    只用水平约束（x,y 由脚点定），竖直方向是"横向"测量，几乎不受深度歧义影响。
    """
    ests = []
    for j, ratio in ((0, H_RATIO[0]), (5, H_RATIO[5]), (11, H_RATIO[11])):
        p = _closest_on_vertical(cam.C, dirs[j], foot_mid)
        if p is not None and 0.10 < p[2] < 2.8:
            ests.append(p[2] / ratio)
    if not ests:
        return float("nan")
    return float(np.median(np.sort(np.asarray(ests))))


def _roots_quadratic(C: np.ndarray, d: np.ndarray, anchor: np.ndarray,
                     radius: float) -> list[float]:
    """射线 C+s·d 与球(anchor, radius) 的交点参数（0 个或 2 个正解）。"""
    v = anchor - C
    b = float(d @ v)
    c = float(v @ v) - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return []
    sq = math.sqrt(disc)
    return [s for s in (b - sq, b + sq) if s > 0.5]


def _pick_pelvis_root(C: np.ndarray, d: np.ndarray, foot_mid: np.ndarray, radius: float,
                      z_pelvis: float) -> float | None:
    """骨盆深度：两解里挑"高度最接近 0.53S"的那个。

    为什么不用"距离先验"择优：框高换算出的距离误差能有 30%（透视压缩 + 框不含头
    顶/鞋底），足以把骨盆选到球的后交点，整副骨架沿射线平移 1m+。骨盆高度是强约束
    （站立 0.53S，蹲下也在 0.35S 以上），用它判根误差只有几厘米。
    """
    roots = _roots_quadratic(C, d, foot_mid, radius)
    if not roots:
        # 不相交（腿长估计偏短）：退化为离球心最近的射线点
        v = foot_mid - C
        return max(float(d @ v), 0.5)
    best, best_cost = roots[0], 1e9
    for s in roots:
        z = float(C[2] + d[2] * s)
        cost = abs(z - z_pelvis)
        if z < 0.12 or z > 1.40:              # 非人类骨盆高度，重罚
            cost += 5.0
        if cost < best_cost:
            best, best_cost = s, cost
    return best


def _pick_bone_root(C: np.ndarray, d: np.ndarray, anchor: np.ndarray, length: float,
                    prev: float | None, z_ref: float, stature: float) -> float:
    """子关节深度：在骨长方程的（至多两个）解里挑几何/时序最合理的那个。

    判据 = 上一帧深度（若有，权重 2）+ "离父关节高度不太远"（权重 1）。父关节高度
    是合法域代理：肢体的两端高度差不可能超过骨长，两解里通常只有一个满足。
    首帧没有时序信息，高度判据就是唯一的歧义消解（纯深度先验误差 30% 以上，见
    eval_pose_3d.py 的相机分组表）。
    """
    roots = _roots_quadratic(C, d, anchor, length)
    if not roots:
        # 骨长被 2D 投影压没了：取离锚点最近的射线点
        v = anchor - C
        s = max(float(d @ v), 0.5)
        return s
    if len(roots) == 1:
        return roots[0]
    best, best_cost = roots[0], 1e9
    for s in roots:
        z = float(C[2] + d[2] * s)
        cost = abs(z - z_ref) / max(stature, 0.5)
        if z < 0.05 or z > 2.45:
            cost += 5.0
        if prev is not None:
            cost += 2.0 * abs(s - prev) / max(stature, 0.5)
        if cost < best_cost:
            best, best_cost = s, cost
    return best


def _plane_hit(C: np.ndarray, d: np.ndarray, axis: int, value: float) -> float | None:
    """射线与平面 X[axis] = value 的交点参数 s。"""
    if abs(d[axis]) < 1e-9:
        return None
    s = (value - C[axis]) / d[axis]
    return s if s > 0.2 else None


def _closest_on_vertical(C: np.ndarray, d: np.ndarray, foot: np.ndarray) -> np.ndarray | None:
    """射线 C+s·d 上水平位置最接近 foot 的点（只约束 x,y，z 自由）。"""
    den = d[0] * d[0] + d[1] * d[1]
    if den < 1e-12:                         # 正下视：水平不可分辨
        return None
    s = (d[0] * (foot[0] - C[0]) + d[1] * (foot[1] - C[1])) / den
    if not np.isfinite(s) or s <= 0.3:
        return None
    return C + d * s


__all__ = ["Pose3DLifter", "LiftResult", "CourtCamera", "camera_from_homography",
           "REF_STATURE", "H_RATIO", "PARENT", "BONE_RATIO", "COCO17_NAMES", "VIS_CONF"]