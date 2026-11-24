"""研究原型：许可干净（自研/自训）的 COCO17 → 3D 轻量提升器，与几何基线做闭环对照。

**这不是生产代码**，只服务于 docs/research/pose_2d_to_3d.md 的数字复现。不改 bsk/。
权重写到 data/models/pose_lift_tcn.pt（已 ignore）。

设计要点（为什么这样设计）
--------------------------
1. 输入不是裸像素，而是**相机射线斜率 + 角尺度 + 可见性掩码**：
   K 由已有的 H 分解得到（`pose_3d.camera_from_homography`），所以
   (u−cx)/f, (v−cy)/f 是"相机坐标系里的方向"，网络天然对相机姿态不变；
   log(scale_px / f) 是"角尺度"，与人体尺寸先验相乘即得距离——这是单目尺度
   可观测的唯一来源，几何法用骨长表+脚点间接拿到，这里显式喂给网络。
2. 输出是**相机坐标系下、相对骨盆的米制 3D**（和 VideoPose3D 同构），
   `align_to_world()` 再用已标定相机的像素射线把它解到球场世界坐标：
   解 min ‖(I − dᵢdᵢᵀ)(X₀ + s·pᵢ − C)‖² 求 X₀∈R³、s∈R（线性最小二乘），
   并对 s 加"接近 1"的弱先验（训练数据是米制的）。
3. 时序严格因果：TCN 的膨胀卷积只做左侧 padding，第 t 帧只用 ≤t 的帧；
   窗口 K=16（≈0.5s@30fps），首帧退化=重复当前帧（与几何法首帧无历史等价）。
4. 训练数据为**自造合成数据**（本项目人体比例表 + 随机广播机位 + 连续关节角 +
   骨长个体差异 ±6% + 像素噪声 + 随机遮挡），不碰任何第三方权重/数据集，
   Apache-2.0 项目可自持。

用法
----
    .venv/Scripts/python.exe docs/research/pose_2d_to_3d_bench.py all
    .venv/Scripts/python.exe docs/research/pose_2d_to_3d_bench.py train --seqs 4000 --epochs 30
    .venv/Scripts/python.exe docs/research/pose_2d_to_3d_bench.py eval
    .venv/Scripts/python.exe docs/research/pose_2d_to_3d_bench.py real
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bsk.court import FIBA                               # noqa: E402
from bsk.perceive.pose_3d import (                       # noqa: E402
    ANKLE_H, COCO17_NAMES, FOREARM, HEAD, HIP_HALF, NOSE_EAR, NOSE_EYE,
    REF_STATURE, SHOULDER_HALF, SHIN, THIGH, TORSO, UPPER_ARM,
    Pose3DLifter, camera_from_homography,
)
from eval_pose_3d import SynthCamera, make_body, random_camera  # noqa: E402

MODEL_PATH = Path("data/models/pose_lift_tcn.pt")
K_WIN = 16                       # 因果窗口（帧）
N_IN = 17 * 2 + 17 + 1           # 射线斜率 34 + 掩码 17 + 角尺度 1
N_OUT = 17 * 3
EDGES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6))
UPPER = (5, 6, 7, 8, 9, 10)      # 上半身（前后深度最难的关节）
WRISTS = (9, 10)


# ======================================================================
#  自造人体：连续关节角（比 eval_pose_3d 的 4 个固定姿势覆盖更广）
# ======================================================================

def make_body_rand(rng, stature: float, pos, facing: float, jit: float) -> np.ndarray:
    """随机关节角 + 骨长个体差异的 17×3 世界坐标人体，双脚落地。

    比例与 pose_3d 同源（这一点必须诚实：闭环评估会"偏乐观"，真实域精度只能看
    域差实验与真实帧合理性检查）。
    """
    S = float(stature)
    k = 1.0 + rng.uniform(-jit, jit, size=17)
    thigh, shin = THIGH * S * k[13], SHIN * S * k[15]
    upper, fore = UPPER_ARM * S * k[7], FOREARM * S * k[9]
    torso, sh_half, hip_half = TORSO * S, SHOULDER_HALF * S, HIP_HALF * S
    head = HEAD * S
    d = math.radians

    # 关节角（度）：髋屈(+前摆)、膝屈、髋外展、肩屈、肘屈、肩外展、躯干前倾、躯干扭转
    th = rng.uniform(-45.0, 65.0, size=2)
    kf = rng.uniform(0.0, 95.0, size=2)
    h_ab = rng.uniform(-8.0, 20.0, size=2)
    sh = rng.uniform(-55.0, 175.0, size=2)
    ef = rng.uniform(0.0, 130.0, size=2)
    s_ab = rng.uniform(-12.0, 45.0, size=2)
    lean = rng.uniform(-15.0, 30.0)
    twist = rng.uniform(-25.0, 25.0)

    out = np.zeros((17, 3))
    out[11] = np.array([0.0, +hip_half, 0.0])
    out[12] = np.array([0.0, -hip_half, 0.0])
    for side, (jh, jk, ja) in enumerate(((11, 13, 15), (12, 14, 16))):
        a = d(th[side])
        b = d(h_ab[side])
        v1 = np.array([math.sin(a) * math.cos(b), math.sin(b), -math.cos(a) * math.cos(b)])
        knee = out[jh] + v1 / np.linalg.norm(v1) * thigh
        a2 = d(th[side] - kf[side])
        v2 = np.array([math.sin(a2) * math.cos(b), math.sin(b), -math.cos(a2) * math.cos(b)])
        out[jk] = knee
        out[ja] = knee + v2 / np.linalg.norm(v2) * shin

    up = np.array([math.sin(d(lean)), 0.0, math.cos(d(lean))])
    neck = up * torso
    ct, st = math.cos(d(twist)), math.sin(d(twist))
    for (jsh, sgn) in ((5, +1.0), (6, -1.0)):
        off = np.array([-st * sgn * sh_half, ct * sgn * sh_half, 0.0])
        out[jsh] = neck + off
    for side, (jsh, jel, jwr) in enumerate(((5, 7, 9), (6, 8, 10))):
        a = d(sh[side])
        b = d(s_ab[side]) * (1.0 if side == 0 else -1.0)
        v = np.array([-math.sin(a) * math.cos(b), math.sin(b), -math.cos(a) * math.cos(b)])
        elbow = out[jsh] + v / np.linalg.norm(v) * upper
        a2 = d(sh[side] - ef[side])
        v2 = np.array([-math.sin(a2) * math.cos(b), math.sin(b), -math.cos(a2) * math.cos(b)])
        out[jel] = elbow
        out[jwr] = elbow + v2 / np.linalg.norm(v2) * fore
    out[0] = neck + up * head
    out[1] = out[0] + np.array([0.35, +1.0, 0.9]) * (head * 0.22)
    out[2] = out[0] + np.array([0.35, -1.0, 0.9]) * (head * 0.22)
    out[3] = out[0] + np.array([0.0, +1.0, 0.2]) * (head * 0.42)
    out[4] = out[0] + np.array([0.0, -1.0, 0.2]) * (head * 0.42)

    out[:, 2] -= (out[[15, 16], 2].min() - ANKLE_H * S)
    c, s = math.cos(facing), math.sin(facing)
    rot = np.stack([c * out[:, 0] - s * out[:, 1], s * out[:, 0] + c * out[:, 1], out[:, 2]], -1)
    rot[:, 0] += pos[0]
    rot[:, 1] += pos[1]
    return rot


# ======================================================================
#  特征：像素 + 相机 → 网络输入（相机姿态不变）
# ======================================================================

def cam_frame(cam: SynthCamera) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """返回 (Rc, C, f, cx, cy)：相机→世界旋转、相机中心、焦距、主点。"""
    Rc = cam.Rc                     # 行向量 = 相机轴在世界里（与 project 一致）
    return Rc, cam.pos, cam.f, cam.K[0, 2], cam.K[1, 2]


def featurize(uv: np.ndarray, conf: np.ndarray, cam) -> np.ndarray:
    """(17,2) 像素 + 置信 → (52,) 网络输入。缺失关节射线置 0、掩码 0。"""
    Rc, C, f, cx, cy = cam_frame(cam)
    x = (uv[:, 0] - cx) / f
    y = (uv[:, 1] - cy) / f
    m = (conf >= 0.15).astype(np.float64)
    m[~np.isfinite(uv).all(1)] = 0.0
    pel = 0.5 * (uv[11] + uv[12])
    rad = np.linalg.norm(uv - pel, axis=1)
    scale_px = float(np.median(rad[m > 0])) if (m > 0).any() else 50.0
    scale_px = max(scale_px, 4.0)
    feat = np.concatenate([x * m, y * m, m, [math.log(scale_px / f)]])
    return feat


def target_cam_rel(body_world: np.ndarray, cam) -> np.ndarray:
    """世界人体 → 相机坐标系、相对骨盆的米制 3D（(51,)）。"""
    Rc, C, _f, _cx, _cy = cam_frame(cam)
    pc = (body_world - C) @ Rc.T
    root = 0.5 * (pc[11] + pc[12])
    return (pc - root).reshape(-1)


# ======================================================================
#  合成序列（训练/域内评估）
# ======================================================================

def gen_sequences(n_seq: int, seed: int, K: int = K_WIN, jit: float = 0.06,
                  noise: tuple = (0.5, 3.0), drop_p: float = 0.25,
                  size=(960, 540)):
    """返回 X(n,K,52)、Y(n,K,51) 以及元数据。相机在一段序列内固定（真实转播一秒内基本不变）。"""
    rng = np.random.default_rng(seed)
    X = np.empty((n_seq, K, N_IN), dtype=np.float32)
    Y = np.empty((n_seq, K, N_OUT), dtype=np.float32)
    n_ok = 0
    tries = 0
    while n_ok < n_seq and tries < n_seq * 6:
        tries += 1
        cam = random_camera(rng, size)
        stature = float(rng.uniform(1.68, 2.16))
        th0 = rng.uniform(-45, 60)
        dirn = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(0.0, 2.2) / 30.0            # 米/帧
        pos0 = np.array([rng.uniform(-12.0, 12.0), rng.uniform(-6.0, 6.0)])
        facing0 = rng.uniform(0, 2 * math.pi)
        d_face = rng.uniform(-0.01, 0.01)
        sigma = float(rng.uniform(*noise))
        body0 = make_body_rand(rng, stature, pos0, facing0, jit)
        seq_x, seq_y = [], []
        ok = True
        for t in range(K):
            # 站位随时间平移，"肢体摆动"用关键点沿朝向的小幅摆动近似（低成本动作模型）
            body = body0.copy()
            ph = 0.9 * math.sin(2 * math.pi * t / 14.0)
            fx, fy = math.cos(facing0), math.sin(facing0)
            body[9] += np.array([fx, fy, 0.0]) * 0.10 * ph
            body[10] -= np.array([fx, fy, 0.0]) * 0.10 * ph
            body[15] += np.array([fx, fy, 0.0]) * 0.05 * ph
            body[16] -= np.array([fx, fy, 0.0]) * 0.05 * ph
            body[:, 0] += math.cos(dirn) * speed * t
            body[:, 1] += math.sin(dirn) * speed * t
            uv, z = cam.project(body)
            if z.min() <= 3.0:
                ok = False
                break
            W, H = size
            if not (uv[:, 0].min() > -20 and uv[:, 0].max() < W + 20
                    and uv[:, 1].min() > -20 and uv[:, 1].max() < H + 20):
                ok = False
                break
            uv_n = uv + rng.normal(0.0, sigma, uv.shape)
            conf = np.full(17, 0.9)
            # 随机遮挡：整条肢体或单关节丢失（basketball 转播里常见）
            if rng.random() < drop_p:
                j = int(rng.integers(0, 17))
                group = {5: [5, 7, 9], 6: [6, 8, 10], 11: [11, 13, 15], 12: [12, 14, 16]}.get(j, [j])
                for g in group:
                    uv_n[g] = 0.5 * (uv_n[11] + uv_n[12])
                    conf[g] = 0.0
            seq_x.append(featurize(uv_n, conf, cam))
            seq_y.append(target_cam_rel(body, cam))
        if not ok:
            continue
        X[n_ok] = np.stack(seq_x)
        Y[n_ok] = np.stack(seq_y)
        n_ok += 1
    return X[:n_ok], Y[:n_ok]


# ======================================================================
#  模型：因果 TCN（ONNX 友好）
# ======================================================================

def build_model(width: int = 128, depth: int = 4):
    import torch.nn as nn

    class CausalBlock(nn.Module):
        def __init__(self, c, dil):
            super().__init__()
            self.pad = (3 - 1) * dil                      # k=3, 左 padding
            self.conv = nn.Conv1d(c, c, 3, dilation=dil)
            self.act = nn.GELU()
            self.norm = nn.BatchNorm1d(c)

        def forward(self, x):
            y = nn.functional.pad(x, (self.pad, 0))
            return self.act(self.norm(self.conv(y))) + x

    class Lifter(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(N_IN, width, 1)
            self.blocks = nn.Sequential(*[CausalBlock(width, 1 << (i % 3)) for i in range(depth)])
            self.out = nn.Conv1d(width, N_OUT, 1)

        def forward(self, x):                              # (B,K,52)
            h = self.inp(x.transpose(1, 2))
            return self.out(self.blocks(h)).transpose(1, 2)  # (B,K,51)

    return Lifter()


# ======================================================================
#  训练
# ======================================================================

def train(seqs: int = 20000, epochs: int = 300, width: int = 192, depth: int = 5,
          seed: int = 101, out: Path = MODEL_PATH, drop_p: float = 0.25) -> None:
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    print(f"[数据] 生成 {seqs} 段 × {K_WIN} 帧合成序列（自造，无第三方数据）…")
    t0 = time.time()
    X, Y = gen_sequences(seqs, seed, drop_p=drop_p)
    Xv, Yv = gen_sequences(max(seqs // 10, 64), seed + 777, drop_p=drop_p)
    print(f"  训练 {X.shape} / 验证 {Xv.shape}，耗时 {time.time()-t0:.1f}s")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(width, depth).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[模型] 参数 {n_par/1e3:.0f}K，设备 {dev}")

    xt = torch.from_numpy(X).to(dev)
    yt = torch.from_numpy(Y).to(dev).reshape(len(Y), K_WIN, 17, 3)
    xv = torch.from_numpy(Xv).to(dev)
    yv = torch.from_numpy(Yv).to(dev).reshape(len(Yv), K_WIN, 17, 3)
    # 骨长一致性（训练目标里加物理约束，抑制"腿被压短"的回归均值现象）
    bone_pairs = torch.tensor([(a, b) for a, b in EDGES], device=dev)
    bsz = 256
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = epochs * max(1, len(xt) // bsz)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    lossf = nn.MSELoss()
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(xt), device=dev)
        tot = 0.0
        for i in range(0, len(xt) - bsz + 1, bsz):
            idx = perm[i:i + bsz]
            pred = model(xt[idx]).reshape(-1, K_WIN, 17, 3)
            gt = yt[idx]
            loss = lossf(pred, gt)
            bl_p = (pred[..., bone_pairs[:, 0], :] - pred[..., bone_pairs[:, 1], :]).norm(dim=-1)
            bl_g = (gt[..., bone_pairs[:, 0], :] - gt[..., bone_pairs[:, 1], :]).norm(dim=-1)
            loss = loss + 0.5 * (bl_p - bl_g).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(idx)
        if ep % 25 == 0 or ep == epochs - 1:
            with torch.no_grad():
                vp = model(xv).reshape(-1, K_WIN, 17, 3)
                vloss = float(lossf(vp, yv))
                vmp = float((vp - yv).norm(dim=-1).mean())
                wrist = float((vp - yv).norm(dim=-1)[..., [9, 10]].mean())
                leg = float((vp[..., [11, 12], 2].mean(-1)
                             - vp[..., [15, 16], 2].mean(-1)).mean()
                            - (yv[..., [11, 12], 2].mean(-1)
                               - yv[..., [15, 16], 2].mean(-1)).mean())
            print(f"  ep{ep:3d} train {tot/len(xt):.5f}  val {vloss:.5f} "
                  f"rel MPJPE {vmp:.3f}m 腕 {wrist:.3f}m 腿长偏差 {leg:+.3f}m  "
                  f"{time.time()-t0:.0f}s")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "width": width, "depth": depth,
                "k_win": K_WIN, "note": "self-trained on synthetic data; Apache-2.0 project"}, out)
    print(f"[保存] {out}")


def load_model(path: Path = MODEL_PATH):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    m = build_model(d["width"], d["depth"])
    m.load_state_dict(d["state"])
    m.eval()
    return m, d


# ======================================================================
#  世界对齐：root-relative（相机系）→ 球场米制
# ======================================================================

def _perpendicular_solve(p_w, cam, uv, w, x0_init, gamma):
    """s 固定为 1，只解平移 X₀：Σ w‖Pᵢ(X₀+pᵢ−C)‖² + γ‖X₀−X₀ⁱⁿⁱᵗ‖²。

    为什么固定 s：透视在 20m 外的"尺度—深度"耦合几乎不可观测（自由解 s 会跑到
    20 倍以上），而网络输出本身已是米制（训练目标就是米制），s=1 是强先验。
    γ 的作用是不让噪声把根节点拽到"沿射线的另一个解"上——这是纯射线解在
    远距离 + 2px 噪声下的典型失败模式。
    """
    Rc, C, f, cx, cy = cam_frame(cam)
    d = np.stack([(uv[:, 0] - cx) / f, (uv[:, 1] - cy) / f, np.ones(17)], 1) @ Rc
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for i in range(17):
        P = np.eye(3) - np.outer(d[i], d[i])
        A += w[i] * P
        b += w[i] * (P @ (C - p_w[i]))
    A += gamma * np.eye(3)
    b += gamma * x0_init
    return np.linalg.solve(A + 1e-9 * np.eye(3), b)


def place_on_court(pred_rel_cam: np.ndarray, cam, uv: np.ndarray, conf: np.ndarray,
                   stature: float = REF_STATURE, gamma: float = 0.35) -> np.ndarray:
    """学习骨架（相机系、相对骨盆、米制）→ 球场世界坐标。

    1) 脚锚：双踝像素射线 ∩ 地面 z=0（几何法里最可靠的一步，误差厘米级）；
    2) 把骨架平移，使"姿态里的踝中点"落在脚锚上方 ANKLE_H 处；
    3) 用全部可见关节的射线做 γ 正则的最小二乘微调（不引入深度自由解）。
    """
    Rc, C, f, cx, cy = cam_frame(cam)
    p_w = pred_rel_cam @ Rc
    d = np.stack([(uv[:, 0] - cx) / f, (uv[:, 1] - cy) / f, np.ones(17)], 1) @ Rc
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    feet = []
    for j in (15, 16):
        if conf[j] >= 0.15 and d[j, 2] < -1e-3:
            s = -C[2] / d[j, 2]
            if 0.5 < s < 400.0:
                feet.append(C + d[j] * s)
    if feet:
        anchor = np.mean(feet, axis=0) + np.array([0.0, 0.0, ANKLE_H * stature])
    else:
        # 无踝：用所有可见关节射线的"最低点"近似地面锚
        pts = np.stack([C + d[i] * max(-C[2] / d[i, 2], 0.5) if d[i, 2] < -1e-3 else C
                        for i in range(17)])
        anchor = pts[np.argmin(pts[:, 2])] + np.array([0.0, 0.0, ANKLE_H * stature])
    p_ank_mid = 0.5 * (p_w[15] + p_w[16])
    x0 = anchor - p_ank_mid
    w = np.where(conf >= 0.15, 1.0, 0.0)      # 缺失关节的射线不可用，权重必须为 0
    x0 = _perpendicular_solve(p_w, cam, uv, w, x0, gamma)
    xyz = x0 + p_w
    # 触地修正：最低踝收敛到地面（脚不可能穿地）
    lo = float(min(xyz[15, 2], xyz[16, 2]))
    if lo < ANKLE_H * stature * 0.6:
        xyz[:, 2] += ANKLE_H * stature - lo
    return xyz


def align_to_world(pred_rel_cam: np.ndarray, cam: SynthCamera, uv: np.ndarray,
                   w: np.ndarray | None = None, s_prior: float = 1.0,
                   lam: float = 0.10) -> np.ndarray:
    """解 X₀、s：让 X₀ + s·pᵢ 尽量落在各自的像素射线上（线性最小二乘）。

    pred_rel_cam: (17,3) 相机系、相对骨盆的米制 3D
    uv: (17,2) 像素（用于构造射线）
    w: (17,) 每关节权重（缺失关节降低权重）
    """
    Rc, C, f, cx, cy = cam_frame(cam)
    p_w = pred_rel_cam @ Rc                      # 相机轴分量 → 世界轴
    d = np.stack([(uv[:, 0] - cx) / f, (uv[:, 1] - cy) / f, np.ones(17)], 1) @ Rc
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    if w is None:
        w = np.ones(17)
    A, b = [], []
    for i in range(17):
        P = np.eye(3) - np.outer(d[i], d[i])
        A.append(w[i] * np.hstack([P, (P @ p_w[i])[:, None]]))
        b.append(w[i] * (P @ C))
    A.append(np.array([[0.0, 0.0, 0.0, lam]]))
    b.append(np.array([lam * s_prior]))
    A = np.vstack(A)
    b = np.concatenate(b)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    X0, s = sol[:3], float(sol[3])
    return X0 + s * p_w                          # (17,3) 世界坐标


def lift_learned(model, uv: np.ndarray, conf: np.ndarray, cam, hist=None):
    """因果窗口推理：hist 是最近 K-1 帧特征（无则重复当前帧）。"""
    import torch
    feat = featurize(uv, conf, cam)
    if hist is None:
        win = np.repeat(feat[None], K_WIN, axis=0)
    else:
        win = np.vstack([hist[-(K_WIN - 1):], feat[None]]) if len(hist) else np.repeat(feat[None], K_WIN, 0)
        if len(win) < K_WIN:
            win = np.vstack([np.repeat(win[:1], K_WIN - len(win), 0), win])
    with torch.no_grad():
        out = model(torch.from_numpy(win[None].astype(np.float32)))[0, -1].numpy()
    rel = out.reshape(17, 3)
    xyz = place_on_court(rel, cam, uv, conf)
    return xyz, feat, rel


# ======================================================================
#  评估：同一批合成样本上，几何法 vs 学习法
# ======================================================================

def synth_samples(n_cams: int = 24, seed: int = 7, noise: float = 1.5,
                  size=(960, 540)):
    """与 eval_pose_3d.synth_suite 同分布（同 seed/同姿势/同相机采样），但只产样本。"""
    rng = np.random.default_rng(seed)
    poses = ("stand", "walk", "arms_up", "crouch")
    out = []
    n_ok = 0
    tries = 0
    while n_ok < n_cams and tries < n_cams * 10:
        tries += 1
        cam = random_camera(rng, size)
        got = 0
        for pose in poses:
            stature = float(rng.uniform(1.72, 2.10))
            pos = (rng.uniform(-11.0, 11.0), rng.uniform(-5.5, 5.5))
            facing = rng.uniform(0, 2 * math.pi)
            body = make_body(pose, stature, pos, facing)
            uv, z = cam.project(body)
            if z.min() <= 2.0:
                continue
            W, H = size
            if not (uv[:, 0].min() > 5 and uv[:, 0].max() < W - 5
                    and uv[:, 1].min() > 5 and uv[:, 1].max() < H - 5):
                continue
            k2 = uv + rng.normal(0.0, noise, uv.shape)
            out.append(dict(pose=pose, cam=cam, body=body, uv=k2, uv_clean=uv,
                            stature=stature))
            got += 1
        if got:
            n_ok += 1
    return out


def metrics(est: np.ndarray, body: np.ndarray, cam: SynthCamera) -> dict:
    d3 = np.linalg.norm(est - body, axis=1)
    Rc = cam.Rc
    fwd = Rc[2].copy()                    # 相机主轴（前向）
    fwd[2] = 0.0
    fwd /= max(np.linalg.norm(fwd), 1e-9)  # 水平前向：前后深度方向
    fa = np.abs((est - body) @ fwd)
    root = 0.5 * (body[11] + body[12])
    ankle = 0.5 * (body[15] + body[16])
    return dict(
        mpjpe=float(d3.mean()), p90=float(np.percentile(d3, 90)),
        root=float(np.linalg.norm(0.5 * (est[11] + est[12]) - root)),
        ankle=float(np.linalg.norm(0.5 * (est[15] + est[16]) - ankle)),
        wrist=float(d3[list(WRISTS)].mean()),
        upper_fa=float(fa[list(UPPER)].mean()),      # 上半身"前后深度"误差
        all_fa=float(fa.mean()),
        ground_err=float(abs(min(est[15, 2], est[16, 2]))),
    )


def eval_synth(model, n_cams: int = 24, seed: int = 7, occlude: bool = False) -> dict:
    samples = synth_samples(n_cams, seed)
    print(f"[闭环评估] {len(samples)} 个样本（{n_cams} 机位 × 4 姿势，σ=1.5px"
          f"{'，左臂遮挡' if occlude else ''}）")
    geo = Pose3DLifter(court=FIBA, image_size=(960, 540))
    rows = {"geo": [], "learned": [], "hybrid": [], "gated": []}
    for s in samples:
        uv = s["uv"].copy()
        conf = np.full(17, 0.9)
        if occlude:
            for j in (5, 7, 9):
                uv[j] = 0.5 * (uv[11] + uv[12])
                conf[j] = 0.0
        kpts = np.column_stack([uv, conf])
        box = (uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max())
        geo.reset()
        res = geo.lift_result(kpts, box, s["cam"].H, 1, 0)
        if res is not None:
            rows["geo"].append(metrics(res.kpts3d, s["body"], s["cam"]))
        est, _, _ = lift_learned(model, uv, conf, s["cam"])
        rows["learned"].append(metrics(est, s["body"], s["cam"]))
        # 混合：学习法给"姿态形状"，几何法给"根节点地面锚"——把学习骨架整体平移到
        # 几何法的骨盆位置（保留学习的关节几何，丢弃其整体平移的不确定性）
        if res is not None:
            sh = res.root - 0.5 * (est[11] + est[12])
            rows["hybrid"].append(metrics(est + sh, s["body"], s["cam"]))
            # 逐关节门控：2D 缺失的关节交回几何法（几何法的优点是逐关节独立，
            # 缺点正是没有跨关节先验；学习法反过来），避免遮挡污染的关节拖累整体
            gated = (est + sh).copy()
            miss = conf < 0.15
            if miss.any():
                gated[miss] = res.kpts3d[miss]
            rows["gated"].append(metrics(gated, s["body"], s["cam"]))
    return rows


def fmt(rows: list[dict], name: str) -> str:
    if not rows:
        return f"  {name:<8} —"
    a = lambda k: np.array([r[k] for r in rows])
    return (f"  {name:<8} n={len(rows):<3} MPJPE {a('mpjpe').mean():.3f}(p90 {np.percentile(a('p90'),50):.2f}) "
            f"根 {a('root').mean():.3f} 踝 {a('ankle').mean():.3f} 腕 {a('wrist').mean():.3f} "
            f"上半身前向 {a('upper_fa').mean():.3f} 全向 {a('all_fa').mean():.3f}")


def temporal_gain(model, n_seq: int = 40, K: int = 24, seed: int = 31) -> None:
    """因果窗口的收益：同一段匀速横移序列里，首帧（无历史）vs 第 8 帧后（稳态）。"""
    rng = np.random.default_rng(seed)
    geo = Pose3DLifter(court=FIBA, image_size=(960, 540))
    g_first, g_steady, l_first, l_steady = [], [], [], []
    for _ in range(n_seq):
        cam = random_camera(rng)
        st = float(rng.uniform(1.75, 2.05))
        hist = []
        for t in range(K):
            body = make_body("walk", st, (-4.0 + 0.12 * t, -1.0 + 0.04 * t),
                             facing=1.1 + 0.01 * t)
            uv, z = cam.project(body)
            if z.min() <= 2.0:
                hist = None
                break
            uv = uv + rng.normal(0.0, 1.5, uv.shape)
            conf = np.full(17, 0.9)
            kpts = np.column_stack([uv, conf])
            box = (uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max())
            res = geo.lift_result(kpts, box, cam.H, 1, t)
            if res is not None:
                e = float(np.linalg.norm(res.kpts3d - body, axis=1).mean())
                (g_first if t == 0 else g_steady if t >= 8 else []).append(e)                     if (t == 0 or t >= 8) else None
            est, feat, _ = lift_learned(model, uv, conf, cam, hist if t else None)
            e2 = float(np.linalg.norm(est - body, axis=1).mean())
            if t == 0:
                l_first.append(e2)
            elif t >= 8:
                l_steady.append(e2)
            hist = (hist or []) + [feat]
            hist = hist[-K_WIN:]
        if hist is not None:
            pass
    f = lambda a: np.mean(a) if a else float("nan")
    print(f"[时序收益] {n_seq} 段 × {K} 帧（σ=1.5px，稳态=第8帧后）")
    print(f"  几何   首帧 {f(g_first):.3f}m → 稳态 {f(g_steady):.3f}m")
    print(f"  学习   首帧 {f(l_first):.3f}m → 稳态 {f(l_steady):.3f}m（因果 TCN，窗口 {K_WIN} 帧）")


# ======================================================================
#  真实帧：合理性 / 能否对齐球场（无 GT，只能看自洽性）
# ======================================================================

def real_check(n_frames: int = 24) -> None:
    from eval_pose_filter import collect
    frame_ids, obs, w, h = collect(2000, n_frames, 960)
    H = None
    try:
        d = json.loads(Path("data/derived/harden-55/calibration.json").read_text("utf-8"))
        H = np.array(d["H"], dtype=np.float64)
        cam = camera_from_homography(H, (w, h))
        if cam is None:
            H = None
    except Exception:
        pass
    if H is None:
        print("[真实帧] 无可用 H（标定链路当前不可信），跳过米制检查，只做骨骼自洽性")
    elif camera_from_homography(H, (w, h)) is None:
        print("[真实帧] calibration.json 的 H 退化（单应分解失败）→ 无法给米制世界坐标")
        return
    model, _ = load_model()
    geo = Pose3DLifter(court=FIBA, image_size=(w, h))
    stats = {"geo": [], "learned": []}
    for fi, (fid, ob) in enumerate(zip(frame_ids, obs)):
        for tid, (box, kpts) in list(ob.items())[:4]:
            k = np.asarray(kpts, dtype=np.float64)
            conf = k[:, 2].copy()
            if H is not None:
                geo.reset()
                res = geo.lift_result(k, box, H, tid, frame=fi)
                if res is not None:
                    stats["geo"].append(_selfcheck(res.kpts3d, conf))
            est, _, _ = lift_learned(model, k[:, :2], conf, _RealCam(cam, H))
            stats["learned"].append(_selfcheck(est, conf))
    for name, rows in stats.items():
        if not rows:
            print(f"  {name}: 无样本")
            continue
        a = lambda key: np.array([r[key] for r in rows])
        print(f"  {name:<8} n={len(rows):<3} 骨长变异 {a('bone_cv').mean():.3f} "
              f"身高 {a('stature').mean():.2f}m 脚离地 {a('foot_z').mean():.3f}m "
              f"骨盆高 {a('pelvis_z').mean():.2f}m 越界 {a('outside').mean()*100:.0f}%")


class _RealCam:
    """把 CourtCamera 包成 SynthCamera 的接口（project/cam_frame 只需寥寥字段）。"""

    def __init__(self, cam, H):
        self._c = cam
        self.H = H
        self.f = cam.f
        self.pos = cam.C
        self.Rc = cam.R
        self.K = cam.K

    def project(self, X):
        return self._c.project(X)


def _selfcheck(xyz: np.ndarray, conf: np.ndarray) -> dict:
    bones = []
    for a, b in EDGES:
        bones.append(np.linalg.norm(xyz[a] - xyz[b]))
    bones = np.array(bones)
    st = float(np.clip(xyz[0, 2] / 0.93, 1.5, 2.3))
    pel = 0.5 * (xyz[11] + xyz[12])
    foot = float(min(xyz[15, 2], xyz[16, 2]))
    outside = float(abs(pel[0]) > FIBA.half_length + 3 or abs(pel[1]) > FIBA.half_width + 3)
    return dict(bone_cv=float(bones.std() / max(bones.mean(), 1e-6)), stature=st,
                foot_z=foot, pelvis_z=float(pel[2]), outside=outside)


# ======================================================================
#  延迟 / 显存 / ONNX
# ======================================================================

def bench_speed(model) -> None:
    import torch
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.normal(size=(1, K_WIN, N_IN)).astype(np.float32))
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[开销] 参数 {n_par/1e3:.0f}K，窗口 {K_WIN} 帧")

    with torch.no_grad():
        for _ in range(20):
            model(x)
        t0 = time.perf_counter()
        for _ in range(200):
            model(x)
        cpu_ms = (time.perf_counter() - t0) / 200 * 1000
    print(f"  torch CPU  单次窗口 {cpu_ms:.2f} ms（≈{cpu_ms/K_WIN:.2f} ms/帧）")

    if torch.cuda.is_available():
        m = model.cuda()
        xg = x.cuda()
        with torch.no_grad():
            for _ in range(20):
                m(xg)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(200):
                m(xg)
            torch.cuda.synchronize()
            gpu_ms = (time.perf_counter() - t0) / 200 * 1000
            peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"  torch GPU  单次窗口 {gpu_ms:.2f} ms（≈{gpu_ms/K_WIN:.2f} ms/帧），"
              f"峰值显存 {peak:.1f} MB")
        model.cpu()

    out = Path("data/models/pose_lift_tcn.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(model, x, str(out), input_names=["x"], output_names=["y"],
                      opset_version=17, dynamic_axes={"x": {0: "B"}, "y": {0: "B"}})
    print(f"  ONNX 导出 {out}（{out.stat().st_size/1e3:.0f} KB）")
    try:
        import onnxruntime as ort
        for prov in (["CPUExecutionProvider"],
                     (["CUDAExecutionProvider", "CPUExecutionProvider"]
                      if "CUDAExecutionProvider" in ort.get_available_providers() else None)):
            if prov is None:
                continue
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            sess = ort.InferenceSession(str(out), so, providers=prov)
            xn = x.numpy()
            for _ in range(10):
                sess.run(None, {"x": xn})
            t0 = time.perf_counter()
            for _ in range(200):
                sess.run(None, {"x": xn})
            ms = (time.perf_counter() - t0) / 200 * 1000
            print(f"  ORT {prov[0]:<22} {ms:.2f} ms/窗口（≈{ms/K_WIN:.2f} ms/帧）")
    except Exception as exc:
        print(f"  ONNX Runtime 不可用：{exc}")


# ======================================================================
#  main
# ======================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "real", "speed", "temp", "all"])
    ap.add_argument("--seqs", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--cams", type=int, default=24)
    ap.add_argument("--ckpt", default=str(MODEL_PATH))
    ap.add_argument("--drop", type=float, default=0.25, help="训练时整肢/单点丢失概率")
    args = ap.parse_args()

    if args.cmd in ("train", "all"):
        train(args.seqs, args.epochs, out=Path(args.ckpt), drop_p=args.drop)
    if args.cmd in ("eval", "all"):
        model, _ = load_model(Path(args.ckpt))
        t0 = time.time()
        rows = eval_synth(model, args.cams)
        print("  结果（世界坐标米制，越小人越好；hybrid=学习形状+几何根节点）")
        for k in ("geo", "learned", "hybrid", "gated"):
            print(fmt(rows[k], k))
        rows_occ = eval_synth(model, args.cams, occlude=True)
        print("  遮挡（左臂 3 点全丢）")
        for k in ("geo", "learned", "hybrid", "gated"):
            print(fmt(rows_occ[k], k))
        print(f"  （评估耗时 {time.time()-t0:.0f}s）")
    if args.cmd in ("temp", "all"):
        model, _ = load_model(Path(args.ckpt))
        temporal_gain(model)
    if args.cmd in ("speed", "all"):
        model, _ = load_model(Path(args.ckpt))
        bench_speed(model)
    if args.cmd == "real":
        real_check()


if __name__ == "__main__":
    main()