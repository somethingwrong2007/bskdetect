"""评估 2D→3D 骨架提升（bsk.perceive.pose_3d）：合成相机闭环 + 真实帧连通性检查。

为什么用合成闭环：单目 3D 没有真实标注，唯一可复现的"真值"是"已知 3D 人体 + 已知
相机 → 投影成 2D → 再提升回 3D"。这里用 bsk.court_3d 的机位风格生成随机广播相机，
用与 lifter 同源的人体比例造 17 关节人体，投影（可选加像素噪声）后喂给 lifter，
比较 3D 关节误差（米）与重投影误差（像素）。

指标口径
- MPJPE：逐关节 3D 欧氏距离均值（米），分别给"全部关节"与"可见关节(conf≥0.15)"；
- 根误差：骨盆 3D 误差、双踝误差；
- 身高：估计身高 vs 真值（相对误差）；
- 重投影：用**估计相机**把 3D 骨架投回 2D，与原像素的中位误差（应≈像素噪声）；
- 相机：估计焦距/相机中心与真值的偏差（衡量单应分解质量）。

用法：
    .venv/Scripts/python.exe eval_pose_3d.py                 # 合成 24 机位 × 4 姿势 + 开销
    .venv/Scripts/python.exe eval_pose_3d.py --cams 40
    .venv/Scripts/python.exe eval_pose_3d.py --real          # 真实帧（H 可用时）
    .venv/Scripts/python.exe eval_pose_3d.py --real --fit-h  # 真实帧 + 机位近似拟合
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time

import numpy as np

from bsk.court import FIBA, project_court_polygon
from bsk.perceive.pose_3d import (ANKLE_H, BONE_RATIO, COCO17_NAMES, FOREARM, HEAD,
                                  HIP_HALF, NOSE_EAR, NOSE_EYE, PARENT,
                                  Pose3DLifter, REF_STATURE, SHOULDER_HALF,
                                  SHIN, THIGH, TORSO, UPPER_ARM,
                                  camera_from_homography)

EDGES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6))


# ======================================================================
#  合成相机（与 court_3d 同一约定：pos/look_at/up/focal）
# ======================================================================

class SynthCamera:
    """pos/look_at/up/focal(image_size) → K/R/t/H（court→pixel）。"""

    def __init__(self, pos, look_at, focal: float, size=(960, 540)):
        self.pos = np.asarray(pos, dtype=np.float64)
        fwd = np.asarray(look_at, dtype=np.float64) - self.pos
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        self.Rc = np.stack([right, -up, fwd])          # 图像 y 向下
        self.f = float(focal)
        W, H = size
        self.K = np.array([[self.f, 0.0, W * 0.5], [0.0, self.f, H * 0.5], [0.0, 0.0, 1.0]])
        self.t = -self.Rc @ self.pos
        self.P = self.K @ np.column_stack([self.Rc, self.t])
        self.H = self.K @ np.column_stack([self.Rc[:, 0], self.Rc[:, 1], self.t])  # 地面 z=0
        self.size = (W, H)
        self.dist = float(np.linalg.norm(self.pos - np.array([0.0, 0.0, 0.0])))
        self.height = float(self.pos[2])

    def project(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        xc = X @ self.Rc.T + self.t
        z = xc[:, 2]
        return np.column_stack([self.K[0, 0] * xc[:, 0] / z + self.K[0, 2],
                                self.K[1, 1] * xc[:, 1] / z + self.K[1, 2]]), z

    def ray_to_ground(self, uv) -> np.ndarray:
        """像素射线与地面 z=0 的交点（世界坐标）。"""
        d = np.linalg.inv(self.K) @ np.array([uv[0], uv[1], 1.0])
        d = self.Rc.T @ d
        s = -self.pos[2] / d[2]
        return self.pos + d * s


def random_camera(rng, size=(960, 540)) -> SynthCamera:
    """广播机位采样：距离 10~32m、高度 6~24m、焦距 500~1500px（960 宽）。"""
    dist = rng.uniform(10.0, 32.0)
    az = rng.uniform(0.0, 2 * math.pi)
    hgt = rng.uniform(6.0, 24.0)
    pos = (dist * math.cos(az), dist * math.sin(az), hgt)
    look = (rng.uniform(-4.0, 4.0), rng.uniform(-2.5, 2.5), 0.0)
    focal = float(rng.uniform(500.0, 1500.0))
    return SynthCamera(pos, look, focal, size)


# ======================================================================
#  合成人体（17 关节，比例与 lifter 同源；可加骨长个体差异）
# ======================================================================

def _rot_z(v, ang):
    c, s = math.cos(ang), math.sin(ang)
    return np.stack([c * v[..., 0] - s * v[..., 1],
                     s * v[..., 0] + c * v[..., 1],
                     v[..., 2]], axis=-1)


def make_body(pose: str, stature: float = REF_STATURE, pos=(0.0, 0.0),
              facing: float = 0.0, bone_jitter: float = 0.0,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """造一个 17×3 的世界坐标人体，脚踩地（z 最低的踝 = ANKLE_H）。

    姿势用关节角参数化（站立 / 举臂 / 走 / 蹲），身体先建在"骨盆在原点、面朝 +x"
    的局部系，再绕 z 转向并平移到球场位置。骨长取自与 lifter 同一张比例表（可用
    bone_jitter 给个体差异，检验比例表失配的影响）。
    """
    S = float(stature)
    k = np.ones(17)
    if bone_jitter > 0 and rng is not None:
        k = 1.0 + rng.uniform(-bone_jitter, bone_jitter, size=17)
    thigh, shin = THIGH * S * k[13], SHIN * S * k[15]
    upper, fore = UPPER_ARM * S * k[7], FOREARM * S * k[9]
    torso, sh_half, hip_half = TORSO * S, SHOULDER_HALF * S, HIP_HALF * S
    head = HEAD * S

    # 姿势参数：髋屈/膝屈/踝（度），肩屈/肘屈，躯干前倾
    th_l = th_r = kf_l = kf_r = 0.0
    sh_l = sh_r = ef_l = ef_r = 0.0
    lean = 0.0
    if pose == "arms_up":
        sh_l = sh_r = 160.0
        ef_l = ef_r = 10.0
    elif pose == "walk":
        th_l, th_r = 28.0, -18.0
        kf_l, kf_r = 12.0, 38.0
        sh_l, sh_r = -25.0, 30.0
        ef_l = ef_r = 25.0
        lean = 4.0
    elif pose == "crouch":
        th_l = th_r = 55.0
        kf_l = kf_r = 70.0
        lean = 22.0
        sh_l = sh_r = 35.0
        ef_l = ef_r = 55.0
    elif pose == "dunk":
        sh_l = sh_r = 165.0
        ef_l = ef_r = 5.0
        th_l, th_r = 12.0, -8.0
        kf_l, kf_r = 8.0, 22.0
        lean = -6.0
    d = math.radians

    out = np.zeros((17, 3))
    pelvis = np.zeros(3)                       # 骨盆在局部原点
    # 髋（沿局部 y：左 +y）
    out[12] = pelvis + np.array([0.0, -hip_half, 0.0])   # right_hip
    out[11] = pelvis + np.array([0.0, +hip_half, 0.0])   # left_hip
    # 腿：thigh 从髋向下，方向 (sinθ, 0, -cosθ)；shin 再叠加膝屈
    for side, (j_hip, j_knee, j_ank, th, kf) in enumerate((
            (11, 13, 15, th_l, kf_l), (12, 14, 16, th_r, kf_r))):
        hip = out[j_hip]
        a1 = d(th)
        knee = hip + np.array([math.sin(a1), 0.0, -math.cos(a1)]) * thigh
        a2 = d(th - kf)                        # 膝屈：小腿相对大腿向后
        ank = knee + np.array([math.sin(a2), 0.0, -math.cos(a2)]) * shin
        out[j_knee] = knee
        out[j_ank] = ank
    # 颈/躯干：前倾 lean
    a = d(lean)
    up_dir = np.array([math.sin(a), 0.0, math.cos(a)])
    neck = pelvis + up_dir * torso
    out[5] = neck + np.array([0.0, +sh_half, 0.0])       # left_shoulder
    out[6] = neck + np.array([0.0, -sh_half, 0.0])       # right_shoulder
    # 手臂：肩屈（绕局部 y 轴，向前为正）+ 肘屈
    for (j_sh, j_el, j_wr, sh, ef) in ((5, 7, 9, sh_l, ef_l), (6, 8, 10, sh_r, ef_r)):
        shp = out[j_sh]
        a1 = d(90.0 - sh) if sh >= 0 else d(90.0 - sh)
        # 自然下垂：sh=0 → 向下；sh=90 → 向前；sh=160 → 向上
        v = np.array([-math.sin(d(sh)), 0.0, -math.cos(d(sh))])
        elbow = shp + v * upper
        v2 = np.array([-math.sin(d(sh - ef)), 0.0, -math.cos(d(sh - ef))])
        out[j_el] = elbow
        out[j_wr] = elbow + v2 * fore
    # 头：鼻在颈上方 head，眼/耳在鼻附近
    nose = neck + up_dir * head
    out[0] = nose
    out[1] = nose + np.array([0.35, +1.0, 0.9]) * (head * 0.22)
    out[2] = nose + np.array([0.35, -1.0, 0.9]) * (head * 0.22)
    out[3] = nose + np.array([0.0, +1.0, 0.2]) * (head * 0.42)
    out[4] = nose + np.array([0.0, -1.0, 0.2]) * (head * 0.42)

    # 落地：最低踝压到 ANKLE_H（脚踩地）
    out[:, 2] -= (out[[15, 16], 2].min() - ANKLE_H * S)
    # 转到世界：绕 z 转 facing，再平移到 pos
    out = _rot_z(out, facing)
    out[:, 0] += pos[0]
    out[:, 1] += pos[1]
    return out


# ======================================================================
#  指标
# ======================================================================

def triage(results: list[dict], key: str, bins: tuple) -> list[tuple[str, list[dict]]]:
    groups: list[tuple[str, list[dict]]] = []
    lo = -1e9
    for hi in bins:
        groups.append((f"[{lo:.0f},{hi:.0f})", [r for r in results if lo <= r[key] < hi]))
        lo = hi
    groups.append((f"[{lo:.0f},inf)", [r for r in results if r[key] >= lo]))
    return [(n, g) for n, g in groups if g]


def summarize(rows: list[dict]) -> str:
    if not rows:
        return "—"
    mp = np.array([r["mpjpe"] for r in rows])
    rt = np.array([r["root_err"] for r in rows])
    ak = np.array([r["ankle_err"] for r in rows])
    hs = np.array([r["h_rel"] for r in rows])
    rp = np.array([r["reproj"] for r in rows])
    rpg = np.array([r["reproj_gt"] for r in rows])
    return (f"MPJPE {mp.mean():.3f}m(p90 {np.percentile(mp,90):.2f}) "
            f"根 {rt.mean():.3f}m 踝 {ak.mean():.3f}m "
            f"身高 {100*hs.mean():+.1f}% 重投影 {rp.mean():.1f}px(真相机 {rpg.mean():.1f}px)")


def synth_suite(n_cams: int = 24, noise: float = 1.5, seed: int = 7,
                size=(960, 540), image_size=None, verbose: bool = True) -> list[dict]:
    """主实验：随机相机 × 姿势 × 球位，闭环比较 3D/重投影误差。"""
    rng = np.random.default_rng(seed)
    lifter = Pose3DLifter(court=FIBA, image_size=image_size or size)
    poses = ("stand", "walk", "arms_up", "crouch")
    trials = []
    n_good_cams = 0
    attempts = 0
    while n_good_cams < n_cams and attempts < n_cams * 10:
        attempts += 1
        cam = random_camera(rng, size)
        got = 0
        for pi, pose in enumerate(poses):
            stature = float(rng.uniform(1.72, 2.10))
            pos = (rng.uniform(-11.0, 11.0), rng.uniform(-5.5, 5.5))
            facing = rng.uniform(0, 2 * math.pi)
            body = make_body(pose, stature, pos, facing)
            uv, z = cam.project(body)
            if z.min() <= 2.0:
                continue
            W, H = size
            if not (np.all(uv[:, 0] > 5) and np.all(uv[:, 0] < W - 5)
                    and np.all(uv[:, 1] > 5) and np.all(uv[:, 1] < H - 5)):
                continue                        # 人出画：该机位不采
            k2 = uv + rng.normal(0.0, noise, uv.shape) if noise > 0 else uv.copy()
            conf = np.full(17, 0.9)
            kpts = np.column_stack([k2, conf])
            bbox = (k2[:, 0].min(), k2[:, 1].min(), k2[:, 0].max(), k2[:, 1].max())
            lifter.reset()                      # 单帧独立评估（时序平滑单独测）
            res = lifter.lift_result(kpts, bbox, cam.H, track_id=1, frame=0)
            if res is None:
                trials.append(dict(ok=False, cam=n_good_cams, pose=pose, dist=cam.dist,
                                   hgt=cam.height, f=cam.f, mpjpe=np.nan, root_err=np.nan,
                                   ankle_err=np.nan, h_rel=np.nan, reproj=np.nan,
                                   reproj_gt=np.nan, f_err=np.nan, cam_err=np.nan))
                continue
            got += 1
            d3 = np.linalg.norm(res.kpts3d - body, axis=1)
            root = 0.5 * (body[11] + body[12])
            ankle = 0.5 * (body[15] + body[16])
            proj_est, _ = lifter.camera(cam.H).project(res.kpts3d)
            proj_gt, _ = cam.project(res.kpts3d)
            trials.append(dict(
                ok=True, cam=n_good_cams, pose=pose, dist=cam.dist, hgt=cam.height, f=cam.f,
                mpjpe=float(d3.mean()), root_err=float(np.linalg.norm(res.root - root)),
                ankle_err=float(np.linalg.norm(res.ground - ankle)),
                h_rel=float(res.stature / stature - 1.0),
                reproj=float(np.median(np.linalg.norm(proj_est - uv, axis=1))),
                reproj_gt=float(np.median(np.linalg.norm(proj_gt - uv, axis=1))),
                f_err=float(lifter.camera(cam.H).f / cam.f - 1.0),
                cam_err=float(np.linalg.norm(lifter.camera(cam.H).C - cam.pos)),
                joints=d3, body=body, est=res.kpts3d, uv=uv, res=res,
            ))
        if got:
            n_good_cams += 1
    if verbose:
        print(f"\n[合成闭环] 采到 {len(trials)} 个有效样本（{n_cams} 机位 × {len(poses)} 姿势，"
              f"像素噪声 σ={noise}px）")
        ok = [t for t in trials if t["ok"]]
        print(f"  成功率 {100*len(ok)/max(len(trials),1):.0f}%")
        print(f"  总体：{summarize(ok)}")
        fe = np.array([abs(t["f_err"]) for t in ok])
        ce = np.array([t["cam_err"] for t in ok])
        print(f"  相机分解：|Δf| 中位 {100*np.median(fe):.1f}%、p90 {100*np.percentile(fe,90):.1f}%；"
              f"相机中心误差 中位 {np.median(ce):.2f}m")
        for key, bins, title in (("dist", (15.0, 22.0), "相机距离(m)"),
                                 ("hgt", (8.0, 14.0), "相机高度(m)"),
                                 ("f", (800.0, 1100.0), "焦距(px@960)")):
            print(f"  按{title}：")
            for name, rows in triage(ok, key, bins):
                print(f"    {name:<12} n={len(rows):<3} {summarize(rows)}")
        print("  按姿势：")
        for pose in poses:
            rows = [t for t in ok if t["pose"] == pose]
            if rows:
                print(f"    {pose:<10} n={len(rows):<3} {summarize(rows)}")
        # 逐关节（全部姿势）
        J = np.stack([t["joints"] for t in ok])
        print("  逐关节 3D 误差（米，均值/p90）：")
        line = []
        for j, nm in enumerate(COCO17_NAMES):
            line.append(f"{nm}:{J[:,j].mean():.2f}/{np.percentile(J[:,j],90):.2f}")
        for i in range(0, 17, 4):
            print("    " + "  ".join(line[i:i + 4]))
    return trials


def bone_jitter_test(n: int = 12, sigma: float = 0.07, seed: int = 11) -> None:
    """比例表失配敏感性：给每个人 ±σ 的骨长个体差异，看 3D 误差涨多少。"""
    rng = np.random.default_rng(seed)
    for label, jit in (("标准比例", 0.0), (f"个体差异 ±{int(sigma*100)}%", sigma)):
        lifter = Pose3DLifter(image_size=(960, 540))
        errs, hs = [], []
        for i in range(n):
            cam = random_camera(rng)
            body = make_body(("stand", "walk")[i % 2], 1.85, (-6 + i, 0.5), facing=0.7,
                             bone_jitter=jit, rng=rng)
            uv, _ = cam.project(body)
            k2 = uv + rng.normal(0, 1.0, uv.shape)
            lifter.reset()
            res = lifter.lift_result(np.column_stack([k2, np.full(17, 0.9)]), None, cam.H, 1, 0)
            if res is None:
                continue
            errs.append(float(np.linalg.norm(res.kpts3d - body, axis=1).mean()))
            hs.append(res.stature / 1.85 - 1.0)
        print(f"  {label:<18} MPJPE {np.mean(errs):.3f}m  身高偏差 {100*np.mean(hs):+.1f}%")


def temporal_test(n_frames: int = 24, noise: float = 1.5, seed: int = 21) -> None:
    """时序平滑收益：一段匀速横移的球员序列（首帧无历史 vs 稳态）。

    单帧评估里"深度两解"只能靠高度参考择优；连续帧里上一帧深度就是最强的先验，
    这正是实时链路的真实工作方式。这里量化"第 1 帧"与"第 8 帧之后"的误差差。
    """
    rng = np.random.default_rng(seed)
    cam = random_camera(rng)
    lifter = Pose3DLifter(image_size=(960, 540))
    errs = []
    for t in range(n_frames):
        pos = (-4.0 + 0.12 * t, -1.0 + 0.04 * t)
        body = make_body("walk", 1.88, pos, facing=1.1 + 0.01 * t)
        uv, _ = cam.project(body)
        k2 = uv + rng.normal(0, noise, uv.shape)
        box = (k2[:, 0].min(), k2[:, 1].min(), k2[:, 0].max(), k2[:, 1].max())
        res = lifter.lift_result(np.column_stack([k2, np.full(17, 0.9)]), box, cam.H, 3, t)
        if res is not None:
            errs.append(float(np.linalg.norm(res.kpts3d - body, axis=1).mean()))
    if not errs:
        print("  时序样本为空")
        return
    e = np.array(errs)
    print(f"  首帧 {e[0]:.3f}m → 第 4~7 帧 {e[4:8].mean():.3f}m → 稳态(8 帧后) "
          f"{e[8:].mean():.3f}m（{n_frames} 帧序列，横移 3 米）")


def timing_test(n: int = 400) -> None:
    """开销：单帧单人 lift 的平均耗时（含相机缓存命中路径）。"""
    rng = np.random.default_rng(3)
    cam = random_camera(rng)
    lifter = Pose3DLifter(image_size=(960, 540))
    body = make_body("walk", 1.9, (2.0, 1.0), 0.4)
    uv, _ = cam.project(body)
    kpts = np.column_stack([uv + rng.normal(0, 1.0, uv.shape), np.full(17, 0.9)])
    box = (uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max())
    for i in range(20):
        lifter.lift(kpts, box, cam.H, 1, i)
    t0 = time.perf_counter()
    for i in range(n):
        lifter.lift(kpts, box, cam.H, 1, 20 + i)
    dt = (time.perf_counter() - t0) / n * 1000
    # 相机重算（H 每帧变化的最坏情况）
    t0 = time.perf_counter()
    for i in range(50):
        lifter.lift(kpts, box, cam.H * (1.0 + 1e-9 * i), 1, 1000 + i)
    dt2 = (time.perf_counter() - t0) / 50 * 1000
    print(f"\n[开销] 缓存命中 {dt:.3f} ms/人/帧；H 每帧变化 {dt2:.3f} ms/人/帧（目标 ≤1.5ms）")


# ======================================================================
#  真实帧（harden-55）：脚点贴地 / 身高 / 重投影 + 叠加图
# ======================================================================

def _draw_skeleton(img, uv, color, radius=2, thick=2):
    import cv2
    for a, b in EDGES:
        cv2.line(img, (int(uv[a, 0]), int(uv[a, 1])), (int(uv[b, 0]), int(uv[b, 1])),
                 color, thick, cv2.LINE_AA)
    for p in uv:
        cv2.circle(img, (int(p[0]), int(p[1])), radius, color, -1, cv2.LINE_AA)


def real_frame_check(fit_h: bool = False, frm: int = 2000, frames: int = 60,
                     width: int = 960) -> None:
    import cv2
    from bsk.perceive.pose_filter import PoseFilter, PoseFilterConfig

    try:
        from eval_pose_filter import collect, to_poses
    except Exception as exc:                        # pragma: no cover
        print(f"[真实帧] 无法导入 eval_pose_filter：{exc}")
        return

    frame_ids, obs, w, h = collect(frm, frames, width)
    H = None
    src = ""
    try:
        from bsk import ingest
        from bsk.store import Library
        lib = Library()
        v = lib.get("harden-55")
        table = lib.table("harden-55")
        anchor = table.key_before(frm)
        rgb0 = None
        for idx, _p, rgb in ingest.frames(v.path, start=anchor, size=(w, h), stride=1,
                                          seek_us=table.pts_us[anchor - 1]):
            if idx >= frm:
                rgb0 = np.asarray(rgb).copy()
                break
    except Exception as exc:                        # pragma: no cover
        print(f"[真实帧] 取帧失败：{exc}")
        rgb0 = None

    # 1) 项目自带线分割标定（另一代理在改这块，可能暂时不可用）
    if rgb0 is not None:
        try:
            from bsk.line_calib import LineCalibrator
            cal = LineCalibrator().calibrate(rgb0)
            if cal is not None:
                H = np.array(cal.H, dtype=np.float64)
                src = f"line_calib(reproj {cal.reproj_error_px:.1f}px)"
        except Exception as exc:
            print(f"[真实帧] line_calib 不可用：{exc}")
    # 2) 机位近似拟合（只用于链路连通性检查，米制精度不可信）
    if H is None and fit_h and rgb0 is not None:
        H, score = fit_broadcast_camera(rgb0)
        src = f"chamfer 机位近似(score {score:.1f}px)"
    # 3) 仓库里的标定文件（若不是退化解）
    if H is None:
        try:
            import json
            from pathlib import Path
            p = Path("data/derived/harden-55/calibration.json")
            d = json.loads(p.read_text(encoding="utf-8"))
            Hc = np.array(d["H"], dtype=np.float64)
            poly = project_court_polygon(Hc, FIBA)
            span = float(np.ptp(poly[:, 0]) + np.ptp(poly[:, 1]))
            if span > 0.1 * w:
                H = Hc
                src = f"calibration.json(reproj {d.get('reproj_error_px', -1):.0f}px)"
            else:
                print("[真实帧] calibration.json 是退化解（球场投影塌缩），跳过")
        except Exception:
            pass
    if H is None:
        print("[真实帧] 无可用单应 H：3D 米制结果无法验证，跳过真实帧检查"
              "（--fit-h 可跑机位近似，仅用于链路连通性）")
        return

    cam = camera_from_homography(H, (w, h))
    if cam is None:
        print("[真实帧] H 无法分解出相机，跳过")
        return
    print(f"[真实帧] H 来源：{src}；相机高度 {cam.height:.2f}m、焦距 {cam.f:.0f}px"
          f"{'' if cam.f_estimated else '（兜底值）'}")

    lifter = Pose3DLifter(image_size=(w, h))
    pf = PoseFilter(PoseFilterConfig())
    heights, foot_z, reproj, n_ok, n_all = [], [], [], 0, 0
    last = None
    for i, (frame, fobs) in enumerate(zip(frame_ids, obs)):
        fobs = {k: (v[0], v[1].copy()) for k, v in fobs.items()}
        poses, boxes = to_poses(frame, fobs, legacy=False)
        out, _occ = pf.apply(frame, poses, boxes)
        res = lifter.lift_batch(out, boxes, H, frame)
        for tid, r in res.items():
            if not r.on_court or r.stature < 1.2:
                continue
            n_all += 1
            vis = r.visible
            if vis.sum() < 8:
                continue
            n_ok += 1
            heights.append(r.stature)
            foot_z.append(float(r.kpts3d[15:17, 2].mean()))
            reproj.append(r.reproj_err_px)
            if last is None and vis[5] and vis[11]:
                last = (tid, r, fobs.get(tid, (None, None))[1])
    if not heights:
        print("[真实帧] 3D 提升没有有效样本")
        return
    hs = np.array(heights)
    print(f"  有效提升 {n_ok}/{n_all} 人次；身高 中位 {np.median(hs):.2f}m "
          f"(p10 {np.percentile(hs,10):.2f} / p90 {np.percentile(hs,90):.2f})，"
          f"落在 1.7~2.2m 的比例 {100*np.mean((hs >= 1.7) & (hs <= 2.2)):.0f}%")
    fz = np.abs(np.array(foot_z))
    print(f"  踝点 |z| 中位 {np.median(fz)*100:.1f}cm（触地修正后应≈0，腾空脚会抬高）")
    print(f"  重投影误差中位 {np.median(reproj):.2f}px（≈0 说明 3D 点严格落在像素射线上）")
    print(f"  根/地面：示例 track {last[0] if last else '-'}；"
          f"骨盆 z {last[1].root[2]:.2f}m，身高 {last[1].stature:.2f}m" if last else "")

    # 叠加图：原图上画 2D 关键点（绿）+ 3D 骨架按估计相机回投影（红）+ 球场外框（青）
    if rgb0 is not None and last is not None:
        img = cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR)
        poly = project_court_polygon(H, FIBA)
        cv2.polylines(img, [poly.astype(np.int32)], True, (255, 255, 0), 2)
        tid, r, k2 = last
        uv3, _d = cam.project(r.kpts3d)
        if k2 is not None:
            _draw_skeleton(img, np.asarray(k2)[:, :2], (0, 255, 0), 1, 1)
        _draw_skeleton(img, uv3, (0, 0, 255), 3, 2)
        cv2.putText(img, f"3D lift: h={r.stature:.2f}m ankle_z={r.kpts3d[15:17,2].mean():+.2f}m "
                         f"reproj={r.reproj_err_px:.2f}px", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, f"3D lift: h={r.stature:.2f}m ankle_z={r.kpts3d[15:17,2].mean():+.2f}m "
                         f"reproj={r.reproj_err_px:.2f}px", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(".tmp_3d_check.jpg", img)
        print("  叠加图 → .tmp_3d_check.jpg（绿=2D 关键点，红=3D 骨架回投影，青=球场外框）")


def fit_broadcast_camera(rgb: np.ndarray):
    """机位近似拟合（chamfer）：白线 + 模板线 + 随机搜索/Powell。

    **不是标定替代品**：只在项目标定链路不可用时用于真实帧连通性检查，误差可达
    几十像素，米制结果只能看量级。正式 3D 用 bsk.line_calib 的 H。
    """
    import cv2
    from bsk.line_calib import court_line_samples
    from scipy.optimize import minimize

    Himg, Wimg = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 2] > 170) & (hsv[:, :, 1] < 75)).astype(np.uint8)
    wood = ((hsv[:, :, 0] > 8) & (hsv[:, :, 0] < 32)
            & (hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 85)).astype(np.uint8)
    wood = cv2.morphologyEx(wood, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    near = cv2.dilate(wood, np.ones((25, 25), np.uint8))
    lines = white & (near > 0)
    lines[:int(0.21 * Himg), :] = 0
    lines[int(0.86 * Himg):, :] = 0
    dt_img = cv2.distanceTransform(1 - lines, cv2.DIST_L2, 5)
    ys, xs = np.nonzero(lines)
    if len(xs) < 500:
        return None, float("inf")
    sel = np.random.default_rng(1).choice(len(xs), size=min(4000, len(xs)), replace=False)
    LP = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)
    samples = court_line_samples(FIBA, 0.6)
    pts = np.concatenate([v for _k, v in samples.items()], 0)
    ph = np.column_stack([pts, np.ones(len(pts))])

    def make(p):
        px, py, pz, lx, ly, lf = p
        f = math.exp(lf)
        pos = np.array([px, py, pz])
        look = np.array([lx, ly, 0.0])
        fwd = look - pos
        fwd /= np.linalg.norm(fwd) + 1e-12
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        rn = np.linalg.norm(right)
        if rn < 1e-8:
            return None
        right /= rn
        up = np.cross(right, fwd)
        Rc = np.stack([right, -up, fwd])
        K = np.array([[f, 0, Wimg / 2], [0, f, Himg / 2], [0, 0, 1.0]])
        t = -Rc @ pos
        return K @ np.column_stack([Rc[:, 0], Rc[:, 1], t])

    def cost(p):
        Hc = make(p)
        if Hc is None:
            return 1e6
        q = (Hc @ ph.T).T
        z = q[:, 2]
        uv = q[:, :2] / np.where(np.abs(z) > 1e-9, z, 1e-9)[:, None]
        ok = (z > 1e-6) & (uv[:, 0] >= -20) & (uv[:, 0] < Wimg + 20) \
            & (uv[:, 1] >= -20) & (uv[:, 1] < Himg + 20)
        if ok.mean() < 0.55:
            return 1e6
        m = np.zeros((Himg, Wimg), np.uint8)
        for x, y in uv[ok]:
            cv2.circle(m, (int(x), int(y)), 2, 255, -1)
        if m.sum() < 100:
            return 1e6
        ys2, xs2 = np.nonzero(m)
        d1 = np.minimum(dt_img[ys2, xs2], 20.0)
        dt_t = cv2.distanceTransform(1 - (m > 0).astype(np.uint8), cv2.DIST_L2, 3)
        d2 = np.minimum(dt_t[LP[:, 1].astype(int), LP[:, 0].astype(int)], 20.0)
        return float(d1.mean() + 0.7 * d2.mean())

    rng = np.random.default_rng(2)
    best = None
    for _i in range(1500):
        p = [rng.uniform(-4, 8), rng.uniform(-30, -12), rng.uniform(6, 20),
             rng.uniform(-2, 6), rng.uniform(-1, 3), math.log(rng.uniform(600, 1500))]
        c = cost(p)
        if best is None or c < best[0]:
            best = (c, p)
    for _t in range(3):
        res = minimize(cost, np.array(best[1]), method="Powell",
                       options={"maxiter": 3000, "xtol": 1e-3, "ftol": 1e-4})
        if res.fun < best[0]:
            best = (float(res.fun), list(res.x))
    return make(best[1]), best[0]


# ======================================================================

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", type=int, default=24)
    ap.add_argument("--noise", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--real", action="store_true", help="跑 harden-55 真实帧检查")
    ap.add_argument("--fit-h", action="store_true", help="标定不可用时用机位近似拟合 H")
    ap.add_argument("--frm", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()

    trials = synth_suite(args.cams, args.noise, args.seed)
    ok = [t for t in trials if t["ok"]]
    # 相机分解自检：H 应能被精确重算出来
    rng = np.random.default_rng(0)
    cam = random_camera(rng)
    cc = camera_from_homography(cam.H, cam.size)
    print(f"\n[分解自检] 相机中心误差 {np.linalg.norm(cc.C - cam.pos):.4f}m，"
          f"焦距 {cc.f:.1f}（真 {cam.f:.1f}，{100*(cc.f/cam.f-1):+.2f}%）")
    print("\n[比例表敏感度]")
    bone_jitter_test()
    print("\n[时序平滑收益]")
    temporal_test()
    timing_test()
    if args.real:
        print()
        real_frame_check(fit_h=args.fit_h, frm=args.frm, frames=args.frames)


if __name__ == "__main__":
    main()