"""3D 骨架的人体运动约束层（anatomical ROM + kinematic/temporal priors）。

定位
----
`pose_3d.Pose3DLifter` 用"单应分解相机 + 像素射线 + 骨长"解出 17×3，几何上自洽
（骨长严格、点在自己射线上），但**单目深度歧义**会让解出来的姿态出现人体根本做不出
的构型：膝反关节、肘过伸、腿往侧边拐、头翻到肩下面、整副骨架沿视线跳 0.5m。
本层不重新解算，只在提升结果之后做**可行性修正**，把这些非人体构型拉回合法域，
并加上时序（速度/触地）约束。全部因果：只用当前帧与该 track 的历史状态。

为什么单独一层
--------------
骨长/射线是"测量约束"，关节活动度/速度/触地是"人体约束"，两者的求解代价和失效
模式完全不同。分开写可以：① 单独评估（`eval_pose_3d.py --prior`）；② 单独回滚
（`BSK_POSE3D_PRIOR=0`）；③ 学习法分支（`pose_lift_net`）替换上半身形状后复用同一套
人体校验（`pose_only`）。

四类约束与它们的修正算子
------------------------
1. **关节活动度（ROM）**：膝/肘只允许在前屈方向弯（过伸 ≤ ext_limit），侧向弯折
   （内/外翻）受 `lateral` 限制；髋有屈/伸/展限；肩限"上臂甩到躯干后方"的角度；
   躯干与头颈限倾角。算子只有两种，都**保持骨长**：
   - `bend`：绕 `v1×v2` 转轴，把两骨夹角改到限内（改变屈伸角度）；
   - `twist`：绕母骨 `v1` 自转，把屈伸平面转回矢状面（改变侧向分量、不动屈伸角度）。
   修正递归作用到整条子链（改肘带着腕、改髋带着膝踝），否则肢体会被折断。
2. **骨长恒定**：上游 `_solve` 里有几处"直接改 z"的兜底（踝贴地、骨盆限高），会破坏
   骨长；本层结尾自上而下（骨盆→髋→膝→踝、颈→肩→肘→腕）把每根骨重新拉到名义长度。
3. **根节点运动学**：骨盆水平速度 ≤ `root_speed`、单帧位移 ≤ `root_speed*dt`、
   竖直速度 ≤ `vert_speed`。超限时**在射线域**求解修正（逐关节沿自己射线找最接近
   "目标平移"的点）——沿视线方向正是单目唯一自由、也唯一错得离谱的方向，所以修正
   不增加重投影误差；垂直于射线的残差留给下一帧的时序 EMA。
4. **足地交互**：踝/膝不许穿透地面（整体竖直抬高）；上一帧触地且当前仍较低的那只脚
   视为支撑脚，用两段腿 IK 把踝相对骨盆的滑移拉回上一帧位置。IK 严格保持大腿/小腿
   骨长，单帧修正限幅并做因果 EMA；只修较低脚，避免双脚约束互相打架。

代价与边界（不要过度承诺）
--------------------------
- ROM 修正是刚体旋转，**必然**让关节离开自己的像素射线 → 重投影变大。所以有
  `reproj_budget_px` 总预算：超限就把整批修正按比例衰减，宁可留一点不合法也不
  伪造测量。20 机位 × 4 姿势、σ=1.5px 的静态合成闭环中，重投影中位约 1.8→1.9px，
  ROM 超限样本率 67.6%→59.2%，但最大超限角 78.1°→7.9°（见 eval_pose_3d.py）。
- 32 帧因果扰动序列中，MPJPE 0.777→0.567m、踝误差 0.398→0.270m、踝穿透归零，
  ROM 超限率 31.2%→12.5%，支撑相滑移中位 61.9→52.1mm。jerk 中位仍有约 9% 回退，
  p90 约 14% 回退：单目 2D 噪声下的硬约束不是免费平滑，不能把它宣称为零副作用。
- 关节活动度表是"保守上限"（AAOS/ADA 常模再放宽 10~15°），目的只是拦住非人体构型，
  不是肌肉骨骼仿真；把表调严会直接伤 MPJPE。
- 单目沿视线不可观测这件事，本层治不了：它只能把跳变压到可达域内，压不掉歧义本身。
- 4GB 档/标定不可信时上游就不调本层（见 `server_rt._lift_players` 的闸门）。
- 以上全部是合成闭环/扰动序列指标，不代表真实转播视频精度。

开销
----
纯 numpy 小向量运算；独立耗时复测缓存命中约 1.08ms/人/帧、H 每帧变化约 1.24ms/人/帧
（关闭约束约 0.25/0.37ms），满足"≤1.5ms/人/帧"目标；A/B 批量评估路径约 1.4ms。

接口
----
    prior = MotionPrior()                       # MotionPriorConfig 可整体调参
    xyz, st_stats = prior.apply(xyz, dirs, C, dt, state, stature, conf, px_per_m)
    xyz2, st2 = prior.pose_only(xyz, stature)   # 无时序状态：学习分支复用

`state` 是调用方（lifter）按 track_id 维护的 dict，本层在里面读写 `"prior"` 键，
生命周期与 track 状态一致（断档/新 track 自动冷启动）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-9
D2R = math.pi / 180.0
R2D = 180.0 / math.pi

# 与 pose_3d 同一套索引；这里把 17 关节的父子链写成显式表，避免依赖 numpy 视图。
NOSE, NECK, PELVIS = 0, 17, 18
CHILDREN: dict[int, tuple[int, ...]] = {
    NECK: (NOSE, 5, 6),
    NOSE: (1, 2, 3, 4),
    5: (7,), 7: (9,),
    6: (8,), 8: (10,),
    PELVIS: (11, 12),
    11: (13,), 13: (15,),
    12: (14,), 14: (16,),
}


def _make_subchains() -> dict[int, tuple[int, ...]]:
    chains = {}
    for root in range(17):
        nodes, stack = [], [root]
        while stack:
            j = stack.pop()
            nodes.append(j)
            stack.extend(CHILDREN.get(j, ()))
        chains[root] = tuple(nodes)
    return chains


# 子链不变，预计算即可；避免每帧每个铰链重复建栈/列表。
SUBCHAINS = _make_subchains()
# 自上而下的骨长恢复顺序（父, 子, 名义骨长的身高比例）
# 眼/耳不在此列：它们随鼻刚体移动，真实头型与标准比例的偏差大于 2D 误差。
BONE_ORDER = ((PELVIS, 11, 0.055), (PELVIS, 12, 0.055),
              (11, 13, 0.245), (12, 14, 0.245),
              (13, 15, 0.246), (14, 16, 0.246),
              (PELVIS, NECK, 0.288),
              (NECK, 5, 0.105), (NECK, 6, 0.105),
              (5, 7, 0.186), (6, 8, 0.186),
              (7, 9, 0.146), (8, 10, 0.146),
              (NECK, NOSE, 0.112))
# 髋两段只报告、不修正：它们在上游由各自像素射线与腰高水平面交点确定，已被观测
# 钉住。按名义骨盆宽挪髋会推翻 2D 测量，并把误差整条传到踝。其余骨段仍严格恢复。
RESTORE_SKIP = frozenset({(PELVIS, 11), (PELVIS, 12)})
# 头（鼻/眼/耳）相对颈的刚体关系：旋转头部时整簇一起动
HEAD_CLUSTER = (NOSE, 1, 2, 3, 4)


@dataclass(slots=True)
class MotionPriorConfig:
    """单位：角度＝度，长度＝米，速度＝米/秒。数值来自 AAOS/ADA 常模 + 10~15° 放宽。"""

    # 膝：屈 0~145（深蹲）、过伸 ≤8、侧向（内/外翻）≤16
    knee_flex: float = 145.0
    knee_ext: float = 8.0
    knee_lateral: float = 16.0
    # 肘：屈 0~150、过伸 ≤8；侧向放宽（肩上举位前臂本来就不在矢状面）
    elbow_flex: float = 150.0
    elbow_ext: float = 8.0
    elbow_lateral: float = 42.0
    # 髋：屈 ≤145、伸 ≤30、展 ≤70（大跨步/防守站姿）
    hip_flex: float = 145.0
    hip_ext: float = 30.0
    hip_abduct: float = 70.0
    # 肩：上臂允许甩到躯干后方 ≤65°（投篮引球）
    shoulder_back: float = 65.0
    # 躯干（骨盆→颈）离竖直最多 80°（扑救/贴身），头颈离躯干轴最多 75°
    torso_tilt: float = 80.0
    neck_tilt: float = 75.0
    # 屈伸平面只有在这个角度以上才有意义（接近伸直时法向是噪声）
    plane_min_flex: float = 12.0
    # 时序
    root_speed: float = 10.0          # 峰值冲刺 ≈9.3 m/s，留余量
    root_accel: float = 45.0          # 单帧位移突变上限（m/s²）
    vert_speed: float = 6.0           # 起跳初速 ≈4.5 m/s + 余量
    root_damp: float = 0.65           # 修正只吃残差的一部分，避免与测量硬拼
    vel_alpha: float = 0.5            # 骨盆速度 EMA
    output_tau: float = 0.04          # 约束增量因果 EMA，压住修正引入的 jerk
    max_root_fix: float = 1.20        # 单次整体平移上限（米）
    # 足地
    contact_z: float = 0.16           # 踝离地阈值（ANKLE_H≈0.072，留鞋/噪声）
    plant_speed: float = 1.1          # 相对骨盆水平速度低于此＝支撑相
    plant_slip: float = 0.10          # 允许的打滑速度（m/s）
    max_plant_fix: float = 0.06       # 单帧支撑腿 IK 修正上限（米）
    plant_tau: float = 0.06           # 支撑腿 IK 修正量的因果 EMA
    # 预算
    pose_gain: float = 0.65          # 姿态/骨长修正增益：留 35% 残差，避免追噪声引起 jerk
    reproj_budget_px: float = 6.0     # 允许中位重投影比修正前高出这么多像素
    floor_margin: float = 0.0


@dataclass
class MotionPrior:
    """无状态调用 + 每 track 一份时序状态（写在调用方的 state["prior"] 里）。"""

    cfg: MotionPriorConfig = field(default_factory=MotionPriorConfig)
    rom_hits: int = 0
    root_hits: int = 0
    plant_hits: int = 0
    floor_hits: int = 0
    calls: int = 0

    def reset(self) -> None:
        self.rom_hits = self.root_hits = self.plant_hits = self.floor_hits = 0
        self.calls = 0

    # ── 对外：完整约束（含时序）────────────────────────────
    def apply(self, xyz: np.ndarray, dirs: np.ndarray | None, C: np.ndarray | None,
              dt: float, state: dict, stature: float, conf: np.ndarray | None = None,
              px_per_m: float | None = None) -> tuple[np.ndarray, dict]:
        """xyz (17,3) → 修正后的 (17,3) + 统计。dirs/C 给了才做射线域修正。

        `px_per_m` 用于把"离开像素射线的距离"折算成像素，超 `reproj_budget_px` 就
        按比例衰减姿态类修正（宁可不修正，也不能把测量改得面目全非）。
        """
        self.calls += 1
        S = float(stature) if np.isfinite(stature) and stature > 0.5 else 1.85
        cfg = self.cfg
        st = state.get("prior")
        if st is None or not np.isfinite(dt) or dt <= 0.0 or dt > 0.35:
            st = {"frame": None, "pelvis": None, "vel": None, "feet": {}}
            state["prior"] = st

        out = np.array(xyz, dtype=np.float64, copy=True)
        stats = {"rom": 0, "rom_deg": 0.0, "root_fix": 0.0, "plant_fix": 0.0,
                 "floor_fix": 0.0, "bone_fix": 0.0, "reproj_d": 0.0, "damped": 1.0}

        # 1) 先按射线域做"位置类"修正（不增加重投影），再做姿态类修正
        if dirs is not None and C is not None and dt > 0.0 and st["pelvis"] is not None:
            out, s = self._kinematic(out, dirs, C, dt, st, S)
            stats.update(s)

        # 2) 关节活动度（骨长保持的刚体旋转）+ 3) 骨长恢复 + 4) 地面，整体可重跑一次
        # ROM/骨长交替跑两遍：第一轮改姿态，第二轮在恢复后的骨长上重新判超限。
        # pose_gain<1 是故意的：单目 2D 噪声会把合法关节推过阈值，每帧全额纠回去
        # 反而制造更大的帧间 jerk；保留部分残差比"漂亮但抖动"更符合实时链路。
        k = max(0.0, min(1.0, cfg.pose_gain))
        base = out.copy()
        for attempt in range(2):
            cur = out if attempt == 0 else base.copy()
            rom_n, rom_deg, bone_fix = 0, 0.0, 0.0
            for _ in range(2):
                cur, s = self._rom(cur, S, conf, k)
                rom_n += s["rom"]
                rom_deg = max(rom_deg, s["rom_deg"])
                cur, s = self._restore_bones(cur, S, k)
                bone_fix += s["bone_fix"]
            out = cur
            stats["rom"], stats["rom_deg"] = rom_n, rom_deg
            stats["bone_fix"] = bone_fix
            out, s = self._floor(out, S)
            stats["floor_fix"] = s["floor_fix"]
            stats["bone_fix"] += s.get("bone_fix", 0.0)
            if attempt:
                stats["damped"] = round(k / max(cfg.pose_gain, EPS), 3)
            if dirs is None or C is None or px_per_m is None or attempt:
                break
            drift = self._reproj_drift(base, out, dirs, C) * float(px_per_m)
            if drift > cfg.reproj_budget_px:
                k = max(0.0, min(k, cfg.reproj_budget_px / max(drift, EPS)))

        if dirs is not None and C is not None:
            stats["reproj_d"] = round(self._reproj_drift(base, out, dirs, C), 4)

        # 5) 约束后的短因果 EMA。平滑"相对修正量"而不是绝对坐标：绝对侧向残差
        # 包含相机几何与真实运动，跨帧平均会把合法姿态拉歪；只平均 ROM/骨长/
        # 地面/脚锚引入的增量，才能压 jerk 而不伤当前帧测量。
        smooth_m = 0.0
        prev_dd = st.get("depth_delta")
        prev_do = st.get("off_delta")
        if (dirs is not None and C is not None and cfg.output_tau > 0.0
                and prev_dd is not None and prev_dd.shape == (17,)
                and np.isfinite(prev_dd).all() and dt > 0.0):
            base_depth = np.einsum("ij,ij->i", base - C, dirs)
            base_off = base - C - dirs * base_depth[:, None]
            cur_depth = np.einsum("ij,ij->i", out - C, dirs)
            cur_off = out - C - dirs * cur_depth[:, None]
            valid = (cur_depth > 0.2) & (base_depth > 0.2)
            if np.any(valid):
                a = float(np.clip(-np.expm1(-dt / cfg.output_tau), 0.15, 1.0))
                dd = cur_depth - base_depth
                do = cur_off - base_off
                sm_dd, sm_do = dd.copy(), do.copy()
                sm_dd[valid] = a * dd[valid] + (1.0 - a) * prev_dd[valid]
                if prev_do is not None and prev_do.shape == (17, 3) and np.isfinite(prev_do).all():
                    sm_do[valid] = a * do[valid] + (1.0 - a) * prev_do[valid]
                out = C + dirs * (base_depth + sm_dd)[:, None] + base_off + sm_do
                smooth_m = float(np.median(np.linalg.norm(
                    (sm_dd[valid] - dd[valid])[:, None] * dirs[valid]
                    + sm_do[valid] - do[valid], axis=1)))
        stats["smooth_m"] = round(smooth_m, 4)

        # 平滑不能把严重反关节"平均掉"。仅对 >20° 的硬异常做一次全额刚性修正，
        # 普通阈值附近的残差仍保留阻尼，避免追噪声。
        hard = max(rom_violations(out, S=S).values(), default=0.0)
        if hard > 20.0:
            out, s = self._rom(out, S, conf, 1.0)
            out, b = self._restore_bones(out, S, 1.0)
            stats["rom"] += s["rom"]
            stats["rom_deg"] = max(stats["rom_deg"], s["rom_deg"], hard)
            stats["bone_fix"] = round(stats["bone_fix"] + b["bone_fix"], 4)

        # 6) 先守地面；7) 最后做支撑腿 IK。IK 保持腿长，之后只允许整体竖直平移。
        # 约束增量 EMA 的状态只保存姿态/地面修正，不能把脚锚刚体平移也存进去，
        # 否则下一帧会再把上一帧的脚锚平移叠加一次，形成累积漂移。
        state_out = out.copy()
        if dirs is not None and C is not None and dt > 0.0:
            out, s = self._floor(out, S)
            stats["floor_fix"] = round(stats["floor_fix"] + s["floor_fix"], 4)
            stats["bone_fix"] = round(stats["bone_fix"] + s.get("bone_fix", 0.0), 4)
            state_out = out.copy()
            out, plant_fix = self._plant_anchor(out, S, dt, st)
            stats["plant_fix"] = round(plant_fix, 4)
            # IK 可能让膝/踝轻微穿地；只做整体竖直平移，不再横向恢复骨长。
            z = float(out[[15, 16, 13, 14], 2].min())
            if z < cfg.floor_margin:
                extra = cfg.floor_margin - z
                out[:, 2] += extra
                stats["floor_fix"] = round(stats["floor_fix"] + extra, 4)
            stats["reproj_d"] = round(self._reproj_drift(base, out, dirs, C), 4)
        if dirs is not None and C is not None:
            base_depth = np.einsum("ij,ij->i", base - C, dirs)
            final_depth = np.einsum("ij,ij->i", state_out - C, dirs)
            st["depth_delta"] = final_depth - base_depth
            st["off_delta"] = (state_out - C - dirs * final_depth[:, None]
                               - (base - C - dirs * base_depth[:, None]))
        else:
            st["depth_delta"] = st["off_delta"] = None
        self._remember(out, dt, st)
        return out, stats

    # ── 对外：只做人体可行性校验（无时序状态）───────────────
    def pose_only(self, xyz: np.ndarray, stature: float,
                  conf: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        S = float(stature) if np.isfinite(stature) and stature > 0.5 else 1.85
        out = np.array(xyz, dtype=np.float64, copy=True)
        gain = max(0.0, min(1.0, self.cfg.pose_gain))
        rom_n, rom_deg, bone_fix = 0, 0.0, 0.0
        for _ in range(2):
            out, s = self._rom(out, S, conf, gain)
            rom_n += s["rom"]
            rom_deg = max(rom_deg, s["rom_deg"])
            out, s = self._restore_bones(out, S, gain)
            bone_fix += s["bone_fix"]
        out, st3 = self._floor(out, S)
        return out, {"rom": rom_n, "rom_deg": rom_deg,
                     "bone_fix": bone_fix + st3.get("bone_fix", 0.0),
                     "floor_fix": st3["floor_fix"]}

    # ── 关节活动度 ──────────────────────────────────────────
    def _rom(self, x: np.ndarray, S: float, conf, k: float = 1.0) -> tuple[np.ndarray, dict]:
        cfg = self.cfg
        n_hits, max_deg = 0, 0.0

        def rig(pivot: np.ndarray, axis: np.ndarray, ang: float, nodes) -> None:
            """绕 pivot、沿 axis 旋转 ang*k 弧度，作用于 nodes（含递归子链）。"""
            ang = ang * k
            if abs(ang) < 1e-4:
                return
            for j in nodes:
                x[j] = _rot(x[j] - pivot, axis, ang) + pivot

        def hinge(a: int, b: int, c: int, ideal_axis: np.ndarray, flex_max: float,
                  ext_max: float, lateral_max: float, normal_sign: int,
                  nodes_c) -> tuple[int, float]:
            """三段链 a→b→c 的铰链约束：带符号屈伸限位 + 侧向平面拉回。

            屈伸角取 u1=母骨、u2=子骨 的夹角 θ（0＝完全伸直），**符号**由旋转平面法向
            k=u1×u2 与人体左右轴 m 的点积决定：反关节（膝往前弯、肘往后弯）与正关节弯
            的 θ 完全一样，只有 k 的朝向不同——所以必须用带符号角才能拦住"膝反关节"，
            单看角度大小永远发现不了。修正绕 σ·k̂ 转（σ＝符号因子），既能把过弯压回
            限内，也能把反关节往"伸直"方向拉（拉平比把整条小腿绕母骨翻 180° 的
            重投影代价小得多，且视觉上不违和）。
            """
            hits, deg = 0, 0.0
            if _low_conf(conf, (b,) + tuple(nodes_c)):
                return 0, 0.0                     # 全链不可见：修正在画面上无意义
            hs = hinge_state(x, a, b, c, ideal_axis, normal_sign)
            if hs is None:
                return 0, 0.0
            signed, axis, _, _ = hs
            target = min(flex_max, max(-ext_max, signed))
            if abs(target - signed) > 0.5:
                rig(x[b], axis, (target - signed) * D2R, nodes_c)
                hits += 1
                deg = abs(target - signed)

            # 侧向分量：绕母骨自转，把屈伸平面转回矢状面。没有屈伸修正时 x 未变，
            # 直接复用状态，省掉一次法向/反三角运算。
            hs2 = hinge_state(x, a, b, c, ideal_axis, normal_sign) if hits else hs
            if hs2 is None or hs2[3] <= cfg.plane_min_flex:
                return hits, deg
            _, _, lat, theta = hs2
            excess = lat - lateral_max
            if excess <= max(2.0, 0.25 * lateral_max):
                return hits, deg
            v1, v2 = x[b] - x[a], x[c] - x[b]
            n1, n2 = _norm(v1), _norm(v2)
            if n1 < EPS or n2 < EPS:
                return hits, deg
            u1, u2 = v1 / n1, v2 / n2
            kk = _unit(_cross(u1, u2))
            m = _unit(ideal_axis)
            u = m - (_dot(m, u1)) * u1                 # 理想轴在⊥u1 平面上的投影
            if _norm(u) < EPS:
                return hits, deg
            u = _unit(u)
            if float(_dot(kk, u)) < 0.0:               # 法向双指：取更近的一侧
                u = -u
            # 越接近伸直，平面越不可辨识 → 少动
            w = min(1.0, theta / max(cfg.plane_min_flex * 2.0, EPS))
            rig(x[b], u1, _signed_about(u1, kk, u) * (excess / max(lat, EPS)) * w, nodes_c)
            return hits + 1, max(deg, excess)

        def cone(a: int, b: int, down: np.ndarray, back_limit: float,
                 nodes_b, ml: np.ndarray) -> tuple[int, float]:
            """a→b 骨在矢状面内相对"下垂方向"的后摆限位（肩/髋共用）。"""
            if _low_conf(conf, (b,) + nodes_b):
                return 0, 0.0
            v = x[b] - x[a]
            n = _norm(v)
            if n < EPS:
                return 0, 0.0
            u = v / n
            d = _unit(down)
            if _norm(d) < EPS:
                return 0, 0.0
            fwd = _unit(_cross(d, ml))   # 矢状面内的"前"
            if _norm(fwd) < EPS:
                return 0, 0.0
            a_f, a_d = float(_dot(u, fwd)), float(_dot(u, d))
            if a_d < 0.15:
                return 0.0, 0.0                   # 已抬到接近水平：后摆概念不适用
            ang = math.degrees(math.atan2(a_f, a_d))       # 0＝自然下垂，负＝甩到身后
            over = -back_limit - ang
            if over > 0.5:
                axis = _cross(d, u)
                if _norm(axis) < EPS:
                    return 0, 0.0
                rig(x[a], _unit(axis), over * D2R, nodes_b)
                return 1, over
            return 0, 0.0

        # 身体坐标系（每次从当前姿态重算，供各约束共享）
        pel = 0.5 * (x[11] + x[12])
        nk = 0.5 * (x[5] + x[6])
        up = _unit(nk - pel)
        if _norm(up) < EPS:
            up = np.array([0.0, 0.0, 1.0])
        ml = _unit(x[5] - x[6])                    # 肩线＝左右轴
        if _norm(ml) < EPS:
            ml = np.array([0.0, 1.0, 0.0])
        down = -up
        hip_ml = _unit(x[11] - x[12])
        if _norm(hip_ml) < EPS:
            hip_ml = ml

        # 躯干倾角（骨盆→颈）
        tv = nk - pel
        nt = _norm(tv)
        if nt > EPS:
            tilt = math.degrees(math.acos(_clip_unit(_dot(tv / nt, WORLD_UP))))
            if tilt > cfg.torso_tilt:
                axis = _cross(tv / nt, WORLD_UP)
                if _norm(axis) > EPS:
                    rig(pel, _unit(axis), (tilt - cfg.torso_tilt) * D2R,
                        (5, 6, NOSE, 1, 2, 3, 4, 7, 8, 9, 10))
                    n_hits += 1
                    max_deg = max(max_deg, tilt - cfg.torso_tilt)

        # 头颈：鼻相对颈的倾角（整簇刚体一起转）
        hv = x[NOSE] - nk
        nh = _norm(hv)
        if nh > EPS:
            tilt = math.degrees(math.acos(_clip_unit(_dot(hv / nh, up))))
            if tilt > cfg.neck_tilt:
                axis = _cross(hv / nh, up)
                if _norm(axis) > EPS:
                    rig(nk, _unit(axis), (tilt - cfg.neck_tilt) * D2R, HEAD_CLUSTER)
                    n_hits += 1
                    max_deg = max(max_deg, tilt - cfg.neck_tilt)

        # 髋（骨盆→髋→膝）：屈/伸 + 侧展
        for (j_hip, j_knee) in ((11, 13), (12, 14)):
            v = x[j_knee] - x[j_hip]
            n = _norm(v)
            if n < EPS:
                continue
            u = v / n
            ab = math.degrees(math.acos(_clip_unit(abs(_dot(u, ml)))))
            # ab＝与肩线的夹角；90° 表示纯竖直/前后，越小越往侧面张开
            abd = 90.0 - ab
            if abd > cfg.hip_abduct:
                axis = _cross(u, ml)
                if _norm(axis) > EPS:
                    rig(x[j_hip], _unit(axis), (abd - cfg.hip_abduct) * D2R,
                        SUBCHAINS[j_knee])
                    n_hits += 1
                    max_deg = max(max_deg, abd - cfg.hip_abduct)
            h, d = cone(j_hip, j_knee, down, cfg.hip_ext, SUBCHAINS[j_knee], ml)
            n_hits += h
            max_deg = max(max_deg, d)

        # 膝（髋→膝→踝）：正常屈曲方向＝小腿往后折 → k·m>0 → normal_sign=+1
        for (j_hip, j_knee, j_ank) in ((11, 13, 15), (12, 14, 16)):
            h, d = hinge(j_hip, j_knee, j_ank, hip_ml, cfg.knee_flex, cfg.knee_ext,
                         cfg.knee_lateral, +1, SUBCHAINS[j_ank])
            n_hits += h
            max_deg = max(max_deg, d)

        # 肩（颈→肩→肘）：上臂不许甩到躯干后方太多
        for (j_sh, j_el) in ((5, 7), (6, 8)):
            h, d = cone(j_sh, j_el, down, cfg.shoulder_back, SUBCHAINS[j_el], ml)
            n_hits += h
            max_deg = max(max_deg, d)

        # 肘（肩→肘→腕）：正常屈曲＝前臂往身前抬 → k·m<0 → normal_sign=-1
        for (j_sh, j_el, j_wr) in ((5, 7, 9), (6, 8, 10)):
            h, d = hinge(j_sh, j_el, j_wr, ml, cfg.elbow_flex, cfg.elbow_ext,
                         cfg.elbow_lateral, -1, SUBCHAINS[j_wr])
            n_hits += h
            max_deg = max(max_deg, d)

        self.rom_hits += 1 if n_hits else 0
        return x, {"rom": n_hits, "rom_deg": max_deg}

    # ── 骨长恒定 ────────────────────────────────────────────
    def _restore_bones(self, x: np.ndarray, S: float, gain: float = 1.0) -> tuple[np.ndarray, dict]:
        total = 0.0
        # 19 点工作区：17 关节 + 颈 + 骨盆（派生点先补上）
        w = np.zeros((19, 3))
        w[:17] = x
        w[NECK] = 0.5 * (x[5] + x[6])
        w[PELVIS] = 0.5 * (x[11] + x[12])
        for (p, c, ratio) in BONE_ORDER:
            if (p, c) in RESTORE_SKIP:
                continue
            v = w[c] - w[p]
            n = _norm(v)
            if n < EPS:
                continue
            L = ratio * S
            err = abs(n - L)
            if err > 0.004:
                total += err * abs(gain)
                w[c] = w[p] + v * (1.0 + (L / n - 1.0) * gain)
        # 眼/耳是"头部细节点"：2D 本身就定得准，且真实头型与 1.85m 标准比例必然有偏差，
        # 强行套名义骨长会把脸压歪（实测对合法人体引入 ~1.7cm 位移）。改成跟随鼻子平移，
        # 保持观测到的相对形状。
        for j in (1, 2, 3, 4):
            w[j] = w[NOSE] + (x[j] - x[NOSE])
        return w[:17].copy(), {"bone_fix": round(total, 4)}

    # ── 地面 ────────────────────────────────────────────────
    def _floor(self, x: np.ndarray, S: float) -> tuple[np.ndarray, dict]:
        cfg = self.cfg
        z = x[[15, 16, 13, 14], 2].min()
        fix = 0.0
        if z < cfg.floor_margin:
            fix = min(cfg.floor_margin - float(z), 0.60)
            x[:, 2] += fix
            x, s = self._restore_bones(x, S)
            # 骨长恢复可能再次把踝压到地面下；再做的纯平移不会破坏骨长。
            z2 = float(x[[15, 16, 13, 14], 2].min())
            if z2 < cfg.floor_margin:
                extra = cfg.floor_margin - z2
                x[:, 2] += extra
                fix += extra
            return x, {"floor_fix": round(fix, 4), "bone_fix": s["bone_fix"]}
        return x, {"floor_fix": 0.0, "bone_fix": 0.0}

    # ── 根节点运动学 + 触地锚 ───────────────────────────────
    def _kinematic(self, x: np.ndarray, dirs: np.ndarray, C: np.ndarray,
                   dt: float, st: dict, S: float) -> tuple[np.ndarray, dict]:
        cfg = self.cfg
        stats = {"root_fix": 0.0}
        pel = 0.5 * (x[11] + x[12])
        prev_p = st["pelvis"]
        prev_v = st["vel"]
        pred = prev_p + prev_v * dt
        delta = np.zeros(3)

        # (a) 水平可达域：单帧位移上限 + 速度上限（防止沿视线的深度跳变）
        d_xy = pel[:2] - pred[:2]
        reach = max(cfg.root_speed * dt, 0.05)
        if prev_v is not None and np.isfinite(prev_v).all():
            reach = min(reach, max(_norm(prev_v) * dt +
                                   cfg.root_accel * dt * dt, 0.05))
        r_xy = math.hypot(float(d_xy[0]), float(d_xy[1]))
        if r_xy > reach:
            over = min(r_xy - reach, cfg.max_root_fix)
            delta[:2] -= (d_xy / max(r_xy, EPS)) * over
            self.root_hits += 1
            stats["root_fix"] = over
        # (b) 竖直：起跳/落地是真实快速运动，只限幅不平滑
        dz = float(pel[2] - pred[2])
        lim_z = max(cfg.vert_speed * dt, 0.05)
        if abs(dz) > lim_z:
            delta[2] -= math.copysign(min(abs(dz) - lim_z, cfg.max_root_fix), dz)
            self.root_hits += 1
            stats["root_fix"] = max(stats["root_fix"], abs(dz) - lim_z)

        if _norm(delta) > 1e-4:
            x = _shift_on_rays(x, dirs, C, delta)
        return x, stats

    def _plant_anchor(self, x: np.ndarray, S: float, dt: float,
                      st: dict) -> tuple[np.ndarray, float]:
        """支撑脚锚：用两段腿 IK 修正踝相对骨盆的滑移，严格保持腿长。"""
        cfg = self.cfg
        candidates = [j for j in (15, 16)
                      if j in st["feet"] and x[j, 2] < cfg.contact_z]
        if not candidates:
            st["plant_delta"] = np.zeros(2)
            return x, 0.0
        j = min(candidates, key=lambda k: x[k, 2])
        hip, knee = (11, 13) if j == 15 else (12, 14)
        fs = st["feet"][j]
        pel = 0.5 * (x[11] + x[12])
        anchor = fs["xy"] + (pel[:2] - fs["pel"])
        want = anchor - x[j, :2]
        sp = math.hypot(float(want[0]), float(want[1]))
        allow = cfg.plant_slip * dt
        if sp <= allow:
            st["plant_delta"] = np.zeros(2)
            return x, 0.0
        raw = (want / max(sp, EPS)) * min(sp - allow, cfg.max_plant_fix)
        prev = st.get("plant_delta")
        if prev is None or prev.shape != (2,) or not np.isfinite(prev).all():
            prev = np.zeros(2)
        a = float(np.clip(-np.expm1(-dt / cfg.plant_tau), 0.2, 1.0))
        delta = a * raw + (1.0 - a) * prev
        st["plant_delta"] = delta

        L1, L2 = 0.245 * S, 0.246 * S
        accepted = 0.0
        # 目标通常很小；若 IK 反而制造硬 ROM 异常，就衰减修正，不硬拽腿。
        for scale in (1.0, 0.5, 0.25, 0.0):
            if scale == 0.0:
                break
            target = x[j].copy()
            target[:2] += delta * scale
            d = target - x[hip]
            dn = _norm(d)
            if dn < EPS:
                continue
            dist = float(np.clip(dn, abs(L1 - L2) + 1e-4, L1 + L2 - 1e-4))
            u = d / dn
            along = (L1 * L1 - L2 * L2 + dist * dist) / (2.0 * dist)
            height = math.sqrt(max(L1 * L1 - along * along, 0.0))
            center = x[hip] + u * along
            bend = x[knee] - center
            bend -= u * _dot(bend, u)
            bn = _norm(bend)
            if bn < EPS:
                bend = np.array([0.0, 1.0, 0.0]) - u * float(u[1])
                bn = _norm(bend)
            if bn < EPS:
                continue
            trial = x.copy()
            trial[knee] = center + bend * (height / bn)
            trial[j] = x[hip] + u * dist
            hard = max(rom_violations(trial, S=S).values(), default=0.0)
            if hard <= 20.0:
                x, accepted = trial, scale
                break
        st["plant_delta"] = delta * accepted
        fixed = math.hypot(float(delta[0]), float(delta[1])) * accepted
        if fixed > 1e-4:
            self.plant_hits += 1
        return x, fixed

    def _remember(self, x: np.ndarray, dt: float, st: dict) -> None:
        pel = 0.5 * (x[11] + x[12])
        if st["pelvis"] is not None and dt > 0:
            inst = (pel - st["pelvis"]) / dt
            v = st["vel"]
            a = self.cfg.vel_alpha
            st["vel"] = inst if v is None else (1 - a) * inst + a * v
        else:
            st["vel"] = np.zeros(3)
        st["pelvis"] = pel.copy()
        feet = {}
        for j in (15, 16):
            if x[j, 2] < self.cfg.contact_z + 0.05:
                feet[j] = {"xy": x[j, :2].copy(), "pel": pel[:2].copy()}
        st["feet"] = feet

    # ── 诊断 ────────────────────────────────────────────────
    def _reproj_drift(self, before: np.ndarray, after: np.ndarray,
                      dirs: np.ndarray, C: np.ndarray) -> float:
        """射线域残差（≈重投影代价）：修正后离开自己射线的距离中位数（米）。"""
        d = after - C
        s = np.einsum("ij,ij->i", d, dirs)
        off = d - dirs * s[:, None]
        return float(np.median(np.linalg.norm(off, axis=1)))

    def status(self) -> dict:
        return {"calls": self.calls, "rom": self.rom_hits, "root": self.root_hits,
                "plant": self.plant_hits, "floor": self.floor_hits}

WORLD_UP = np.array([0.0, 0.0, 1.0])


# ======================================================================
#  小算子
# ======================================================================

def _norm(v: np.ndarray) -> float:
    # 3D 小向量用标量运算：np.linalg.norm 的通用调度开销比这里一次乘法还大。
    return math.sqrt(float(v[0]) * float(v[0]) + float(v[1]) * float(v[1])
                     + float(v[2]) * float(v[2]))


def _dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]], dtype=np.float64)


def _clip_unit(v: float) -> float:
    return min(1.0, max(-1.0, v))


def _unit(v: np.ndarray) -> np.ndarray:
    n = _norm(v)
    return v / n if n > EPS else np.zeros(3)


def _rot(v: np.ndarray, axis: np.ndarray, ang: float) -> np.ndarray:
    """Rodrigues 旋转（右手系，绕单位轴 axis 转 ang 弧度）。"""
    n = _norm(axis)
    if n < EPS:
        return v
    k = axis / n
    c, s = math.cos(ang), math.sin(ang)
    return v * c + _cross(k, v) * s + k * _dot(k, v) * (1.0 - c)


def _low_conf(conf, idx) -> bool:
    """整段链都没有可见性：修正它们不会改变画面，只会白付开销。"""
    if conf is None or len(conf) <= max(idx):
        return False
    # 小索引集合用 Python 标量比较，避免每帧为 np.any 建立临时数组。
    return not any(conf[j] >= 0.15 for j in idx)


def _signed_about(axis: np.ndarray, from_v: np.ndarray, to_v: np.ndarray) -> float:
    """绕单位轴 axis，从 from_v 转到 to_v 所需的有符号角（弧度）。"""
    return float(math.atan2(_dot(_cross(from_v, to_v), axis), _dot(from_v, to_v)))


def _shift_on_rays(x: np.ndarray, dirs: np.ndarray, C: np.ndarray,
                   delta: np.ndarray) -> np.ndarray:
    """把整体平移 delta 尽量落在各关节自己的射线上（射线内零重投影代价）。

    沿视线的误差只能用射线自由度消除，而射线恰好覆盖这个方向；垂直于射线的残差不强求
    （那是 2D 已经定死的方向，动了就是伪造测量）。
    """
    t = x + delta
    s = np.einsum("ij,ij->i", t - C, dirs)
    ok = s > 0.2
    out = x.copy()
    out[ok] = C + dirs[ok] * s[ok][:, None]
    return out


__all__ = ["MotionPrior", "MotionPriorConfig", "rom_violations", "kinematic_flags",
           "hinge_state", "BONE_ORDER", "CHILDREN"]


# ======================================================================
#  评估用：只看、不改
# ======================================================================

def hinge_state(x: np.ndarray, a: int, b: int, c: int, m: np.ndarray,
                normal_sign: int) -> tuple[float, np.ndarray, float, float] | None:
    """三段链 a→b→c 的铰链状态（带符号屈伸角，度）。

    返回 `(signed_flex, rot_axis, lateral_deg, theta_deg)`；退化（骨长≈0、完全伸直、
    左右轴不可辨识）时返回 None。`rot_axis` 已含符号因子 σ：绕它转 Δ 度就让
    signed_flex 变化 Δ 度（正＝往正常屈曲方向）。`lateral_deg` 是屈伸平面法向与
    理想人体左右轴的夹角（0＝纯矢状面屈伸，越大越"往侧边拐"）。
    """
    v1, v2 = x[b] - x[a], x[c] - x[b]
    n1, n2 = _norm(v1), _norm(v2)
    mu = _unit(m)
    mn = _norm(mu)
    if n1 < EPS or n2 < EPS or mn < EPS:
        return None
    u1, u2 = v1 / n1, v2 / n2
    theta = math.degrees(math.acos(_clip_unit(_dot(u1, u2))))
    k = _cross(u1, u2)
    nk = _norm(k)
    if nk < EPS:
        return None
    kk = k / nk
    sigma = float(normal_sign) * (1.0 if _dot(kk, mu) >= 0.0 else -1.0)
    lat = math.degrees(math.acos(_clip_unit(abs(_dot(kk, mu)))))
    return sigma * theta, sigma * kk, lat, theta


def rom_violations(x: np.ndarray, cfg: MotionPriorConfig | None = None,
                   S: float = 1.85) -> dict[str, float]:
    """逐关节超限角度（度，0＝合法）。给 eval/自检用，不修改输入。"""
    cfg = cfg or MotionPriorConfig()
    x = np.asarray(x, dtype=np.float64)
    out: dict[str, float] = {}

    def ang(a, b):
        na, nb = _norm(a), _norm(b)
        if na < EPS or nb < EPS:
            return 0.0
        return math.degrees(math.acos(_clip_unit(_dot(a, b) / (na * nb))))

    pel = 0.5 * (x[11] + x[12])
    nk = 0.5 * (x[5] + x[6])
    ml = _unit(x[5] - x[6])
    up = _unit(nk - pel)
    if _norm(up) < EPS:
        up = WORLD_UP
    hip_ml = _unit(x[11] - x[12])
    if _norm(hip_ml) < EPS:
        hip_ml = ml
    out["torso"] = max(0.0, ang(nk - pel, WORLD_UP) - cfg.torso_tilt)
    out["neck"] = max(0.0, ang(x[NOSE] - nk, up) - cfg.neck_tilt)
    for (ha, kn, an, tag) in ((11, 13, 15, "left"), (12, 14, 16, "right")):
        hs = hinge_state(x, ha, kn, an, hip_ml, +1)
        if hs is None:
            out[f"knee_{tag}"] = 0.0
        else:
            signed = hs[0]
            out[f"knee_{tag}"] = max(0.0, signed - cfg.knee_flex, -cfg.knee_ext - signed)
        u = _unit(x[kn] - x[ha])
        out[f"hip_{tag}"] = max(0.0, 90.0 - math.degrees(
            math.acos(_clip_unit(abs(_dot(u, ml))))) - cfg.hip_abduct)
    for (sh, el, wr, tag) in ((5, 7, 9, "left"), (6, 8, 10, "right")):
        hs = hinge_state(x, sh, el, wr, ml, -1)
        if hs is None:
            out[f"elbow_{tag}"] = 0.0
        else:
            signed = hs[0]
            out[f"elbow_{tag}"] = max(0.0, signed - cfg.elbow_flex,
                                      -cfg.elbow_ext - signed)
        u = _unit(x[el] - x[sh])
        d = -up
        fwd = _unit(_cross(d, ml))
        if _norm(fwd) > EPS and float(_dot(u, d)) >= 0.15:
            a = math.degrees(math.atan2(float(_dot(u, fwd)), float(_dot(u, d))))
            out[f"shoulder_{tag}"] = max(0.0, -cfg.shoulder_back - a)
        else:
            out[f"shoulder_{tag}"] = 0.0
    return out


def kinematic_flags(prev: np.ndarray, curr: np.ndarray, dt: float,
                    cfg: MotionPriorConfig | None = None) -> dict[str, float]:
    """时序越界量（米/米每秒）：根节点位移、竖直速度、踝穿透。给 eval 用。"""
    cfg = cfg or MotionPriorConfig()
    a, b = np.asarray(prev, float), np.asarray(curr, float)
    p = 0.5 * (a[11] + a[12])
    q = 0.5 * (b[11] + b[12])
    v = (q - p) / max(dt, EPS)
    return {"root_speed": float(np.linalg.norm(v[:2])),
            "vert_speed": float(abs(v[2])),
            "speed_over": max(0.0, float(np.linalg.norm(v[:2])) - cfg.root_speed),
            "vert_over": max(0.0, float(abs(v[2])) - cfg.vert_speed),
            "penetration": float(max(0.0, -min(np.asarray(curr, float)[15, 2],
                                               np.asarray(curr, float)[16, 2])))}
