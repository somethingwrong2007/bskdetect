"""骨架时序状态估计层（2D 先行）：抗抖动 + 遮挡预测 + 骨长校准。

为什么单独一层：RTMO / YOLOv8-pose 都是逐帧单发模型，帧与帧之间没有任何约束，
静止站立时关键点也会在 1-3px 范围里抖，被挡住时干脆跳到骨架外面。这里把"逐帧
观测"升级成"轨迹状态"，全部是因果（在线）的，只用当前帧与历史状态，不预取未来帧
（后续接直播流，只有当前帧可用）。

1. 位置：One Euro Filter。速度自适应截止频率：站着不动时强平滑压抖动，快速挥臂
   时自动放开截止频率减小滞后。坐标按身高归一化，参数与分辨率/远近无关。
2. 置信度门控（带迟滞）：低置信关节不喂滤波器，用恒速预测顶替并降置信；已跟踪的
   关节用较低阈值维持、新出现的用较高阈值进入，避免在阈值附近来回闪。
3. 关节级遮挡：连续丢失时恒速外推，最多 max_predict 帧（~0.5s），速度逐帧衰减。
4. 人体级遮挡：整框丢失时按框速度维持 track，超过 max_track_age 才丢弃；重新检回
   后用 reacquire_frames 帧（~0.2s）混合过渡，避免"弹回"。
5. 骨长约束：每根骨在线学一个参考长度，突变超过容差时只按比例回拉一部分（不旋转
   骨骼方向，姿态语义不受影响）。
6. 左右镜像补全：只在关节状态彻底死亡且连续丢失 >= mirror_min_miss 帧时启用，补出的
   点不回流滤波。旧实现每帧都补且写回 kpts，是左右肢抽搐的来源之一。
7. 躯干相对先验补全（头部 5 点 / 双腕等）：RTMO 对眼/耳基本测不到（conf 中位数 0.1-0.2），
   手腕也常来回掉。但头相对脖子、腕相对肘是"刚性/准刚性"关系，可以逐 track 在线学
   偏移（以躯干长归一化），观测不可信时用父关节 + 学习偏移补出来，conf 取中间值
   （可见但明显低于实测）。这是覆盖率提升的主要来源，补位点同样不回流滤波。

3D 扩展：把 kpts 换成世界坐标、把像素身高换成米制比例即可复用同一状态机；骨长约束
天然以米为单位，足底触地/关节角合法域需要逐帧单应矩阵，留到 3D 层做（状态里保留了
每个关节的观测/预测标志，可直接接）。
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np

from bsk import schema

# 骨长约束用的父子骨：只校正有明确端点方向的骨，横向肩/髋连线与头颈不动
BONES = (
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)
# 左右对称点对（镜像补全用）
MIRROR_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
BONE_PARENT = np.array([a for a, _b in BONES], dtype=np.intp)
BONE_CHILD = np.array([b for _a, b in BONES], dtype=np.intp)
MIRROR_L = np.array([l for l, _r in MIRROR_PAIRS], dtype=np.intp)
MIRROR_R = np.array([r for _l, r in MIRROR_PAIRS], dtype=np.intp)

# 先验补全的父关节：头部 5 点挂"脖子"虚拟节点（颈 5/6 中点，索引 17），腕挂肘、踝挂膝。
# 肩/髋也挂脖子，是为了让"偏移归一化尺度"统一（躯干长），并不是为了补肩髋。
NECK = 17
PARENT_OF = np.array([NECK, NECK, NECK, NECK, NECK, NECK, NECK,
                      5, 6, 7, 8, NECK, NECK, 11, 12, 13, 14], dtype=np.intp)
HEAD = np.arange(0, 5)
# 准刚性关系：头/肩/髋相对脖子几乎只随躯干整体动，可以用先验撑很久；
# 腕/踝是末端，姿势记忆会很快过期（0.3s 内还靠谱，久了必须交回外推/隐藏）。
RIGID = np.array([0, 1, 2, 3, 4, 11, 12], dtype=np.intp)
# 末端关节：RTMO 测到时 conf 常在 0.15-0.3 之间游走，弱观测不如用"肘/膝 + 学习偏移"补
TIPS = np.array([9, 10, 15, 16], dtype=np.intp)
# 头部模板（肩线坐标系，x 沿左肩→右肩，x 向右为正；y 向下为正），以躯干长（颈→髋中点）归一化。
# 来自 harden-55 上 419-490 个高置信头部观测的中位数：鼻 (0,-0.33)、眼 (±0.07,-0.42)、
# 耳 (±0.205,-0.43)。只在该 track 还没学到自己的偏移时兜底用。
HEAD_T_U = np.array([0.0, -0.07, 0.07, -0.205, 0.205])
HEAD_T_V = np.array([-0.33, -0.42, -0.42, -0.43, -0.43])

CONF_PRED = 0.10    # 预测点置信度：低于渲染阈值 0.15 → 画灰点、不连边
CONF_MIRROR = 0.30  # 镜像补全点：能连边但明显低于实测，提示"这是猜的"
CONF_PRIOR = 0.35   # 先验补全点（父关节 + 学习偏移）：比镜像可信、比实测低
CONF_TEMPLATE = 0.20  # 纯模板补出的头部点（还没学到个体偏移）：可见但更可疑
CONF_TIP = 0.17     # 腕/踝纯外推点：刚过渲染阈值 0.15，勉强连边，提醒"这是预测"


@dataclass
class PoseFilterConfig:
    fps: float = 30.0
    mode: str = "oneeuro"        # "oneeuro" | "kalman"（对比用，实测 One Euro 更优）
    min_cutoff: float = 2.0      # Hz，静止截止频率：越小越稳、越"粘"
    beta: float = 0.35           # 速度系数（身高/秒），越大快速动作滞后越小
    d_cutoff: float = 1.2        # 速度估计的截止频率
    kalman_k: float = 0.55       # 恒速卡尔曼位置增益（稳态 α）
    kalman_b: float = 0.12       # 恒速卡尔曼速度增益（稳态 β）
    hi_gate: float = 0.25        # 未跟踪关节进入观测的门限
    lo_gate: float = 0.12        # 已跟踪关节维持观测的门限（迟滞）
    max_predict: int = 15        # 关节级最大外推帧数（~0.5s @30fps）
    max_track_age: int = 10      # 人体级最大维持帧数（~0.33s，久了预测框会飘）
    min_confirm: int = 3         # 至少被检测到几帧才允许人体级维持（滤掉一闪而过的误检）
    vel_damp: float = 0.70       # 外推时每帧速度衰减
    max_vel: float = 0.30        # 归一化速度上限（身高/帧），约 9 身高/秒
    reacquire_frames: int = 6    # 重现后的混合过渡帧数（~0.2s）
    reacq_min_miss: int = 2      # 丢失多少帧后重现才当成"重捕获"（1 帧抖动走常规滤波）
    bone_tol: float = 0.15       # 骨长突变容差（15% 以内不校正）
    bone_gain: float = 0.35      # 骨长校正强度（只走 35%，避免破坏姿态）
    mirror_min_miss: int = 6     # 连续丢失多少帧后才允许镜像补全（~0.2s）
    reset_gap: int = 12          # 帧号跳变超过此值直接清状态（拖进度条/换片段）
    torso_fallback: bool = True  # 髋部缺失（半身目标）时用 0.26×框高 + 画面朝上兜底
    # ── 躯干相对先验补全（头部/手腕可见覆盖率）──────────────────
    prior_on: bool = True        # 总开关
    head_obs_conf: float = 0.25  # 头部 5 点低于此 conf 不喂滤波（RTMO 眼/耳噪声大，改用先验）
    limb_obs_conf: float = 0.18  # 腕/踝弱观测阈值：低于它渲染也是隐形，直接换成先验点
    prior_min_obs: int = 2       # 父关节相对偏移至少学到几帧才敢用来补
    prior_learn: float = 0.25    # 偏移学习 EMA 增益
    prior_rigid_miss: int = 60   # 头/肩/髋这类准刚性关节最多先验补多少帧（~2s）
    prior_limb_miss: int = 14    # 腕/踝/肘/膝先验最多顶多少帧（~0.5s，跨帧遮挡不消失）
    prior_blend: float = 5.0     # 末端关节"外推 vs 姿势记忆"的过渡帧数（miss/5 权重）
    tip_pred_visible: bool = True  # 腕/踝无父关节可挂靠时，外推点抬到 0.17 勉强可见


def _alpha(cutoff, dt: float):
    """One Euro 的指数平滑系数：tau = 1/(2πf)，alpha = dt/(tau+dt)。"""
    tau = 1.0 / (2.0 * np.pi * np.maximum(cutoff, 1e-3))
    return dt / (tau + dt)


def _alpha_s(cutoff: float, dt: float) -> float:
    """标量版（省掉 numpy 调用开销）。"""
    return dt / (1.0 / (2.0 * math.pi * max(cutoff, 1e-3)) + dt)


class _Track:
    """单个人的滤波/遮挡状态。

    位置一律存像素坐标：输出时不能乘回"身高参考"，否则框高的单帧抖动会变成整副骨架的
    缩放抖动（远处小球员尤其明显）。身高只用于把速度换算成"身高/秒"、让滤波参数与
    尺度无关。
    """

    __slots__ = ("tid", "frame", "pos", "out", "vel", "dxhat", "conf", "obs", "seen",
                 "miss", "rec", "h", "bbox", "bbox_vel", "age", "bone", "bone_seen", "bone_tick", "n_obs",
                 "off", "off_seen", "video_id", "model", "schema")

    def __init__(self, tid: int, frame: int, bbox, kpts: np.ndarray,
                 cfg: PoseFilterConfig, src: schema.Pose2D):
        n = kpts.shape[0]
        h = max(float(bbox[3] - bbox[1]), 8.0)
        self.tid = tid
        self.frame = frame
        self.h = h
        self.bbox = np.asarray(bbox, dtype=np.float64)
        self.bbox_vel = np.zeros(4)
        self.age = 0
        self.n_obs = 0
        # 新轨迹还没位置，先落在框中心，等观测进来再覆盖
        cx = (bbox[0] + bbox[2]) * 0.5
        cy = (bbox[1] + bbox[3]) * 0.5
        self.pos = np.tile(np.array([[cx, cy]]), (n, 1))
        self.out = self.pos.copy()  # 渲染用的显示位置（重捕获期做混合过渡）
        self.vel = np.zeros((n, 2))       # 像素/帧
        self.dxhat = np.zeros((n, 2))     # 像素/秒（One Euro 的速度估计）
        self.conf = np.zeros(n)
        self.obs = np.zeros(n, dtype=bool)
        self.seen = np.zeros(n, dtype=bool)
        self.miss = np.zeros(n, dtype=np.int32)
        self.rec = np.full(n, cfg.reacquire_frames, dtype=np.int32)
        self.bone = np.zeros(len(BONES))   # 参考骨长，以身高归一化
        self.bone_seen = np.zeros(len(BONES), dtype=bool)
        self.bone_tick = 0
        self.off = np.zeros((n, 2))        # 父关节相对偏移（躯干长归一化），先验补全用
        self.off_seen = np.zeros(n, dtype=np.int32)
        self.video_id = src.video_id
        self.model = src.model
        self.schema = src.schema


class PoseFilter:
    """按 track_id 维护在线状态；线程安全（服务端多线程取帧）。"""

    def __init__(self, cfg: PoseFilterConfig | None = None):
        self.cfg = cfg or PoseFilterConfig()
        self._fmax = self.cfg.fps * 0.45   # 截止频率上限（奈奎斯特），防止 alpha > 1
        # 逐关节门限：头部 5 点单独提高（RTMO 眼/耳 conf 中位数只有 0.1-0.2，喂进来只会
        # 让骨架抖；宁可交给先验补全保持形状稳定）
        self._lo = np.full(17, self.cfg.lo_gate)
        self._hi = np.full(17, self.cfg.hi_gate)
        # 先验补全的静态查表（每帧每人都要用，别在热路径里算 np.isin/np.where）
        self._rigid = np.zeros(17, dtype=bool)
        self._rigid[RIGID] = True
        self._weak_thr = np.zeros(17)              # 低于它就算"弱观测"（0 = 不启用）
        if self.cfg.prior_on:
            self._lo[HEAD] = max(self.cfg.lo_gate, self.cfg.head_obs_conf)
            self._hi[HEAD] = max(self.cfg.hi_gate, self.cfg.head_obs_conf)
            self._weak_thr[HEAD] = self.cfg.head_obs_conf
            self._weak_thr[TIPS] = self.cfg.limb_obs_conf
            self._cap = np.where(self._rigid, self.cfg.prior_rigid_miss, self.cfg.prior_limb_miss)
        else:
            self._cap = np.zeros(17, dtype=np.int32)
        self._tracks: dict[int, _Track] = {}
        self._frame = 0
        self._lock = threading.Lock()

    def reset(self) -> None:
        self._tracks.clear()
        self._frame = 0

    # ── 对外接口 ────────────────────────────────────────────
    def apply(
        self,
        frame: int,
        poses: list[schema.Pose2D],
        boxes: dict[int, tuple],
    ) -> tuple[list[schema.Pose2D], dict[int, tuple]]:
        """摄入当前帧观测，输出滤波后的骨架 + 人体级遮挡时的预测框。

        返回 (poses, occluded_boxes)。occluded_boxes 是"这帧没检测到、但状态还在维持"
        的 track：键为 track_id，值为预测框，调用方可据此补 Tracked(visible=False)。
        """
        cfg = self.cfg
        with self._lock:
            # 帧号回退/大跳变（拖进度条、换片段）→ 清状态，别拿陈旧速度去外推
            gap = frame - self._frame if self._frame else 1
            if self._frame and (frame <= self._frame or gap > cfg.reset_gap):
                self.reset()
                gap = 1
            self._frame = frame
            gap = max(int(gap), 1)

            observed: dict[int, tuple[np.ndarray, tuple, schema.Pose2D]] = {}
            for p in poses:
                if p.track_id in observed:
                    continue  # 同 ID 两条观测不该出现，保留第一条
                kp = np.asarray(p.kpts, dtype=np.float64)
                if kp.ndim != 2 or kp.shape[0] != 17 or kp.shape[1] < 3:
                    continue
                box = boxes.get(p.track_id)
                if box is None:
                    box = (float(kp[:, 0].min()), float(kp[:, 1].min()),
                           float(kp[:, 0].max()), float(kp[:, 1].max()) + 1.0)
                observed[p.track_id] = (kp, tuple(float(v) for v in box), p)

            out: list[schema.Pose2D] = []
            for tid, (kp, box, src) in observed.items():
                st = self._tracks.get(tid)
                if st is None:
                    st = _Track(tid, frame, box, kp, cfg, src)
                    self._tracks[tid] = st
                self._update(st, kp, box, frame, gap)
                st.age = 0
                st.n_obs += 1
                st.frame = frame
                out.append(self._to_pose(st))

            # 人体级遮挡：这帧没被检测到但状态未超龄 → 按框速度维持 track
            # 未确认的轨迹（只闪了 1-2 帧的误检）不维持，否则观众席假阳会被拖着画满 0.5s
            occ_boxes: dict[int, tuple] = {}
            for tid in [t for t in self._tracks if t not in observed]:
                st = self._tracks[tid]
                st.age += gap
                if st.age > cfg.max_track_age or st.n_obs < cfg.min_confirm:
                    del self._tracks[tid]
                    continue
                self._predict(st, frame, gap)
                occ_boxes[tid] = tuple(float(v) for v in st.bbox)
                out.append(self._to_pose(st))
            return out, occ_boxes

    # ── 单轨迹更新 ──────────────────────────────────────────
    def _update(self, st: _Track, kpts: np.ndarray, box, frame: int, gap: int) -> None:
        cfg = self.cfg
        dt = max(gap / cfg.fps, 1e-3)
        h = max(float(box[3] - box[1]), 8.0)
        # 身高参考做 EMA：单帧框抖动会直接变成整体缩放抖动，不能每帧硬切
        st.h = h if st.h <= 0 else st.h * 0.85 + h * 0.15
        hr = st.h

        # 框速度：人体级遮挡时靠它维持 track（尺寸不预测，人不会突然变大变小）
        prev_box = st.bbox
        st.bbox = np.asarray(box, dtype=np.float64)
        if st.age == 0:
            st.bbox_vel = 0.6 * st.bbox_vel + 0.4 * ((st.bbox - prev_box) / gap)

        obs_xy = kpts[:, :2]                     # 像素
        conf = kpts[:, 2]
        finite = np.isfinite(obs_xy).all(1)
        vmax = cfg.max_vel * hr               # 像素/帧

        # 门控带迟滞：已见过的关节用低阈值维持，没见过的用高阈值进入
        valid = finite & (conf >= np.where(st.seen, self._lo, self._hi))
        steady = valid.all() and not st.miss.any()

        if steady:
            # 常见路径（全部关节连续观测）：只做 One Euro，别为分支付 numpy 开销
            self._smooth(st, obs_xy, hr, gap, dt)
            st.rec[:] = cfg.reacquire_frames
            st.conf[:] = conf
            st.out[:] = st.pos
            st.obs = valid
            self._bone_constraint(st, valid)
            if cfg.prior_on:
                self._prior_learn(st)
            st.frame = frame
            return

        first = valid & ~st.seen
        reacq = valid & st.seen & (st.miss >= cfg.reacq_min_miss)  # 遮断多帧后重现：预测已不可信
        if reacq.any():
            rec_sel = st.rec[reacq].copy()
            prev_sel = st.out[reacq].copy()

        # 1) 首次观测 / 重新捕获：滤波器状态直接重置到测量值
        #    预测漂了 0.2s 之后，任何"慢慢收敛"都会让骨架拖在真实位置后面，
        #    所以状态立刻归位，视觉上的过渡交给下面的输出混合去做。
        reset = first | reacq
        if reset.any():
            gap_to_prev = np.maximum(st.miss[reset] + 1, 1).astype(np.float64)
            st.vel[reset] = np.clip((obs_xy[reset] - st.pos[reset]) / gap_to_prev[:, None], -vmax, vmax)
            st.pos[reset] = obs_xy[reset]
            st.dxhat[reset] = st.vel[reset] * cfg.fps
            st.seen[reset] = True
            st.vel[first] = 0.0
            st.dxhat[first] = 0.0

        # 2) 丢失关节：恒速外推 + 速度衰减，超过 max_predict 就定住（不再越漂越远）
        lost = ~valid
        if lost.any():
            step = min(gap, 3)
            st.pos[lost] += st.vel[lost] * step
            st.vel[lost] *= cfg.vel_damp ** step
            st.miss[lost] += gap
            st.rec[lost] = 0
            dead = st.miss > cfg.max_predict
            st.vel[dead] = 0.0

        # 3) 持续观测关节：One Euro（速度自适应截止频率）/ 恒速卡尔曼（对比用）
        upd = valid & ~reset
        if upd.any():
            self._smooth(st, obs_xy, hr, gap, dt, mask=upd)

        if valid.any():
            st.miss[valid] = 0
            st.rec[valid] += 1
            st.conf[valid] = conf[valid]

        # 4) 骨长约束（轻度比例校正，不改骨骼方向）
        self._bone_constraint(st, valid)

        # 5) 显示位置：正常帧直接取状态；重现后的 reacquire_frames 帧内从上一帧显示位置
        #    线性过渡过去，避免"弹回"（渲染看到的是这条轨迹，状态本身已经归位）
        np.copyto(st.out, st.pos)
        if reacq.any():
            r = np.minimum(rec_sel.astype(np.float64), cfg.reacquire_frames)
            w = (0.35 + 0.65 * r / max(cfg.reacquire_frames, 1))[:, None]
            st.out[reacq] = (1.0 - w) * prev_sel + w * st.pos[reacq]

        st.obs = valid
        if cfg.prior_on:
            self._prior_learn(st)
        st.frame = frame

    def _smooth(self, st: _Track, obs_xy: np.ndarray, hr: float, gap: int, dt: float,
                mask: np.ndarray | None = None) -> None:
        """观测关节的位置更新（One Euro 或恒速卡尔曼），含速度上限。"""
        cfg = self.cfg
        x = obs_xy if mask is None else obs_xy[mask]
        prev = st.pos if mask is None else st.pos[mask]
        vel = st.vel if mask is None else st.vel[mask]
        if cfg.mode == "kalman":
            resid = x - (prev + vel * gap)
            new = prev + vel * gap + cfg.kalman_k * resid
            vel = vel + cfg.kalman_b * resid / gap
            dxhat = vel * cfg.fps
        else:
            dx = (x - prev) / dt
            a_d = _alpha_s(cfg.d_cutoff, dt)
            dxhat = a_d * dx + (1.0 - a_d) * (st.dxhat if mask is None else st.dxhat[mask])
            # 速度换算成"身高/秒"再算截止频率，参数与远近/分辨率无关
            cutoff = cfg.min_cutoff + cfg.beta * np.abs(dxhat) / hr
            np.minimum(cutoff, self._fmax, out=cutoff)
            a = dt / (0.5 / (np.pi * cutoff) + dt)   # tau=1/(2πf)，内联省一次调用
            new = a * x + (1.0 - a) * prev
            vel = 0.6 * vel + 0.4 * (dxhat / cfg.fps)
        if mask is None:
            st.pos[:] = new
            st.dxhat[:] = dxhat
            st.vel[:] = vel
        else:
            st.pos[mask] = new
            st.dxhat[mask] = dxhat
            st.vel[mask] = vel
        # 速度上限：单帧位移不可能超过 max_vel 个身高，超了就是模型跑飞
        vmax = cfg.max_vel * hr
        sp = np.sqrt(st.vel[:, 0] ** 2 + st.vel[:, 1] ** 2)
        bad = sp > vmax
        if bad.any():
            st.vel[bad] *= (vmax / sp[bad])[:, None]

    def _predict(self, st: _Track, frame: int, gap: int) -> None:
        """整框丢失：框按速度平移（尺寸保持不变），关节由 _update 的外推逻辑接管。"""
        box = st.bbox + st.bbox_vel * min(gap, 3)
        self._update(st, np.full((17, 3), np.nan), box, frame, gap)
        st.bbox_vel *= 0.95 ** gap  # 遮挡久了人可能停下来，速度慢慢收敛
        st.obs[:] = False

    def _bone_constraint(self, st: _Track, valid: np.ndarray) -> None:
        """骨长轻度校正：整组骨一次性向量化回拉（只改长度、不改方向）。

        每 2 帧跑一次：参考骨长是慢学量，10-15Hz 足够，省一半 CPU。
        """
        cfg = self.cfg
        st.bone_tick += 1
        if st.bone_tick & 1:
            return
        pos = st.pos
        hr = st.h if st.h > 8.0 else 8.0
        d = pos[BONE_CHILD] - pos[BONE_PARENT]            # (10,2)
        length = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) / hr   # 以身高归一化
        alive = st.seen[BONE_PARENT] & st.seen[BONE_CHILD] & (length > 1e-4)
        if not alive.any():
            return
        # 参考骨长慢学：单帧坏姿态拉不动它
        learn = alive & valid[BONE_PARENT] & valid[BONE_CHILD]
        if learn.any():
            st.bone[learn] = np.where(st.bone_seen[learn],
                                      st.bone[learn] * 0.9 + length[learn] * 0.1,
                                      length[learn])
            st.bone_seen[learn] = True
        use = alive & (st.bone > 0)
        if not use.any():
            return
        ratio = np.clip(st.bone[use] / length[use], 1.0 - cfg.bone_tol, 1.0 + cfg.bone_tol)
        if not (np.abs(ratio - 1.0) > 1e-3).any():
            return
        scale = 1.0 + cfg.bone_gain * (ratio - 1.0)
        pos[BONE_CHILD[use]] = pos[BONE_PARENT[use]] + d[use] * scale[:, None]

    # ── 躯干相对先验（头部/手腕覆盖率）────────────────────────
    def _torso(self, st: _Track) -> tuple[np.ndarray, np.ndarray] | None:
        """取当前帧的"脖子中点 + 躯干尺度 + 肩线坐标系"，不可靠时返回 None。

        尺度首选躯干长（颈→髋中点）：框高被手臂/伸展污染，颈到髋是刚性段。整条腿都
        没测到的半身目标（画面边缘、被人挡住）退回 0.26×框高——实测 torso/框高 中位数
        就是 0.265，误差够小，好过整片头部直接消失。
        返回 (neck, (scale, u, v))：u 沿左肩→右肩，v 垂直向下（身体下方）。
        """
        if not (st.obs[5] and st.obs[6]):
            return None
        o = st.out
        neck = (o[5] + o[6]) * 0.5
        u = o[6] - o[5]
        nu = float(math.hypot(u[0], u[1]))
        if nu < 0.06 * st.h:
            return None
        u = u / nu
        v = np.array([-u[1], u[0]])
        if st.obs[11] and st.obs[12]:
            hip = (o[11] + o[12]) * 0.5
            scale = float(math.hypot(neck[0] - hip[0], neck[1] - hip[1]))
            if scale >= 0.10 * st.h:
                if v @ (hip - neck) < 0.0:      # v 永远指向身体下方
                    v = -v
                return neck, np.array([scale, u[0], u[1], v[0], v[1]])
        # 半身目标：没有髋就没法定向，按"画面里头朝上"的常规视角处理（只翻 v，不动 u，
        # 否则左右眼/耳会互换）
        if not self.cfg.torso_fallback:
            return None
        if v[1] < 0.0:
            v = -v
        scale = 0.26 * st.h
        return neck, np.array([max(scale, 1.0), u[0], u[1], v[0], v[1]])

    def _prior_learn(self, st: _Track) -> None:
        """在线学"父关节 → 子关节"的偏移（躯干长归一化），每 2 帧一次。"""
        if st.bone_tick & 1:
            return
        torso = self._torso(st)
        if torso is None:
            return
        neck, ref = torso
        scale = ref[0]
        anchor = np.empty((18, 2))
        anchor[:17] = st.out
        anchor[NECK] = neck
        ok = np.ones(17, dtype=bool)
        m = PARENT_OF != NECK                 # 脖子是虚拟节点，可信性已由 _torso 保证
        ok[m] = st.obs[PARENT_OF[m]]
        sel = st.obs & ok
        if not sel.any():
            return
        delta = (st.out[sel] - anchor[PARENT_OF[sel]]) / scale
        g = self.cfg.prior_learn
        fresh = st.off_seen[sel] == 0
        st.off[sel] = np.where(fresh[:, None], delta, st.off[sel] * (1.0 - g) + delta * g)
        st.off_seen[sel] += 1

    def _prior_fill(self, st: _Track, kpts: np.ndarray) -> None:
        """用父关节 + 学习偏移（头部退化到模板）补出"不可信"的关节。

        触发条件：关节这帧没被采信，或被采信但 conf 低到渲染层会当隐形点（头 <0.25、
        末端 <0.18），且父关节观测可靠。刚性组（头/肩/髋）直接取先验；末端组在
        "外推 vs 姿势记忆"之间按丢失帧数渐变，丢了 0.3s 以上就只用姿势记忆（外推此时
        已经飘了）。头部个体偏移学到 prior_min_obs 帧之前用肩线坐标系的对称模板兜底，
        保住"有头没脸"的观感。补位点 conf 取 0.2-0.35，可见但明显低于实测。
        """
        cfg = self.cfg
        miss_now = ~st.obs
        weak = st.obs & (st.conf < self._weak_thr)
        sel = miss_now | weak
        if not sel.any():
            return
        torso = self._torso(st)
        if torso is None:
            return
        neck, ref = torso
        scale, ux, uy, vx, vy = float(ref[0]), ref[1], ref[2], ref[3], ref[4]
        u = np.array([ux, uy])
        v = np.array([vx, vy])
        o = st.out
        par_ok = np.ones(17, dtype=bool)
        m = PARENT_OF != NECK
        par_ok[m] = st.obs[PARENT_OF[m]]               # _torso 已保证脖子的两侧肩可信
        learned = st.off_seen >= cfg.prior_min_obs
        rigid = self._rigid
        cap = self._cap

        # 1) 头部 5 点：学习偏移为主，没学到的点用肩线模板兜底
        hs = HEAD[sel[HEAD] & (st.miss[HEAD] <= cfg.prior_rigid_miss)]
        if hs.size:
            lw = np.clip(st.off_seen[hs] / 6.0, 0.0, 1.0)[:, None] * learned[hs][:, None]
            tpl = (HEAD_T_U[hs][:, None] * u + HEAD_T_V[hs][:, None] * v) * scale
            prior = lw * (st.off[hs] * scale) + (1.0 - lw) * tpl
            kpts[hs, :2] = neck + prior
            kpts[hs, 2] = np.where(learned[hs], CONF_PRIOR, CONF_TEMPLATE)

        # 2) 其他关节：父关节 + 学习偏移；末端按丢失帧数从外推渐变到姿势记忆
        fill = sel & par_ok & learned & (st.miss <= cap)
        fill[HEAD] = False
        if not fill.any():
            return
        anchor = np.empty((18, 2))
        anchor[:17] = o
        anchor[NECK] = neck
        prior = anchor[PARENT_OF[fill]] + st.off[fill] * scale
        w = np.clip(1.0 - st.miss[fill].astype(np.float64) / cfg.prior_blend, 0.0, 1.0)
        w[rigid[fill]] = 0.0                            # 刚性关节不走外推，直接取先验
        kpts[fill, :2] = w[:, None] * o[fill] + (1.0 - w[:, None]) * prior
        kpts[fill, 2] = CONF_PRIOR

    # ── 输出 ────────────────────────────────────────────────
    def _to_pose(self, st: _Track) -> schema.Pose2D:
        cfg = self.cfg
        hr = st.h
        kpts = np.empty((17, 3), dtype=np.float64)
        kpts[:, :2] = st.out
        conf = np.where(st.obs, st.conf,
                        np.where(st.miss <= cfg.max_predict, CONF_PRED, 0.0))
        # 腕/踝外推点抬到渲染阈值之上：整条手臂都没测到时先验也无从挂靠，只能靠状态外推，
        # 这时"消失"比"短暂预测"更难用；只给 0.17（勉强可见），超过 max_predict 直接归零。
        if cfg.tip_pred_visible:
            tips = TIPS[(~st.obs[TIPS]) & st.seen[TIPS] & (st.miss[TIPS] <= cfg.max_predict)]
            if tips.size:
                conf[tips] = CONF_TIP
        kpts[:, 2] = conf
        if cfg.prior_on:
            self._prior_fill(st, kpts)
        self._mirror_fill(st, kpts)
        return schema.Pose2D(
            video_id=st.video_id, frame=st.frame, track_id=st.tid,
            kpts=kpts.tolist(), schema=st.schema, model=st.model,
            height_px=float(hr),
            occluded=bool((~st.obs).any()),
        )

    def _mirror_fill(self, st: _Track, kpts: np.ndarray) -> None:
        """左右镜像补全：只补"状态已死"且丢失足够久的关节，补出的点不回流滤波。

        整组点对一次性判定（向量化），同一对不会互相拿推测点当依据。
        """
        cfg = self.cfg
        conf = kpts[:, 2]
        dead = (conf <= CONF_PRED * 1.5) & (st.miss >= cfg.mirror_min_miss) & ~st.obs
        if not dead.any():
            return
        cx = (st.bbox[0] + st.bbox[2]) * 0.5
        # 左缺右可信 → 补左；右缺左可信 → 补右（conf 阈值保证不会两边互推）
        fix_l = dead[MIRROR_L] & (conf[MIRROR_R] >= 0.6)
        fix_r = dead[MIRROR_R] & (conf[MIRROR_L] >= 0.6)
        if fix_l.any():
            li, ri = MIRROR_L[fix_l], MIRROR_R[fix_l]
            kpts[li, 0] = 2.0 * cx - kpts[ri, 0]
            kpts[li, 1] = kpts[ri, 1]
            kpts[li, 2] = CONF_MIRROR
        if fix_r.any():
            li, ri = MIRROR_L[fix_r], MIRROR_R[fix_r]
            kpts[ri, 0] = 2.0 * cx - kpts[li, 0]
            kpts[ri, 1] = kpts[li, 1]
            kpts[ri, 2] = CONF_MIRROR


__all__ = ["PoseFilter", "PoseFilterConfig", "CONF_PRED", "CONF_MIRROR", "CONF_PRIOR",
           "CONF_TEMPLATE", "CONF_TIP"]