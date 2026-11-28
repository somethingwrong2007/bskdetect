"""骨架时序状态估计层（2D 先行）：抗抖动 + 遮挡预测 + 骨长校准。

为什么单独一层：RTMO 等逐帧单发模型，帧与帧之间没有任何约束，
静止站立时关键点也会在 1-3px 范围里抖，被挡住时干脆跳到骨架外面。这里把"逐帧
观测"升级成"轨迹状态"，全部是因果（在线）的，只用当前帧与历史状态，不预取未来帧
（后续接直播流，只有当前帧可用）。

1. 位置：框中心刚体平移前馈 + 相对姿态 One Euro。速度自适应截止频率：站着不动时强平滑
   压抖动，快速挥臂时自动放开截止频率减小滞后。只平滑相对框的形变，避免整人移动也被
   低通而落后于同帧检测框。坐标按身高归一化，参数与分辨率/远近无关。
2. 置信度门控（带迟滞）：低置信关节不喂滤波器，用恒速预测顶替并降置信；已跟踪的
   关节用较低阈值维持、新出现的用较高阈值进入，避免在阈值附近来回闪。
3. 关节级遮挡：连续丢失时恒速外推，最多 max_predict 帧（~0.5s），速度逐帧衰减。
4. 人体级遮挡：整框丢失时按框速度维持 track，超过 max_track_age 才丢弃；重新检回
   后用 reacquire_frames 帧（~0.2s）混合过渡，避免"弹回"。hold 期间**锁尺度**：
   框只平移不缩放（尺度用丢失前的一帧冻结），骨骼形状也整体冻结只做刚体平移，
   恢复时身高参考按 reacquire_frames 帧渐变回实测——旧实现让每根骨各自按速度外推，
   0.3s 后骨长能拉长 20%，恢复瞬间整副骨架"弹回"（实测恢复首帧位移 20%→8%）。
5. 骨长约束：每根骨在线学一个参考长度，突变超过容差时只按比例回拉一部分（不旋转
   骨骼方向，姿态语义不受影响）。
6. 左右镜像补全：只在关节状态彻底死亡且连续丢失 >= mirror_min_miss 帧时启用，补出的
   点不回流滤波。旧实现每帧都补且写回 kpts，是左右肢抽搐的来源之一。镜像轴用**躯干
   中线**（肩/髋中点连线，身体坐标系）而不是 bbox 中心：球员侧身或单臂伸展时框中心
   根本不在人体对称面上，按框中心镜像会把左腕补到右肩外侧（左腕补位误差 12%→6%）。
7. 躯干相对先验补全（头部 5 点 / 双腕等）：RTMO 对眼/耳基本测不到（conf 中位数 0.1-0.2），
   手腕也常来回掉。但头相对脖子、腕相对肘是"刚性/准刚性"关系，可以逐 track 在线学
   偏移（以躯干长归一化），观测不可信时用父关节 + 学习偏移补出来，conf 取中间值
   （可见但明显低于实测）。这是覆盖率提升的主要来源，补位点同样不回流滤波。
   末端（腕/踝）再加一路"沿父骨方向外推"的动态先验（肘→腕、膝→踝，长度用在线学的
   骨长）：静态偏移记不住挥臂，动态外推记不住屈肘，两者各占一半时左右腕补位误差
   差异最小（A/B 见 eval_pose_filter.py）。
8. 不可见点冻结：conf 归零的关节（既没实测也没补位）输出坐标保持上一帧不变。否则
   外推到天边的点会在"重新可见"的那一帧变成几十个身高的跳变，既污染统计也骗评估。

3D 扩展：把 kpts 换成世界坐标、把像素身高换成米制比例即可复用同一状态机；骨长约束
天然以米为单位，足底触地/关节角合法域需要逐帧单应矩阵，留到 3D 层做（状态里保留了
每个关节的观测/预测标志，可直接接）。

参数含义与调参建议（全部在 PoseFilterConfig；数值都是 960 宽、30fps 实测出来的）
- fps：必须等于真实帧率，否则所有"多少帧"的语义（0.5s 外推、0.2s 恢复）都会错。
- min_cutoff / beta：抗抖 ↔ 跟手的一对旋钮。min_cutoff 越小，静止越稳、跟手越"粘"；
  beta 越大，快速动作滞后越小、毛刺越多。默认 2.0/0.35 → 抖动 1.7%/帧²、滞后 1 帧。
- hi_gate / lo_gate：迟滞门限。误检多、模型抖得厉害就往上抬；小目标丢点多就往下压。
  head_obs_conf 只作用于头部 5 点：RTMO 的眼/耳 conf 中位数只有 0.10-0.20，按普通
  门限喂进去只会让头抖，所以头部单独用 0.25 并交给先验补全。
- max_predict：单关节外推上限（15 帧 = 0.5s）。再长基本是瞎猜，不如让点消失。
- max_extrap：外推点离最后一次实测的最远距离（身高）。快攻/扣篮多的素材可放到
  0.35，静态战术片段可收到 0.15，防止遮挡期"手臂飞出去"。
- reset_gap：帧号跳变超过此值直接清状态（拖进度条/换片段）。
- reacquire_frames / reacq_step：遮挡恢复的过渡窗口与每帧追平速度。追平速度 =
  reacq_step × fps = 3.6 身高/秒，0.3 身高误差 ≈ 0.08s 追完，看不出一帧跳变。
  hold 期锁尺度后，恢复首帧位移由 20.7% 降到 9.9%（≤12% 目标）。
- max_track_age / min_confirm：人体级维持帧数 / 最小确认帧数。观众席（坐着的人）
  大量误检进骨架时，先减少 max_track_age、再增大 min_confirm。
- bone_tol / bone_gain：骨长校正容差与强度。姿态语义（手臂方向）被拉歪了就调小
  bone_gain；短臂/透视压缩明显就放宽 bone_tol。
- prior_min_obs / prior_learn：父关节相对偏移的学习门槛与 EMA 增益（越大学得越快、
  越容易被单帧坏姿态污染）。prior_rigid_miss / prior_limb_miss：先验能顶多少帧，
  头部这种刚性关系可以顶 2s，手腕这种末端最好 ≤0.5s。
- adapt_static / adapt_gain / adapt_v0：静止自适应（默认关）。复核（120 帧/12 人）：
  开＝抖动 1.72%→1.68%（静止子集 0.85%→0.82%），代价是跟随滞后 4.63%→4.72%、
  时移残差 3.45%→3.48%。收益 <3%、代价可测，维持默认关。
- hold_rigid / scale_ease / freeze_invisible：人体级遮挡三件套（默认全开）。关掉
  hold_rigid 退回逐关节外推：遮挡期骨架平均误差 11.0%→11.5~12.6% 身高、预测框
  IoU 0.61→0.53。freeze_invisible 关掉会让"重见"的关节从外推位置一帧跳到补位位置
  （单帧 >100% 身高），统计和观感都会坏。
- tip_dir_prior / tip_dir_w / torso_carry：末端补位与身体坐标系。注入式单关节遮挡
  A/B（左右腕各 80 帧）：旧行为 左 10.7% / 右 6.2%（差 4.5pp）→ 默认 8.0% / 6.3%
  （差 1.7pp）。主要贡献来自 torso_carry（肩线过短的帧占 20%，旧行为直接放弃先验），
  tip_dir_w=0.5 略优于纯静态（8.1%）与纯方向外推（8.7%）。

调参速查：抖动大 → 降 min_cutoff；跟手差 → 升 beta；遮挡期乱飞 → 降 max_extrap 或
max_predict；恢复弹跳 → 降 reacq_step；头/腕常年没点 → 调 head_obs_conf、prior_min_obs；
骨架整体缩放抖 → 检查 height/框高 EMA（st.h），别让下游用单帧框高。
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
# 末端的动态先验链：腕挂肘再挂肩，踝挂膝再挂髋；TIP_BONE 是"父→子"骨在 BONES 里的下标
TIP_PARENT = np.zeros(17, dtype=np.intp)
TIP_GRAND = np.zeros(17, dtype=np.intp)
TIP_BONE = np.zeros(17, dtype=np.intp)
TIP_LEN_RATIO = np.ones(17)   # 子骨/父骨的名义长度比（在线骨长还没学出来时的兜底）
TIP_PARENT[TIPS] = (7, 8, 13, 14)
TIP_GRAND[TIPS] = (5, 6, 11, 12)
TIP_BONE[TIPS] = (1, 3, 7, 9)          # (7,9) (8,10) (13,15) (14,16)
TIP_LEN_RATIO[TIPS] = (0.785, 0.785, 1.004, 1.004)  # 前臂/上臂、小腿/大腿（人体测量比例）
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
    # 静止自适应（默认关）：A/B 见 eval_pose_filter.py，静态抖动 -6% 但快速动作误差 +0.1pp，
    # 收益太小不值得改变线上手感，留作可选项
    adapt_static: bool = False
    adapt_gain: float = 0.7      # 静止时 min_cutoff 的下调比例（0.7 = 2.0Hz→0.6Hz）
    adapt_v0: float = 0.15       # 自适应速度尺度（身高/秒）：只有比这更慢的关节才算"静止"
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
    reacq_step: float = 0.12     # 重捕获过渡期每帧最多追多少身高（防"弹回"）
    max_extrap: float = 0.25     # 外推点离最后一次实测最远多少个身高（防遮挡期飘飞）
    reacq_min_miss: int = 2      # 丢失多少帧后重现才当成"重捕获"（1 帧抖动走常规滤波）
    bone_tol: float = 0.15       # 骨长突变容差（15% 以内不校正）
    bone_gain: float = 0.35      # 骨长校正强度（只走 35%，避免破坏姿态）
    mirror_min_miss: int = 6     # 连续丢失多少帧后才允许镜像补全（~0.2s）
    mirror_torso: bool = True    # 镜像轴：True=躯干中线（身体系），False=bbox 中心（旧行为）
    reset_gap: int = 12          # 帧号跳变超过此值直接清状态（拖进度条/换片段）
    torso_fallback: bool = True  # 髋部缺失（半身目标）时用 0.26×框高 + 画面朝上兜底
    # ── 人体级遮挡 hold：锁尺度 + 刚体平移 ─────────────────────
    hold_rigid: bool = True      # hold 期骨骼整体刚体平移（关掉＝旧的逐关节外推）
    scale_ease: bool = True      # 恢复后身高参考按 reacquire_frames 帧渐变回实测
    freeze_invisible: bool = True  # conf=0 的关节坐标冻结在上一帧（防"重见"时大跳）
    tip_dir_w: float = 0.5       # 末端动态先验权重（肘/膝方向外推 vs 静态学习偏移）
    tip_dir_prior: bool = True   # 末端用"父亲骨方向 + 在线骨长"外推兜底
    torso_carry: bool = True     # 肩线在图像上过短时沿用上一帧身体坐标系（关掉＝旧行为）
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
                 "last", "disp",
                 "off", "off_seen", "video_id", "model", "schema",
                 "holding", "hold_pose", "hold_shift", "hold_box", "h_target", "h_ease",
                 "tu", "tv")

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
        self.last = self.pos.copy()   # 每个关节最后一次实测位置（外推距离上限的锚点）
        self.disp = self.pos.copy()   # 上一帧真正输出到画面的位置（含先验/镜像补位）
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
        # 人体级 hold（锁尺度）
        self.holding = False
        self.hold_pose = np.zeros((n, 2))  # hold 开始时的输出姿态（形状/尺度冻结）
        self.hold_shift = np.zeros(2)      # hold 期累计刚体平移
        self.hold_box = self.bbox.copy()   # hold 开始时冻结的框（尺度不动）
        self.h_target = h                  # 恢复后要渐变到的身高参考
        self.h_ease = 0                    # 身高参考渐变剩余帧数
        # 最近一次有效的身体坐标系（侧身时肩线退化，用它保持朝向连续）
        self.tu: np.ndarray | None = None
        self.tv: np.ndarray | None = None
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
        self._tip_mask = np.zeros(17, dtype=bool)
        self._tip_mask[TIPS] = True
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

    def forget(self, track_ids) -> None:
        """删除已确认离开有效区域的轨迹，阻止下一帧再次外推出场外骨架。"""
        with self._lock:
            for tid in track_ids:
                self._tracks.pop(tid, None)

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
        observed = bool(np.isfinite(kpts[:, 0]).any())
        if observed and st.holding:
            # 人体级遮挡结束：身高参考从"冻结值"渐变到实测值。直接硬切会让骨架在恢复
            # 那一帧整体缩放（远处小球员尤其明显），渐变 6 帧肉眼看不出。
            st.holding = False
            if cfg.scale_ease:
                st.h_target = h
                st.h_ease = max(int(cfg.reacquire_frames), 1)
        # 身高参考做 EMA：单帧框抖动会直接变成整体缩放抖动，不能每帧硬切
        if st.h <= 0:
            st.h = h
        elif st.h_ease > 0:
            st.h += (st.h_target - st.h) / st.h_ease
            st.h_ease -= 1
        else:
            st.h = st.h * 0.85 + h * 0.15
        hr = st.h

        # 框速度：人体级遮挡时靠它维持 track。**尺度分量恒为 0**：hold 期宽高一律沿用
        # 丢失前那一帧（人不会在 0.3s 内变大变小），旧实现把宽高也做了速度预测，
        # 框越飘越大，恢复时骨架跟着弹。
        prev_box = st.bbox
        st.bbox = np.asarray(box, dtype=np.float64)
        # 刚体平移前馈：骨架直接继承当前帧框位移，One Euro 只平滑相对框的姿态形变。
        # 否则匀速移动时整副骨架也被低通，稳定滞后约 v*(1-alpha)/alpha（合成 1.2 身高/秒
        # 为 5.8% 身高）。首帧和整人重捕获没有可比的上一实测框，不能补这一次位移。
        body_shift = np.zeros(2)
        if st.n_obs:
            body_shift = 0.5 * ((st.bbox[:2] + st.bbox[2:]) - (prev_box[:2] + prev_box[2:]))
        if st.age == 0:
            st.bbox_vel[:2] = 0.6 * st.bbox_vel[:2] + 0.4 * ((st.bbox[:2] - prev_box[:2]) / gap)
            st.bbox_vel[2:] = 0.0

        obs_xy = kpts[:, :2]                     # 像素
        conf = kpts[:, 2]
        finite = np.isfinite(obs_xy).all(1)
        vmax = cfg.max_vel * hr               # 像素/帧

        # 门控带迟滞：已见过的关节用低阈值维持，没见过的用高阈值进入
        valid = finite & (conf >= np.where(st.seen, self._lo, self._hi))
        steady = valid.all() and not st.miss.any()

        if steady:
            # 常见路径（全部关节连续观测）：只做 One Euro，别为分支付 numpy 开销
            self._smooth(st, obs_xy, hr, gap, dt, body_shift=body_shift)
            st.rec[:] = cfg.reacquire_frames
            st.conf[:] = conf
            st.out[:] = st.pos
            st.last[:] = st.pos
            st.disp[:] = st.pos
            st.seen[:] = True      # 稳态帧也必须记"见过"：否则首次即全可见的 track
            st.obs = valid         # 永远被当成 first，reacquire 混合与镜像冻结都不会生效
            self._bone_constraint(st, valid)
            if cfg.prior_on:
                self._prior_learn(st)
            st.frame = frame
            return

        first = valid & ~st.seen
        reacq = valid & st.seen & (st.miss >= cfg.reacq_min_miss)  # 遮断多帧后重现：预测已不可信

        # 1) 首次观测 / 重新捕获：滤波器状态直接重置到测量值
        #    预测漂了 0.2s 之后，任何"慢慢收敛"都会让骨架拖在真实位置后面，
        #    所以状态立刻归位，视觉上的过渡交给下面的输出混合去做。
        reset = first | reacq
        if reset.any():
            gap_to_prev = np.maximum(st.miss[reset] + 1, 1).astype(np.float64)
            st.vel[reset] = np.clip((obs_xy[reset] - st.pos[reset] - body_shift) / gap_to_prev[:, None], -vmax, vmax)
            st.pos[reset] = obs_xy[reset]
            st.last[reset] = obs_xy[reset]
            st.dxhat[reset] = st.vel[reset] * cfg.fps
            st.seen[reset] = True
            st.vel[first] = 0.0
            st.dxhat[first] = 0.0

        # 2) 丢失关节：恒速外推 + 速度衰减，超过 max_predict 就定住（不再越漂越远）
        lost = ~valid
        if lost.any():
            step = min(gap, 3)
            # vel 存相对框速度；丢失关节也随当前帧框一起移动，再外推自身形变。
            st.pos[lost] += body_shift
            st.pos[lost] += st.vel[lost] * step
            st.vel[lost] *= cfg.vel_damp ** step
            st.miss[lost] += gap
            st.rec[lost] = 0
            dead = st.miss > cfg.max_predict
            st.vel[dead] = 0.0
            # 位移上限：以"最后一次实测位置"为锚，外推最远 max_extrap 个身高。恒速外推在
            # 0.5s 量级上很容易跑飞（手一挥就飘出去半个人），这是"遮挡期内骨架不飞"的兜底。
            li = np.nonzero(lost)[0]
            off = st.pos[li] - st.last[li]
            dist = np.sqrt(off[:, 0] ** 2 + off[:, 1] ** 2)
            cap = cfg.max_extrap * hr
            far = dist > cap
            if far.any():
                st.pos[li[far]] = st.last[li[far]] + off[far] * (cap / dist[far])[:, None]

        # 3) 持续观测关节：One Euro（速度自适应截止频率）/ 恒速卡尔曼（对比用）
        upd = valid & ~reset
        if upd.any():
            self._smooth(st, obs_xy, hr, gap, dt, mask=upd, body_shift=body_shift)

        if valid.any():
            st.miss[valid] = 0
            st.rec[valid] += 1
            st.conf[valid] = conf[valid]
            st.last[valid] = st.pos[valid]
            st.seen[valid] = True

        # 4) 骨长约束（轻度比例校正，不改骨骼方向）
        self._bone_constraint(st, valid)

        # 5) 显示位置：正常帧直接取状态。观测恢复后 reacquire_frames 帧内做限速过渡
        #    （每帧最多走 reacq_step 个身高）：状态本身已经归位，渲染的是"从补位/外推点
        #    平滑追到实测"的轨迹——0.25 身高误差 2-3 帧追完（<0.1s），不会一帧弹半个身子。
        np.copyto(st.out, st.pos)
        conv = valid & ~first & (st.rec < cfg.reacquire_frames)
        if conv.any():
            idx = np.nonzero(conv)[0]
            delta = st.pos[idx] - st.disp[idx]
            dist = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)
            maxstep = cfg.reacq_step * hr
            far = dist > maxstep
            if far.any():
                sc = (maxstep / np.maximum(dist[far], 1e-6))[:, None]
                st.out[idx[far]] = st.disp[idx[far]] + delta[far] * sc

        st.obs = valid
        if cfg.prior_on:
            self._prior_learn(st)
        st.frame = frame

    def _smooth(self, st: _Track, obs_xy: np.ndarray, hr: float, gap: int, dt: float,
                mask: np.ndarray | None = None, body_shift: np.ndarray | None = None) -> None:
        """观测关节的位置更新（One Euro 或恒速卡尔曼），含速度上限。

        body_shift 是当前帧框中心的刚体位移；滤波状态由“绝对位置 + 绝对速度”改为
        “当前位置 + 相对速度”，因此持续观测和丢失外推都使用同一坐标约定。
        """
        cfg = self.cfg
        shift = np.zeros(2) if body_shift is None else body_shift
        x = obs_xy if mask is None else obs_xy[mask]
        prev = st.pos if mask is None else st.pos[mask]
        vel = st.vel if mask is None else st.vel[mask]
        if cfg.mode == "kalman":
            pred = prev + vel * gap + shift
            resid = x - pred
            new = pred + cfg.kalman_k * resid
            vel = vel + cfg.kalman_b * resid / gap
            dxhat = vel * cfg.fps
        else:
            dx = (x - prev - shift) / dt
            a_d = _alpha_s(cfg.d_cutoff, dt)
            dxhat = a_d * dx + (1.0 - a_d) * (st.dxhat if mask is None else st.dxhat[mask])
            # 速度换算成"身高/秒"再算截止频率，参数与远近/分辨率无关
            spd = np.abs(dxhat) / hr
            if cfg.adapt_static:
                # 静止/慢走时把截止频率压到 min_cutoff*(1-gain)：这正是肉眼最容易被抖到的
                # 时刻；一旦动起来（spd >= adapt_v0）指数项归零，回到原 min_cutoff + beta*spd，
                # 快速动作的滞后一点都不增加。
                cutoff = cfg.min_cutoff * (1.0 - cfg.adapt_gain * np.exp(-spd / cfg.adapt_v0)) \
                    + cfg.beta * spd
            else:
                cutoff = cfg.min_cutoff + cfg.beta * spd
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
        """整框丢失：框按速度**平移**（宽高冻结），骨骼按刚体平移维持。

        旧的逐关节恒速外推在 0.3s 尺度上会让每根骨各自飞，骨长能拉长 20%，
        恢复瞬间整副骨架弹回。这里改成：hold 开始时把输出姿态快照下来，之后只
        累加一个整体平移量（框速度，逐帧衰减，位移上限 max_extrap 个身高），
        关节速度清零交给 _update 的丢失分支只记 miss，不再改写位置。
        """
        cfg = self.cfg
        step = min(gap, 3)
        shift = st.bbox_vel[:2] * step
        if cfg.hold_rigid:
            if not st.holding:
                st.holding = True
                st.hold_pose[:] = st.disp          # 形状/尺度：丢失前的最后一帧输出
                st.hold_shift[:] = 0.0
                st.hold_box[:] = st.bbox
            st.hold_shift += shift
            cap = cfg.max_extrap * max(st.h, 1.0)
            d = float(math.hypot(st.hold_shift[0], st.hold_shift[1]))
            if d > cap:                            # 遮挡久了人可能停下/变向，别让框飞出去
                st.hold_shift *= cap / d
            st.pos[:] = st.hold_pose + st.hold_shift
            st.last[:] = st.pos                    # 锚点跟着走，避免再被 max_extrap 拉回
            st.vel[:] = 0.0
            box = np.array([st.hold_box[0] + st.hold_shift[0], st.hold_box[1] + st.hold_shift[1],
                            st.hold_box[2] + st.hold_shift[0], st.hold_box[3] + st.hold_shift[1]])
        else:
            box = st.bbox + st.bbox_vel * step
        self._update(st, np.full((17, 3), np.nan), box, frame, gap)
        st.bbox_vel[:2] *= 0.95 ** gap             # 遮挡久了人可能停下来，速度慢慢收敛
        st.bbox_vel[2:] = 0.0
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
        if nu >= 0.06 * st.h:
            u = u / nu
            v = np.array([-u[1], u[0]])
            st.tu = u
            st.tv = v
        elif st.tu is not None and self.cfg.torso_carry:
            # 侧身/正对镜头轴：肩线在图像上只剩几个像素，u 的噪声会被放大成 90° 旋转。
            # 朝向是慢变量，沿用上一帧有效的身体坐标系（实测占 20% 的帧，左腕补位
            # 误差一半以上来自这里）。
            u, v = st.tu, st.tv
        elif not self.cfg.torso_carry:
            return None                  # A/B 用：旧行为＝肩线太短就整帧放弃先验
        else:
            u = np.array([1.0, 0.0])
            v = np.array([0.0, 1.0])
        scale = 0.0
        if st.obs[11] and st.obs[12]:
            hip = (o[11] + o[12]) * 0.5
            scale = float(math.hypot(neck[0] - hip[0], neck[1] - hip[1]))
            if scale >= 0.10 * st.h:
                if v @ (hip - neck) < 0.0:      # v 永远指向身体下方
                    v = -v
                    st.tv = v
                return neck, np.array([scale, u[0], u[1], v[0], v[1]])
        # 半身目标：没有髋就没法定向，按"画面里头朝上"的常规视角处理（只翻 v，不动 u，
        # 否则左右眼/耳会互换）
        if not self.cfg.torso_fallback:
            return None
        if v[1] < 0.0 and st.tu is None:
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
        idx = np.nonzero(fill)[0]
        prior = anchor[PARENT_OF[idx]] + st.off[idx] * scale
        if cfg.tip_dir_prior and cfg.tip_dir_w > 0.0:
            # 末端动态先验：腕 = 肘 + 前臂长 × (肘-肩) 方向，踝同理。
            # 静态学习偏移记的是"平均手臂姿势"，挥臂时全错；方向外推记的是"手臂此刻
            # 指向"，屈肘时全错。两者各占一半，实测左右腕补位误差最接近、总体最优。
            loc = np.nonzero(self._tip_mask[idx] & par_ok[idx] & st.obs[PARENT_OF[idx]])[0]
            if loc.size:
                gi = idx[loc]
                par, gp, bi = TIP_PARENT[gi], TIP_GRAND[gi], TIP_BONE[gi]
                d = o[par] - o[gp]
                dl = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1])
                ok = (dl > 0.08 * st.h) & st.obs[gp]
                if ok.any():
                    ll = np.where(st.bone_seen[bi] & (st.bone[bi] > 0),
                                  st.bone[bi] * st.h, dl * TIP_LEN_RATIO[gi])
                    dirv = d / np.maximum(dl, 1e-6)[:, None]
                    pred = o[par] + dirv * ll[:, None]
                    wd = cfg.tip_dir_w
                    sub = loc[ok]
                    prior[sub] = wd * pred[ok] + (1.0 - wd) * prior[sub]
        w = np.clip(1.0 - st.miss[idx].astype(np.float64) / cfg.prior_blend, 0.0, 1.0)
        w[rigid[idx]] = 0.0                             # 刚性关节不走外推，直接取先验
        kpts[idx, :2] = w[:, None] * o[idx] + (1.0 - w[:, None]) * prior
        kpts[idx, 2] = CONF_PRIOR

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
        # 仍然不可见的关节：坐标冻结在上一帧输出。0.5s 外推 + 再被先验补出来，
        # 两次跳变叠在一起能到几十个身高；这个点本来就不画，冻住即可。
        if cfg.freeze_invisible:
            vis = kpts[:, 2] > 0.0
            if not vis.all():
                kpts[~vis, :2] = st.disp[~vis]
        st.disp[:] = kpts[:, :2]
        return schema.Pose2D(
            video_id=st.video_id, frame=st.frame, track_id=st.tid,
            kpts=kpts.tolist(), schema=st.schema, model=st.model,
            height_px=float(hr),
            occluded=bool((~st.obs).any()),
        )

    def _mirror_fill(self, st: _Track, kpts: np.ndarray) -> None:
        """左右镜像补全：只补"状态已死"且丢失足够久的关节，补出的点不回流滤波。

        镜像轴优先用**躯干中线**（颈→髋中点，身体坐标系）：bbox 中心只有在球员正对
        镜头、双臂对称时才等于人体对称面，侧身/单臂伸展时按框中心镜像会把点补到
        身体外面（这正是左腕补位误差 12% 的主要来源）。拿不到躯干时退回框中心。
        """
        cfg = self.cfg
        conf = kpts[:, 2]
        dead = (conf <= CONF_PRED * 1.5) & (st.miss >= cfg.mirror_min_miss) & ~st.obs
        if not dead.any():
            return
        # 左缺右可信 → 补左；右缺左可信 → 补右（conf 阈值保证不会两边互推）
        fix_l = dead[MIRROR_L] & (conf[MIRROR_R] >= 0.6)
        fix_r = dead[MIRROR_R] & (conf[MIRROR_L] >= 0.6)
        if not (fix_l.any() or fix_r.any()):
            return
        torso = self._torso(st) if cfg.mirror_torso else None
        if torso is not None:
            neck, ref = torso
            u = np.array([ref[1], ref[2]])   # 左肩→右肩，身体横轴（单位向量）
            # 关于"过颈点、沿身体纵轴"的平面对称：减掉两倍横轴分量
            def _mirror(src: np.ndarray) -> np.ndarray:
                d = src - neck
                return src - 2.0 * (d @ u)[:, None] * u
        else:
            cx = 0.5 * (st.bbox[0] + st.bbox[2])
            def _mirror(src: np.ndarray) -> np.ndarray:
                out = src.copy()
                out[:, 0] = 2.0 * cx - src[:, 0]
                return out
        if fix_l.any():
            li, ri = MIRROR_L[fix_l], MIRROR_R[fix_l]
            kpts[li, :2] = _mirror(kpts[ri, :2])
            kpts[li, 2] = CONF_MIRROR
        if fix_r.any():
            li, ri = MIRROR_L[fix_r], MIRROR_R[fix_r]
            kpts[ri, :2] = _mirror(kpts[li, :2])
            kpts[ri, 2] = CONF_MIRROR


__all__ = ["PoseFilter", "PoseFilterConfig", "CONF_PRED", "CONF_MIRROR", "CONF_PRIOR",
           "CONF_TEMPLATE", "CONF_TIP"]