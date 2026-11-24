"""球衣号码识别：躯干裁片 → 两位数字分类 → 轨迹级时序融合。

需求与验收见 docs/jersey_number_detection.md。这里的三个设计决定，理由如下：

1. 用"两位数字分类"（推荐 B）而不是 CRNN。号码只有 0–99，两位独立分类的类别数只有
   10+10，合成数据就能收敛，导出 ONNX 是定长 22 维输出，TensorRT profile 与 RTMO 一样
   好写；CRNN 要处理变长序列与 CTC 解码，实时链路上多一套状态机，收益不明显。
   十位头额外带一个 NONE 类（索引 0），用来表达"个位数"，因此 10×10 的复合类别与
   0–99 一一对应，没有二义性。
2. 输出必须是时序融合后的稳定结论。逐帧 OCR 直出的典型症状是同一个人在 23/28/23 之间
   闪（球衣褶皱、运动模糊、单帧看不清），而号码是整场比赛都不变的强先验，值得用轨迹
   投票换稳定性。融合在概率空间做 EMA：至少累计 N 次观测且后验过阈值才发布，发布后
   只允许在高置信新证据下改一次，改完就锁死（防闪烁 ≤1 次）。
3. 读不出来就留 None，绝不猜。宁可少几个号码，也不能给出错号——错号会污染得分/犯规
   归属，且下游没有任何手段发现它是错的。

运行时只依赖 numpy + OpenCV + ONNX Runtime（与 bsk/perceive/persons.py 一致，torch 只在
训练侧出现）。权重缺失或 onnxruntime 不可用时整个模块退化成 no-op：不写 jersey 字段、
不影响现有实时链路。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path("data/models")
DEFAULT_WEIGHTS = MODEL_DIR / "jersey_small.onnx"

# 模型输入尺寸（H×W）。躯干裁片接近正方形，不需要 CRNN 那种 64×160 的细长条；
# 改这里必须同时重训，尺寸会写进 sidecar，加载时对不上直接报错而不是静默错位。
INPUT_H, INPUT_W = 64, 96
N_TENS = 10          # 0 = NONE（个位数），1..9 = 十位数字
N_UNITS = 10         # 个位数字
LOGIT_LEN = N_TENS + N_UNITS + 2   # 十位 10 + 个位 10 + (有号码/空白) 2
IDX_TENS = slice(0, N_TENS)
IDX_UNITS = slice(N_TENS, N_TENS + N_UNITS)
IDX_BLANK = slice(N_TENS + N_UNITS, LOGIT_LEN)
N_NUMBERS = 100      # 号码值域 0..99

# 复合类别 (t, u) → 号码：t==0 表示个位数，取 u；否则 t*10+u。双射，无碰撞。
_COMPOSITE_TO_NUM = np.array(
    [u if t == 0 else t * N_UNITS + u for t in range(N_TENS) for u in range(N_UNITS)],
    dtype=np.intp,
)

# COCO-17 肩点索引：躯干裁片与侧身/背影判定都要用
_L_SHOULDER, _R_SHOULDER = 5, 6


@dataclass
class JerseyConfig:
    """识别与融合的旋钮。默认值按 docs 的需求取，实际数值在 eval_jersey.py 上标定。"""

    # ── 裁片几何（bbox 左上为原点的比例，见需求 §3.1）
    x_inset: float = 0.15          # 左右各收 15%，削掉手臂与相邻球员
    y_top: float = 0.10            # 从头顶往下 10% 开始（跳过脖子/领口上方的模糊区）
    y_bot: float = 0.55            # 到 55% 结束（号码通常压在胸腹之间）
    min_crop_w: int = 18           # 低于这个像素尺寸不做识别：号码笔画已经没了
    min_crop_h: int = 24

    # ── 可信度门控（不满足就丢弃本次观测，不投票）
    min_kpt_conf: float = 0.30     # 肩点可用判定
    min_shoulder_frac: float = 0.16  # 肩宽 / 框宽：侧身、背影时肩宽塌缩

    # ── 采样（需求 §3.1：不是每帧都跑）
    every_n: int = 5               # 每条轨迹每 5 帧取一次（30fps 下 ≈ 0.17s）
    max_crops: int = 6             # 单帧最多识别的人数，按框面积从大到小取
    batch_size: int = 8            # 一次前向的裁片数上限

    # ── 时序融合（需求 §3.2）
    fps: float = 30.0
    ema_alpha: float = 0.40        # 单次观测在融合后验里的权重
    min_obs: int = 3               # 至少累计这么多次观测才发布
    min_conf: float = 0.60         # 发布门限（融合后验）
    max_switch: int = 1            # 整条轨迹允许的号码切换次数（防闪烁）
    gap_reset_s: float = 1.0       # 轨迹丢失超过这么久，状态清零

    # ── 冲突处理（需求 §3.3）
    conflict_penalty: float = 0.5  # 同队同号码：置信度打折并记日志

    # ── 运行
    weights: str | Path = DEFAULT_WEIGHTS
    device: str = "auto"           # auto → TensorRT/CUDA/CPU 依次尝试
    use_trt: bool = True           # 复用 persons.py 的 TRT 会话；失败自动降级


@dataclass
class CropResult:
    """一次躯干裁片的产出与判定。ok=False 时 reason 说明为什么这次观测不可信。"""

    ok: bool
    reason: str = ""
    crop: np.ndarray | None = None      # (h, w, 3) uint8, RGB
    score: float = 1.0                  # 几何可信度（0–1），用于同帧内排序

    def __bool__(self) -> bool:  # pragma: no cover - 便捷写法
        return self.ok


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


# ── 裁片与预处理 ────────────────────────────────────────────

def torso_bbox(bbox, cfg: JerseyConfig = JerseyConfig()) -> tuple[int, int, int, int] | None:
    """人体框 → 躯干（号码区）框，整数像素并夹在合理范围。返回 None 表示太小。"""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None
    cx0 = int(round(x0 + cfg.x_inset * w))
    cx1 = int(round(x1 - cfg.x_inset * w))
    cy0 = int(round(y0 + cfg.y_top * h))
    cy1 = int(round(y0 + cfg.y_bot * h))
    if cx1 - cx0 < cfg.min_crop_w or cy1 - cy0 < cfg.min_crop_h:
        return None
    return cx0, cy0, cx1, cy1


def crop_torso(rgb: np.ndarray, bbox, kpts: np.ndarray | None = None,
               cfg: JerseyConfig = JerseyConfig()) -> CropResult:
    """按躯干区域裁片，并判断这次观测值不值得采信。

    背影/侧身不采信的依据是肩宽塌缩：正面时肩宽约占框宽 0.45–0.6，侧身降到 0.2 以下，
    而号码此时基本被手臂或身体侧面挡住。判据太严会直接没号码，宁可用 min_conf 卡发布。
    """
    box = torso_bbox(bbox, cfg)
    if box is None:
        return CropResult(False, "too_small")
    cx0, cy0, cx1, cy1 = box
    H, W = rgb.shape[:2]
    cx0, cy0 = max(cx0, 0), max(cy0, 0)
    cx1, cy1 = min(cx1, W), min(cy1, H)
    if cx1 - cx0 < cfg.min_crop_w or cy1 - cy0 < cfg.min_crop_h:
        return CropResult(False, "out_of_frame")
    crop = rgb[cy0:cy1, cx0:cx1]

    bw = max(float(bbox[2] - bbox[0]), 1.0)
    if kpts is None or len(kpts) <= _R_SHOULDER:
        # 没有关键点时不否决：准确档骨架一般都有，缺了也不该让号码功能整体停摆
        return CropResult(True, "", crop, min(1.0, (cx1 - cx0) / 96.0))
    k = np.asarray(kpts, dtype=np.float64)
    l, r = k[_L_SHOULDER], k[_R_SHOULDER]
    if l[2] < cfg.min_kpt_conf or r[2] < cfg.min_kpt_conf:
        return CropResult(False, "shoulder_low_conf", crop)
    sw = float(np.hypot(l[0] - r[0], l[1] - r[1])) / bw
    if sw < cfg.min_shoulder_frac:
        return CropResult(False, "side_or_back", crop, sw)
    return CropResult(True, "", crop, min(1.0, sw / 0.45))


_CLAHE = None  # 复用同一个对象：createCLAHE 有内部状态分配，逐帧新建会拖慢实时链路


def gray_input(crop: np.ndarray, out_size: tuple[int, int] = (INPUT_W, INPUT_H)) -> np.ndarray:
    """裁片 → 灰度 + CLAHE + 缩放到模型尺寸的 uint8 (H,W)。

    这一步单独暴露出来，是为了让训练侧与推理侧共用同一条预处理链（bsk/train_jersey.py
    直接 import）：预处理不一致是最难查的 bug——不报错，只是线上精度莫名其妙掉一截。
    uint8 输出还能原样落盘，训练时不必把整份数据集展开成 float32 占内存。

    先转灰度 + CLAHE 再识别，是为了不把"球衣底色"当成特征学：本项目素材有青色主场也有
    白色客场，直接喂彩色的话模型很容易记住底色配对而不是数字笔画；灰度丢掉颜色后，
    对比度归一化负责把浅色/深色球衣拉到同一量级。
    """
    global _CLAHE
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    gray = _CLAHE.apply(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY))
    w, h = out_size
    interp = cv2.INTER_AREA if (gray.shape[1] > w or gray.shape[0] > h) else cv2.INTER_LINEAR
    return cv2.resize(gray, (w, h), interpolation=interp)


def preprocess(crops: list[np.ndarray], cfg: JerseyConfig = JerseyConfig()) -> np.ndarray:
    """裁片 → 模型输入 (N,3,H,W) float32，归一化到 [-1,1]。

    三通道复制是为了与常见主干（MobileNet 系第一层 3 输入通道）兼容，以后想上彩色
    分支不用改接口和权重输入契约。
    """
    out = np.empty((len(crops), 3, INPUT_H, INPUT_W), np.float32)
    for i, c in enumerate(crops):
        x = (gray_input(c).astype(np.float32) - 127.5) / 127.5
        out[i, 0] = x
        out[i, 1] = x
        out[i, 2] = x
    return out


def logits_to_dist(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """22 维 logits → (号码分布 (N,100), 空白概率 (N,))。

    十位与个位相乘而不是拼 100 类，是为了让"2"在 12/23/20 之间共享一个可迁移的笔画
    概念；合成数据里两位数字的表现本来就不一致，拆开训收敛快得多。
    """
    logits = np.asarray(logits, np.float32)
    if logits.ndim == 1:
        logits = logits[None, :]
    pt = _softmax(logits[:, IDX_TENS], axis=1)
    pu = _softmax(logits[:, IDX_UNITS], axis=1)
    pc = (pt[:, :, None] * pu[:, None, :]).reshape(logits.shape[0], -1)
    pnum = np.take_along_axis(pc, _COMPOSITE_TO_NUM[None, :].repeat(logits.shape[0], 0), axis=1)
    pb = _softmax(logits[:, IDX_BLANK], axis=1)
    return pnum * (1.0 - pb[:, 1:2]), pb[:, 1]


# ── 推理后端（ONNX，可选 TensorRT）────────────────────────────

class JerseyModel:
    """ONNX 会话封装。sidecar（同名 .json）记录输入尺寸与头部布局，防止权重和预处理错位。"""

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, device: str = "auto",
                 use_trt: bool = True) -> None:
        self.path = Path(weights)
        self.ep: str | None = None
        self.meta: dict = {}
        self.sess = None
        self.error: str | None = None
        self._n_pre = 0
        self._ms = 0.0
        if not self.path.exists():
            self.error = f"权重不存在: {self.path}"
            return
        side = self.path.with_suffix(".json")
        if side.exists():
            try:
                self.meta = json.loads(side.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover
                self.error = f"sidecar 解析失败: {exc}"
                return
            ih, iw = self.meta.get("input_hw", [INPUT_H, INPUT_W])
            if (int(ih), int(iw)) != (INPUT_H, INPUT_W):
                self.error = f"权重输入 {ih}x{iw} 与代码 {INPUT_H}x{INPUT_W} 不一致，请重训或改常量"
                return
        try:
            import onnxruntime  # noqa: F401
        except Exception as exc:
            self.error = f"onnxruntime 不可用: {exc}"
            return
        self._open(device, use_trt)

    def _open(self, device: str, use_trt: bool) -> None:
        import onnxruntime as ort

        _add_dll_dirs()
        if use_trt and device in ("auto", "cuda"):
            # 复用 RTMO 那套 TRT 会话构造；引擎首次构建要几十秒，所以失败一次就降级
            try:
                from bsk.perceive.persons import make_session_trt
                self.sess, self.ep = make_session_trt(
                    self.path, cache_dir=MODEL_DIR / "trt_cache")
                return
            except Exception:
                self.sess = None
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        provs = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if device in ("auto", "cuda") else ["CPUExecutionProvider"])
        try:
            self.sess = ort.InferenceSession(str(self.path), opts, providers=provs)
            self.ep = self.sess.get_providers()[0]
        except Exception as exc:
            self.error = f"会话创建失败: {exc}"

    @property
    def ready(self) -> bool:
        return self.sess is not None

    @property
    def mean_ms(self) -> float:
        """单次前向的平均耗时（毫秒），供验收报告用。"""
        return self._ms / self._n_pre if self._n_pre else 0.0

    def predict(self, batch: np.ndarray) -> np.ndarray:
        if self.sess is None:
            raise RuntimeError(self.error or "模型未就绪")
        t0 = time.perf_counter()
        name = self.sess.get_inputs()[0].name
        out = self.sess.run(None, {name: np.ascontiguousarray(batch, np.float32)})[0]
        self._ms += (time.perf_counter() - t0) * 1000.0
        self._n_pre += 1
        return np.asarray(out, np.float32).reshape(batch.shape[0], LOGIT_LEN)

    def reset_stats(self) -> None:
        self._ms = 0.0
        self._n_pre = 0


def _add_dll_dirs() -> None:
    """Windows 下 CUDA DLL 目录：与 persons.py 同一套处理，但这里失败不致命。"""
    if os.name != "nt":
        return
    try:
        import site
        from pathlib import Path as _P
        for sp in site.getsitepackages():
            for sub in (("torch", "lib"), ("nvidia", "cudnn", "bin"),
                        ("nvidia", "cuda_runtime", "bin")):
                p = _P(sp).joinpath(*sub)
                if p.is_dir():
                    try:
                        os.add_dll_directory(str(p))
                    except OSError:
                        pass
    except Exception:
        pass


# ── 单条轨迹的号码状态 ────────────────────────────────────────

@dataclass
class _TrackState:
    last_frame: int = -10**9
    post: np.ndarray | None = None      # (100,) 融合后的号码后验
    blank: float = 0.0
    obs: int = 0
    team: str | None = None
    jersey: int | None = None           # 已发布号码
    jersey_conf: float = 0.0
    switches: int = 0
    locked: bool = False

    def update(self, pnum: np.ndarray, pblank: float, n: int,
               cfg: JerseyConfig) -> tuple[int | None, float, int]:
        """吃进一次观测，返回 (当前发布号码, 置信度, 本次是否发生切换)。

        切换计数只在"已经发布过"之后再累加，所以首次发布不算闪烁。
        """
        if self.post is None:
            self.post = pnum.copy()
            self.blank = pblank
        else:
            a = cfg.ema_alpha
            self.post = a * pnum + (1.0 - a) * self.post
            self.blank = a * pblank + (1.0 - a) * self.blank
        self.obs += 1
        self.last_frame = n
        if self.locked:
            return self.jersey, self.jersey_conf, 0
        best = int(np.argmax(self.post))
        conf = float(self.post[best])
        if conf >= cfg.min_conf and self.obs >= cfg.min_obs:
            if self.jersey is None:
                self.jersey, self.jersey_conf = best, conf
                return best, conf, 0
            if best != self.jersey:
                if self.switches >= cfg.max_switch:
                    self.locked = True
                    return self.jersey, self.jersey_conf, 0
                self.switches += 1
                self.jersey, self.jersey_conf = best, conf
                if self.switches >= cfg.max_switch:
                    # 配额用完即锁：改完这一次就不许再改，否则"≤1 次切换"要靠后续
                    # 每帧都不出错才成立，实际素材上做不到
                    self.locked = True
                return best, conf, 1
        elif self.jersey is not None:
            self.jersey_conf = max(self.jersey_conf, conf)
        return self.jersey, self.jersey_conf, 0


# ── 对外主类 ────────────────────────────────────────────────

class JerseyRecognizer:
    """在已有 track_id 上挂稳定号码：采样躯干裁片 → 批量前向 → 轨迹级融合。

    状态按 (video_id, track_id) 存放，多视频/多后端不串号；所有逻辑只用当前帧与历史
    状态，不预取未来帧（后续要接直播流）。
    """

    def __init__(self, weights: str | Path | None = None, cfg: JerseyConfig | None = None,
                 device: str = "auto") -> None:
        self.cfg = cfg or JerseyConfig()
        path = Path(weights) if weights else Path(self.cfg.weights)
        self.model = JerseyModel(path, device=device, use_trt=self.cfg.use_trt)
        self._states: dict[tuple[str, int], _TrackState] = {}
        self._crop_ms = 0.0
        self._infer_ms = 0.0
        self._n_crop = 0
        self._published: dict[str, dict[tuple[str | None, int], int]] = {}
        self.events: list[dict] = []   # 发布/切换/冲突日志，评估与排障都看这里

    @property
    def ready(self) -> bool:
        return self.model.ready

    @property
    def unavailable_reason(self) -> str | None:
        return self.model.error

    # 兼容需求文档里的写法
    @property
    def enabled(self) -> bool:  # pragma: no cover
        return self.model.ready

    def reset(self, vid: str | None = None) -> None:
        if vid is None:
            self._states.clear()
            self._published.clear()
        else:
            for k in [k for k in self._states if k[0] == vid]:
                del self._states[k]
            self._published.pop(vid, None)

    def _log(self, kind: str, vid: str, tid: int, n: int, **kw) -> None:
        if len(self.events) < 2000:
            self.events.append({"kind": kind, "vid": vid, "track_id": tid, "frame": n, **kw})
        if kind == "conflict":
            print(f"[jersey] 冲突: vid={vid} track={tid} #{kw.get('jersey')} 已被 "
                  f"track={kw.get('other_track')} 占用，置信度打折")

    def recognize(self, vid: str, n: int, rgb, objects, poses=None) -> None:
        """就地写入 objects 里各 Tracked 的 jersey/jersey_conf（含时序融合）。

        poses 可选：给关键点是用来做侧身/背影门控的，线上传的是 Pose2D 列表。
        """
        if not self.model.ready:
            return
        cfg = self.cfg
        kpt_by_id: dict[int, np.ndarray] = {}
        if poses:
            for p in poses:
                if p.kpts:
                    kpt_by_id[p.track_id] = np.asarray(p.kpts, np.float64)

        # 采样节拍与门控都过了才占 batch 名额
        cands = [o for o in (objects or [])
                 if o.category == "player" and o.visible and o.bbox_px]
        # 老化/换队清零：画面里一个人都没有时也要跑，否则空场镜头后状态永远不清
        self._expire(vid, n, cands, cfg)
        if not cands:
            return

        picked: list[tuple[object, np.ndarray, float]] = []
        crops: list[np.ndarray] = []
        t0 = time.perf_counter()
        for o in cands:
            st = self._states.get((vid, o.track_id))
            if st is not None and st.jersey is not None:
                # 已发布的号每帧都要回写：只在观测帧写会让框标签一帧有一帧没（闪烁）
                o.jersey, o.jersey_conf = st.jersey, st.jersey_conf
            if st is not None and st.locked:
                continue            # 已锁死（切换配额用完），不再花算力
            if st is not None and (n - st.last_frame) < cfg.every_n:
                continue            # 没到采样节拍
            if st is None:
                st = self._states[(vid, o.track_id)] = _TrackState(team=o.team)
            st.last_frame = n       # 尝试过就推进节拍：读不出的帧同样占一次采样机会
            cr = crop_torso(rgb, o.bbox_px, kpts=kpt_by_id.get(o.track_id), cfg=cfg)
            if cr.ok and cr.crop is not None:
                picked.append((o, cr.crop, cr.score))
        self._crop_ms += (time.perf_counter() - t0) * 1000.0

        if not picked:
            return
        # 人多时按"裁片面积 × 几何可信度"取前 N：远端球员的号码本来就读不清，
        # 把算力留给近处真能读出的人
        picked.sort(key=lambda it: -(it[1].shape[0] * it[1].shape[1] * it[2]))
        picked = picked[:cfg.max_crops]
        self._n_crop += len(picked)

        for i in range(0, len(picked), cfg.batch_size):
            chunk = picked[i:i + cfg.batch_size]
            t1 = time.perf_counter()
            batch = preprocess([c for _o, c, _s in chunk], cfg)
            logits = self.model.predict(batch)
            self._infer_ms += (time.perf_counter() - t1) * 1000.0
            pnum, pblank = logits_to_dist(logits)
            for j, (o, _c, _s) in enumerate(chunk):
                self._apply(vid, o, n, pnum[j], float(pblank[j]), cfg)

    def _expire(self, vid: str, n: int, cands, cfg: JerseyConfig) -> None:
        alive = {o.track_id for o in cands}
        team_of = {o.track_id: o.team for o in cands}
        max_age = int(round(cfg.gap_reset_s * cfg.fps))
        for tid in [t for (v, t) in self._states if v == vid]:
            st = self._states[(vid, tid)]
            if tid not in alive and n - st.last_frame > max_age:
                del self._states[(vid, tid)]
                self._release(vid, tid, st)
            elif st.team is not None and team_of.get(tid) is not None and st.team != team_of[tid]:
                st.team = team_of[tid]
                self._release(vid, tid, st)
                st.post, st.obs, st.jersey, st.locked, st.switches = None, 0, None, False, 0

    def _release(self, vid: str, tid: int, st: _TrackState) -> None:
        reg = self._published.get(vid)
        if reg and st.jersey is not None:
            key = (st.team, st.jersey)
            if reg.get(key) == tid:
                del reg[key]

    def _apply(self, vid: str, o, n: int, pnum: np.ndarray, pblank: float,
               cfg: JerseyConfig) -> None:
        key = (vid, o.track_id)
        st = self._states.get(key)
        if st is None:
            st = self._states[key] = _TrackState(team=o.team)
        if pblank > 0.5:
            # 空白票也要进状态：背影/遮挡帧占多数，否则"读不出"被当成"没观测"，
            # 融合后验会一直停在早期的高分上
            pnum = np.zeros_like(pnum)
        before = st.jersey
        jersey, conf, switched = st.update(pnum, pblank, n, cfg)
        if switched:
            self._log("switch", vid, o.track_id, n, from_=before, to=jersey, conf=round(conf, 3))
        elif jersey is not None and before is None:
            self._log("publish", vid, o.track_id, n, jersey=jersey, conf=round(conf, 3),
                      obs=st.obs)
        if jersey is None:
            return
        if conf < cfg.min_conf:
            return
        reg = self._published.setdefault(vid, {})
        other = reg.get((o.team, jersey))
        if other is not None and other != o.track_id:
            conf *= cfg.conflict_penalty
            self._log("conflict", vid, o.track_id, n, jersey=jersey,
                      other_track=other, conf=round(conf, 3))
        else:
            reg[(o.team, jersey)] = o.track_id
        # 打折后的置信度也要写回状态：下一帧走的是"回写已发布号码"的路径，
        # 不写回去就等于把冲突惩罚悄悄抹掉了
        st.jersey_conf = float(conf)
        o.jersey = jersey
        o.jersey_conf = float(conf)

    # ── 统计（验收报告口径）
    def published_numbers(self, vid: str | None = None) -> dict[int, int]:
        """当前已发布的 track_id → 号码，评估与人工纠错面板都用这个入口。"""
        out = {}
        for (v, tid), st in self._states.items():
            if vid is None or v == vid:
                if st.jersey is not None:
                    out[tid] = st.jersey
        return out

    def stats(self) -> dict:
        return {
            "ep": self.model.ep,
            "ready": self.model.ready,
            "crops": self._n_crop,
            "crop_ms": round(self._crop_ms, 3),
            "infer_ms": round(self._infer_ms, 3),
            "forward_ms": round(self.model.mean_ms, 3),
            "tracks": len(self._states),
        }

    def reset_stats(self) -> None:
        self._crop_ms = self._infer_ms = 0.0
        self._n_crop = 0
        self.model.reset_stats()


# ── M1：裁片采集/标注辅助 ──────────────────────────────────────

def collect_crops(vid: str = "harden-55", frm: int = 1, nframes: int = 600, width: int = 960,
                  out_dir: str | Path = "jersey_samples", every_n: int = 3,
                  max_per_track: int = 6, cfg: JerseyConfig | None = None,
                  backend: str = "accurate") -> dict:
    """顺序解码 → 检测+跟踪 → 导出躯干裁片与清单（需求 §4.2 的真实标注数据来源）。

    产物：
      out_dir/crops/<vid>/f<帧号>_t<track_id>.png
      out_dir/manifest.jsonl   每张图的 frame/track_id/bbox/crop/尺寸/门控原因
      out_dir/labels.csv       给人工填 label 列的模板（frame,track_id,crop,label）
    模型只跑一遍就把 crop 与 track_id 对齐存下来，比事后重新解码便宜得多；
    manifest 里保留被门控丢弃的框（dropped=true），调 min_shoulder_frac 时能直接复盘。
    """
    from bsk import ingest  # 延迟导入：运行链路不该被 av 绑住
    from bsk.analyze import Tracker
    from bsk.store import Library

    cfg = cfg or JerseyConfig()
    out = Path(out_dir)
    (out / "crops" / vid).mkdir(parents=True, exist_ok=True)
    lib = Library()
    v = lib.get(vid)
    table = lib.table(vid)
    anchor = table.key_before(frm)
    size = (width, int(round(width * v.height / max(v.width, 1))))
    if backend == "accurate":
        from bsk.perceive.persons import Rtmo
        pose = Rtmo()
    else:
        from bsk.perceive.fast_pose import YoloV8Pose
        pose = YoloV8Pose(device="cuda", cache_dir=MODEL_DIR)
    tracker = Tracker()

    rows: list[dict] = []
    seen: dict[int, int] = {}
    last: dict[int, int] = {}
    n_drop = 0
    for idx, _pts, rgb in ingest.frames(v.path, start=anchor, size=size, stride=1,
                                        seek_us=table.pts_us[anchor - 1]):
        if idx < frm:
            continue
        if idx >= frm + nframes:
            break
        ppl = pose(rgb, thr=0.3)
        keep = []
        for p in ppl:
            bw, bh = p.bbox[2] - p.bbox[0], p.bbox[3] - p.bbox[1]
            if bh / max(bw, 1) < 0.6 or bh / max(bw, 1) > 5.0:
                continue
            if bh < max(36, rgb.shape[0] * 0.03) or bw < max(18, rgb.shape[1] * 0.015):
                continue
            if p.kpts is None or np.sum(p.kpts[:, 2] > 0.5) < 2:
                continue
            keep.append(p)
        ids = tracker.assign([p.bbox for p in keep])
        for p, tid in zip(keep, ids):
            if idx - last.get(tid, -10**9) < every_n:
                continue
            if seen.get(tid, 0) >= max_per_track:
                continue
            cr = crop_torso(rgb, p.bbox, kpts=p.kpts, cfg=cfg)
            rec = {"video_id": vid, "frame": int(idx), "track_id": int(tid),
                   "bbox": [float(x) for x in p.bbox], "det_conf": round(float(p.score), 4),
                   "team": None, "label": "", "dropped": not cr.ok, "reason": cr.reason}
            if cr.ok and cr.crop is not None:
                name = f"f{idx:05d}_t{tid}.png"
                cv2.imwrite(str(out / "crops" / vid / name),
                            cv2.cvtColor(cr.crop, cv2.COLOR_RGB2BGR))
                rec["crop"] = f"crops/{vid}/{name}"
                rec["crop_wh"] = [int(cr.crop.shape[1]), int(cr.crop.shape[0])]
                seen[tid] = seen.get(tid, 0) + 1
                last[tid] = idx
            else:
                n_drop += 1
            rows.append(rec)
        print(f"\r[collect] 帧 {idx}｜裁片 {sum(seen.values())}｜丢弃 {n_drop}", end="", flush=True)
    print()

    with open(out / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "labels.csv", "w", encoding="utf-8") as fh:
        fh.write("crop,frame,track_id,label\n")
        for r in rows:
            if not r["dropped"]:
                fh.write(f"{r['crop']},{r['frame']},{r['track_id']},\n")
    summary = {"vid": vid, "frames": nframes, "crops": sum(1 for r in rows if not r["dropped"]),
               "dropped": n_drop, "out": str(out)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"[collect] {summary}")
    return summary


__all__ = [
    "CropResult",
    "DEFAULT_WEIGHTS",
    "INPUT_H",
    "INPUT_W",
    "JerseyConfig",
    "JerseyModel",
    "JerseyRecognizer",
    "LOGIT_LEN",
    "collect_crops",
    "crop_torso",
    "gray_input",
    "logits_to_dist",
    "preprocess",
    "torso_bbox",
]


if __name__ == "__main__":
    # M1 采片入口：python -m bsk.perceive.jersey --vid harden-55 --from 400 --nframes 600
    import argparse

    ap = argparse.ArgumentParser(description="导出真实躯干裁片与标注清单（需求 §4.2 / M1）")
    ap.add_argument("--vid", default="harden-55")
    ap.add_argument("--from", dest="frm", type=int, default=1, help="起始帧号")
    ap.add_argument("--nframes", type=int, default=600)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--out", default="jersey_samples")
    ap.add_argument("--every-n", dest="every_n", type=int, default=3,
                    help="同一 track 最多每 N 帧采一张")
    ap.add_argument("--max-per-track", dest="max_per_track", type=int, default=6)
    ap.add_argument("--backend", default="accurate", choices=["accurate", "fast"])
    args = ap.parse_args()
    collect_crops(args.vid, args.frm, args.nframes, args.width, args.out,
                  args.every_n, args.max_per_track, backend=args.backend)
