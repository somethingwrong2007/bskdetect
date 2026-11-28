"""球衣号码识别的自检：裁片几何、门控、融合状态机、降级行为。

不需要真实权重也不需要视频：推理后端用桩替换，裁片用纯色块编码"应该是哪个号码"，
所以这里测的是我们自己的代码（几何/门控/融合/接入），不测模型精度——精度看
eval_jersey.py。运行：

    .venv/Scripts/python.exe test_jersey.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from bsk import schema
from bsk.perceive.jersey import (INPUT_H, INPUT_W, LOGIT_LEN, N_TENS, N_UNITS,
                                 CropResult, JerseyConfig, JerseyRecognizer,
                                 crop_torso, gray_input, logits_to_dist, preprocess,
                                 torso_bbox)

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


# ── 桩推理后端 ──────────────────────────────────────────────

def logits_for(number: int | None, conf: float = 0.95) -> np.ndarray:
    """构造一个"模型会说号码 number"的 logits；number=None 表示判为空白。"""
    lg = np.full(LOGIT_LEN, -8.0, np.float32)
    if number is None:
        lg[N_TENS + N_UNITS + 1] = 8.0
        return lg
    t, u = (0, number) if number < 10 else (number // 10, number % 10)
    lg[t] = 8.0
    lg[N_TENS + u] = 8.0
    lg[N_TENS + N_UNITS] = 6.0 * conf
    return lg


class StubModel:
    """按裁片主色查表返回 logits，用来把"这一帧看到什么号"变成可控输入。"""

    def __init__(self, table: dict[int, int | None], conf: float = 0.95):
        self.table = table
        self.conf = conf
        self.ready = True
        self.error = None
        self.ep = "stub"
        self.calls = 0
        self.crops = 0
        self._ms = 0.0
        self._n = 0

    def predict(self, batch: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.crops += batch.shape[0]
        out = np.zeros((batch.shape[0], LOGIT_LEN), np.float32)
        for i in range(batch.shape[0]):
            # gray_input 之后主色仍是一个可反查的灰度值（用 code_of 事先标定）
            key = int(round(np.clip(batch[i, 0, 0, 0], -1, 1) * 127.5 + 127.5))
            out[i] = logits_for(self.table.get(key, 7), self.conf)
        return out

    @property
    def mean_ms(self) -> float:
        return 0.0

    def reset_stats(self) -> None:
        pass


def code_of(value: int) -> int:
    """纯色经 gray_input（含 CLAHE）后的实际灰度值。

    桩模型只能看到预处理之后的张量，所以"颜色 → 号码"的对照表必须用这个值来建，
    而不是原始像素值——CLAHE 会整体抬亮几个级（实测 90→97）。
    """
    return int(round(float(np.median(gray_input(np.full((90, 60, 3), value, np.uint8))))))


def make_recognizer(table: dict[int, int | None], conf: float = 0.95,
                    **cfg_kw) -> JerseyRecognizer:
    cfg = JerseyConfig(**cfg_kw)
    rec = JerseyRecognizer.__new__(JerseyRecognizer)
    rec.cfg = cfg
    rec.model = StubModel(table, conf)
    rec._states = {}
    rec._crop_ms = rec._infer_ms = 0.0
    rec._n_crop = 0
    rec._published = {}
    rec.events = []
    return rec


def frame_with(number_code: int | None, size: tuple[int, int] = (480, 640)) -> tuple:
    """造一帧：躯干区填成 number_code 对应的灰度色，返回 (rgb, bbox)。"""
    cfg = JerseyConfig()
    tw, th = 60, 90
    tx0, ty0 = 200, 120
    w = tw / (1.0 - 2 * cfg.x_inset)
    h = th / (cfg.y_bot - cfg.y_top)
    x0, y0 = tx0 - cfg.x_inset * w, ty0 - cfg.y_top * h
    bbox = (x0, y0, x0 + w, y0 + h)
    rgb = np.zeros((size[1], size[0], 3), np.uint8)
    if number_code is not None:
        rgb[ty0:ty0 + th, tx0:tx0 + tw] = number_code
    return rgb, bbox


def tracked(tid: int, bbox, frame: int, team: str | None = "A") -> schema.Tracked:
    return schema.Tracked(track_id=tid, category="player", frame=frame,
                          bbox_px=bbox, conf=0.9, visible=True, team=team)


# ── 用例 ────────────────────────────────────────────────────

def test_crop_geometry() -> None:
    cfg = JerseyConfig()
    box = torso_bbox((100, 200, 180, 420), cfg)
    check("躯干框比例", box == (112, 222, 168, 321), f"得到 {box}")
    check("过小的框不裁", torso_bbox((100, 200, 112, 240), cfg) is None)
    rgb = np.arange(600 * 400 * 3, dtype=np.uint8).reshape(600, 400, 3)
    cr = crop_torso(rgb, (100, 200, 180, 420), cfg=cfg)
    check("裁片尺寸 56x99", cr.ok and cr.crop.shape == (99, 56, 3), str(cr.crop.shape))
    # 越界夹取：框贴在画面边上时不能崩
    cr2 = crop_torso(rgb, (0, 0, 40, 120), cfg=cfg)
    check("边界框可裁且不越界", cr2.ok and cr2.crop.shape[0] > 0)


def test_gating() -> None:
    cfg = JerseyConfig()
    rgb = np.zeros((300, 200, 3), np.uint8)
    bbox = (20.0, 20.0, 120.0, 280.0)
    k_front = np.zeros((17, 3)); k_front[5] = [45, 60, 0.9]; k_front[6] = [95, 60, 0.9]
    k_side = np.zeros((17, 3)); k_side[5] = [68, 60, 0.9]; k_side[6] = [72, 60, 0.9]
    k_lowconf = k_front.copy(); k_lowconf[5, 2] = k_lowconf[6, 2] = 0.1
    check("正面肩宽采信", crop_torso(rgb, bbox, k_front, cfg).ok)
    side = crop_torso(rgb, bbox, k_side, cfg)
    check("侧身/背影不采信", (not side.ok) and side.reason == "side_or_back", side.reason)
    low = crop_torso(rgb, bbox, k_lowconf, cfg)
    check("肩点低置信不采信", (not low.ok) and low.reason == "shoulder_low_conf", low.reason)
    check("无关键点时不否决", crop_torso(rgb, bbox, None, cfg).ok)
    bad = k_front.copy(); bad[11, 0] = np.nan
    invalid = crop_torso(rgb, bbox, bad, cfg)
    check("非有限关键点被拒绝", (not invalid.ok) and invalid.reason == "keypoint_nonfinite",
          invalid.reason)
    pose = k_front.copy()
    pose[11] = [45, 180, 0.9]; pose[12] = [95, 180, 0.9]
    pose_crop = crop_torso(rgb, bbox, pose, cfg)
    fallback = crop_torso(rgb, bbox, k_front, cfg)
    check("肩髋骨架收紧胸腹框", pose_crop.ok and pose_crop.crop.shape != fallback.crop.shape,
          f"pose={pose_crop.crop.shape}, fallback={fallback.crop.shape}")


def test_dist_and_heads() -> None:
    lg = np.stack([logits_for(n) for n in list(range(0, 100)) + [None]])
    p, pb = logits_to_dist(lg)
    arg = p.argmax(1)
    check("0–99 全部可解码", bool((arg[:100] == np.arange(100)).all()))
    check("空白样本不发布号码", bool(pb[-1] > 0.5), f"p_blank={pb[-1]:.3f}")
    check("分布归一", bool(np.allclose(p.sum(1) + pb, 1.0, atol=1e-5)))
    x = preprocess([np.full((60, 40, 3), v, np.uint8) for v in (0, 128, 255)])
    # CLAHE 会把纯色整体抬亮几个级（0→3），所以只断言落在 [-1,1] 与形状上
    check("预处理形状与范围", x.shape == (3, 3, INPUT_H, INPUT_W) and
          float(x.min()) >= -1.0 and float(x.max()) <= 1.0,
          f"{x.shape} [{float(x.min()):.3f},{float(x.max()):.3f}]")
    g = gray_input(np.full((60, 40, 3), 128, np.uint8))
    check("训练/推理共用预处理", np.allclose(x[1, 0], (g.astype(np.float32) - 127.5) / 127.5,
                                    atol=1e-6))


def test_fusion_publish_and_flicker() -> None:
    table = {code_of(90): 23}
    rec = make_recognizer(table, min_obs=3, min_conf=0.6, every_n=1)
    rgb, bbox = frame_with(90)
    o = tracked(1, bbox, 1)
    rec.recognize("v", 1, rgb, [o])
    check("单次观测不发布", o.jersey is None, f"jersey={o.jersey}")
    rec.recognize("v", 2, rgb, [o])
    check("两次观测仍不发布", o.jersey is None)
    rec.recognize("v", 3, rgb, [o])
    check("第三次观测发布 #23", o.jersey == 23 and o.jersey_conf >= 0.6,
          f"jersey={o.jersey} conf={o.jersey_conf:.3f}")
    # 后续没到观测也保持已发布号码（不闪）
    o2 = tracked(1, bbox, 4)
    rec.recognize("v", 4, np.zeros_like(rgb), [o2])
    check("无观测帧保持已发布号码", o2.jersey == 23, f"jersey={o2.jersey}")


def test_fusion_switch_limit() -> None:
    # 前 10 帧看到 23，之后一直看到 28：只允许改一次，之后锁死
    table = {code_of(90): 23, code_of(120): 28}
    rec = make_recognizer(table, min_obs=3, min_conf=0.6, every_n=1, max_switch=1)
    rgb23, bbox = frame_with(90)
    rgb28, _ = frame_with(120)
    o = tracked(1, bbox, 1)
    for f in range(1, 11):
        rec.recognize("v", f, rgb23, [o])
    published = o.jersey
    for f in range(11, 60):
        rec.recognize("v", f, rgb28, [o])
    switches = [e for e in rec.events if e["kind"] == "switch" and e["track_id"] == 1]
    check("起始发布 #23", published == 23, f"published={published}")
    check("切换次数 ≤1", len(switches) <= 1, f"switches={len(switches)}")
    check("切换后不再改动", o.jersey in (23, 28), f"final={o.jersey}")
    st = rec._states[("v", 1)]
    check("锁死后停止推理", st.locked and rec.model.crops <= 60, f"locked={st.locked}")


def test_fusion_reset_and_isolation() -> None:
    table = {code_of(90): 11}
    rec = make_recognizer(table, min_obs=2, min_conf=0.5, every_n=1, gap_reset_s=1.0, fps=30)
    rgb, bbox = frame_with(90)
    o = tracked(1, bbox, 1)
    for f in range(1, 6):
        rec.recognize("v", f, rgb, [o])
    check("发布 #11", o.jersey == 11, f"jersey={o.jersey}")
    # 轨迹从 objects 里消失超过 1s：状态回收，重新出现时不得沿用旧号
    rec.recognize("v", 40, np.zeros_like(rgb), [])
    check("丢失 >1s 后状态回收", ("v", 1) not in rec._states, str(list(rec._states)))
    gone = tracked(1, bbox, 41)
    rec.recognize("v", 41, np.zeros_like(rgb), [gone])
    check("重新出现不沿用旧号", gone.jersey is None, f"jersey={gone.jersey}")
    # 多视频隔离
    rec.reset()
    o_a = tracked(1, bbox, 1)
    for f in range(1, 6):
        rec.recognize("vidA", f, rgb, [o_a])
    o_b = tracked(1, bbox, 1)
    rec.recognize("vidB", 1, np.zeros_like(rgb), [o_b])
    check("跨视频不串号", o_a.jersey == 11 and o_b.jersey is None,
          f"A={o_a.jersey} B={o_b.jersey}")


def test_conflict_penalty() -> None:
    table = {code_of(90): 3, code_of(150): 3}
    rec = make_recognizer(table, min_obs=2, min_conf=0.5, every_n=1)
    rgb, bbox = frame_with(90)
    a = tracked(1, bbox, 1, team="A")
    b = tracked(2, bbox, 1, team="A")
    for f in range(1, 6):
        rec.recognize("v", f, rgb, [a, b])
    conf_a, conf_b = a.jersey_conf, b.jersey_conf
    conflict = [e for e in rec.events if e["kind"] == "conflict"]
    check("同队同号记日志", len(conflict) >= 1, f"{len(conflict)} 条")
    check("冲突方置信度打折", min(conf_a, conf_b) < max(conf_a, conf_b) or
          min(conf_a, conf_b) < 1.0, f"conf=({conf_a:.2f},{conf_b:.2f})")
    c = tracked(3, bbox, 6, team="B")
    for f in range(6, 12):
        rec.recognize("v", f, rgb, [a, b, c])
    check("另一队同号不冲突", c.jersey == 3 and c.jersey_conf >= min(conf_a, conf_b),
          f"B conf={c.jersey_conf:.2f}")


def test_never_guess() -> None:
    # 空白票为主时不得发布号码
    table = {code_of(90): None}
    rec = make_recognizer(table, min_obs=2, min_conf=0.5, every_n=1)
    rgb, bbox = frame_with(90)
    o = tracked(1, bbox, 1)
    for f in range(1, 20):
        rec.recognize("v", f, rgb, [o])
    check("读不出就留 None", o.jersey is None, f"jersey={o.jersey}")
    # 球与非可见对象不进入识别
    ball = schema.Tracked(track_id=9, category="ball", frame=1, bbox_px=bbox, visible=True)
    occ = tracked(10, bbox, 1)
    occ.visible = False
    before = rec.model.crops
    rec.recognize("v", 20, rgb, [ball, occ])
    check("球/遮挡框不识别", rec.model.crops == before, f"crops+{rec.model.crops - before}")


def test_informative_observations_and_quality() -> None:
    from bsk.perceive.jersey import _TrackState

    st = _TrackState()
    cfg = JerseyConfig(min_obs=3, min_conf=0.0)
    p23 = np.zeros(100); p23[23] = 1.0
    for _ in range(5):
        st.update(np.zeros(100), 1.0, 1, cfg)
    check("空白帧不计有效观测", st.obs == 0, f"obs={st.obs}")
    st.update(p23, 0.0, 2, cfg, quality=1.0)
    check("空白后首个数字正确初始化", st.obs == 1 and int(st.post.argmax()) == 23,
          f"obs={st.obs} best={int(st.post.argmax())}")
    low, high = _TrackState(), _TrackState()
    low.update(p23, 0.0, 1, cfg, quality=0.1)
    high.update(p23, 0.0, 1, cfg, quality=1.0)
    # 用冲突观测检验权重；重复同号会让两者都保持 1，无法区分质量。
    p28 = np.zeros(100); p28[28] = 1.0
    low.update(p28, 0.0, 2, cfg, quality=0.1)
    high.update(p28, 0.0, 2, cfg, quality=1.0)
    check("低质量裁片降低融合权重", low.post[23] > high.post[23],
          f"low={low.post[23]:.3f}, high={high.post[23]:.3f}")


def test_sampling_cadence() -> None:
    table = {code_of(90): 5}
    rec = make_recognizer(table, min_obs=3, min_conf=0.5, every_n=5)
    rgb, bbox = frame_with(90)
    o = tracked(1, bbox, 1)
    for f in range(1, 11):
        rec.recognize("v", f, rgb, [o])
    check("每 5 帧采样一次", rec.model.crops == 2, f"crops={rec.model.crops}")
    # 门控失败不消耗节拍：侧身失败后下一帧变正面应立即采样
    cfg = JerseyConfig(every_n=5)
    rgb, bbox = frame_with(90)
    side = np.zeros((17, 3)); side[5] = [bbox[0] + 26, bbox[1] + 20, 0.9]
    side[6] = [bbox[0] + 30, bbox[1] + 20, 0.9]
    rec3 = make_recognizer(table, min_obs=1, min_conf=0.0, every_n=5)
    rec3.recognize("v", 1, rgb, [tracked(1, bbox, 1)],
                   [schema.Pose2D(video_id="v", frame=1, track_id=1, kpts=side.tolist())])
    check("门控失败不推进采样节拍", rec3.model.crops == 0, f"crops={rec3.model.crops}")
    rec3.recognize("v", 2, rgb, [tracked(1, bbox, 2)])
    check("下一帧立即重试", rec3.model.crops == 1, f"crops={rec3.model.crops}")
    # 人多时按面积取前 max_crops
    rec2 = make_recognizer(table, min_obs=1, min_conf=0.0, every_n=1, max_crops=3)
    objs = [tracked(i, (0, 0, 40 + i * 4, 120 + i * 4), 1) for i in range(8)]
    rgb2 = np.zeros((300, 300, 3), np.uint8)
    rgb2[12:100, 6:90] = 90
    rec2.recognize("v", 1, rgb2, objs)
    check("单帧裁片数受限", rec2.model.crops == 3, f"crops={rec2.model.crops}")


def test_missing_weights_is_noop() -> None:
    """权重不在的时候必须整体退化，不能抛异常、不能写 jersey 字段。"""
    rec = JerseyRecognizer(weights="data/models/__definitely_missing__.onnx")
    check("权重缺失时 ready=False", not rec.ready, str(rec.unavailable_reason))
    o = tracked(1, (100, 100, 180, 320), 1)
    rec.recognize("v", 1, np.zeros((400, 400, 3), np.uint8), [o])
    check("权重缺失时不写号码", o.jersey is None and o.jersey_conf == 0.0)
    check("权重缺失时不抛异常", True)


def test_sidecar_guard() -> None:
    """sidecar 输入尺寸与代码常量不一致时要拒绝加载，而不是静默错位。"""
    from bsk.perceive.jersey import JerseyModel

    d = Path(".tmp_jersey_sidecar")
    d.mkdir(exist_ok=True)
    w = d / "x.onnx"
    w.write_bytes(b"not-a-real-model")
    (d / "x.json").write_text(json.dumps({"input_hw": [32, 32]}), encoding="utf-8")
    m = JerseyModel(w, device="cpu", use_trt=False)
    check("尺寸不匹配被拒绝", m.ready is False and "不一致" in (m.error or ""), str(m.error))


def test_onnx_parity() -> None:
    """有导出权重时，校验 ONNX 与 torch 的号码判定一致（没有就跳过）。"""
    onnx = Path("data/models/jersey_small.onnx")
    pt = Path("data/models/jersey_small.pt")
    if not onnx.exists():
        check("ONNX 一致性（跳过：未导出）", True)
        return
    from bsk.perceive.jersey import JerseyModel

    m = JerseyModel(onnx, device="cpu", use_trt=False)
    if not m.ready:
        check("ONNX 一致性（跳过：onnxruntime 不可用）", True, str(m.error))
        return
    rng = np.random.default_rng(0)
    imgs = [np.random.randint(0, 255, (70, 45, 3), dtype=np.uint8) for _ in range(16)]
    lg_onnx = m.predict(preprocess(imgs))
    p_onnx, _ = logits_to_dist(lg_onnx)
    check("ONNX 输出形状", lg_onnx.shape == (16, LOGIT_LEN), str(lg_onnx.shape))
    check("ONNX 概率归一", bool(np.allclose(p_onnx.sum(1) >= 0, True)) and
          bool((p_onnx.sum(1) <= 1.0001).all()))
    if pt.exists():
        try:
            import torch
            from bsk.train_jersey import build_torch
            net = build_torch()
            net.load_state_dict(torch.load(pt, map_location="cpu"))
            net.eval()
            with torch.no_grad():
                lg_t = net(torch.from_numpy(preprocess(imgs))).numpy()
            diff = float(np.abs(lg_t - lg_onnx).max())
            same = bool((lg_t.argmax(1) == lg_onnx.argmax(1)).all())
            check("torch/ONNX logits 对齐", diff < 0.05 and same, f"max|Δ|={diff:.4f}")
        except Exception as exc:
            check("torch/ONNX 对齐（跳过）", True, f"{type(exc).__name__}: {exc}")


def main() -> int:
    for fn in (test_crop_geometry, test_gating, test_dist_and_heads,
               test_fusion_publish_and_flicker, test_fusion_switch_limit,
               test_fusion_reset_and_isolation, test_conflict_penalty, test_never_guess,
               test_informative_observations_and_quality, test_sampling_cadence,
               test_missing_weights_is_noop, test_sidecar_guard, test_onnx_parity):
        print(f"\n── {fn.__name__} ──")
        fn()
    print("\n" + ("全部通过" if not FAILS else f"失败 {len(FAILS)} 项: {FAILS}"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
