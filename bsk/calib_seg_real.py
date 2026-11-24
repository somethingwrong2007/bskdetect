"""真实域混合数据生成：3D 球场线渲染叠加到真实转播画面上。

旧版合成数据是"干净渲染图"（纯色地板 + 白线），真实转播里有木地板纹理、
彩色漆区、球员遮挡、观众、压缩噪声，训出来的模型在真实画面上完全失效。

做法：
1. 从真实视频抽一批帧作为背景池；
2. 用 Court3DRenderer 随机相机（高度/距离/焦距/朝向随机）渲染 6 类线掩码；
3. 把线以白/浅色画到真实背景上（颜色、透明度抖动）；
4. 贴随机遮挡块并同步清空对应标签（模拟球员/器材挡住线）；
5. 光度增强 + 模糊 + JPEG 压缩。

输出目录结构与 calib_seg_data 一致（images/ masks/ meta.json），
可直接用 bsk.calib_seg_train 训练。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from bsk.court import FIBA, NBA, Court
from bsk.court_3d import (
    NUM_CLASSES,
    Court3DRenderer,
    CourtMesh3D,
    random_broadcast_camera,
)


def clean_court_lines(frame: np.ndarray) -> np.ndarray:
    """Inpaint 掉背景里的白色球场线。

    合成样本的标签只覆盖我们画上去的线；若背景里残留未标注的真线，
    模型会学到"忽略真线"，正好把要检测的目标压掉。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    white = ((hsv[:, :, 2] > 140) & (hsv[:, :, 1] < 80)).astype(np.uint8)
    white = cv2.dilate(white, np.ones((3, 3), np.uint8), iterations=2)
    if white.mean() > 0.25:  # 大面积白色（特写/白板）时不清，避免糊掉整幅
        return frame
    return cv2.inpaint(frame, white, 5, cv2.INPAINT_TELEA)


def extract_backgrounds(
    video_path: str | Path,
    out_dir: str | Path,
    n_frames: int = 240,
    size: tuple[int, int] = (960, 540),
    clean_lines: bool = True,
) -> int:
    """从视频均匀抽帧存为 JPEG。返回实际抽到的帧数。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        raise RuntimeError(f"无法读取视频帧数: {video_path}")
    idxs = np.linspace(0, total - 1, n_frames).astype(int)
    saved = 0
    for i, fidx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        if clean_lines:
            frame = cv2.cvtColor(clean_court_lines(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                                 cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out / f"bg_{i:04d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        saved += 1
    cap.release()
    return saved


def _line_color(rng: np.random.Generator) -> np.ndarray:
    """球场线颜色：以白为主，少量偏黄的漆线。"""
    if rng.random() < 0.15:
        base = np.array([235.0, 220.0, 120.0])
    else:
        base = np.array([240.0, 240.0, 240.0])
    return np.clip(base * rng.uniform(0.82, 1.06), 0, 255)


def _random_occluders(img: np.ndarray, label: np.ndarray, bg: np.ndarray,
                      rng: np.random.Generator, max_n: int = 6) -> None:
    """贴随机遮挡块，模拟球员和器材挡住线，同步清空标签。

    优先从背景别处取真实纹理贴片，少量纯色块用来覆盖模型没见过的物体。
    """
    h, w = label.shape
    for _ in range(int(rng.integers(0, max_n + 1))):
        bw = int(rng.uniform(0.03, 0.14) * w)
        bh = int(rng.uniform(0.08, 0.35) * h)
        if bw < 4 or bh < 4:
            continue
        x = int(rng.integers(0, max(1, w - bw)))
        y = int(rng.integers(0, max(1, h - bh)))
        r = rng.random()
        if r < 0.75:
            # 从背景别处取一块贴过来（更接近真实人物纹理）
            sx = int(rng.integers(0, max(1, bg.shape[1] - bw)))
            sy = int(rng.integers(0, max(1, bg.shape[0] - bh)))
            patch = cv2.resize(bg[sy:sy + bh, sx:sx + bw], (bw, bh))
            img[y:y + bh, x:x + bw] = patch
            label[y:y + bh, x:x + bw] = 0
        elif r < 0.9:
            color = rng.integers(20, 235, 3)
            img[y:y + bh, x:x + bw] = color
            label[y:y + bh, x:x + bw] = 0
        else:
            mask = np.zeros((h, w), np.uint8)
            cv2.ellipse(mask, (x + bw // 2, y + bh // 2), (bw // 2, bh // 2), 0, 0, 360, 255, -1)
            m = mask.astype(bool)
            img[m] = rng.integers(20, 235, 3)
            label[m] = 0


def _photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.astype(np.float32)
    # 亮度/对比度
    out = out * rng.uniform(0.82, 1.18) + rng.uniform(-18, 18)
    # 色相/饱和抖动（转 HSV 做）
    hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + int(rng.integers(-10, 11))) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.8, 1.25), 0, 255)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    # 噪声
    out += rng.normal(0, rng.uniform(1.0, 7.0), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    # 模糊（模拟运动模糊/失焦，偶尔强一点）
    r = rng.random()
    if r < 0.35:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    elif r < 0.45:
        k = int(rng.choice([5, 7]))
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        out = cv2.filter2D(out, -1, kernel)
    return out


def compose_sample(
    renderers: dict[str, Court3DRenderer],
    court: Court,
    bg_pool: list[np.ndarray],
    size: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """生成一张真实背景 + 合成球线的样本。"""
    h, w = size
    bg_src = bg_pool[int(rng.integers(0, len(bg_pool)))]
    # 随机裁切/缩放背景，模拟构图变化
    scale = rng.uniform(0.9, 1.3)
    bh, bw = bg_src.shape[:2]
    cw, ch = int(bw / scale), int(bh / scale)
    if cw >= bw or ch >= bh:
        bg = cv2.resize(bg_src, (w, h))
    else:
        x0 = int(rng.integers(0, bw - cw + 1))
        y0 = int(rng.integers(0, bh - ch + 1))
        bg = cv2.resize(bg_src[y0:y0 + ch, x0:x0 + cw], (w, h))

    # 缩放到目标尺寸后再清一次线，避免残留真线进入未标注背景
    bg = clean_court_lines(bg)

    renderer = renderers[court.model]
    # 8% 无线帧负样本：教模型不要从球衣/广告牌里"脑补"出线
    if rng.random() < 0.08:
        return _photometric(bg, rng), np.zeros((h, w), np.uint8)

    cam = random_broadcast_camera((h, w), rng)
    cam["focal_length"] = float(rng.uniform(500, 1700))
    cam["pos"] = (
        float(rng.uniform(-32, 32)),
        float(rng.uniform(-32, 32)),
        float(rng.uniform(5, 22)),
    )
    raw = renderer.render_label(cam)

    # 只要线的轮廓：涂出来的漆区/中圈是填充面，做标定训练必须转成线
    label = np.zeros((h, w), np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    for cid in range(1, NUM_CLASSES):
        m = (raw == cid).astype(np.uint8)
        if not m.any():
            continue
        edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, kernel)
        edge = cv2.dilate(edge, kernel, iterations=1)
        label[edge > 0] = cid

    img = bg.astype(np.float32)
    for cid in range(1, NUM_CLASSES):
        m = label == cid
        if not m.any():
            continue
        color = _line_color(rng)
        alpha = float(rng.uniform(0.65, 1.0))
        img[m] = img[m] * (1 - alpha) + color[None, :] * alpha

    _random_occluders(img, label, bg, rng)
    img = _photometric(img, rng)

    # JPEG 压缩痕迹
    if rng.random() < 0.5:
        q = int(rng.integers(45, 90))
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            img = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return img, label


def build_mixed_dataset(
    out_dir: str | Path,
    bg_dir: str | Path,
    n_samples: int = 6000,
    size: tuple[int, int] = (360, 640),
    courts: list[Court] | None = None,
    seed: int = 0,
) -> Path:
    """生成混合数据集（真实背景 + 3D 渲染球线）。"""
    courts = courts or [FIBA, NBA]
    bg_files = sorted(Path(bg_dir).glob("*.jpg"))
    if not bg_files:
        raise FileNotFoundError(f"背景池为空: {bg_dir}（先运行 extract_backgrounds）")
    bg_pool = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in bg_files]

    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    renderers = {c.model: Court3DRenderer(CourtMesh3D.build(c)) for c in courts}
    rng = np.random.default_rng(seed)

    meta = []
    for i in range(n_samples):
        court = courts[i % len(courts)]
        img, label = compose_sample(renderers, court, bg_pool, size, rng)
        cv2.imwrite(str(out / "images" / f"{i:06d}.jpg"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        cv2.imwrite(str(out / "masks" / f"{i:06d}.png"), label)
        meta.append({"idx": i, "court_model": court.model})
        if (i + 1) % 500 == 0:
            print(f"  generated {i + 1}/{n_samples}")

    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"dataset saved to {out} ({n_samples} samples, size={size[1]}x{size[0]})")
    return out


__all__ = ["extract_backgrounds", "compose_sample", "build_mixed_dataset"]