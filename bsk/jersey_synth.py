"""合成球衣号码裁片生成器：不依赖外部数据集授权，直接产出带标签训练数据。

为什么先合成再标真实：需求文档要求人工标 300–800 张真实裁片，但那批数据到手之前，管线
（预处理 → 训练 → 导出 → 融合 → 评估）也需要能跑通并有一个基线指标。合成数据把"数字
形状识别"这件事的分布铺开（字体、球衣底色、透视、褶皱、运动模糊、遮挡、压缩），先训出来
的权重可以直接当真实标注的微调起点，而不是从零开始。

真实迁移的边界要说清楚：这里的字体是系统 TrueType（粗体/无衬线为主），和 NBA 球衣的定制
字体不完全一致；所以合成指标只能证明"管线正确 + 模型容量够"，不能替代 docs §6 的验收数字
——那一步必须等真实裁片标注。背景与退化尽量贴近转播画面（远端小目标、JPEG、运动模糊），
但球衣纹理毕竟是程序生成的。

风格与 bsk/calib_seg_real.py 一致：几何/光照随机化都在 numpy+cv2 里做，无额外依赖。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from bsk.perceive.jersey import INPUT_H, INPUT_W

# 球衣配色：本项目素材有青色主场/白色客场，另外补几支常见配色，避免模型只认一种底色
PALETTE = (
    ((0, 122, 125), (180, 220, 230)),   # 青色系（主场）
    ((232, 232, 236), (90, 90, 100)),   # 白色系（客场）
    ((28, 32, 40), (200, 205, 215)),    # 深藏青
    ((150, 26, 40), (235, 235, 240)),   # 深红
    ((36, 56, 140), (240, 240, 245)),   # 蓝
    ((240, 190, 40), (40, 40, 50)),     # 金黄
    ((52, 52, 56), (230, 230, 235)),    # 灰黑
    ((110, 40, 130), (240, 240, 245)),  # 紫
)

# 优先挑球衣上常见的粗体/几何无衬线字体；缺字体就退回 PIL 默认
FONT_CANDIDATES = (
    "arialbd.ttf", "arblk.ttf", "bahnschrift.ttf", "impact.ttf", "verdanab.ttf",
    "tahoma.ttf", "calibrib.ttf", "Candarab.ttf", "gadget.ttf", "FRADM.TTF",
    "FRADMCN.TTF", "STENCIL.TTF", "ARIALN.TTF", "ARIBL.TTF", "comici.ttf",
)

SKIN_TONES = ((196, 150, 110), (150, 100, 70), (110, 72, 50), (222, 180, 140))


@dataclass
class SynthConfig:
    canvas: int = 192              # 生成画布边长（最终再缩到 INPUT_W×INPUT_H）
    font_size: tuple[int, int] = (72, 132)
    blank_ratio: float = 0.10      # 无号码样本比例：教会模型"读不出"而不是硬猜
    heavy_degrade_ratio: float = 0.35
    jpeg_quality: tuple[int, int] = (40, 92)
    motion_kernel: tuple[int, int] = (0, 7)
    max_occlude: float = 0.35
    fonts: list[str] = field(default_factory=list)


def find_fonts(cfg: SynthConfig = SynthConfig()) -> list[str]:
    """扫描系统字体目录，命中候选表就用；一个都没有时返回 []，由调用方退回位图字体。"""
    if cfg.fonts:
        return [f for f in cfg.fonts if Path(f).exists()]
    roots = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts"), Path("/System/Library/Fonts")]
    out: list[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for name in FONT_CANDIDATES:
            for cand in (root / name, root / name.upper(), root / name.lower()):
                if cand.is_file():
                    out.append(str(cand))
                    break
    return sorted(set(out))


def _fabric(size: int, base: tuple, rng: np.random.Generator) -> np.ndarray:
    """球衣底布：底色 + 织物噪声 + 条纹/色块 + 褶皱光照。"""
    hi, lo = np.array(base[0], np.float32), np.array(base[1], np.float32)
    img = np.full((size, size, 3), hi, np.float32)
    # 织物质感：低频噪声比白噪声更像布纹（小尺寸噪声上采样后天然带模糊）
    noise = cv2.GaussianBlur(rng.standard_normal((size // 3 + 1, size // 3 + 1))
                             .astype(np.float32), (0, 0), 1.2)
    noise = cv2.resize(noise, (size, size), interpolation=cv2.INTER_LINEAR)
    img += (noise[..., None] * rng.uniform(4, 14))
    mode = rng.integers(0, 4)
    if mode == 1:                      # 竖条纹
        w = int(size * rng.uniform(0.06, 0.14))
        step = w * int(rng.integers(2, 4))
        m = (np.arange(size) % step) < w
        img[:, m] = img[:, m] * 0.25 + lo * 0.75
    elif mode == 2:                    # 斜向色块（球衣侧片）
        xs, yy = np.meshgrid(np.arange(size), np.arange(size))
        band = np.abs(((xs + yy) % int(size * 0.5)) - size * 0.25) < size * 0.07
        img[band] = img[band] * 0.4 + lo * 0.6
    elif mode == 3:                    # 下摆深色边
        h = int(size * rng.uniform(0.10, 0.22))
        img[-h:] = img[-h:] * 0.45 + lo * 0.55
    # 褶皱：正弦起伏 + 局部阴影，模拟布料反光
    xs, yy = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    shade = (np.sin(xs * rng.uniform(3, 9) + rng.uniform(0, 6.28)) *
             np.cos(yy * rng.uniform(2, 6) + rng.uniform(0, 6.28)))
    shade += rng.uniform(-0.4, 0.4, (size, size)).astype(np.float32) * 0.35
    shade = cv2.GaussianBlur(shade, (0, 0), 3.0)
    img *= (1.0 + 0.10 * shade[..., None])
    return np.clip(img, 0, 255)


def _draw_number(img: np.ndarray, text: str, font_path: str | None, size: int,
                 rng: np.random.Generator) -> np.ndarray:
    """把号码画进画布：字色与底色对比度随机，带描边/阴影（真实球衣大多是贴布+缝边）。"""
    from PIL import Image, ImageDraw, ImageFont

    h, w = img.shape[:2]
    pil = Image.fromarray(np.uint8(img))
    d = ImageDraw.Draw(pil)
    if font_path:
        font = ImageFont.truetype(font_path, size)
    else:
        font = ImageFont.load_default(size)
    box = d.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    cx, cy = w // 2 + int(rng.integers(-w * 0.05, w * 0.06)), int(h * rng.uniform(0.42, 0.58))
    x, y = cx - tw // 2 - box[0], cy - th // 2 - box[1]
    lum = float(np.mean(img[..., :3]))
    if lum > 150:                      # 浅底用深字，深底用浅字，偶尔撞色（真实球衣也有）
        fill = tuple(int(v) for v in rng.choice([[20, 20, 26], [30, 30, 90], [110, 20, 30]]))
        trim = tuple(int(v) for v in rng.choice([[235, 235, 240], [240, 190, 40]]))
    else:
        fill = tuple(int(v) for v in rng.choice([[245, 245, 248], [250, 245, 225]]))
        trim = tuple(int(v) for v in rng.choice([[20, 20, 30], [40, 40, 120], [150, 30, 40]]))
    if rng.random() < 0.7:             # 描边
        r = max(1, int(size * rng.uniform(0.02, 0.06)))
        for dx in range(-r, r + 1, max(1, r // 2)):
            for dy in range(-r, r + 1, max(1, r // 2)):
                if dx * dx + dy * dy <= r * r:
                    d.text((x + dx, y + dy), text, font=font, fill=trim)
    if rng.random() < 0.5:             # 硬阴影（贴布厚度）
        off = max(1, int(size * 0.03))
        d.text((x + off, y + off), text, font=font, fill=(0, 0, 0, 180))
    d.text((x, y), text, font=font, fill=fill)
    return np.asarray(pil.convert("RGB"), np.float32)


def _perspective(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """轻微 keystone：转播里号码区总在斜面上，正视图会让模型不会读侧向透视。"""
    s = img.shape[0]
    j = s * 0.10
    src = np.float32([[0, 0], [s, 0], [s, s], [0, s]])
    dst = np.float32([[rng.uniform(0, j), rng.uniform(0, j)],
                      [s - rng.uniform(0, j), rng.uniform(0, j * 0.4)],
                      [s - rng.uniform(0, j * 0.5), s - rng.uniform(0, j)],
                      [rng.uniform(0, j * 0.4), s - rng.uniform(0, j * 0.5)]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (s, s), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def _occlude(img: np.ndarray, rng: np.random.Generator, max_frac: float) -> np.ndarray:
    """手臂/队友身体从边缘压进来：真实素材里号码经常只露出一半。"""
    h, w = img.shape[:2]
    tone = tuple(int(v) for v in rng.choice(SKIN_TONES))
    n = int(rng.integers(1, 3))
    for _ in range(n):
        side = rng.integers(0, 4)
        r = int(min(h, w) * rng.uniform(0.18, 0.18 + max_frac))
        centers = {0: (int(rng.uniform(0, w)), -r // 2), 1: (int(rng.uniform(0, w)), h + r // 2),
                   2: (-r // 2, int(rng.uniform(0, h))), 3: (w + r // 2, int(rng.uniform(0, h)))}
        cv2.circle(img, centers[int(side)], r, tone, -1, cv2.LINE_AA)
        cv2.circle(img, centers[int(side)], int(r * 0.72),
                   tuple(int(v * 0.85) for v in tone), -1, cv2.LINE_AA)
    return img


def _motion_blur(img: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    if k <= 1:
        return img
    ker = np.zeros((k, k), np.float32)
    ang = float(rng.uniform(0, np.pi))
    c = k // 2
    for i in range(k):
        xx = c + int(round((i - c) * np.cos(ang)))
        yy = c + int(round((i - c) * np.sin(ang)))
        if 0 <= xx < k and 0 <= yy < k:
            ker[yy, xx] = 1.0
    return cv2.filter2D(img, -1, ker / max(ker.sum(), 1e-6))


def make_crop(number: int | None, rng: np.random.Generator | None = None,
              cfg: SynthConfig = SynthConfig(), fonts: list[str] | None = None,
              crop_frac: tuple[float, float] = (0.55, 0.85)) -> tuple[np.ndarray, dict]:
    """生成一张躯干裁片（RGB uint8, INPUT_H×INPUT_W）。number=None 表示无号码样本。

    裁片窗口只覆盖画布的 crop_frac 比例并随机偏移，等价于真实管线里 bbox 不准、号码
    贴边的情形——训练分布必须包含"号码不完整"，否则上线后边缘样本全废。
    """
    rng = rng or np.random.default_rng()
    cfg = cfg or SynthConfig()
    fonts = fonts if fonts is not None else find_fonts(cfg)
    s = cfg.canvas
    base = PALETTE[int(rng.integers(0, len(PALETTE)))]
    img = _fabric(s, base, rng)
    meta: dict = {"number": None if number is None else int(number), "blank": number is None}
    if number is not None:
        text = str(int(number))
        fnt = fonts[int(rng.integers(0, len(fonts)))] if fonts else None
        fs = int(rng.integers(cfg.font_size[0], cfg.font_size[1] + 1))
        img = _draw_number(img, text, fnt, fs, rng)
        meta.update(font=str(fnt or "default"), font_size=fs)
    img = _perspective(img, rng)
    if rng.random() < 0.6:
        img = _occlude(img, rng, cfg.max_occlude)
    # 光照：整体亮度偏移 + 局部高光（球场顶灯）
    img = img * rng.uniform(0.65, 1.35) + rng.uniform(-18, 22)
    img = np.clip(img, 0, 255)
    # 取样窗口
    frac = rng.uniform(*crop_frac)
    win = max(int(s * frac), 24)
    ox = int(rng.integers(0, s - win + 1))
    oy = int(rng.integers(0, s - win + 1))
    out = img[oy:oy + win, ox:ox + win]
    # 退化链：模糊 → 下采样 → JPEG（转播压缩噪声是主要干扰源）
    k = int(rng.integers(cfg.motion_kernel[0], cfg.motion_kernel[1] + 1))
    if rng.random() < cfg.heavy_degrade_ratio:
        out = _motion_blur(out, max(k, 3), rng)
    elif k > 1:
        out = _motion_blur(out, k, rng)
    out = cv2.resize(out, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
    q = int(rng.integers(*cfg.jpeg_quality))
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(np.uint8(out), cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        out = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    meta["jpeg"] = q
    return np.ascontiguousarray(out, np.uint8), meta


def build(n_per_number: int = 60, blank_per: int = 8, seed: int = 7,
          cfg: SynthConfig = SynthConfig(), max_number: int = 99,
          single_extra: int = 2):
    """生成 (rgb, label) 流：0–99 每个号码 n_per_number 张，外加无号码样本。

    个位数多补 single_extra 张：真实素材里个位号码出现频次明显高于 0–99 均匀分布，
    而且"无十位"这个判定本身就是最容易翻车的地方。
    """
    rng = np.random.default_rng(seed)
    fonts = find_fonts(cfg)
    for num in range(0, max_number + 1):
        reps = n_per_number + (single_extra if num < 10 else 0)
        for _ in range(reps):
            yield make_crop(num, rng, cfg, fonts)
    for _ in range(blank_per * (max_number + 1) // 4):
        yield make_crop(None, rng, cfg, fonts)


def build_to_dir(out_dir: str | Path = "data/jersey_synth", n_per_number: int = 60,
                 seed: int = 7, cfg: SynthConfig = SynthConfig(), split_ratio: float = 0.9,
                 max_number: int = 99) -> dict:
    """落盘为 PNG + manifest.jsonl（train/val 划分），供 bsk/train_jersey.py 直接读。"""
    import json

    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    fonts = find_fonts(cfg)
    rows: list[dict] = []
    i = 0

    def emit(num: int | None):
        nonlocal i
        rgb, meta = make_crop(num, rng, cfg, fonts)
        name = f"{i:06d}_{'blank' if num is None else num}.png"
        cv2.imwrite(str(out / "images" / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        rows.append({"image": f"images/{name}", "label": (-1 if num is None else int(num)),
                     **meta})
        i += 1

    for num in range(0, max_number + 1):
        for _ in range(n_per_number + (2 if num < 10 else 0)):
            emit(num)
    for _ in range(max(16, n_per_number * 4)):
        emit(None)
    rng.shuffle(rows)
    n_tr = int(len(rows) * split_ratio)
    for idx, r in enumerate(rows):
        r["split"] = "train" if idx < n_tr else "val"
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"dir": str(out), "total": len(rows), "train": n_tr, "val": len(rows) - n_tr,
               "fonts": len(fonts), "input_hw": [INPUT_H, INPUT_W]}
    print(f"[synth] {summary}")
    return summary


__all__ = ["PALETTE", "SynthConfig", "build", "build_to_dir", "find_fonts", "make_crop"]
