"""训练/导出球衣号码模型：合成数据（或已标注真实裁片）→ 双头分类 → ONNX。

模型是一个轻量深度可分离 CNN（约 0.15M 参数），输出 22 维 logits：
十位 10（索引 0 = 无十位）+ 个位 10 + 空白 2。选它而不是直接上 MobileNetV3-small 的理由：
64×96 的输入上 MobileNetV3 有 2.5M 参数、训练与 TRT 构建都更慢，而号码识别的信息量
（100 类 + 空白）远用不满；先把小模型跑通并留出延迟余量（docs §5 要求新增 ≤2ms/帧），
以后需要提精度再换主干，输入输出契约不变。

用法：
    # 纯合成（零标注，验证管线 + 出基线权重）
    python -m bsk.train_jersey --synth --per-number 60 --epochs 12
    # 合成预训 + 真实标注微调
    python -m bsk.train_jersey --manifest jersey_samples/labels.csv --finetune data/models/jersey_small.pt
    # 只要真实数据
    python -m bsk.train_jersey --manifest jersey_samples/labels.csv --epochs 30

产物（都不入库）：data/models/jersey_small.pt（权重）、jersey_small.onnx（推理）、
jersey_small.json（sidecar：输入尺寸、头部布局、指标，加载时校验用）。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np

from bsk.jersey_synth import SynthConfig, find_fonts, make_crop
from bsk.perceive.jersey import INPUT_H, INPUT_W, LOGIT_LEN, N_TENS, N_UNITS, gray_input

MODEL_DIR = Path("data/models")
BLANK_LABEL = -1


# ── 模型 ────────────────────────────────────────────────────

def build_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def block(cin, cout, stride, se=False):
        return nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
            nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    class JerseyNet(nn.Module):
        """stem + 5 个深度可分离块 + 三个分类头，输入 (N,3,64,96)，输出 (N,22)。"""

        def __init__(self, ch=(32, 64, 96, 128, 160)):
            super().__init__()
            c = [3, *ch]
            self.stem = nn.Sequential(nn.Conv2d(3, c[1], 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(c[1]), nn.ReLU(inplace=True))
            self.b1 = block(c[1], c[2], 2)
            self.b2 = block(c[2], c[3], 2)
            self.b3 = block(c[3], c[4], 2)
            self.b4 = block(c[4], c[5], 2)
            self.b5 = block(c[5], c[5], 2)
            self.head = nn.Sequential(nn.Conv2d(c[5], c[5], 3, 1, 1, groups=c[5], bias=False),
                                      nn.BatchNorm2d(c[5]), nn.ReLU(inplace=True))
            self.drop = nn.Dropout(0.15)
            self.tens = nn.Linear(c[5], N_TENS)
            self.units = nn.Linear(c[5], N_UNITS)
            self.blank = nn.Linear(c[5], 2)

        def forward(self, x):
            x = self.stem(x)
            for b in (self.b1, self.b2, self.b3, self.b4, self.b5):
                x = b(x)
            x = self.head(x)
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
            x = self.drop(x)
            return torch.cat([self.tens(x), self.units(x), self.blank(x)], dim=1)

    return JerseyNet()


def encode(number: int) -> np.ndarray:
    """号码 → (十位类, 个位类)。个位数用十位=0（NONE）表示，与解码规则一一对应。"""
    if number < 10:
        return np.array([0, number], np.int64)
    return np.array([number // 10, number % 10], np.int64)


# ── 数据 ────────────────────────────────────────────────────

def label_to_targets(label: int) -> tuple[int, int, int]:
    if label == BLANK_LABEL:
        return 0, 0, 1                    # 空白：十位/个位类不参与训练（ignore_index）
    t, u = encode(label)
    return int(t), int(u), 0


def synth_batch(n_per_number: int, seed: int, cfg: SynthConfig, max_number: int = 99,
                single_extra: int = 2):
    """在内存里生成合成集，返回 (N,H,W,3) uint8 与 (N,) int 标签。

    内存换时间：64×96 的 12k 张图约 220MB，直接驻留内存可以让 epoch 之间不用重复解码，
    CPU 训练时这几乎是唯一能接受的方案；上千类样本要落盘再走 DataLoader 时改用
    build_to_dir + manifest。
    """
    rng = np.random.default_rng(seed)
    fonts = find_fonts(cfg)
    imgs, labs = [], []
    for num in range(max_number + 1):
        for _ in range(n_per_number + (single_extra if num < 10 else 0)):
            im, _meta = make_crop(num, rng, cfg, fonts)
            imgs.append(im)
            labs.append(num)
    for _ in range(max(32, n_per_number * 4)):
        im, _meta = make_crop(None, rng, cfg, fonts)
        imgs.append(im)
        labs.append(BLANK_LABEL)
    return np.stack(imgs), np.array(labs, np.int64)


def load_manifest(path: str | Path, root: str | Path | None = None):
    """读真实标注清单（csv 或 jsonl）。列/字段：crop|image|path + label（空表示未标注）。"""
    path = Path(path)
    root = Path(root) if root else path.parent
    rows = []
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    else:
        import csv
        with open(path, encoding="utf-8-sig") as fh:
            rows = [r for r in csv.DictReader(fh)]
    imgs, labs = [], []
    skipped = 0
    for r in rows:
        raw = r.get("label", "")
        if raw is None or str(raw).strip() == "" or not str(raw).strip().lstrip("-").isdigit():
            skipped += 1
            continue
        lab = int(raw)
        rel = r.get("crop") or r.get("image") or r.get("path")
        p = Path(rel)
        if not p.is_absolute():
            p = root / p
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            skipped += 1
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        imgs.append(im)
        labs.append(lab if 0 <= lab <= 99 else BLANK_LABEL)
    print(f"[manifest] {path}: 有效 {len(imgs)}，跳过未标注/读不到 {skipped}")
    return (np.stack(imgs) if imgs else np.zeros((0, INPUT_H, INPUT_W, 3), np.uint8)), \
        np.array(labs, np.int64)


# ── 训练 ────────────────────────────────────────────────────

def _decode(logits, torch):
    """22 维 logits → 号码预测（-1 = 判为空白）。"""
    tens = logits[:, :N_TENS].argmax(1)
    units = logits[:, N_TENS:N_TENS + N_UNITS].argmax(1)
    blank = logits[:, N_TENS + N_UNITS:].argmax(1)
    num = torch.where(tens == 0, units, tens * N_UNITS + units)
    return torch.where(blank == 1, torch.full_like(num, -1), num)


def train(args):
    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dev = torch.device(args.device if args.device != "auto" else
                       ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type == "cuda" and torch.cuda.get_device_capability()[0] >= 12 and \
            torch.version.cuda and torch.version.cuda.startswith("11"):
        print("[warn] CUDA 11.x 轮子不支持 sm_120（Blackwell），已自动改用 CPU")
        dev = torch.device("cpu")

    cfg = SynthConfig()
    X, y = None, None
    if args.synth or not args.manifest:
        print(f"[data] 合成数据 per-number={args.per_number} …")
        t0 = time.time()
        X, y = synth_batch(args.per_number, args.seed, cfg, max_number=args.max_number)
        print(f"[data] 合成 {len(X)} 张，耗时 {time.time()-t0:.1f}s")
    if args.manifest:
        Xm, ym = load_manifest(args.manifest, args.manifest_root)
        if len(Xm):
            X = Xm if X is None else np.concatenate([X, Xm])
            y = ym if y is None else np.concatenate([y, ym])
            print(f"[data] 并入真实标注 {len(Xm)} 张")
    if X is None or len(X) == 0:
        raise SystemExit("没有可用训练数据：给 --synth 或有效的 --manifest")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    n_tr = int(len(X) * args.split)
    tr, va = perm[:n_tr], perm[n_tr:]
    # 同一张图不能同时出现在训练与验证里：permutation 划分本身不重叠，这里只是防手滑
    assert not (set(tr.tolist()) & set(va.tolist())), "训练/验证划分重叠"

    model = build_torch().to(dev)
    if args.finetune and Path(args.finetune).exists():
        st = torch.load(args.finetune, map_location=dev)
        model.load_state_dict(st.get("model", st) if isinstance(st, dict) else st)
        print(f"[finetune] 载入 {args.finetune}")

    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] 参数量 {n_par/1e6:.2f}M  device={dev}  样本 {len(X)}"
          f"（训练 {len(tr)} / 验证 {len(va)}）  输入 1x3x{INPUT_H}x{INPUT_W}")

    # 预处理与线上共用 gray_input()。它输出 uint8（CLAHE 的结果本来就是 uint8），
    # 归一化只是 (g-127.5)/127.5 的线性变换，放 batch 里做即可：
    # 省掉把整份数据集摊成 float32 的 ~0.9GB 内存，且与 preprocess() 数值等价。
    tG = torch.from_numpy(np.stack([gray_input(x) for x in X])).to(dev)
    ty = torch.from_numpy(y).to(dev)
    blank = ty < 0
    digit = ty.clamp(min=0)
    zero = torch.zeros_like(digit)
    tens_t = torch.where(blank | (digit < 10), zero, digit // 10)
    tens_t = torch.where(blank, torch.full_like(digit, -100), tens_t).long()
    units_t = torch.where(blank, torch.full_like(digit, -100), digit % 10).long()
    blank_t = blank.long()
    ce = nn.CrossEntropyLoss(ignore_index=-100)

    def mb(idx, bs, shuffle=False):
        order = rng.permutation(idx) if shuffle else idx
        for i in range(0, len(order), bs):
            s = torch.from_numpy(np.ascontiguousarray(order[i:i + bs])).to(dev)
            g = tG[s].float()
            x = ((g - 127.5) / 127.5).unsqueeze(1).expand(-1, 3, -1, -1)
            yield x, tens_t[s], units_t[s], blank_t[s]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, math.ceil(len(tr) / args.batch))
    total_steps = steps_per_epoch * args.epochs
    if total_steps >= 16:
        # OneCycle 在小数据集上几轮就能收敛；warmup 段按总步数算，步数太少会除零退化
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.2)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    best = (-1.0, None)
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot = 0.0
        for xb, a, b, c in mb(tr, args.batch, shuffle=True):
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = ce(out[:, :N_TENS], a) + ce(out[:, N_TENS:N_TENS + N_UNITS], b) + \
                ce(out[:, N_TENS + N_UNITS:], c)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss)
        model.eval()
        with torch.no_grad():
            pred = []
            for xb, a, b, c in mb(va, 256):
                pred.append(_decode(model(xb), torch))
            pv = torch.cat(pred)
            gt = ty[va]
            acc = float((pv == gt).float().mean()) if len(va) else 0.0
            blank_ok = ""
            if (gt < 0).any():
                blank_ok = f" blank_recall={float((pv[gt < 0] < 0).float().mean()):.3f}"
        dt = time.time() - t0
        print(f"[ep {ep+1:02d}] loss={tot/max(1,math.ceil(len(tr)/args.batch)):.4f} "
              f"val_top1={acc:.4f}{blank_ok} {dt:.1f}s")
        if acc >= best[0]:
            best = (acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pt = MODEL_DIR / args.name
    torch.save(best[1], pt.with_suffix(".pt"))
    model.load_state_dict(best[1])
    metrics = {"val_top1": round(best[0], 4), "samples": int(len(X)), "epochs": args.epochs,
               "params_M": round(n_par / 1e6, 3), "device": str(dev),
               "synth_per_number": args.per_number, "manifest": args.manifest}
    if not args.no_onnx:
        onnx_path = export_onnx(model, pt.with_suffix(".onnx"))
        metrics["onnx"] = str(onnx_path)
    (pt.with_suffix(".json")).write_text(json.dumps(
        {"input_hw": [INPUT_H, INPUT_W], "logit_len": LOGIT_LEN,
         "heads": {"tens": N_TENS, "units": N_UNITS, "blank": 2},
         "normalize": "gray+clahe, x/127.5-1, rgb=copy", "metrics": metrics},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {pt.with_suffix('.pt')}  val_top1={best[0]:.4f}")
    print(json.dumps(metrics, ensure_ascii=False))
    return metrics


def export_onnx(model, path: Path):
    import torch

    model.eval()
    dummy = torch.zeros(1, 3, INPUT_H, INPUT_W)
    torch.onnx.export(model, dummy, str(path), opset_version=17,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    print(f"[onnx] 导出 {path}（batch 维动态；TRT 固定 profile 见 docs 使用说明）")
    return path


def main():
    ap = argparse.ArgumentParser(description="训练/导出球衣号码模型")
    ap.add_argument("--synth", action="store_true", help="使用合成数据")
    ap.add_argument("--manifest", default=None, help="真实标注 csv/jsonl")
    ap.add_argument("--manifest-root", default=None, help="清单里相对路径的根目录")
    ap.add_argument("--finetune", default=None, help="从这个 .pt 继续训")
    ap.add_argument("--per-number", type=int, default=60)
    ap.add_argument("--max-number", type=int, default=99)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--split", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--name", default="jersey_small.onnx", help="输出文件名（onnx/pt/json 同名）")
    ap.add_argument("--no-onnx", action="store_true")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
