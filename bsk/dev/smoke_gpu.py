"""环境自检：验证最可能坑掉整条管线的四个假设。

1. sm_120 上 CUDA 与 torch 是否真的可用
2. ONNX Runtime CUDA EP 能否加载并执行
3. FP16 与 FP32 数值是否一致（Blackwell 上有 FP16 kernel 算错的公开案例，
   必须在写模型之前排除，否则以后所有误差都无从归因）
4. 显存基线，用来判断 4GB 档还剩多少空间

用法：python -m bsk.dev.smoke_gpu
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

SHAPE = (1, 3, 512, 512)
CHANNELS = (32, 64, 128)


def _build_onnx(dtype: str) -> Path:
    """同一组权重的 FP32 / FP16 两份图，用于数值对比。"""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    np_dtype = np.float32 if dtype == "float32" else np.float16
    tp = TensorProto.FLOAT if dtype == "float32" else TensorProto.FLOAT16

    nodes = []
    inits = []
    prev = "x"
    for i, c_out in enumerate(CHANNELS):
        c_in = SHAPE[1] if i == 0 else CHANNELS[i - 1]
        w = (np.random.default_rng(i).standard_normal((c_out, c_in, 3, 3)) * 0.05).astype(np_dtype)
        b = np.zeros((c_out,), dtype=np_dtype)
        inits += [numpy_helper.from_array(w, f"w{i}"), numpy_helper.from_array(b, f"b{i}")]
        y = f"y{i}"
        nodes.append(helper.make_node("Conv", [prev, f"w{i}", f"b{i}"], [y],
                                      kernel_shape=[3, 3], pads=[1, 1, 1, 1], name=f"conv{i}"))
        prev = f"r{i}"
        nodes.append(helper.make_node("Relu", [y], [prev], name=f"relu{i}"))
    nodes.append(helper.make_node("GlobalAveragePool", [prev], ["out"], name="gap"))

    graph = helper.make_graph(
        nodes, "probe",
        [helper.make_tensor_value_info("x", tp, list(SHAPE))],
        [helper.make_tensor_value_info("out", tp, [SHAPE[0], CHANNELS[-1]])],
        initializer=inits,
    )
    model = helper.make_model(graph, producer_name="bskdetect")
    model.ir_version = 10
    onnx.checker.check_model(model)
    path = Path(tempfile.gettempdir()) / f"bsk_probe_{dtype}.onnx"
    onnx.save(model, str(path))
    return path


def check_torch() -> None:
    import torch

    assert torch.cuda.is_available(), "torch 看不到 CUDA"
    cap = torch.cuda.get_device_capability(0)
    print(f"[torch] {torch.__version__} device={torch.cuda.get_device_name(0)} capability=sm_{cap[0]}{cap[1]}")
    x = torch.randn(2048, 2048, device="cuda")
    err = (x @ x - (x.double() @ x.double()).float()).abs().max().item()
    print(f"[torch] fp32 matmul 对 fp64 最大偏差 {err:.4f}")


def check_ort() -> None:
    import onnxruntime as ort

    print(f"[ort] {ort.__version__} available={ort.get_available_providers()}")
    x32 = np.random.default_rng(0).standard_normal(SHAPE).astype(np.float32)

    sess = ort.InferenceSession(str(_build_onnx("float32")), providers=["CUDAExecutionProvider"])
    if "CUDAExecutionProvider" not in sess.get_providers():
        raise RuntimeError(f"CUDA EP 未生效，实际使用 {sess.get_providers()}")
    ref = sess.run(None, {"x": x32})[0].astype(np.float32)
    print(f"[ort] fp32 执行成功 out={ref.shape} sum={ref.sum():.4f}")

    sess16 = ort.InferenceSession(str(_build_onnx("float16")), providers=["CUDAExecutionProvider"])
    got = sess16.run(None, {"x": x32.astype(np.float16)})[0].astype(np.float32)
    diff = np.abs(got - ref)
    rel = diff.max() / max(np.abs(ref).max(), 1e-9)
    print(f"[ort] fp16 对 fp32 最大绝对偏差 {diff.max():.5f} 相对 {rel:.5f}")
    if rel > 5e-2:
        raise RuntimeError("FP16 kernel 数值异常，Blackwell 上先按 FP32 部署")

    t0 = time.perf_counter()
    for _ in range(30):
        sess.run(None, {"x": x32})
    print(f"[ort] 3conv@512x512 单帧 {((time.perf_counter() - t0) / 30) * 1000:.2f} ms")


def check_nvcodec() -> None:
    try:
        import PyNvVideoCodec as nvc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"PyNvVideoCodec 导入失败，硬件解码路径不可用: {exc}") from exc
    names = [a for a in dir(nvc) if not a.startswith("_")]
    print(f"[nvcodec] {getattr(nvc, '__version__', '?')} 顶层符号={names[:12]}")
    print("[nvcodec] 注意：此处仅验证可导入，真正解码验证见 check_decode.py")


def check_vram() -> None:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(",")
    total, used = int(out[0]), int(out[1])
    print(f"[vram] 总计 {total} MiB 已用 {used} MiB 空闲 {total - used} MiB")
    print(f"[vram] 4GB 档预算参考：空闲需 > {4096 - used} MiB 才够另开一路完整管线")


def main() -> int:
    failed = []
    for fn in (check_torch, check_ort, check_nvcodec, check_vram):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"[{fn.__name__}] FAILED: {type(exc).__name__}: {exc}")
            failed.append(fn.__name__)
    print("\nRESULT:", "ALL PASS" if not failed else f"FAILED -> {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
