"""进程内 ONNX Runtime 会话注册表。

不同计算图不能共用一个 InferenceSession；这里共享的是会话创建、provider 选择、
锁和生命周期。相同权重在不同视频/任务之间只创建一次，轨迹状态仍由各业务模块隔离。
本模块不修改 persons.py 的 TensorRT 配置，默认走通用 CUDA/CPU 会话。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

_SESSIONS: dict[tuple[str, str, bool], tuple[object, str]] = {}
_LOCK = threading.RLock()
_CUDA_DLLS_READY = False


def _prepare_cuda_dlls() -> None:
    """Load CUDA EP DLLs from the active ONNX Runtime wheel on Windows.

    Standalone ``onnxruntime-gpu`` environments do not have torch's ``lib``
    directory. Adding only PATH is insufficient for dependent DLL resolution;
    ORT's preload hook must run before the first CUDA InferenceSession.
    """
    global _CUDA_DLLS_READY
    if _CUDA_DLLS_READY or os.name != "nt":
        return
    import site
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return
    for site_packages in site.getsitepackages():
        root = Path(site_packages) / "nvidia"
        for component in ("cublas", "cuda_runtime", "cudnn", "cuda_nvrtc", "cufft"):
            directory = root / component / "bin"
            if not directory.is_dir():
                continue
            try:
                os.add_dll_directory(str(directory))
                os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
            except OSError:
                continue
    preload = getattr(ort, "preload_dlls", None)
    if preload is not None:
        preload(directory="")
    _CUDA_DLLS_READY = True


def get_session(
    path: str | Path,
    *,
    device: str = "auto",
    use_trt: bool = False,
    cache_dir: str | Path = "data/models/trt_cache",
) -> tuple[object, str]:
    """返回共享的 ``(session, execution_provider)``。

    ``use_trt=True`` 保留给输入契约与 persons.py 固定 profile 完全一致的模型；
    球场关键点模型默认 ``False``，避免误用 RTMO 的 640×640/profile 配置。
    """
    resolved = str(Path(path).resolve())
    key = (resolved, str(device), bool(use_trt))
    with _LOCK:
        cached = _SESSIONS.get(key)
        if cached is not None:
            return cached
        if use_trt and device in ("auto", "cuda"):
            from bsk.perceive.persons import make_session, make_session_trt

            try:
                value = make_session_trt(resolved, cache_dir=cache_dir)
            except Exception:
                # 与原球衣/RTMO 路径一致：TRT 不可用时回落通用 CUDA/CPU。
                value = make_session(resolved, prefer_gpu=True)
        else:
            if device in ("auto", "cuda"):
                _prepare_cuda_dlls()
            from bsk.perceive.persons import make_session

            value = make_session(resolved, prefer_gpu=device in ("auto", "cuda"))
        _SESSIONS[key] = value
        return value


def clear_sessions() -> None:
    """测试或显式重载使用；不清空文件上的引擎缓存。"""
    with _LOCK:
        _SESSIONS.clear()
