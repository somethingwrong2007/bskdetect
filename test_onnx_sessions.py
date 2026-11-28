"""进程内 ONNX 会话注册表：复用、隔离、TRT 回退与并发只创建一次。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from bsk.perceive import onnx_sessions


def setup_function():
    onnx_sessions.clear_sessions()


def teardown_function():
    onnx_sessions.clear_sessions()


def test_same_weight_reuses_session_and_different_weight_does_not(monkeypatch):
    from bsk.perceive import persons

    calls = []

    def fake(path, prefer_gpu=True):
        calls.append((path, prefer_gpu))
        return object(), "CPUExecutionProvider"

    monkeypatch.setattr(persons, "make_session", fake)
    a = onnx_sessions.get_session("a.onnx", device="cpu")
    b = onnx_sessions.get_session("a.onnx", device="cpu")
    c = onnx_sessions.get_session("b.onnx", device="cpu")
    assert a is b
    assert a[0] is not c[0]
    assert len(calls) == 2


def test_cuda_dll_preparation_runs_before_generic_gpu_session(monkeypatch):
    from bsk.perceive import persons

    calls = []

    def prepare():
        calls.append("prepare")

    def make(path, prefer_gpu=True):
        calls.append(("session", path, prefer_gpu))
        return object(), "CPUExecutionProvider"

    monkeypatch.setattr(onnx_sessions, "_prepare_cuda_dlls", prepare)
    monkeypatch.setattr(persons, "make_session", make)
    onnx_sessions.get_session("gpu.onnx", device="auto")
    onnx_sessions.get_session("cpu.onnx", device="cpu")

    assert calls[0] == "prepare"
    assert calls[1][0] == "session" and calls[1][1].endswith("gpu.onnx") and calls[1][2]
    assert calls[2][0] == "session" and calls[2][1].endswith("cpu.onnx") and not calls[2][2]


def test_trt_failure_falls_back_without_changing_contract(monkeypatch):
    from bsk.perceive import persons

    expected = (object(), "CPUExecutionProvider")

    def fail(*args, **kwargs):
        raise RuntimeError("TensorRT unavailable")

    monkeypatch.setattr(persons, "make_session_trt", fail)
    monkeypatch.setattr(persons, "make_session", lambda *a, **k: expected)
    assert onnx_sessions.get_session("a.onnx", device="cuda", use_trt=True) is expected


def test_concurrent_creation_is_serialized(monkeypatch):
    from bsk.perceive import persons

    calls = []

    def fake(path, prefer_gpu=True):
        calls.append(path)
        return object(), "CPUExecutionProvider"

    monkeypatch.setattr(persons, "make_session", fake)
    with ThreadPoolExecutor(max_workers=8) as pool:
        sessions = list(pool.map(lambda _: onnx_sessions.get_session("same.onnx"), range(24)))
    assert len(calls) == 1
    assert all(session is sessions[0] for session in sessions)
