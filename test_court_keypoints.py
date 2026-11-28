"""球场 16 点配准单元测试：全部使用合成图像/假模型，不依赖训练权重。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from bsk.calibrate import pixel_to_court
from bsk.perceive import court_keypoints as ck


@pytest.fixture
def H_inv() -> np.ndarray:
    return np.asarray([
        [58.0, 6.0, 960.0],
        [-1.0, 39.0, 540.0],
        [0.0004, -0.0002, 1.0],
    ], np.float64)


def image(h=1080, w=1920, value=110):
    return np.full((h, w, 3), value, np.uint8)


class FakeModel:
    ready = True

    def __init__(self, H_inv):
        self.H_inv = H_inv
        self.conf = 0.92
        self.drop: set[int] = set()
        self.nan = False
        self.calls = 0

    def predict(self, rgb):
        self.calls += 1
        pts = ck._project(self.H_inv, ck.WORLD_POINTS)
        out = []
        for i, ((name, world), (u, v)) in enumerate(zip(ck.KEYPOINT_SPECS, pts)):
            if i in self.drop:
                continue
            x, y = (np.nan, np.nan) if self.nan else (u, v)
            out.append(ck.DetectedKeypoint(name, (float(x), float(y)), self.conf))
        return out


def tracker(H_inv, **kwargs):
    cfg = ck.KeypointConfig(weights="missing-test-model.onnx", **kwargs)
    obj = ck.CourtKeypointTracker(cfg)
    obj.model = FakeModel(H_inv)
    return obj


def test_sixteen_fiba_points_and_aux_geometry():
    assert len(ck.KEYPOINT_IDS) == len(set(ck.KEYPOINT_IDS)) == 16
    pts = dict(ck.KEYPOINT_SPECS)
    assert pts["corner_A_far"] == (14.0, 7.5)
    assert pts["corner_A_near"] == (-14.0, -7.5)
    assert pts["mid_near"] == (0.0, -7.5)
    assert pts["ft_A_far"] == pytest.approx((8.2, 1.8))
    assert pts["three_corner_A_far"] == pytest.approx((14.0, 6.6))
    assert pts["three_apex_A"] == pytest.approx((5.675, 0.0))
    assert ck.AUX_RESTRICTED_POINTS.shape == (8, 2)


def test_semantic_ransac_maps_pixels_back_to_world(H_inv):
    tr = tracker(H_inv)
    result = tr.update(0, image())
    assert result.valid, result.reason
    assert result.source == "onnx"
    assert result.inliers == 16
    assert result.reproj_error_px < 1e-4
    assert len(result.keypoints) == 16
    assert all(k.inlier for k in result.keypoints)
    uv = np.asarray([k.pixel for k in result.keypoints])
    xy = np.asarray([pixel_to_court(result.H, *p) for p in uv])
    np.testing.assert_allclose(xy, ck.WORLD_POINTS, atol=1e-5)
    data = result.to_dict()
    assert data["H"] is not None and data["H_inv"] is not None
    assert data["keypoints"][0]["world"] == [-14.0, -7.5]


def test_frame_skipping_reuses_causal_result(H_inv):
    tr = tracker(H_inv, every_n=5)
    tr.update(0, image())
    assert tr.model.calls == 1
    carried = tr.update(1, image())
    assert carried.valid and carried.inference_frame == 0
    assert tr.model.calls == 1
    tr.update(4, image())
    assert tr.model.calls == 1
    tr.update(5, image())
    assert tr.model.calls == 2


def test_classical_detection_runs_every_three_frames_and_reuses_h(monkeypatch):
    tr = ck.CourtKeypointTracker(ck.KeypointConfig(
        weights="missing.onnx", every_n=3, retry_every=3))
    calls = []

    def fake_detect(_rgb, _cfg):
        calls.append(len(calls))
        return ck.LineObservation(
            points=[], intersection_keypoints=[],
            segments=[((100, 100), (300, 100)), ((100, 100), (100, 300))],
        )

    monkeypatch.setattr(ck, "detect_classical", fake_detect)
    tr._motion = lambda _rgb: ("static", np.zeros(2), 1.0)
    tr.update(0, image())
    tr.update(1, image())
    tr.update(2, image())
    assert len(calls) == 1
    tr.update(3, image())
    assert len(calls) == 2


def test_four_semantic_points_are_enough_for_homography(H_inv):
    tr = tracker(H_inv)
    tr.model.drop = set(range(4, 16))
    result = tr.update(0, image())
    assert result.valid, result.reason
    assert result.inliers == 4
    assert result.H is not None and result.H_inv is not None
    assert tr.cfg.min_matches == 4 and tr.cfg.every_n == 3


def test_stable_labeled_support_rejects_leave_one_out_outlier():
    H_inv = np.asarray([[400.0 / 28.0, 0.0, 300.0],
                        [0.0, 400.0 / 15.0, 300.0],
                        [0.0, 0.0, 1.0]], np.float64)
    world = ck.WORLD_POINTS[[0, 1, 2, 3, 4]]  # Four corners plus mid_near.
    pixel = ck._project(H_inv, world)
    cfg = ck.KeypointConfig(weights="missing-test-model.onnx")
    support = ck.CourtKeypointTracker._stable_labeled_support

    assert support(world, pixel, H_inv, cfg)
    # Keep the outlier inside the unchanged 160,000 px^2 hull so rejection can
    # only come from deterministic leave-one-out refits, not random RANSAC.
    pixel[-1] = (300.0, 400.0)
    assert cv2.contourArea(cv2.convexHull(pixel.astype(np.float32).reshape(-1, 1, 2))) == 160000.0
    assert not support(world, pixel, H_inv, cfg)


def test_carry_rechecks_decayed_publication_confidence(H_inv):
    tr = tracker(H_inv, every_n=5)
    assert tr.update(0, image()).valid
    tr._last_result.confidence = 0.60
    tr._last_valid = ck.time.time() - 0.05
    carried = tr.update(1, image())
    assert carried.valid and carried.source == "temporal"
    assert carried.confidence < 0.60
    assert tr.model.calls == 1

    # temporal 只是展示来源，不能绕开原始 onnx 标定的发布置信度门。
    tr._last_valid = ck.time.time() - 0.15
    expired = tr.update(2, image())
    assert not expired.valid
    assert expired.reason == "low_confidence"
    assert expired.H is None and expired.H_inv is None
    assert tr.model.calls == 1


def test_invalidate_preserves_newer_concurrent_result(H_inv):
    tr = tracker(H_inv)
    assert tr.update(5, image()).valid
    tr.invalidate("runtime_error", frame=5)
    assert tr._H is not None
    tr.invalidate("runtime_error", frame=6)
    assert tr._H is None and tr._H_inv is None


def test_hard_cut_clears_old_homography(H_inv):
    tr = tracker(H_inv, every_n=1)
    assert tr.update(0, image(value=30)).valid
    old = tr._H.copy()
    tr.model.drop = set(range(16))
    cut = tr.update(1, image(value=230))
    assert tr._H is None and tr._H_inv is None
    assert cut.mode == "hard_cut"
    assert not cut.valid and cut.H is None and cut.H_inv is None
    assert tr.stats["hard_cuts"] == 1
    assert not np.array_equal(old, tr._H)


def test_pan_uses_smoothing_not_raw_matrix(H_inv):
    tr = tracker(H_inv, every_n=1)
    assert tr.update(0, image()).valid
    previous = tr._H_inv.copy()
    shifted = np.asarray([[1, 0, 16], [0, 1, -8], [0, 0, 1]], float) @ H_inv
    tr.model.H_inv = shifted
    tr._motion = lambda rgb: ("pan", np.asarray([16.0, -8.0]), 1.0)
    result = tr.update(1, image())
    assert result.valid
    raw = ck._project(shifted, ck.REFERENCE_POINTS)
    smooth = ck._project(result.H_inv, ck.REFERENCE_POINTS)
    prev = ck._project(previous, ck.REFERENCE_POINTS)
    # 平滑结果必须位于旧机位和新机位之间，而不是直接跳到新矩阵。
    assert np.linalg.norm(smooth - prev) < np.linalg.norm(raw - prev)
    assert np.linalg.norm(smooth - prev) > 1


def test_missing_points_are_hidden_and_valid_with_six(H_inv):
    tr = tracker(H_inv)
    tr.model.drop = {0, 2, 3, 4, 5, 7, 8, 9, 11, 13}
    result = tr.update(0, image())
    assert result.valid
    assert result.inliers == 6
    hidden = [k for k in result.keypoints if not k.inlier]
    assert len(hidden) == 10
    assert all(k.confidence == 0 for k in hidden)


def test_low_confidence_does_not_publish_h(H_inv):
    tr = tracker(H_inv, every_n=1)
    assert tr.update(0, image()).valid
    tr.model.conf = 0.1
    result = tr.update(1, image())
    assert not result.valid
    assert result.H is None and result.H_inv is None
    assert result.reason == "low_confidence"
    assert result.confidence == 0


def test_publication_confidence_penalizes_rejected_detections():
    detections = [
        ck.DetectedKeypoint(name, (0.0, 0.0), 0.9, world, index < 6)
        for index, (name, world) in enumerate(ck.KEYPOINT_SPECS)
    ]
    expected = 0.9 * (0.65 + 0.35 * 6 / 16)
    assert ck.CourtKeypointTracker._publication_confidence(detections) == pytest.approx(expected)


def test_marginal_semantic_confidence_is_not_published(H_inv):
    tr = tracker(H_inv)
    tr.model.conf = 0.45
    result = tr.update(0, image())
    assert not result.valid and result.reason == "low_confidence"
    assert result.H is None and result.H_inv is None


def test_nan_and_inf_model_points_are_rejected(H_inv):
    tr = tracker(H_inv)
    tr.model.nan = True
    result = tr.update(0, image())
    assert not result.valid and result.H is None
    assert tr._H is None


def test_closeup_without_prior_remains_uncalibrated(H_inv):
    tr = tracker(H_inv, retry_every=1)
    tr.model.drop = set(range(3, 16))
    result = tr.update(0, image())
    assert not result.valid
    assert result.inliers == 0
    assert result.H is None
    assert result.reason == "low_confidence"


def test_resolution_change_invalidates_state(H_inv):
    tr = tracker(H_inv, every_n=5)
    assert tr.update(0, image()).valid
    result = tr.update(1, image(h=720, w=1280))
    assert result.valid and result.transition == "initial"


def test_trackers_and_threads_are_isolated(H_inv):
    a, b = tracker(H_inv), tracker(H_inv)
    a.update(0, image())
    assert a._H is not None and b._H is None
    with ThreadPoolExecutor(max_workers=4) as pool:
        observations = list(pool.map(lambda i: ck.detect_classical(image(value=i), ck.KeypointConfig()), range(8)))
    assert len(observations) == 8


def test_line_aid_requires_prior_and_rejects_hard_cut(H_inv):
    tr = tracker(H_inv)
    assert tr.update(0, image()).valid
    tr.model.drop = set(range(16))
    obs = ck.LineObservation(
        points=[], segments=[
            ((150, 220), (1750, 260)), ((160, 180), (200, 820)),
            ((1650, 200), (1710, 850)), ((350, 280), (390, 750)),
        ],
    )
    H, H_inv2, families, quality = tr._line_aid(obs, (1080, 1920), "pan")
    # 与合成 H 不匹配的任意线段不应被当作支撑。
    assert H is None or families >= tr.cfg.min_line_families
    assert tr._line_aid(obs, (1080, 1920), "hard_cut")[0] is None


def test_plausibility_rejects_degenerate_homography():
    tr = tracker(np.eye(3))
    bad = np.asarray([[1, 0, 960], [0, 1, 540], [0, 0, 1e-9]], float)
    assert not tr._plausible(bad, (1080, 1920))


def test_result_json_has_no_nan_numbers(H_inv):
    import json

    tr = tracker(H_inv)
    tr.model.drop = {0}
    result = tr.update(0, image()).to_dict()
    encoded = json.dumps(result, allow_nan=False)
    assert "NaN" not in encoded and "Infinity" not in encoded
    assert result["keypoints"][0]["pixel"] is None


def test_orb_motion_handles_textured_pan_and_cut():
    tr = ck.CourtKeypointTracker(ck.KeypointConfig(weights="missing.onnx"))
    rng = np.random.default_rng(42)
    textured = rng.integers(0, 256, (270, 480, 3), dtype=np.uint8)
    assert tr._motion(textured)[0] == "initial"
    mode, flow, scale = tr._motion(np.roll(textured, 4, axis=1))
    assert mode in ("pan", "static", "zoom", "hard_cut")
    assert np.all(np.isfinite(flow)) and np.isfinite(scale)
    mode, flow, _ = tr._motion(255 - textured)
    assert mode == "hard_cut"


def draw_court(H_inv):
    rgb = np.full((1080, 1920, 3), 45, np.uint8)
    for a, b in ck.COURT_SEGMENTS:
        p = ck._project(H_inv, np.asarray([a, b])).astype(int)
        cv2.line(rgb, tuple(p[0]), tuple(p[1]), (235, 235, 235), 3)
    return rgb


def test_classical_full_court_fallback_uses_lines_and_ransac(H_inv):
    tr = ck.CourtKeypointTracker(ck.KeypointConfig(
        weights="missing.onnx", classical_hypotheses=24))
    rgb = draw_court(H_inv)
    obs = ck.detect_classical(rgb, tr.cfg)
    assert len(obs.points) >= 6 and len(obs.segments) >= 4
    H, H_inv2, matched, err = tr._solve_unlabeled(obs.points, rgb.shape[:2], 0)
    assert H is not None and H_inv2 is not None
    assert len(matched) >= 4 and err < tr.cfg.max_reproj_px
    families, quality, _shift, residual = tr._line_support(H_inv2, obs, rgb.shape[:2])
    assert families >= 2 and quality >= 0.35
    assert residual < max(26.0, np.hypot(*rgb.shape[:2]) * tr.cfg.line_gate_frac)
    tr._H, tr._H_inv = H, H_inv2
    line_obs = ck.LineObservation(points=[], segments=obs.segments)
    aided_H, aided_inv, aided_families, aided_residual = tr._line_aid(
        line_obs, rgb.shape[:2], "pan")
    assert aided_H is not None and aided_inv is not None
    assert aided_families >= 2
    assert aided_residual < max(26.0, np.hypot(*rgb.shape[:2]) * tr.cfg.line_gate_frac)


def test_online_classical_pipeline_publishes_homography(H_inv):
    tr = ck.CourtKeypointTracker(ck.KeypointConfig(
        weights="missing.onnx", classical_hypotheses=24))
    result = tr.update(0, draw_court(H_inv))
    assert result.valid, result.reason
    assert result.source == "classical_ransac"
    assert result.H is not None and result.H_inv is not None
    assert result.inliers >= 4


def test_model_solve_failure_still_runs_classical_fallback(H_inv):
    tr = tracker(H_inv, every_n=1, classical_hypotheses=24)
    # Five confident but degenerate detections pass the count gate while no
    # semantic homography can be fitted. They must not suppress line fallback.
    tr.model.predict = lambda _rgb: [
        ck.DetectedKeypoint(name, (100.0, 100.0), 0.99)
        for name, _world in ck.KEYPOINT_SPECS[:5]
    ]

    result = tr.update(0, draw_court(H_inv))

    assert result.valid, result.reason
    assert result.source in {"classical", "classical_ransac", "keypoints+lines"}
    assert result.H is not None and result.H_inv is not None


def test_occlusion_glare_and_broken_lines_use_prior_matching(H_inv):
    tr = tracker(H_inv)
    rgb = draw_court(H_inv)
    # 球员遮挡少量交点。
    for i, (u, v) in enumerate(ck._project(H_inv, ck.WORLD_POINTS)):
        if i % 3 == 0:
            cv2.rectangle(rgb, (int(u) - 18, int(v) - 28),
                          (int(u) + 18, int(v) + 28), (25, 25, 25), -1)
    # 反光高亮和断续白线；RANSAC 应只保留受支撑的交点。
    glare = np.full_like(rgb, 210)
    rgb = cv2.addWeighted(rgb, 0.88, glare, 0.12, 0)
    for i, (a, b) in enumerate(ck.COURT_SEGMENTS):
        if i % 7 == 0:
            p = ck._project(H_inv, np.asarray([a, b])).astype(int)
            cv2.line(rgb, tuple(p[0]), tuple(p[1]), (45, 45, 45), 12)
    tr._H, tr._H_inv = np.linalg.inv(H_inv), H_inv
    obs = ck.detect_classical(rgb, tr.cfg)
    named = tr._match_prior(obs.points, rgb.shape[:2])
    H, H_inv2, matched, err = tr._solve_labeled(named, rgb.shape[:2], tr.cfg.min_matches)
    assert H is not None and len(matched) >= 4 and err < tr.cfg.max_reproj_px


def test_parallel_line_fragments_do_not_count_as_two_families(H_inv):
    tr = tracker(H_inv)
    assert tr.update(0, image()).valid
    a, b = ck._project(H_inv, np.asarray([[-14, -7.5], [14, -7.5]]))
    midpoint = (a + b) / 2
    obs = ck.LineObservation(points=[], segments=[
        (tuple(a), tuple(midpoint)), (tuple(midpoint), tuple(b)),
    ])
    families, _quality, _shift, _residual = tr._line_support(tr._H_inv, obs, (1080, 1920))
    assert families == 1
    assert tr._line_aid(obs, (1080, 1920), "pan")[0] is None


def test_stale_reused_calibration_expires(H_inv):
    import time

    tr = tracker(H_inv, every_n=100)
    assert tr.update(0, image()).valid
    tr._last_valid = time.time() - 2.0
    tr._motion = lambda rgb: ("static", np.zeros(2), 1.0)
    result = tr.update(1, image())
    assert not result.valid and result.reason == "stale" and result.H is None


def test_skipped_pan_accumulates_flow_and_updates_homography(H_inv):
    tr = tracker(H_inv, every_n=100)
    assert tr.update(0, image()).valid
    previous = tr._H_inv.copy()
    tr._motion = lambda rgb: ("pan", np.asarray([8.0, -4.0]), 1.0)
    first = tr.update(1, image())
    second = tr.update(2, image())
    assert first.valid and second.valid
    assert not np.array_equal(previous, tr._H_inv)
    np.testing.assert_allclose(tr._H_inv[:2, 2], previous[:2, 2] + [16.0, -8.0], atol=1e-6)


def test_onnx_output_contract_and_sidecar_validation(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    from bsk.perceive import onnx_sessions

    weights = tmp_path / "court.onnx"
    weights.write_bytes(b"test")
    meta = {
        "input_hw": [32, 64], "preprocess": "rgb_0_255",
        "coordinate_space": "normalized", "keypoint_ids": list(ck.KEYPOINT_IDS),
    }
    weights.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    pixels = np.column_stack([np.linspace(10, 60, 16), np.linspace(4, 28, 16)])
    output = np.column_stack([pixels[:, 0] / 100, pixels[:, 1] / 80,
                              np.full(16, 0.9)]).astype(np.float32)[None]

    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name="images")]

        def run(self, names, feed):
            assert feed["images"].shape == (1, 3, 32, 64)
            return [output]

    monkeypatch.setattr(onnx_sessions, "get_session", lambda *a, **k: (Session(), "stub"))
    model = ck.CourtKeypointModel(ck.KeypointConfig(weights=weights))
    assert model.ready
    points = model.predict(image(h=80, w=100))
    assert len(points) == 16
    np.testing.assert_allclose([p.pixel for p in points], pixels, atol=1e-5)

    meta["coordinate_space"] = "input_pixels"
    weights.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    output = np.column_stack([pixels[:, 0] * 64 / 100, pixels[:, 1] * 32 / 80,
                              np.full(16, 0.9)]).astype(np.float32)[None]
    model2 = ck.CourtKeypointModel(ck.KeypointConfig(weights=weights))
    np.testing.assert_allclose([p.pixel for p in model2.predict(image(h=80, w=100))],
                               pixels, atol=1e-5)

    meta["keypoint_ids"] = list(reversed(ck.KEYPOINT_IDS))
    weights.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    invalid = ck.CourtKeypointModel(ck.KeypointConfig(weights=weights))
    assert not invalid.ready and "keypoint_ids" in invalid.error


class StubLineTracker:
    def __init__(self):
        self.calls = 0
        self.cal = None
        self.stopped = False

    def update(self, n, rgb):
        self.calls += 1
        return self.cal

    def stop(self):
        self.stopped = True


class StubPoseModel:
    def __call__(self, rgb, thr):
        return []


class StubPersonTracker:
    def assign(self, boxes, frame_idx=None):
        return []


def test_realtime_pipeline_prefers_keypoints_and_clears_line_history(monkeypatch, H_inv):
    from bsk import server_rt

    tr = tracker(H_inv)
    valid = tr.update(0, image())
    line = StubLineTracker()
    line.cal = valid.to_calibration()
    vid = "test-kp-pipeline"
    server_rt._calib_trackers[vid] = line
    roi = server_rt.get_roi_tracker(vid)
    roi.court_mask = np.ones((1080, 1920), np.uint8)
    roi.prev_gray = np.zeros((270, 480), np.uint8)
    monkeypatch.setattr(server_rt, "_H_OVERRIDE_PATH", None)
    monkeypatch.setattr(server_rt, "_get_court_keypoint_tracker", lambda *a: tr)
    monkeypatch.setattr(server_rt, "_get_rtmo", lambda: StubPoseModel())
    monkeypatch.setattr(server_rt, "_get_tracker", lambda *a: StubPersonTracker())
    monkeypatch.setattr(server_rt, "_get_jersey", lambda *a: None)

    result = server_rt.analyze_frame_realtime(vid, 0, image(), 0, backend="accurate")
    assert result.court_keypoints.valid
    assert result.court_keypoints.frame == 0
    assert result.calib is not None
    assert result.calib.source.startswith("court_keypoints:")
    assert line.calls == 0

    cut = ck.empty_result(1, 1.0, 1, "hard_cut", "hard_cut",
                           "hard_cut", "onnx", tr.stats)
    monkeypatch.setattr(tr, "update", lambda n, rgb: cut)
    result = server_rt.analyze_frame_realtime(vid, 1, image(value=230), 1, backend="accurate")
    assert result.court_keypoints is cut and result.calib is None
    assert vid not in server_rt._calib_trackers
    assert line.stopped
    assert roi.court_mask is None and roi.prev_gray is None
    server_rt._court_kp_trackers.pop((vid, "accurate"), None)


def test_realtime_valid_hard_cut_still_clears_line_history(monkeypatch, H_inv):
    from bsk import server_rt

    tr = tracker(H_inv)
    valid = tr.update(0, image(value=30))
    valid.mode = valid.transition = "hard_cut"
    line = StubLineTracker()
    line.cal = valid.to_calibration()
    vid = "test-kp-valid-cut"
    server_rt._calib_trackers[vid] = line
    roi = server_rt.get_roi_tracker(vid)
    roi.court_mask = np.ones((1080, 1920), np.uint8)
    roi.prev_gray = np.zeros((270, 480), np.uint8)
    monkeypatch.setattr(server_rt, "_H_OVERRIDE_PATH", None)
    monkeypatch.setattr(server_rt, "_get_court_keypoint_tracker", lambda *a: tr)
    monkeypatch.setattr(tr, "update", lambda n, rgb: valid)
    monkeypatch.setattr(server_rt, "_get_rtmo", lambda: StubPoseModel())
    monkeypatch.setattr(server_rt, "_get_tracker", lambda *a: StubPersonTracker())
    monkeypatch.setattr(server_rt, "_get_jersey", lambda *a: None)

    result = server_rt.analyze_frame_realtime(vid, 1, image(value=230), 1, backend="accurate")
    assert result.court_keypoints.valid
    assert result.calib is not None and result.calib.source.startswith("court_keypoints:")
    assert vid not in server_rt._calib_trackers
    assert line.stopped
    assert roi.court_mask is None and roi.prev_gray is None
    server_rt._court_kp_trackers.pop((vid, "accurate"), None)


def test_realtime_exception_invalidates_cached_keypoint_h(monkeypatch, H_inv):
    from bsk import server_rt

    tr = tracker(H_inv, every_n=5, retry_every=100)
    assert tr.update(0, image()).valid
    original_update = tr.update

    def fail_update(n, rgb):
        raise RuntimeError("synthetic inference failure")

    vid = "test-kp-runtime-error"
    old_line = StubLineTracker()
    old_line.cal = tr._last_result.to_calibration()
    fresh_line = StubLineTracker()
    server_rt._calib_trackers[vid] = old_line
    roi = server_rt.get_roi_tracker(vid)
    roi.court_mask = np.ones((1080, 1920), np.uint8)
    roi.prev_gray = np.zeros((270, 480), np.uint8)
    monkeypatch.setattr(server_rt, "_H_OVERRIDE_PATH", None)
    monkeypatch.setattr(server_rt, "_get_court_keypoint_tracker", lambda *a: tr)
    monkeypatch.setattr(server_rt, "_get_calib_tracker", lambda *a: fresh_line)
    monkeypatch.setattr(tr, "update", fail_update)
    monkeypatch.setattr(server_rt, "_get_rtmo", lambda: StubPoseModel())
    monkeypatch.setattr(server_rt, "_get_tracker", lambda *a: StubPersonTracker())
    monkeypatch.setattr(server_rt, "_get_jersey", lambda *a: None)

    result = server_rt.analyze_frame_realtime(vid, 1, image(), 1, backend="accurate")
    assert not result.court_keypoints.valid
    assert result.court_keypoints.reason.startswith("runtime_error:")
    assert tr._H is None and tr._H_inv is None
    assert vid not in server_rt._calib_trackers
    assert old_line.stopped
    assert roi.court_mask is None and roi.prev_gray is None
    assert fresh_line.calls == 1

    monkeypatch.setattr(tr, "update", original_update)
    result = server_rt.analyze_frame_realtime(vid, 2, image(), 2, backend="accurate")
    assert not result.court_keypoints.valid
    assert result.court_keypoints.reason == "no_calibration"
    assert tr._H is None and tr._H_inv is None
    server_rt._calib_trackers.pop(vid, None)
    server_rt._court_kp_trackers.pop((vid, "accurate"), None)


def test_keypoints_json_endpoint_is_read_only(monkeypatch, H_inv):
    import json
    from types import SimpleNamespace
    from bsk import server

    tr = tracker(H_inv)
    kp = tr.update(7, image())
    cached = SimpleNamespace(court_keypoints=kp)
    monkeypatch.setattr(server.lib, "table", lambda vid: SimpleNamespace(clamp=lambda n: n))
    monkeypatch.setattr(server, "get_cached", lambda *a: cached)
    monkeypatch.setattr(server, "court_keypoints_config", lambda: {"mode": "auto"})
    response = server.court_keypoints_json("v", 7, "accurate")
    data = json.loads(response.body)
    assert data["status"] == "ok" and data["valid"] is True
    assert data["frame"] == 7 and len(data["keypoints"]) == 16
    assert data["config"]["mode"] == "auto"
