"""3D 提升人体约束与离线链路的回归测试（不依赖模型权重）。"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from bsk import schema
from bsk.calibrate import Calibration
from bsk.perceive.pose_3d import Pose3DLifter
from bsk.perceive.pose_prior import MotionPrior, MotionPriorConfig, rom_violations


def legal_body(stature: float = 1.85) -> np.ndarray:
    s = float(stature)
    x = np.zeros((17, 3))
    x[11], x[12] = (0, 0.055 * s, 1.0), (0, -0.055 * s, 1.0)
    x[13], x[14] = (0, 0.055 * s, 1 - 0.245 * s), (0, -0.055 * s, 1 - 0.245 * s)
    x[15], x[16] = (0, 0.055 * s, 1 - 0.491 * s), (0, -0.055 * s, 1 - 0.491 * s)
    x[5], x[6] = (0, 0.105 * s, 1 + 0.288 * s), (0, -0.105 * s, 1 + 0.288 * s)
    x[7], x[8] = (0, 0.105 * s, 1 + 0.102 * s), (0, -0.105 * s, 1 + 0.102 * s)
    x[9], x[10] = (0, 0.105 * s, 1 - 0.044 * s), (0, -0.105 * s, 1 - 0.044 * s)
    x[0] = (0, 0, 1 + 0.400 * s)
    x[1], x[2] = x[0] + (0.01, 0.02, 0.02), x[0] + (0.01, -0.02, 0.02)
    x[3], x[4] = x[0] + (0, 0.03, 0.01), x[0] + (0, -0.03, 0.01)
    return x


def synth_camera():
    pos = np.array([0.0, -20.0, 10.0])
    fwd = -pos / np.linalg.norm(pos)
    right = np.cross(fwd, (0.0, 0.0, 1.0))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rot = np.stack([right, -up, fwd])
    trans = -rot @ pos
    k = np.array([[900.0, 0, 480], [0, 900.0, 270], [0, 0, 1]])
    homography = k @ np.column_stack([rot[:, 0], rot[:, 1], trans])
    return rot, trans, k, homography


def test_legal_body_is_unchanged() -> None:
    x = legal_body()
    out, stats = MotionPrior().pose_only(x, 1.85)
    np.testing.assert_allclose(out, x, atol=1e-12)
    assert stats["rom"] == 0
    assert stats["bone_fix"] == 0


def test_reversed_knee_and_elbow_are_corrected() -> None:
    x = legal_body()
    for joint, xyz in ((15, (0.35, 0.055 * 1.85, 0.30)),
                       (9, (-0.35, 0.105 * 1.85, 1.0))):
        bad = x.copy()
        bad[joint] = xyz
        assert max(rom_violations(bad, S=1.85).values()) > 8.0
        out, stats = MotionPrior(MotionPriorConfig(pose_gain=1.0)).pose_only(bad, 1.85)
        assert stats["rom"] >= 1
        assert max(rom_violations(out, S=1.85).values()) < 1e-6


def test_full_pipeline_does_not_discard_rom_result() -> None:
    """回归：apply 的修正循环必须把 cur 写回 out，不能只更新统计。"""
    x = legal_body()
    x[15] = (0.35, 0.055 * 1.85, 0.30)
    dirs = np.tile(np.array([0.0, 0.0, 1.0]), (17, 1))
    out, stats = MotionPrior(MotionPriorConfig(pose_gain=1.0)).apply(
        x, dirs, np.zeros(3), 0.0, {}, 1.85, px_per_m=10.0)
    assert stats["rom"] >= 1
    assert max(rom_violations(out, S=1.85).values()) < 1e-6


def test_support_leg_ik_preserves_bone_lengths() -> None:
    x = legal_body()
    prior = MotionPrior()
    pel = 0.5 * (x[11] + x[12])
    st = {"feet": {15: {"xy": x[15, :2].copy(), "pel": pel[:2].copy()}},
          "plant_delta": np.zeros(2)}
    x[15, 0] += 0.05
    before = np.linalg.norm(x[15, :2] - pel[:2])
    out, fixed = prior._plant_anchor(x, 1.85, 1.0 / 30.0, st)
    after = np.linalg.norm(out[15, :2] - 0.5 * (out[11] + out[12])[:2])
    assert fixed > 0.0 and after < before
    np.testing.assert_allclose(np.linalg.norm(out[13] - out[11]), 0.245 * 1.85, atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(out[15] - out[13]), 0.246 * 1.85, atol=1e-8)


def test_two_frame_pipeline_has_no_temporal_error() -> None:
    rot, trans, k, homography = synth_camera()
    lifter = Pose3DLifter(image_size=(960, 540), prior=True, fps=30.0)
    for frame, dx in ((0, 0.0), (1, 0.05)):
        body = legal_body()
        body[:, 0] += dx
        xc = body @ rot.T + trans
        uv = np.column_stack([k[0, 0] * xc[:, 0] / xc[:, 2] + k[0, 2],
                              k[1, 1] * xc[:, 1] / xc[:, 2] + k[1, 2]])
        box = (uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max())
        res = lifter.lift_result(np.column_stack([uv, np.full(17, 0.9)]), box,
                                 homography, track_id=1, frame=frame)
        assert res is not None and "error" not in res.prior
        assert np.isfinite(res.kpts3d).all()


def test_prior_switch_and_accounting() -> None:
    off = Pose3DLifter(prior=False)
    on = Pose3DLifter(prior=True)
    assert off.prior is None and on.prior is not None
    on._note_prior({"rom": 4})
    on._note_prior({"rom": 0})
    st = on.prior_status()
    assert st["frames"] == 2 and st["hit"] == 1
    assert st["hit_rate_pct"] == 50.0 and st["rom_per_frame"] == 2.0


def test_offline_payload_uses_stored_pose_without_detection() -> None:
    from bsk import server_rt

    rot, trans, k, homography = synth_camera()
    body = legal_body()
    xc = body @ rot.T + trans
    uv = np.column_stack([k[0, 0] * xc[:, 0] / xc[:, 2] + k[0, 2],
                          k[1, 1] * xc[:, 1] / xc[:, 2] + k[1, 2]])
    pose = schema.Pose2D(video_id="offline-test", frame=1, track_id=7,
                         kpts=np.column_stack([uv, np.full(17, 0.9)]).tolist())
    cal = Calibration(H=np.linalg.inv(homography).tolist(), H_inv=homography.tolist(),
                      reproj_error_px=0.0, keypoints_used=4, court_model="FIBA",
                      timestamp=time.time())
    rgb = np.zeros((540, 960, 3), dtype=np.uint8)
    fake = SimpleNamespace(
        get=lambda _vid: SimpleNamespace(width=960, height=540, fps=25.0),
        table=lambda _vid: SimpleNamespace(clamp=lambda n: n, pts_us=[0]),
        poses=lambda _vid: {1: [pose]},
        frame_records=lambda _vid: {},
        picker=lambda _vid: SimpleNamespace(frames_at=lambda ids, size=None: [(ids[0], 0, rgb)]),
    )
    server_rt._cache._cache.clear()
    server_rt._lifters.pop("offline-test", None)
    try:
        with patch("bsk.store.Library", return_value=fake), \
                patch("bsk.server_rt._calibrate_timeline_frame", return_value=cal), \
                patch("bsk.server_rt.profile_allows_3d", return_value=True):
            data = server_rt.frame_3d_payload("offline-test", 1, "fast")
        assert data["status"] == "ok", data
        assert data["calib_ok"] and data["poses_2d"] == 1
        assert len(data["players"]) == 1
        player = data["players"][0]
        assert player["track_id"] == 7 and len(player["kpts"]) == 17
        assert player["bbox_px"] is not None
        assert data["detail"]["prior"]["enabled"] is True
        assert data["detail"]["prior"]["frames"] == 1
        assert "rom" in player["prior"]
    finally:
        server_rt._cache._cache.clear()
        server_rt._lifters.pop("offline-test", None)


def test_offline_payload_without_pose_skips_calibration() -> None:
    from bsk import server_rt

    rgb = np.zeros((540, 960, 3), dtype=np.uint8)
    fake = SimpleNamespace(
        get=lambda _vid: SimpleNamespace(width=960, height=540, fps=25.0),
        table=lambda _vid: SimpleNamespace(clamp=lambda n: n, pts_us=[0]),
        poses=lambda _vid: {},
        frame_records=lambda _vid: {},
        picker=lambda _vid: SimpleNamespace(frames_at=lambda ids, size=None: [(ids[0], 0, rgb)]),
    )
    server_rt._cache._cache.clear()
    try:
        with patch("bsk.store.Library", return_value=fake), \
                patch("bsk.server_rt.profile_allows_3d", return_value=True), \
                patch("bsk.server_rt._calibrate_timeline_frame",
                      side_effect=AssertionError("无姿态时不应标定")):
            data = server_rt.frame_3d_payload("offline-empty", 1, "fast")
            cached_data = server_rt.frame_3d_payload("offline-empty", 1, "fast")
        assert data["status"] == "no_poses" and data["players"] == []
        assert cached_data["status"] == "no_poses" and cached_data["players"] == []
        cached = server_rt.get_cached("offline-empty", 1, "fast")
        assert cached is not None and cached.status_3d["reason"] == "no_poses"
    finally:
        server_rt._cache._cache.clear()


def test_offline_calibration_error_is_contained() -> None:
    from bsk import server_rt

    rgb = np.zeros((540, 960, 3), dtype=np.uint8)
    kpts = [[480.0 + i, 270.0 + i, 0.9] for i in range(17)]
    pose = schema.Pose2D(video_id="offline-error", frame=1, track_id=1, kpts=kpts)
    fake = SimpleNamespace(
        get=lambda _vid: SimpleNamespace(width=960, height=540, fps=25.0),
        table=lambda _vid: SimpleNamespace(clamp=lambda n: n, pts_us=[0]),
        poses=lambda _vid: {1: [pose]},
        frame_records=lambda _vid: {},
        picker=lambda _vid: SimpleNamespace(frames_at=lambda ids, size=None: [(ids[0], 0, rgb)]),
    )
    server_rt._cache._cache.clear()
    try:
        with patch("bsk.store.Library", return_value=fake), \
                patch("bsk.server_rt.profile_allows_3d", return_value=True), \
                patch("bsk.server_rt._calibrate_timeline_frame",
                      side_effect=RuntimeError("标定内部异常")):
            data = server_rt.frame_3d_payload("offline-error", 1, "fast")
        assert data["status"] == "no_calib" and data["players"] == []
        assert server_rt.get_cached("offline-error", 1, "fast") is None
    finally:
        server_rt._cache._cache.clear()
