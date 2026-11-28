"""球员候选、ID 关联和骨架绑定的因果回归测试（假模型，不依赖权重）。"""

from __future__ import annotations

import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from bsk.analyze import Tracker, _feet_in_court
from bsk.perceive.player_gate import (
    conservative_court_polygon, ground_point, player_quality, rfdetr_player_gate,
    safe_keypoints,
)
from bsk.perceive.persons import Person
from bsk.perceive.pose_filter import PoseFilter
from bsk.perceive.rfdetr import Detection


def _kpts(x: float, bottom: float, conf: float = 0.9) -> np.ndarray:
    kp = np.zeros((17, 3), np.float32)
    kp[:, 0] = x + 30
    kp[:, 1] = bottom - 145 + np.arange(17) * 8
    kp[:, 2] = conf
    kp[15] = (x + 22, bottom, conf)
    kp[16] = (x + 40, bottom, conf)
    return kp


def _person(x: float, bottom: float, score: float = 0.9, conf: float = 0.9) -> Person:
    return Person((x, bottom - 160, x + 60, bottom), score, _kpts(x, bottom, conf))


def test_global_tracking_is_order_independent_and_one_to_one():
    tr = Tracker()
    first = tr.assign([(0, 0, 60, 160), (300, 0, 360, 160)])
    assert first == [1, 2]
    # 模型输出顺序变化不能改变身份。
    second = tr.assign([(305, 0, 365, 160), (5, 0, 65, 160)])
    assert second == [2, 1]
    assert len(set(second)) == 2


def test_motion_prediction_keeps_fast_running_player_id():
    tr = Tracker()
    assert tr.assign([(0, 0, 60, 160)], frame_idx=1) == [1]
    assert tr.assign([(40, 0, 100, 160)], frame_idx=2) == [1]
    # 与上一帧没有 IoU，但与两帧位移一致；旧逐框 IoU 跟踪会在这里换 ID。
    assert tr.assign([(90, 0, 150, 160)], frame_idx=3) == [1]


def test_invalid_boxes_do_not_shift_detection_ids():
    tr = Tracker()
    ids = tr.assign([(np.nan, 0, 10, 10), (0, 0, 60, 160)])
    assert ids == [-1, 1]


def test_ground_point_prefers_ankles_and_rejects_bad_ankles():
    box = (1260, 400, 1420, 560)
    kp = _kpts(1260, 560)
    kp[15] = (1280, 550, 0.9)
    kp[16] = (1400, 555, 0.9)
    assert ground_point(box, kp) == pytest.approx((1340, 552.5))
    kp[15, 2] = kp[16, 2] = 0.1
    assert ground_point(box, kp) == (1340, 560)
    kp[15] = (1280, 450, 0.9)  # 跑到身体上方的错误踝点
    kp[16, 2] = 0.1
    assert ground_point(box, kp) == (1340, 560)


def test_partial_court_roi_extends_near_side_without_changing_normal_hull():
    normal = np.array([[100, 100], [1100, 100], [1100, 600], [100, 600]], np.float32)
    np.testing.assert_array_equal(conservative_court_polygon(normal, (720, 1280)), normal)
    partial = np.array([[391, 144], [365, 64], [1060, 445], [732, 430]], np.float32)
    expanded = conservative_court_polygon(partial, (720, 1280))
    assert len(expanded) >= 4 and abs(cv2.contourArea(expanded)) > abs(cv2.contourArea(partial))
    # 近端真实球员落点被保留，远端观众席仍被排除。
    assert cv2.pointPolygonTest(expanded.reshape(-1, 1, 2), (725, 536), True) > 0
    assert cv2.pointPolygonTest(expanded.reshape(-1, 1, 2), (358, 273), True) < 0


def test_ankle_inside_court_accepts_box_bottom_outside():
    poly = np.array([[200, 100], [1320, 100], [1320, 700], [200, 700]], np.float32)
    box = (1250, 400, 1430, 560)
    kp = _kpts(1250, 560)
    kp[15] = (1280, 550, 0.9)
    kp[16, 2] = 0.1
    assert _feet_in_court(poly, box, 10, None) is False
    assert _feet_in_court(poly, box, 10, kp) is True


def test_player_quality_keeps_running_players_and_filters_bad_poses():
    running = _person(100, 500)
    running.bbox = (100, 330, 180, 500)  # 跑动展开后约 1.07，仍应是球员
    assert player_quality(running, (900, 1600), "accurate")
    assert player_quality(running, (900, 1600), "fast")
    assert not player_quality(_person(100, 500, conf=0.1), (900, 1600), "accurate")
    assert not player_quality(_person(100, 500, score=0.2), (900, 1600), "accurate")
    seated = _person(100, 500)
    seated.bbox = (100, 420, 320, 500)
    assert not player_quality(seated, (900, 1600), "accurate")
    upper_only = _person(100, 500)
    upper_only.kpts[11:, 2] = 0.1  # 看台/摄影师只测到上半身，不能仅凭肩腕点绑骨架
    assert not player_quality(upper_only, (900, 1600), "accurate")


def test_safe_keypoints_contains_nonfinite_values():
    kp = _kpts(100, 500)
    kp[5, 0] = np.nan
    clean = safe_keypoints(kp)
    assert np.isfinite(clean).all()
    assert clean[5, 2] == 0
    assert np.isnan(kp[5, 0])  # 不修改模型返回的原始数组


def test_rfdetr_gate_keeps_only_current_frame_players():
    player = _person(700, 550)
    spectator = _person(720, 550)
    referee = _person(740, 550)
    detections = [
        Detection(3, "player", 0.9, player.bbox),
        Detection(8, "referee", 0.9, referee.bbox),
        Detection(2, "number", 0.9, (710, 300, 730, 330)),
    ]
    assert rfdetr_player_gate([spectator, referee, player], detections) == [player]
    assert rfdetr_player_gate([player], []) == []
    # 真实权重中的裁判误报约 0.38，不能靠降低阈值把它放进轨迹。
    assert rfdetr_player_gate(
        [spectator], [Detection(3, "player", 0.39, spectator.bbox)]
    ) == []
    # 已确认球员在运动模糊帧降到 0.35 时继续保留；陌生低分候选仍不能入场。
    assert rfdetr_player_gate(
        [player], [Detection(3, "player", 0.35, player.bbox)],
        known_boxes=[player.bbox],
    ) == [player]
    assert rfdetr_player_gate(
        [player], [Detection(3, "player", 0.35, player.bbox)],
        known_boxes=[(1000, 100, 1060, 260)],
    ) == []
    # 同位置 referee 置信度更高时，必须在人类类别间先 NMS，而不是只对 player 类 NMS。
    assert rfdetr_player_gate(
        [referee],
        [Detection(3, "player", 0.70, referee.bbox),
         Detection(8, "referee", 0.90, referee.bbox)],
    ) == []
    # 同一球员的动作类重复框不能放行相邻人员。
    assert rfdetr_player_gate(
        [spectator, player],
        [Detection(3, "player", 0.90, player.bbox),
         Detection(5, "player-jump-shot", 0.85, player.bbox)],
    ) == [player]


class _PoseSequence:
    def __init__(self, frames):
        self.frames = list(frames)

    def __call__(self, rgb, thr):
        return self.frames.pop(0)


class _Roi:
    court_mask = None
    prev_gray = None
    stable_count = 0

    def update_detections(self, detections):
        self.detections = detections


@pytest.fixture
def realtime_stub(monkeypatch):
    from bsk import server_rt

    h_inv = np.array([[40, 0, 800], [0, 40, 450], [0, 0, 1]], np.float64)
    cal = SimpleNamespace(H_inv=h_inv, court_model="FIBA", timestamp=time.time())
    tracker = Tracker()
    pose_filter = PoseFilter()
    roi = _Roi()
    monkeypatch.setattr(server_rt, "_H_OVERRIDE_PATH", None)
    monkeypatch.setattr(server_rt, "_override_calibration", lambda shape: cal)
    monkeypatch.setattr(server_rt, "_get_tracker", lambda vid: tracker)
    monkeypatch.setattr(server_rt, "_get_pose_filter", lambda *args: pose_filter)
    monkeypatch.setattr(server_rt, "get_roi_tracker", lambda vid: roi)
    monkeypatch.setattr(server_rt, "_get_jersey", lambda *args: None)
    monkeypatch.setattr(server_rt, "_get_rfdetr", lambda: None)
    monkeypatch.setattr(server_rt, "_POSE_FILTER_ON", True)
    return server_rt, tracker, pose_filter, roi


def _run_frame(server_rt, model, n):
    return server_rt.analyze_frame_realtime(
        "binding-test", n, np.zeros((900, 1600, 3), np.uint8), n * 33333,
        enable_calib=False, backend="accurate",
    )


def test_offcourt_never_enters_tracking_and_visible_ids_match_poses(realtime_stub, monkeypatch):
    server_rt, tracker, _, roi = realtime_stub
    offcourt = _person(40, 550)     # 脚点 x=62，明显在球场左边线 x=240 外
    player = _person(700, 550)
    model = _PoseSequence([
        [offcourt, player],  # 场外人员排在前面，旧实现会先拿 ID
        [_person(710, 550), offcourt],
    ])
    monkeypatch.setattr(server_rt, "_get_rtmo", lambda: model)
    first = _run_frame(server_rt, None, 1)
    second = _run_frame(server_rt, None, 2)

    assert [o.track_id for o in first.objects] == [1]
    assert [p.track_id for p in first.poses] == [1]
    assert [o.track_id for o in second.objects] == [1]
    assert [p.track_id for p in second.poses] == [1]
    assert set(tracker._live) == {1}
    assert all(p.bbox[0] >= 240 for p in roi.detections)


def test_occluded_prediction_is_dropped_when_it_leaves_court(realtime_stub, monkeypatch):
    server_rt, _, pose_filter, _ = realtime_stub
    # 连续三帧确认轨迹，并向右线附近移动；第四帧无检测时应外推到线外。
    model = _PoseSequence([
        [_person(1140, 550)],
        [_person(1200, 550)],
        [_person(1260, 550)],
        [_person(1320, 550)],
        [_person(1340, 550)],
        [],
    ])
    monkeypatch.setattr(server_rt, "_get_rtmo", lambda: model)
    for n in (1, 2, 3, 4, 5):
        result = _run_frame(server_rt, None, n)
        assert len(result.poses) == 1
    result = _run_frame(server_rt, None, 6)
    assert result.poses == []
    assert result.objects == []
    assert 1 not in pose_filter._tracks


def test_without_calibration_valid_players_still_get_skeletons(realtime_stub, monkeypatch):
    server_rt, _, _, _ = realtime_stub
    monkeypatch.setattr(server_rt, "_override_calibration", lambda shape: None)
    monkeypatch.setattr(server_rt, "_get_rtmo", lambda: _PoseSequence([[_person(700, 550)]]))
    result = _run_frame(server_rt, None, 1)
    assert len(result.poses) == 1
    assert result.objects[0].track_id == result.poses[0].track_id


def test_stale_track_cannot_relax_new_low_confidence_candidate(realtime_stub, monkeypatch):
    server_rt, tracker, _, _ = realtime_stub
    player = _person(700, 550)
    tracker._live[1] = player.bbox
    tracker._last_frame[1] = 1
    tracker._miss[1] = 0
    detector = SimpleNamespace(
        ready=True,
        predict=lambda rgb, conf: [Detection(3, "player", 0.35, player.bbox)],
    )
    monkeypatch.setattr(server_rt, "_get_rtmo", lambda: _PoseSequence([[player]]))
    monkeypatch.setattr(server_rt, "_get_rfdetr", lambda: detector)
    monkeypatch.setattr(server_rt, "_rfdetr", detector)
    result = _run_frame(server_rt, None, 20)  # 远超 max_lost，旧轨迹不能给新候选背书
    assert result.objects == result.poses == []
    assert tracker._live == {}


def test_rfdetr_prevents_offcourt_person_from_entering_tracking(realtime_stub, monkeypatch):
    server_rt, tracker, _, _ = realtime_stub
    player = _person(700, 550)
    spectator = _person(720, 550)  # 即使落在错误标定 ROI 内，也没有球员类别语义
    player_next = _person(705, 550)
    spectator_next = _person(725, 550)
    detector_frames = [
        [Detection(3, "player", 0.9, player.bbox)],
        [Detection(3, "player", 0.35, player_next.bbox),
         Detection(3, "player", 0.38, spectator_next.bbox)],
    ]
    detector = SimpleNamespace(
        ready=True,
        predict=lambda rgb, conf: detector_frames.pop(0),
    )
    monkeypatch.setattr(server_rt, "_get_rtmo", lambda: _PoseSequence([
        [spectator, player], [spectator_next, player_next],
    ]))
    monkeypatch.setattr(server_rt, "_get_rfdetr", lambda: detector)
    monkeypatch.setattr(server_rt, "_rfdetr", detector)
    first = _run_frame(server_rt, None, 1)
    second = _run_frame(server_rt, None, 2)
    for result in (first, second):
        assert len(result.objects) == len(result.poses) == 1
        assert result.objects[0].track_id == result.poses[0].track_id == 1
        assert result.poses[0].model.startswith("rtmo_s+rfdetr_s")
    assert second.objects[0].bbox_px[0] < 720  # 没有被相邻场外人员替换
    assert set(tracker._live) == {1}
