"""Shi-Tomasi 边线特征检测回归：合成图、异常输入和命令行。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from bsk.perceive.court_line_features import (
    FittedCourtLine,
    LineFeatureConfig,
    LineFeaturePoint,
    LineFeatureResult,
    _angle_distance_clusters,
    _line_intersections,
    _merge_nearby_lines,
    detect_line_features,
    visualize_line_features,
)


def noisy_court(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((480, 800, 3), 45, np.uint8)
    noise = rng.normal(0, 12, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.line(image, (50, 100), (750, 220), (240, 240, 240), 3)
    cv2.line(image, (400, 30), (440, 450), (240, 240, 240), 3)
    # 孤立白点形成可被角点检测到的离群特征，但不应形成直线。
    for x, y in ((80, 420), (720, 80), (260, 390), (680, 360)):
        cv2.circle(image, (x, y), 2, (245, 245, 245), -1)
    return image


def fast_config(**kwargs) -> LineFeatureConfig:
    # 历史回归默认覆盖自动 ROI 分支；固定多边形由新增测试单独验证。
    values = dict(ransac_iterations=150, max_corners=700, min_distance=7,
                  use_fixed_roi=False, line_angle_prior_deg=())
    values.update(kwargs)
    return LineFeatureConfig(**values)


def orthogonal_court() -> np.ndarray:
    image = np.full((480, 800, 3), 45, np.uint8)
    cv2.line(image, (60, 200), (740, 200), (240, 240, 240), 5)
    cv2.line(image, (400, 40), (400, 440), (240, 240, 240), 5)
    return image


def permissive_line_config(**kwargs) -> LineFeatureConfig:
    values = dict(
        ransac_iterations=150, max_corners=500, min_distance=3,
        min_line_points=4, min_cluster_points=4, min_line_length=80.0,
        min_span_frac=0.01, min_span_bins=2, use_roi_auto=False,
        use_fixed_roi=False, line_angle_prior_deg=(),
    )
    values.update(kwargs)
    return LineFeatureConfig(**values)


def test_detects_two_lines_and_rejects_noise_points():
    image = noisy_court()
    result = detect_line_features(image, fast_config())

    assert result.stats["lines"] == 2
    assert len(result.points) > 20
    assert len(result.inlier_points) == len(result.points)
    assert result.stats["filtered_corners"] > result.stats["corners"]
    # 孤立白点虽然通过 Shi-Tomasi，但不能形成最终直线内点。
    assert all(np.hypot(point.x - x, point.y - y) > 8
               for point in result.points for x, y in ((80, 420), (720, 80),
                                                        (260, 390), (680, 360)))
    expected = [np.asarray([700.0, 120.0]), np.asarray([40.0, 420.0])]
    expected = [direction / np.linalg.norm(direction) for direction in expected]
    matched = set()
    for line in result.lines:
        direction = np.asarray(line.direction)
        index = max(range(2), key=lambda i: abs(float(direction @ expected[i])))
        matched.add(index)
        assert abs(float(direction @ expected[index])) > 0.995
        assert line.inlier_count >= 8
        assert line.rms_error < 3.0
        assert np.all(np.isfinite(line.p0)) and np.all(np.isfinite(line.p1))
        for point in line.points:
            distance = abs(float((np.asarray([point.x, point.y]) - line.p0) @ line.normal))
            assert distance <= 5.0 + 1e-6
    assert matched == {0, 1}


def test_target_line_prior_selects_only_matching_direction():
    image = noisy_court()
    result = detect_line_features(
        image, fast_config(), target_line=(50, 100, 750, 220)
    )
    assert result.stats["mode"] == "target_line"
    assert result.stats["roi_source"] == "none"
    assert len(result.lines) == 1
    line = result.lines[0]
    assert abs(line.direction[0]) > 0.98
    assert line.inlier_count >= 8
    target_normal = np.asarray([-120.0, 700.0])
    target_normal /= np.linalg.norm(target_normal)
    for point in result.points:
        distance = abs(float((np.asarray([point.x, point.y]) - [50.0, 100.0]) @ target_normal))
        assert distance <= 12.5 + 1e-6


def test_visualization_uses_red_points_and_blue_lines():
    image = noisy_court()
    result = detect_line_features(image, fast_config())
    output = visualize_line_features(image, result)
    assert output.shape == image.shape
    assert np.any(np.all(output == (0, 0, 255), axis=2))  # BGR red
    assert np.any(np.all(output == (255, 0, 0), axis=2))  # BGR blue
    assert not np.shares_memory(output, image)


def test_blank_noise_and_tiny_images_are_safe():
    cfg = fast_config()
    blank = np.full((240, 320, 3), 120, np.uint8)
    assert detect_line_features(blank, cfg).lines == []

    rng = np.random.default_rng(91)
    noise = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    assert detect_line_features(noise, cfg).lines == []

    tiny = detect_line_features(np.zeros((4, 4, 3), np.uint8), cfg)
    assert tiny.lines == [] and tiny.stats["reason"] == "image_too_small"


def test_roi_limits_valid_feature_points():
    image = noisy_court()
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[:, 350:470] = 1
    result = detect_line_features(image, fast_config(), roi=mask)
    assert result.points
    assert all(350 <= point.x <= 470 for point in result.points)
    # 拟合线可以无限延伸，但输出端点必须裁剪在 ROI 内，不能画到观众席。
    assert all(350 <= value <= 470 for line in result.lines for value in (line.p0[0], line.p1[0]))


def test_full_image_mode_is_reported_without_auto_roi():
    result = detect_line_features(noisy_court(), fast_config(
        use_roi_auto=False, use_fixed_roi=False))
    assert result.stats["mode"] == "full_image"
    assert result.stats["roi_source"] == "none"


def test_colored_floor_roi_excludes_side_advertisement_and_top_stands():
    image = np.full((480, 800, 3), (28, 24, 22), np.uint8)
    # 紫色场地与深色观众席形成明确颜色对比；两侧广告牌故意使用低饱和白线。
    court = np.asarray([[90, 150], [710, 150], [760, 460], [40, 460]], np.int32)
    cv2.fillConvexPoly(image, court, (78, 58, 148))
    cv2.polylines(image, [court], True, (240, 240, 240), 3)
    cv2.line(image, (400, 150), (400, 460), (240, 240, 240), 3)
    cv2.line(image, (90, 305), (710, 305), (240, 240, 240), 3)
    for x in (4, 788):
        cv2.rectangle(image, (x, 130), (x + 8, 450), (235, 235, 235), -1)
        for y in range(145, 450, 28):
            cv2.line(image, (x, y), (x + 8, y), (20, 20, 20), 1)

    result = detect_line_features(image, fast_config())
    assert result.stats["roi_source"] == "auto"
    assert result.points
    assert all(point.y >= 135 for point in result.points)
    assert all(25 <= point.x <= 775 for point in result.points)
    assert len(result.lines) >= 2


def test_floor_support_rejects_player_and_stands_inside_fixed_roi():
    image = np.full((480, 800, 3), (24, 22, 20), np.uint8)
    court = np.asarray([[120, 120], [680, 120], [760, 460], [40, 460]], np.int32)
    cv2.fillConvexPoly(image, court, (55, 105, 185))  # BGR 木质场地
    cv2.polylines(image, [court], True, (240, 240, 240), 3)
    cv2.line(image, (400, 120), (400, 460), (240, 240, 240), 3)
    cv2.line(image, (100, 300), (700, 300), (240, 240, 240), 3)
    # 模拟磨损白线上的小缺口，让 Shi-Tomasi 在整条线上都有稳定响应。
    for y in range(140, 300, 30):
        cv2.line(image, (396, y), (404, y), (55, 105, 185), 1)
    for x in range(140, 700, 40):
        cv2.line(image, (x, 297), (x, 303), (55, 105, 185), 1)
    # 固定多边形内部、场地之外的看台白纹不能形成有效点。
    for x in (82, 682):
        cv2.rectangle(image, (x, 125), (x + 36, 205), (24, 22, 20), -1)
        for y in (140, 160, 180, 200):
            cv2.line(image, (x, y), (x + 36, y), (235, 235, 235), 2)
    player = np.asarray([[350, 320], [450, 320], [470, 410], [330, 410]], np.int32)
    cv2.fillConvexPoly(image, player, (245, 245, 245))
    cv2.line(image, (355, 350), (445, 350), (245, 245, 245), 6)
    cv2.putText(image, "7", (385, 385), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    rng = np.random.default_rng(17)
    image = np.clip(image.astype(np.int16) + rng.normal(0, 4, image.shape).astype(np.int16),
                    0, 255).astype(np.uint8)

    result = detect_line_features(image, LineFeatureConfig(
        ransac_iterations=150, max_corners=700, quality_level=0.005, min_distance=3,
    ))

    assert result.stats["floor_roi_used"] is True
    assert result.stats["obstacle_rejected"] > 0
    assert result.points
    assert len(result.lines) >= 2
    assert all(not (325 <= point.x <= 475 and 315 <= point.y <= 415)
               for point in result.points)
    assert all(not (((80 <= point.x <= 120) or (680 <= point.x <= 720))
                    and 120 <= point.y <= 210)
               for point in result.points)


def test_json_is_finite_and_contains_all_requested_outputs():
    result = detect_line_features(noisy_court(), fast_config())
    payload = result.to_dict()
    encoded = json.dumps(payload, allow_nan=False)
    assert "NaN" not in encoded and "Infinity" not in encoded
    assert {"lines", "points", "stats"} <= payload.keys()
    assert {"p0", "p1", "points", "inlier_count"} <= payload["lines"][0].keys()
    assert {"x", "y", "line_id"} <= payload["points"][0].keys()


@pytest.mark.parametrize("kwargs", [
    {"max_corners": 1},
    {"quality_level": 0.0},
    {"distance_threshold": 0.0},
    {"min_distance": 0},
    {"min_line_points": 1},
    {"min_span_bins": 7},
    {"min_cluster_points": 1},
    {"min_line_length": 0.0},
    {"white_neighborhood_size": 4},
    {"gradient_neighborhood_size": 2},
    {"min_gradient_strength": 0.0},
    {"angle_prior_tolerance_deg": 91.0},
    {"line_angle_prior_deg": (float("nan"),)},
    {"merge_angle_deg": -1.0},
    {"merge_distance_factor": 0.0},
])
def test_invalid_parameters_fail_fast(kwargs):
    with pytest.raises(ValueError):
        LineFeatureConfig(**kwargs)


def test_nonfinite_image_and_target_are_rejected():
    image = np.zeros((32, 32, 3), np.float32)
    image[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        detect_line_features(image)
    with pytest.raises(ValueError, match="target_line"):
        detect_line_features(np.zeros((32, 32, 3), np.uint8), target_line=(0, 0, 0, 0))


def test_detection_is_deterministic():
    image = noisy_court()
    first = detect_line_features(image, fast_config())
    second = detect_line_features(image, fast_config())
    assert [line.to_dict() for line in first.lines] == [line.to_dict() for line in second.lines]


def test_command_line_script_runs_and_saves_visualization(tmp_path: Path):
    source = tmp_path / "court.png"
    output = tmp_path / "result.png"
    assert cv2.imwrite(str(source), noisy_court())
    script = Path(__file__).with_name("detect_court_lines.py")
    completed = subprocess.run(
        [sys.executable, str(script), str(source), "--output", str(output),
         "--max-corners", "700", "--quality-level", "0.01",
         "--distance-threshold", "5", "--min-distance", "7",
         "--ransac-iterations", "150"],
        cwd=script.parent, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.exists() and cv2.imread(str(output)) is not None
    summary = json.loads(completed.stdout)
    assert summary["line_count"] >= 1


def test_auto_roi_blocks_top_advertisement_and_all_points_stay_inside():
    image = np.full((480, 800, 3), 45, np.uint8)
    cv2.line(image, (70, 180), (730, 300), (240, 240, 240), 3)
    cv2.line(image, (390, 160), (430, 450), (240, 240, 240), 3)
    # 实心白色广告牌加黑色网格会产生大量白线区域角点，但不应进入自动球场 ROI。
    cv2.rectangle(image, (90, 8), (710, 88), (245, 245, 245), -1)
    for x in range(120, 710, 45):
        cv2.line(image, (x, 8), (x, 88), (20, 20, 20), 2)
    for y in range(20, 88, 20):
        cv2.line(image, (90, y), (710, y), (20, 20, 20), 2)

    result = detect_line_features(image, fast_config())
    assert result.stats["roi_source"] == "auto"
    assert result.stats["roi_components"] >= 1
    assert result.points
    assert all(point.y > 115 for point in result.points)
    assert len(result.lines) >= 1


def test_hsv_filter_rejects_saturated_fake_line_inside_roi():
    image = noisy_court()
    # 黄色线位于两条白线围成的 ROI 内，但高饱和，必须被 HSV 前置过滤。
    cv2.line(image, (120, 330), (700, 390), (0, 200, 255), 4)
    result = detect_line_features(image, fast_config())
    assert result.stats["white_rejected"] > 0
    assert all(point.evidence >= 0.35 for point in result.points)
    target = np.asarray([580.0, 60.0])
    target /= np.linalg.norm(target)
    target_normal = np.asarray([-target[1], target[0]])
    target_rho = float(np.dot(target_normal, [120.0, 330.0]))
    for line in result.lines:
        line_rho = abs(float(np.dot(target_normal, line.p0)))
        assert abs(line_rho - target_rho) > 15.0


def test_explicit_roi_is_passed_directly_to_corner_detection(monkeypatch):
    image = noisy_court()
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[:, 350:470] = 255
    captured = {}
    original = cv2.goodFeaturesToTrack

    def wrapped(*args, **kwargs):
        captured["mask"] = kwargs.get("mask")
        return original(*args, **kwargs)

    monkeypatch.setattr(cv2, "goodFeaturesToTrack", wrapped)
    result = detect_line_features(image, fast_config(), roi=mask)
    assert result.stats["roi_source"] == "provided"
    assert captured["mask"] is not None
    assert captured["mask"].shape == mask.shape
    assert not captured["mask"][:, :300].any()
    assert all(350 <= point.x <= 470 for point in result.points)


def test_cluster_with_too_few_points_is_discarded():
    image = np.full((420, 720, 3), 45, np.uint8)
    cv2.line(image, (60, 180), (660, 220), (240, 240, 240), 4)
    cv2.line(image, (520, 280), (570, 300), (240, 240, 240), 3)
    result = detect_line_features(image, fast_config(min_line_points=12))
    assert len(result.lines) == 1
    assert result.lines[0].inlier_count >= 12
    assert result.stats["accepted_clusters"] <= result.stats["clusters"]


def test_cli_accepts_explicit_auto_roi_switch_and_mask(tmp_path: Path):
    source = tmp_path / "court.png"
    mask_path = tmp_path / "court_mask.png"
    output = tmp_path / "result.png"
    image = noisy_court()
    assert cv2.imwrite(str(source), image)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[:, 340:480] = 255
    assert cv2.imwrite(str(mask_path), mask)
    script = Path(__file__).with_name("detect_court_lines.py")
    completed = subprocess.run(
        [sys.executable, str(script), str(source), "--output", str(output),
         "--use-roi-auto", "--roi-mask", str(mask_path),
         "--max-corners", "700", "--quality-level", "0.01",
         "--distance-threshold", "5", "--min-distance", "7",
         "--ransac-iterations", "150"],
        cwd=script.parent, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["roi_source"] == "provided"
    assert summary["line_count"] >= 1


def test_polygon_roi_is_passed_directly_to_corner_detection(monkeypatch):
    image = noisy_court()
    polygon = np.asarray([[350, 40], [450, 40], [450, 440], [350, 440]], np.float32)
    captured = {}
    original = cv2.goodFeaturesToTrack

    def wrapped(*args, **kwargs):
        captured["mask"] = kwargs.get("mask")
        return original(*args, **kwargs)

    monkeypatch.setattr(cv2, "goodFeaturesToTrack", wrapped)
    result = detect_line_features(image, fast_config(), roi=polygon)
    assert result.stats["roi_source"] == "provided"
    assert captured["mask"] is not None
    assert not captured["mask"][:, :349].any()
    assert not captured["mask"][:, 451:].any()
    assert all(350 <= point.x <= 450 for point in result.points)


def test_neighborhood_hsv_filter_rejects_isolated_white_pixel(monkeypatch):
    image = np.full((320, 420, 3), 45, np.uint8)
    cv2.line(image, (60, 250), (360, 250), (240, 240, 240), 5)
    image[100, 100] = (255, 255, 255)  # 中心很白，但 5x5 邻域没有连续白线证据。
    corners = np.asarray([
        [[100, 100]], [[60, 248]], [[100, 248]], [[140, 248]],
        [[180, 248]], [[220, 248]], [[260, 248]], [[300, 248]],
    ], np.float32)
    monkeypatch.setattr(cv2, "goodFeaturesToTrack",
                        lambda *args, **kwargs: corners.copy())
    cfg = permissive_line_config(
        max_corners=20, min_distance=1, min_line_length=40.0,
        white_neighborhood_size=5, min_point_evidence=0.35,
    )
    result = detect_line_features(image, cfg, roi=np.ones(image.shape[:2], np.uint8))
    assert result.lines
    assert result.stats["white_rejected"] >= 1
    assert all(np.hypot(point.x - 100, point.y - 100) > 8 for point in result.points)


def test_min_cluster_points_default_five_rejects_small_cluster():
    xy = np.asarray([[20, 100], [50, 100], [80, 100], [110, 100]], np.float64)
    cfg = permissive_line_config(min_line_points=4, min_cluster_points=5)
    clusters, _rejected = _angle_distance_clusters(
        xy, cfg, float(np.hypot(800, 480)), np.random.default_rng(1)
    )
    assert clusters == []
    cfg = permissive_line_config(min_line_points=4, min_cluster_points=4)
    clusters, _rejected = _angle_distance_clusters(
        xy, cfg, float(np.hypot(800, 480)), np.random.default_rng(1)
    )
    assert len(clusters) == 1


def test_minimum_line_length_rejects_short_segment():
    image = np.full((320, 500, 3), 45, np.uint8)
    cv2.line(image, (40, 100), (70, 100), (240, 240, 240), 5)
    cv2.line(image, (120, 220), (460, 220), (240, 240, 240), 5)
    result = detect_line_features(
        image, permissive_line_config(min_line_length=100.0),
        roi=np.ones(image.shape[:2], np.uint8),
    )
    assert result.lines
    assert all(line.span >= 100.0 for line in result.lines)
    assert all(np.linalg.norm(np.asarray(line.p1) - line.p0) >= 100.0
               for line in result.lines)
    assert all(abs(line.direction[1]) < 0.01 for line in result.lines)
    assert all(point.y > 180 for point in result.points)


def test_angle_prior_rejects_non_court_direction():
    image = orthogonal_court()
    cfg = permissive_line_config(
        line_angle_prior_deg=(90.0,), angle_prior_tolerance_deg=2.0,
    )
    result = detect_line_features(image, cfg, roi=np.ones(image.shape[:2], np.uint8))
    assert result.lines
    assert result.stats["angle_lines_rejected"] >= 1
    assert all(abs(line.direction[0]) < 0.05 for line in result.lines)


def test_nearby_duplicate_lines_are_merged_and_refitted():
    cfg = permissive_line_config(min_line_length=40.0)
    first_points = [LineFeaturePoint(float(x), 100.0, 1.0) for x in range(20, 140, 15)]
    second_points = [LineFeaturePoint(float(x), 102.0, 1.0) for x in range(20, 140, 15)]
    first = FittedCourtLine(
        0, (20.0, 100.0), (140.0, 100.0), (1.0, 0.0), (0.0, 1.0),
        first_points, span=120.0, score=20.0,
    )
    second = FittedCourtLine(
        1, (20.0, 102.0), (140.0, 102.0), (1.0, 0.0), (0.0, 1.0),
        second_points, span=120.0, score=18.0,
    )
    white = np.ones((320, 420), np.float32)
    gradient = np.ones((320, 420, 2), np.float32) * 0.5
    merged, merge_count = _merge_nearby_lines([first, second], cfg, white, gradient)
    assert merge_count >= 1
    assert len(merged) == 1
    assert merged[0].inlier_count == len(first_points) + len(second_points)
    assert 100.0 <= np.mean([point.y for point in merged[0].points]) <= 102.0


def test_optional_player_mask_removes_corners_before_clustering():
    image = noisy_court()
    player = np.zeros(image.shape[:2], np.uint8)
    player[95:135, 70:110] = 1
    result = detect_line_features(image, fast_config(), player_mask=player)
    assert result.stats["player_rejected"] > 0
    assert all(not (70 <= point.x <= 110 and 95 <= point.y <= 135)
               for point in result.points)


def test_visualization_does_not_draw_rejected_points():
    image = np.zeros((100, 140, 3), np.uint8)
    accepted = LineFeaturePoint(30.0, 40.0, 1.0, line_id=0)
    rejected = LineFeaturePoint(110.0, 80.0, 1.0, line_id=-1)
    line = FittedCourtLine(
        0, (10.0, 40.0), (80.0, 40.0), (1.0, 0.0), (0.0, 1.0), [accepted],
    )
    result = LineFeatureResult(lines=[line], points=[accepted, rejected])
    output = visualize_line_features(image, result)
    assert np.all(output[80, 110] == 0)
    assert np.any(np.all(output == (0, 0, 255), axis=2))


def test_cli_exposes_new_filter_and_validation_parameters(tmp_path: Path):
    source = tmp_path / "court.png"
    output = tmp_path / "result.png"
    assert cv2.imwrite(str(source), noisy_court())
    script = Path(__file__).with_name("detect_court_lines.py")
    completed = subprocess.run(
        [sys.executable, str(script), str(source), "--output", str(output),
         "--white-threshold", "150", "--max-saturation", "80",
         "--white-neighborhood-size", "5", "--min-point-evidence", "0.3",
         "--gradient-neighborhood-size", "5", "--min-gradient-strength", "0.1",
         "--min-cluster-points", "5", "--min-line-length", "40",
         "--line-angles", "0,45,90,135", "--angle-prior-tolerance", "20",
         "--merge-angle", "8", "--merge-distance-factor", "3",
         "--cluster-angle", "2", "--cluster-distance-factor", "1.5",
         "--max-corners", "700", "--ransac-iterations", "150"],
        cwd=script.parent, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["min_cluster_points"] == 5
    assert summary["min_line_length"] == 40.0
    assert summary["white_neighborhood_size"] == 5
    assert summary["line_count"] >= 1


def test_default_fixed_polygon_blocks_corners_above_floor(monkeypatch):
    image = noisy_court()
    captured = {}
    original = cv2.goodFeaturesToTrack

    def wrapped(*args, **kwargs):
        captured["mask"] = kwargs.get("mask")
        return original(*args, **kwargs)

    monkeypatch.setattr(cv2, "goodFeaturesToTrack", wrapped)
    result = detect_line_features(image, fast_config(use_fixed_roi=True))
    assert result.stats["roi_source"] == "fixed_polygon"
    assert captured["mask"] is not None
    assert not captured["mask"][:100].any()
    assert all(point.y >= 100 for point in result.points)
    assert result.stats["quality_level"] == 0.025
    assert result.stats["min_line_length"] == 80.0


def test_infinite_line_intersections_are_filtered_and_clustered():
    mask = np.ones((300, 400), np.uint8)
    mask[:100] = 0
    horizontal = FittedCourtLine(
        0, (0.0, 150.0), (399.0, 150.0), (1.0, 0.0), (0.0, 1.0),
        confidence=0.9,
    )
    vertical = FittedCourtLine(
        1, (120.0, 0.0), (120.0, 299.0), (0.0, 1.0), (-1.0, 0.0),
        confidence=0.8,
    )
    outside = FittedCourtLine(
        2, (0.0, 50.0), (399.0, 50.0), (1.0, 0.0), (0.0, 1.0),
        confidence=0.7,
    )
    duplicate = FittedCourtLine(
        3, (121.0, 100.0), (121.0, 299.0), (0.0, 1.0), (-1.0, 0.0),
        confidence=0.6,
    )
    parallel = FittedCourtLine(
        4, (0.0, 151.0), (399.0, 151.0), (1.0, 0.0), (0.0, 1.0),
        confidence=0.5,
    )
    intersections, stats = _line_intersections(
        [horizontal, vertical, outside, duplicate, parallel], mask,
        fast_config(use_fixed_roi=True),
    )
    assert len(intersections) == 1
    point = intersections[0]
    assert point.x == pytest.approx(120.0) and point.y == pytest.approx(150.0)
    assert point.line_ids == (0, 1)
    assert point.distance_first <= 12 and point.distance_second <= 12
    assert stats["intersection_outside_rejected"] >= 1
    assert stats["intersection_parallel_rejected"] >= 2


def test_intersection_keypoints_match_dl_dataclass_and_green_visualization():
    from bsk.perceive.court_keypoints import DetectedKeypoint

    image = orthogonal_court()
    result = detect_line_features(
        image, permissive_line_config(min_line_length=80.0),
        roi=np.ones(image.shape[:2], np.uint8),
    )
    assert result.intersection_keypoints
    assert all(isinstance(point, DetectedKeypoint) for point in result.intersection_keypoints)
    assert all(point.world is None for point in result.intersection_keypoints)
    payload = result.to_dict()
    assert len(payload["intersection_keypoints"]) == len(result.intersections)
    output = visualize_line_features(image, result)
    assert np.any(np.all(output == (0, 255, 0), axis=2))


def test_invalid_intersection_parameters_fail_fast():
    for kwargs in (
        {"intersection_max_line_distance": -1.0},
        {"intersection_cluster_distance": 0.0},
        {"min_intersection_angle_deg": 90.0},
        {"fixed_roi_polygon": ((0.0, 0.0), (2.0, 1.0), (0.0, 1.0))},
    ):
        with pytest.raises(ValueError):
            LineFeatureConfig(**kwargs)
