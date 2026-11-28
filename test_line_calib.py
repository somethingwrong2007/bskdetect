from __future__ import annotations

import cv2
import numpy as np

from bsk.court import COURTS
from bsk.line_calib import (
    CL_BOUNDARY,
    CL_CENTER_CIRCLE,
    CL_CENTER_LINE,
    CL_PAINT,
    CL_THREE_POINT,
    LineCalibrator,
    _project,
    court_line_samples,
)


def test_templates_match_webgpu_geometry() -> None:
    for court in (COURTS["FIBA"], COURTS["NBA"]):
        samples = court_line_samples(court, step=0.25)
        boundary = samples[CL_BOUNDARY]
        assert np.isclose(np.max(np.abs(boundary[:, 0])), court.half_length + court.line_half)
        assert np.isclose(np.max(np.abs(boundary[:, 1])), court.half_width + court.line_half)

        center_line = samples[CL_CENTER_LINE]
        assert np.isclose(np.max(np.abs(center_line[:, 1])),
                          court.half_width + court.line_half + court.center_line_extension)

        circle = samples[CL_CENTER_CIRCLE]
        assert np.allclose(np.linalg.norm(circle, axis=1), court.center_circle_center_radius)

        paint = samples[CL_PAINT]
        assert np.isclose(np.max(np.abs(paint[:, 1])),
                          court.paint_width / 2 + court.line_half)

        three = samples[CL_THREE_POINT]
        for side, sign in (("A", 1), ("B", -1)):
            rim_x, _ = court.rim_pos(side)
            expected = np.array([rim_x - sign * court.three_pt_center_radius, 0.0])
            assert np.min(np.linalg.norm(three - expected, axis=1)) < 0.3


def test_paint_initializer_recovers_lane() -> None:
    court = COURTS["FIBA"]
    shape = (540, 960)
    quad = np.array([[120, 430], [840, 430], [760, 80], [220, 80]], np.float32)
    world_rect = np.array([[-court.half_length, -court.half_width],
                           [court.half_length, -court.half_width],
                           [court.half_length, court.half_width],
                           [-court.half_length, court.half_width]], np.float32)
    homography = cv2.getPerspectiveTransform(world_rect, quad)
    pred = np.zeros(shape, np.uint8)
    baseline = court.half_length
    free_throw = baseline - court.free_throw_center_dist
    lane_half = court.paint_width / 2 + court.line_half
    segments = [
        ((baseline, -lane_half), (free_throw, -lane_half)),
        ((baseline, lane_half), (free_throw, lane_half)),
        ((free_throw, -lane_half), (free_throw, lane_half)),
        ((baseline, -court.half_width), (baseline, court.half_width)),
    ]
    for index, (start, end) in enumerate(segments):
        points, _ = _project(homography, np.array([start, end], np.float64))
        class_id = CL_PAINT if index < 3 else CL_BOUNDARY
        cv2.line(pred, tuple(np.round(points[0]).astype(int)),
                 tuple(np.round(points[1]).astype(int)), class_id, 3)

    calibrator = LineCalibrator(court=court, size=shape, weights=None, device="cpu")
    candidates = calibrator._init_from_paint(pred, court, shape)
    lane = np.array([[baseline, -lane_half], [baseline, lane_half],
                     [free_throw, lane_half], [free_throw, -lane_half]], np.float64)
    target, _ = _project(homography, lane)
    errors = [float(np.max(np.linalg.norm(_project(candidate, lane)[0] - target, axis=1)))
              for candidate, _source in candidates]
    assert errors and min(errors) < 3.0
