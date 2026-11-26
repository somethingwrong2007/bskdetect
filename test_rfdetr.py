from __future__ import annotations

import numpy as np

from bsk.perceive.rfdetr import Detection, RfDetrDetector


def _image(color: tuple[int, int, int]) -> np.ndarray:
    image = np.empty((100, 100, 3), np.uint8)
    image[:] = color
    return image


def test_filter_balls_keeps_orange_ball() -> None:
    det = Detection(0, "ball", 0.9, (45.0, 45.0, 55.0, 55.0))
    assert RfDetrDetector.filter_balls(_image((230, 110, 20)), [det]) == [det]


def test_filter_balls_accepts_model_ball_without_color_gate() -> None:
    blue = Detection(0, "ball", 0.9, (40.0, 40.0, 50.0, 50.0))
    number = Detection(2, "number", 0.9, (40.0, 40.0, 50.0, 50.0))
    assert RfDetrDetector.filter_balls(_image((20, 30, 220)), [blue, number]) == [blue]


def test_filter_balls_keeps_only_highest_confidence() -> None:
    low = Detection(0, "ball", 0.6, (40.0, 40.0, 50.0, 50.0))
    high = Detection(0, "ball", 0.9, (45.0, 45.0, 55.0, 55.0))
    assert RfDetrDetector.filter_balls(_image((230, 110, 20)), [low, high]) == [high]
