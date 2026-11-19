"""Rendering/association software fixtures; not measured vision performance."""

import numpy as np

from sonar_vision.core.contracts import Box, Detection
from sonar_vision.desktop.profiles import profiles
from sonar_vision.desktop.render import BUS_COLOR, SIGN_COLOR, render
from sonar_vision.desktop.v3 import BusAndSignDetector


def result_at(left):
    return dict(
        frame=np.full((720, 1280, 3), (14, 29, 45), np.uint8),
        index=12,
        row=dict(
            buses=[dict(box=dict(x1=left, y1=200, x2=800, y2=600), confidence=0.9)],
            findings=[
                dict(
                    box=dict(x1=400, y1=250, x2=700, y2=300),
                    detector_confidence=0.8,
                    readings=[dict(text="FIXTURE RAW", confidence=0.95)],
                    code=None,
                )
            ],
        ),
    )


def test_video_fills_canvas_and_real_findings_use_class_colors():
    result = result_at(200)
    canvas = render(profiles()["curitiba"], result, "fixture", False)
    assert tuple(canvas[450, 200]) == BUS_COLOR[::-1]
    assert tuple(canvas[300, 650]) == SIGN_COLOR[::-1]
    assert np.array_equal(canvas[400, 1100], result["frame"][400, 1100])
    assert np.array_equal(canvas[400, 10], result["frame"][400, 10])
    assert not np.array_equal(canvas[305:340, 400:600], result["frame"][305:340, 400:600])
    assert np.all(result["frame"] == np.array([14, 29, 45]))


def test_boxes_follow_current_frame_without_stale_coordinates():
    first = render(profiles()["curitiba"], result_at(200), "fixture", False)
    second = render(profiles()["curitiba"], result_at(240), "fixture", False)
    assert tuple(first[450, 200]) == BUS_COLOR[::-1]
    assert tuple(second[450, 200]) == (14, 29, 45)
    assert tuple(second[450, 240]) == BUS_COLOR[::-1]


def test_no_bus_or_sign_is_drawn_when_current_frame_has_no_detection():
    row = result_at(200)
    row["row"] = dict(buses=[], findings=[])
    canvas = render(profiles()["curitiba"], row, "fixture", False)
    assert np.array_equal(canvas[40:680], row["frame"][40:680])


def test_composition_uses_only_actual_bus_and_actual_sign_predictions():
    frame = np.zeros((100, 160, 3), np.uint8)
    bus = Detection(box=Box(x1=1, y1=1, x2=150, y2=90), confidence=0.9, label="bus")
    sign = Detection(box=Box(x1=20, y1=10, x2=100, y2=30), confidence=0.8, label="destination_sign")

    class Buses:
        model_version = "fixture-bus-model"

        def detect(self, image):
            assert image is frame
            return [bus, sign]

    class Signs:
        model_version = "fixture-sign-model"

        def detect(self, image):
            assert image is frame
            return [sign, bus]

    detector = BusAndSignDetector(Buses(), Signs())
    assert detector.detect(frame) == [bus, sign]
    assert detector.model_version == "composition:bus=fixture-bus-model;sign=fixture-sign-model"
