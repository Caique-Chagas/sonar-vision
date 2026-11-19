import numpy as np
import pytest

from sonar_vision.br.hierarchy import HierarchicalDetector
from sonar_vision.core.contracts import Box, Detection


class Backend:
    model_version = "test"

    def __init__(self, values):
        self.values, self.calls = values, []

    def detect(self, frame):
        self.calls.append(frame.shape)
        return self.values


def detection(box, label="bus", score=0.9):
    return Detection(
        box=Box(x1=box[0], y1=box[1], x2=box[2], y2=box[3]), confidence=score, label=label
    )


def test_local_coordinates_map_back_and_keep_sign_score():
    bus = Backend([detection([50, 40, 150, 140], score=0.8)])
    sign = Backend([detection([10, 5, 90, 25], "destination_sign", 0.7)])
    result = HierarchicalDetector(bus, sign, margin=0).detect(np.zeros((200, 200, 3), np.uint8))
    assert result[0].box == Box(x1=60, y1=45, x2=140, y2=65)
    assert result[0].confidence == 0.7
    assert sign.calls == [(100, 100, 3)]


def test_no_bus_means_no_sign_inference():
    sign = Backend([])
    assert not HierarchicalDetector(Backend([]), sign).detect(np.zeros((100, 100, 3), np.uint8))
    assert not sign.calls


def test_overlap_suppression_and_crop_bounds():
    bus = Backend([detection([0, 0, 100, 100])] * 10)
    sign = Backend([detection([0, 0, 120, 30], "destination_sign")])
    result = HierarchicalDetector(bus, sign).detect(np.zeros((100, 100, 3), np.uint8))
    assert len(sign.calls) == 8
    assert len(result) == 1
    assert result[0].box.x2 == 100


@pytest.mark.parametrize("margin", [-0.1, 0.2, float("nan")])
def test_invalid_margins_rejected(margin):
    with pytest.raises(ValueError):
        HierarchicalDetector(Backend([]), Backend([]), margin=margin)
