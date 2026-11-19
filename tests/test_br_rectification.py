import numpy as np
import pytest

from sonar_vision.br.ocr_rectification import rectify_quad


def test_observed_quad_rectification_preserves_pixels_and_does_not_mutate():
    im = np.full((100, 200, 3), 80, np.uint8)
    before = im.copy()
    out = rectify_quad(im, [[10, 10], [150, 20], [150, 50], [10, 40]])
    assert out.shape == (30, 140, 3)
    assert np.array_equal(im, before)
    assert out.mean() == 80


@pytest.mark.parametrize(
    "points",
    [
        [],
        [[0, 0]] * 4,
        [[-1, 0], [20, 0], [20, 20], [0, 20]],
        [[0, 0], [200, 0], [200, 20], [0, 20]],
        [[0, 0], [20, 0], [20, float("nan")], [0, 20]],
    ],
)
def test_bad_or_unbounded_polygons_do_not_create_text(points):
    assert rectify_quad(np.zeros((100, 200, 3), np.uint8), points) is None
