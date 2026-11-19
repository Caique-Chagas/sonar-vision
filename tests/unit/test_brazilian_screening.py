"""Software safeguards, not recognition metrics from synthetic images."""

from dataclasses import replace

import numpy as np
import pytest

from sonar_vision.evaluation.screening import (
    MediaPermission,
    exact_route_score,
    normalize_frame,
    sample_video,
)

PERMISSION = MediaPermission(
    "https://commons.wikimedia.org/wiki/File:example.webm", "Author", "CC BY 4.0", True
)


@pytest.mark.parametrize(
    "changes",
    [
        {"evaluation_allowed": False},
        {"author": ""},
        {"license": ""},
        {"source_url": "not a URL"},
        {"source_url": "https://user:secret@example.org/a"},
    ],
)
def test_permission_required_even_for_local_files(tmp_path, changes):
    with pytest.raises(ValueError, match="permissão"):
        list(sample_video(tmp_path / "a.webm", replace(PERMISSION, **changes), [0]))


@pytest.mark.parametrize(
    "actual,expected,score",
    [
        ("6030-10", "6030-10", True),
        ("603O-10", "6030-10", False),
        ("930P-10", "6030-10", False),
        (None, "6030-10", False),
        ("6030-10", None, None),
        ("6030-10", "", None),
    ],
)
def test_independent_ground_truth_and_ocr_confusion(actual, expected, score):
    assert exact_route_score(actual, expected) is score


@pytest.mark.parametrize("frame", [None, np.zeros((2, 3)), np.zeros((0, 2, 3), np.uint8)])
def test_corrupt_frames(frame):
    with pytest.raises(ValueError, match="corrompido"):
        normalize_frame(frame)


def test_large_input_is_reduced_before_model():
    frame = normalize_frame(np.zeros((2300, 4080, 3), np.uint8))
    assert frame.shape[0] <= 1800 and frame.shape[1] <= 3200


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_frames": 121},
        {"max_seconds": 181},
        {"timestamps": [-1]},
        {"timestamps": list(range(121))},
    ],
)
def test_resource_limits_before_open(tmp_path, kwargs):
    kwargs.setdefault("timestamps", [0])
    with pytest.raises(ValueError, match="limites|Limites"):
        list(sample_video(tmp_path / "missing.mp4", PERMISSION, **kwargs))


def test_missing_and_corrupted_media(tmp_path):
    with pytest.raises(ValueError, match="indisponível"):
        list(sample_video(tmp_path / "missing.mp4", PERMISSION, [0]))
    p = tmp_path / "broken.mp4"
    p.write_bytes(b"broken")
    with pytest.raises(ValueError, match="corrompido"):
        list(sample_video(p, PERMISSION, [0]))


def test_explicit_alphanumeric_catalog_entry_is_preserved():
    from sonar_vision.validation.normalizer import LineNormalizer

    assert LineNormalizer({"603O-10", "6030-10"}).normalize("603O-10").route == "603O-10"
    conflict = LineNormalizer({"6030-10"}).normalize("603O-10")
    assert conflict.route is None and conflict.ambiguous
    assert "6030-10" in conflict.candidates
