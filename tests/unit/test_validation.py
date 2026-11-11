import pytest

from sonar_vision.core.contracts import MatchStatus
from sonar_vision.tracking.iou import SignTracker
from sonar_vision.validation.normalizer import LineNormalizer
from sonar_vision.validation.routes import RouteMatcher
from sonar_vision.validation.temporal import Observation, TemporalValidator

ROUTES = {"971D-10", "971D-21", "875P-10", "8000-10", "EXP-1"}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("971d - 10", "971D-10"),
        ("875P–10", "875P-10"),
        ("LINHA 971D-10 CENTRO", "971D-10"),
        ("EXP-1", "EXP-1"),
        ("9 7 1 D - 1 0", "971D-10"),
        ("8000-10", "8000-10"),
        ("LINHA N801-11", "N801-11"),
    ],
)
def test_exact_normalization(text, expected):
    normalized = LineNormalizer(ROUTES).normalize(text)
    assert normalized.route == expected
    assert normalized.raw == text


@pytest.mark.parametrize("text", ["971D", "875P", "971D-IO", "", "CENTRO", "971D-10 875P-10"])
def test_ambiguous_or_partial_never_confirmed(text):
    assert LineNormalizer(ROUTES).normalize(text).route is None


def test_short_line_preserves_alternatives():
    result = LineNormalizer(ROUTES).normalize("971D")
    assert result.candidates == ("971D-10", "971D-21")
    assert result.ambiguous


@pytest.mark.parametrize(
    "detected,requested,expected",
    [
        ("971D-10", "971D-10", MatchStatus.MATCH),
        ("875P-10", "971D-10", MatchStatus.NON_MATCH),
        (None, "971D-10", MatchStatus.UNCERTAIN),
        ("9999-10", "9999-10", MatchStatus.UNCERTAIN),
    ],
)
def test_context_match(detected, requested, expected):
    assert RouteMatcher(ROUTES).match(detected, requested).status == expected


def observation(i, text="971D-10", confidence=0.9):
    return Observation(i, i * 0.2, LineNormalizer(ROUTES).normalize(text), 0.9, confidence)


def test_consensus_requires_distinct_frames_and_time(settings):
    validator = TemporalValidator(settings)
    assert validator.add(1, observation(0)).route is None
    assert validator.add(1, observation(1)).route is None
    result = validator.add(1, observation(2))
    assert result.route == "971D-10" and result.supporting == 3
    with pytest.raises(ValueError, match="distintos"):
        validator.add(1, observation(2))


@pytest.mark.parametrize(
    "texts",
    [
        ["971D-10", "875P-10"] * 4,
        ["971D-10", "971D-10", "", "971D-10"],
        ["971D"] * 8,
    ],
)
def test_unstable_windows_rejected(settings, texts):
    settings.min_ratio = 0.8
    validator = TemporalValidator(settings)
    for i, text in enumerate(texts):
        result = validator.add(1, observation(i, text))
    assert result.route is None


def test_fresh_contradiction_suppresses_old_majority(settings):
    validator = TemporalValidator(settings)
    for i in range(6):
        validator.add(1, observation(i))
    assert validator.add(1, observation(6, "875P-10")).route is None


def test_low_confidence_and_separate_vehicles(settings):
    validator = TemporalValidator(settings)
    for i in range(5):
        assert validator.add(1, observation(i, confidence=0.3)).route is None
        assert validator.add(i + 2, observation(i)).route is None


def test_stale_evidence_expires(settings):
    validator = TemporalValidator(settings)
    for i in range(3):
        validator.add(1, observation(i))
    assert validator.add(1, observation(40)).route is None


def test_tracking_separates_and_expires(sign):
    tracker = SignTracker()
    assert tracker.update([sign], 0)[0][0] == 1
    assert tracker.update([sign], 0.2)[0][0] == 1
    assert tracker.update([sign], 2)[0][0] == 2


def test_ambiguous_crossing_does_not_reuse_track(sign):
    tracker = SignTracker()
    tracker.update([sign], 0)
    assigned = tracker.update([sign, sign], 0.2)
    assert len({key for key, _ in assigned}) == 2
    assert all(key != 1 for key, _ in assigned)
