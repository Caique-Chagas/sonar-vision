from datetime import datetime

import pytest
from pydantic import ValidationError

from sonar_vision.core.contracts import Box, VisionEvent, to_legacy
from sonar_vision.speech.announcements import AnnouncementGate, announcement


def make_event(**kwargs):
    values = dict(
        stop_id="stop",
        waiting_session_id="wait",
        track_id=1,
        route_requested="971D-10",
        source="local_video",
        model_version="test",
        status="UNCERTAIN",
        frames_considered=1,
        supporting_frames=1,
        reason="fixture",
    )
    return VisionEvent(**(values | kwargs))


@pytest.mark.parametrize(
    "invalid",
    [
        {"x1": 5, "y1": 0, "x2": 1, "y2": 2},
        {"x1": -1, "y1": 0, "x2": 1, "y2": 2},
        {"x1": 0, "y1": 0, "x2": float("nan"), "y2": 2},
    ],
)
def test_invalid_boxes(invalid):
    with pytest.raises(ValidationError):
        Box(**invalid)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"status": "CONFIRMED", "route_detected": "971D-10"},
        {"match_status": "MATCH"},
        {"route_detected": "971D-10"},
        {"supporting_frames": 5},
        {"ocr_confidence": 1.5},
        {"timestamp": datetime(2026, 11, 10)},
        {"source": "simulation", "simulated": False},
        {"detection_confidence": float("nan")},
    ],
)
def test_inconsistent_event_rejected(kwargs):
    with pytest.raises(ValidationError):
        make_event(**kwargs)


def test_legacy_does_not_invent_stop_time_or_general_score():
    event = make_event()
    legacy = to_legacy(event)
    assert legacy.tempo_parado is None and legacy.confianca is None
    assert "image" not in event.model_dump_json()


def test_announcement_uncertainty_and_deduplication():
    event = make_event()
    assert "Não foi possível" in announcement(event)
    gate = AnnouncementGate()
    assert gate.accept(event)
    assert not gate.accept(make_event())
    assert gate.accept(make_event(track_id=2))


def test_match_message_never_claims_boarding():
    event = make_event(
        status="CONFIRMED",
        route_detected="971D-10",
        match_status="MATCH",
        frames_considered=3,
        supporting_frames=3,
        detection_confidence=0.9,
        ocr_confidence=0.9,
    )
    text = announcement(event)
    assert "Linha solicitada confirmada" in text
    assert "embarcar" not in text and "destino" not in text


def test_impossible_confirmation_window_rejected():
    from sonar_vision.core.config import Settings

    with pytest.raises(ValidationError, match="Janela temporal"):
        Settings(_env_file=None, min_votes=8, window_frames=3)
