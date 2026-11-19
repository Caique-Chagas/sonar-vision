import numpy as np
import pytest

from sonar_vision.core.contracts import Box, Detection
from sonar_vision.desktop.recovery import PanelRecovery, belongs_to_panel_region
from sonar_vision.desktop.temporal_review import EncounterReview


def detection(x1, y1, x2, y2, label="destination_sign", confidence=0.8):
    return Detection(box=Box(x1=x1, y1=y1, x2=x2, y2=y2), label=label, confidence=confidence)


class Signs:
    model_version = "fixture-sign"

    def __init__(self, predictions):
        self.predictions, self.calls = predictions, 0

    def detect(self, frame):
        self.calls += 1
        return self.predictions


def test_recovery_maps_actual_crop_box_and_retains_score():
    sign = Signs([detection(10, 10, 60, 25)])
    recovery = PanelRecovery(sign)
    result = recovery.detect(
        np.zeros((200, 200, 3), dtype=np.uint8), [detection(20, 20, 120, 180, "bus")], []
    )
    assert len(result) == 1
    assert result[0].box == Box(x1=25, y1=22, x2=75, y2=37)
    assert result[0].confidence == 0.8
    assert recovery.trace[0]["accepted"]


def test_global_panel_skips_recovery():
    sign = Signs([])
    result = PanelRecovery(sign).detect(
        np.zeros((200, 200, 3), dtype=np.uint8),
        [detection(20, 20, 120, 180, "bus")],
        [detection(30, 30, 100, 50)],
    )
    assert result == [] and sign.calls == 0


def test_no_bus_never_invents_box():
    sign = Signs([detection(10, 10, 60, 25)])
    assert PanelRecovery(sign).detect(np.zeros((200, 200, 3), dtype=np.uint8), [], []) == []
    assert sign.calls == 0


@pytest.mark.parametrize("panel", [(20, 130, 100, 150), (20, 20, 30, 45), (0, 20, 60, 40)])
def test_reject_lower_vertical_or_poor_containment(panel):
    assert not belongs_to_panel_region(
        Box(x1=panel[0], y1=panel[1], x2=panel[2], y2=panel[3]), Box(x1=20, y1=20, x2=120, y2=180)
    )


def row(code="637P-10", *, buses=1, reading=None):
    return dict(
        buses=[detection(20, 20, 120, 180, "bus").model_dump()] * buses,
        findings=[
            dict(
                box=detection(30, 30, 100, 50).box.model_dump(),
                detector_confidence=0.8,
                code=code,
                ocr_error=None,
                readings=[dict(text=reading or code or "", confidence=0.9)],
            )
        ],
    )


def test_temporal_requires_real_span_and_repeated_evidence():
    review = EncounterReview()
    assert review.observe(0, 0.0, row())["consensus"][0]["route"] is None
    assert not review.observe(1, 1 / 30, row())["sampled_at_5Hz"]
    assert review.observe(6, 0.2, row())["consensus"][0]["route"] is None
    result = review.observe(12, 0.4, row())
    assert result["consensus"][0]["route"] == "637P-10"
    assert result["operational_confirmation"] is False
    assert result["interpolated_boxes"] is False


def test_conflicting_latest_code_clears_consensus():
    review = EncounterReview()
    for i in range(3):
        review.observe(i * 6, i * 0.2, row())
    assert review.observe(18, 0.6, row("477A-10"))["consensus"][0]["route"] is None


def test_two_overlapping_buses_never_share_panel_votes():
    result = EncounterReview().observe(0, 0, row(buses=2))
    assert result["panel_track_ids"] == [None]
    assert all(r["supporting"] == 0 for r in result["consensus"])


def test_missing_detections_do_not_create_stale_boxes_or_consensus():
    review = EncounterReview()
    review.observe(0, 0, row())
    result = review.observe(6, 0.2, dict(buses=[], findings=[]))
    assert result["bus_track_ids"] == [] and result["consensus"] == []


def test_ineligible_raw_reading_cannot_vote_for_cached_candidate():
    result = EncounterReview().observe(0, 0, row(reading="637P-1O"))
    assert result["consensus"][0]["supporting"] == 0
