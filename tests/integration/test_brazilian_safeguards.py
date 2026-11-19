"""Brazilian-code fixtures test behavior only, never real-world OCR accuracy."""

import pytest
from conftest import FakeDetector, FakeOCR

from sonar_vision.core.contracts import MatchStatus
from sonar_vision.ocr.paddle import OCRFailure
from sonar_vision.processing.pipeline import Context, FrameProcessor


@pytest.mark.parametrize(
    "text,expected",
    [
        ("6030-10", MatchStatus.MATCH),
        ("930P-10", MatchStatus.NON_MATCH),
        ("603O-10", MatchStatus.UNCERTAIN),
        ("PINHEI", MatchStatus.UNCERTAIN),
        ("", MatchStatus.UNCERTAIN),
    ],
)
def test_brazilian_codes_and_audio_deduplication(frame, settings, sign, text, expected):
    p = FrameProcessor(
        FakeDetector([sign]),
        FakeOCR(text),
        Context("6030-10", "test", "test", "local_video"),
        settings,
    )
    events = []
    for i in range(8):
        events.extend(p.process(frame, i, i * 0.2)[1])
    assert events[-1].match_status == expected
    assert len([e for e in events if e.status == "CONFIRMED"]) <= 1


def test_latest_contradiction_is_uncertain(frame, settings, sign):
    ocr = FakeOCR("6030-10")
    p = FrameProcessor(
        FakeDetector([sign]), ocr, Context("6030-10", "test", "test", "local_video"), settings
    )
    for i in range(5):
        p.process(frame, i, i * 0.2)
    ocr.text = "930P-10"
    events = p.process(frame, 5, 1.0)[1]
    assert events and events[-1].status == "UNCERTAIN"


def test_ocr_exception_never_emits_confirmation(frame, settings, sign):
    class FailedOCR:
        def read(self, roi):
            raise OCRFailure("engine unavailable")

    p = FrameProcessor(
        FakeDetector([sign]),
        FailedOCR(),
        Context("6030-10", "test", "test", "local_video"),
        settings,
    )
    with pytest.raises(OCRFailure):
        p.process(frame, 0, 0)
    assert not p.gate.seen
