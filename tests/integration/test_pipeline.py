from conftest import FakeDetector, FakeOCR

from sonar_vision.core.contracts import Box, Detection, MatchStatus
from sonar_vision.processing.pipeline import Context, FrameProcessor


def processor(settings, sign, ocr=None, detections=None):
    return FrameProcessor(
        FakeDetector(detections if detections is not None else [sign]),
        ocr or FakeOCR(),
        Context("971D-10", "stop", "wait", "local_video"),
        settings,
    )


def test_pipeline_positive_negative_and_annotation(frame, settings, sign):
    pipeline = processor(settings, sign)
    events = []
    for i in range(5):
        annotated, new = pipeline.process(frame, i, i * 0.2)
        events.extend(new)
    assert events[-1].match_status == MatchStatus.MATCH
    assert len(events) == 2  # initial uncertainty and one confirmation; never repeat per frame
    assert (annotated != frame).any()
    assert events[-1].destination_detected is None


def test_different_line_and_no_text(frame, settings, sign):
    for text, expected in [("875P-10", MatchStatus.NON_MATCH), ("", MatchStatus.UNCERTAIN)]:
        pipeline = processor(settings, sign, FakeOCR(text))
        events = []
        for i in range(4):
            _, new = pipeline.process(frame, i, i * 0.2)
            events.extend(new)
        assert events[-1].match_status == expected


def test_no_bus_no_events(frame, settings, sign):
    assert processor(settings, sign, detections=[]).process(frame, 0, 0)[1] == []


def test_catalog_format_beyond_numeric_regex(frame, settings, sign):
    pipeline = FrameProcessor(
        FakeDetector([sign]),
        FakeOCR("EXP-1"),
        Context("EXP-1", "stop", "wait", "local_video"),
        settings,
    )
    events = []
    for i in range(3):
        events.extend(pipeline.process(frame, i, i * 0.2)[1])
    assert events[-1].match_status == MatchStatus.MATCH


def test_two_vehicles_do_not_share_votes(frame, settings, sign):
    second = Detection(
        box=Box(x1=230, y1=40, x2=390, y2=90), confidence=0.9, label="destination_sign"
    )
    pipeline = processor(settings, sign, detections=[sign, second])
    _, first = pipeline.process(frame, 0, 0)
    assert {e.track_id for e in first} == {1, 2}
    assert all(e.status == "UNCERTAIN" for e in first)
    _, second_frame = pipeline.process(frame, 1, 0.2)
    assert not second_frame
    _, third = pipeline.process(frame, 2, 0.4)
    assert len(third) == 2 and all(e.supporting_frames == 3 for e in third)
