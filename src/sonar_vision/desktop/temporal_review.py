"""Offline encounter evidence; current-frame boxes only, no operational confirmation."""

from dataclasses import asdict
from typing import Any

from sonar_vision.core.config import Settings
from sonar_vision.core.contracts import Box, Detection
from sonar_vision.desktop.recovery import belongs_to_panel_region
from sonar_vision.tracking.iou import SignTracker
from sonar_vision.validation.normalizer import LineNormalizer, NormalizedLine
from sonar_vision.validation.temporal import Observation, TemporalValidator


class EncounterReview:
    """Bus IDs reduce cross-vehicle mixing; ambiguous/missing OCR casts no vote."""

    def __init__(self) -> None:
        self.tracker = SignTracker()
        self.validator = TemporalValidator(Settings.model_construct())
        self.normalizer = LineNormalizer()
        self.last_sample = float("-inf")

    def observe(self, index: int, timestamp: float, row: dict[str, Any]) -> dict[str, Any]:
        buses = self.tracker.update(
            [Detection.model_validate(b) for b in row.get("buses", [])], timestamp
        )
        associations: dict[int, list[dict[str, Any]]] = {track: [] for track, _ in buses}
        panel_tracks = []
        for finding in row["findings"]:
            box = Box.model_validate(finding["box"])
            matches = [track for track, bus in buses if belongs_to_panel_region(box, bus.box)]
            assigned = matches[0] if len(matches) == 1 else None
            panel_tracks.append(assigned)
            if assigned is not None:
                associations[assigned].append(finding)
        sampled = timestamp - self.last_sample >= 0.2 - 1e-6
        consensus = []
        self.validator.prune(timestamp)
        if sampled:
            self.last_sample = timestamp
            for track, _ in buses:
                findings = associations[track]
                codes = {f["code"] for f in findings if f["code"]}
                code = next(iter(codes)) if len(codes) == 1 else None
                supporting = [
                    (f, r)
                    for f in findings
                    for r in f["readings"]
                    if code
                    and r["confidence"] >= 0.7
                    and self.normalizer.normalize(r["text"]).route == code
                    and not f.get("ocr_error")
                ]
                if supporting:
                    finding, reading = max(supporting, key=lambda pair: pair[1]["confidence"])
                    normalized = self.normalizer.normalize(reading["text"])
                    detection, ocr = finding["detector_confidence"], reading["confidence"]
                else:
                    normalized = NormalizedLine("", (), "missing_or_conflicting", True)
                    detection, ocr = 0.0, 0.0
                result = self.validator.add(
                    track, Observation(index, timestamp, normalized, detection, ocr)
                )
                consensus.append(dict(track_id=track, **asdict(result)))
        return dict(
            bus_track_ids=[track for track, _ in buses],
            panel_track_ids=panel_tracks,
            sampled_at_5Hz=sampled,
            consensus=consensus,
            operational_confirmation=False,
            interpolated_boxes=False,
        )
