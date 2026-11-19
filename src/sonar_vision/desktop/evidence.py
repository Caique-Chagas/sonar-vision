"""Presentation states from preserved frame evidence, without route-reference input."""

import copy
import math
import re
from typing import Any

from sonar_vision.desktop.temporal_review import EncounterReview
from sonar_vision.validation.normalizer import LineNormalizer

COMPLETE_CODE = re.compile(r"(?:[0-9]{3}[A-Z0-9]|[0-9]{3,5}|[A-Z][0-9]{3})-[0-9]{2}")


class PresentationEvidence:
    """CODE_STABLE is textual consensus, never operational LINE_VERIFIED."""

    def __init__(self) -> None:
        self.review = EncounterReview()
        self.normalizer = LineNormalizer()
        self.seen_pixels: set[str] = set()
        self.last_index = -1
        self.last_timestamp = float("-inf")
        self.stable: dict[int, tuple[str, float]] = {}
        self.announced: set[tuple[int, str]] = set()

    def observe(self, record: dict[str, Any]) -> dict[str, Any]:
        index, timestamp = record["source_index"], record["timestamp_seconds"]
        digest = record["input_sha256"]
        if (
            not isinstance(index, int)
            or index <= self.last_index
            or not math.isfinite(timestamp)
            or timestamp <= self.last_timestamp
            or not re.fullmatch("[0-9a-f]{64}", digest)
        ):
            raise ValueError("Frame identity/timestamp/pixel hash must be distinct and ordered")
        if len(self.seen_pixels) >= 10000:
            raise ValueError("Presentation evidence frame budget exceeded")
        self.last_index, self.last_timestamp = index, timestamp
        duplicate = digest in self.seen_pixels
        self.seen_pixels.add(digest)
        original = record["row"]
        voting = copy.deepcopy(original)
        for finding in voting["findings"]:
            if duplicate or finding.get("detection_origin", "detection") in {
                "tracking",
                "interpolated",
                "ground_truth",
            }:
                finding.update(code=None, readings=[])
        review = self.review.observe(index, timestamp, voting)
        for consensus in review["consensus"]:
            track = consensus["track_id"]
            if consensus["route"]:
                self.stable[track] = consensus["route"], timestamp
            else:
                self.stable.pop(track, None)
        current: dict[int, set[str]] = {}
        for finding, track in zip(voting["findings"], review["panel_track_ids"], strict=True):
            code = finding["code"]
            if (
                track is not None
                and code
                and COMPLETE_CODE.fullmatch(code)
                and finding["detector_confidence"] >= 0.35
                and not finding.get("ocr_error")
                and any(
                    r["confidence"] >= 0.7 and self.normalizer.normalize(r["text"]).route == code
                    for r in finding["readings"]
                )
            ):
                current.setdefault(track, set()).add(code)
        for track in list(self.stable):
            code, updated = self.stable[track]
            if current.get(track) != {code} or timestamp - updated > 0.25 or duplicate:
                del self.stable[track]
        stable = [(track, code) for track, (code, _) in self.stable.items()]
        state = "NO_DETECTION" if not original["findings"] else "PANEL_DETECTED"
        if current:
            state = "OCR_CANDIDATE"
        event = None
        if len(stable) == 1:
            track, code = stable[0]
            state = "CODE_STABLE"
            if (track, code) not in self.announced:
                self.announced.add((track, code))
                event = dict(
                    track_id=track,
                    code=code,
                    timestamp_seconds=timestamp,
                    source_index=index,
                    input_sha256=digest,
                    text=f"Código de linha {code} identificado.",
                    operational_line_verified=False,
                )
        return dict(
            state=state,
            stable_codes=[dict(track_id=t, code=c) for t, c in stable],
            temporal=review,
            duplicate_pixels=duplicate,
            audio_event=event,
            line_verified=False,
            criteria=dict(
                min_votes=3,
                min_ratio=0.75,
                window_samples=8,
                sample_hz=5,
                min_span_seconds=0.3,
                min_detection=0.35,
                min_ocr=0.7,
                evidence_seconds=3,
                unique_source_frames=True,
                unique_pixel_hashes=True,
            ),
        )
