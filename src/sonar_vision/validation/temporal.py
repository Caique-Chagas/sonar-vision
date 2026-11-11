from collections import Counter, deque
from dataclasses import dataclass
from statistics import mean

from sonar_vision.core.config import Settings
from sonar_vision.validation.normalizer import NormalizedLine


@dataclass(frozen=True)
class Observation:
    frame_index: int
    timestamp: float
    normalized: NormalizedLine
    detection_confidence: float
    ocr_confidence: float


@dataclass(frozen=True)
class Consensus:
    route: str | None
    frames: int
    supporting: int
    detection_confidence: float | None
    ocr_confidence: float | None
    raw_texts: list[str]
    reason: str


class TemporalValidator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.history: dict[int, deque[Observation]] = {}

    def prune(self, now: float) -> None:
        for track_id in list(self.history):
            if now - self.history[track_id][-1].timestamp > self.settings.evidence_seconds:
                del self.history[track_id]

    def add(self, track_id: int, obs: Observation) -> Consensus:
        self.prune(obs.timestamp)
        items = self.history.setdefault(track_id, deque(maxlen=self.settings.window_frames))
        if items and (
            obs.frame_index <= items[-1].frame_index or obs.timestamp < items[-1].timestamp
        ):
            raise ValueError("Frames precisam ser distintos e cronológicos por track")
        items.append(obs)
        while items and obs.timestamp - items[0].timestamp > self.settings.evidence_seconds:
            items.popleft()
        eligible = [
            o
            for o in items
            if o.normalized.route
            and o.ocr_confidence >= self.settings.min_ocr
            and o.detection_confidence >= self.settings.min_detection
        ]
        votes = Counter(o.normalized.route for o in eligible)
        winner, count = votes.most_common(1)[0] if votes else (None, 0)
        supporting = [o for o in eligible if o.normalized.route == winner]
        stable = (
            winner is not None
            and count >= self.settings.min_votes
            and count / len(items) >= self.settings.min_ratio
            and obs in supporting
            and supporting[-1].timestamp - supporting[0].timestamp >= self.settings.min_span_seconds
        )
        return Consensus(
            route=winner if stable else None,
            frames=len(items),
            supporting=count,
            detection_confidence=mean(o.detection_confidence for o in supporting)
            if stable
            else None,
            ocr_confidence=mean(o.ocr_confidence for o in supporting) if stable else None,
            raw_texts=[o.normalized.raw for o in items],
            reason="Consenso temporal atingido" if stable else "Consenso temporal insuficiente",
        )
