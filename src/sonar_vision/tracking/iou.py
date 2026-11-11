"""Conservative short-lived sign tracks; ambiguous overlaps create fresh IDs."""

from dataclasses import dataclass

from sonar_vision.core.contracts import Box, Detection


def iou(a: Box, b: Box) -> float:
    intersection = max(0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(
        0, min(a.y2, b.y2) - max(a.y1, b.y1)
    )
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - intersection
    return intersection / union if union else 0


@dataclass
class Track:
    box: Box
    last_seen: float


class SignTracker:
    def __init__(self, threshold: float = 0.35, ttl: float = 1.0, margin: float = 0.15):
        self.threshold, self.ttl, self.margin = threshold, ttl, margin
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def update(self, detections: list[Detection], now: float) -> list[tuple[int, Detection]]:
        self.tracks = {k: v for k, v in self.tracks.items() if now - v.last_seen <= self.ttl}
        candidates = {
            i: sorted([(iou(d.box, t.box), k) for k, t in self.tracks.items()], reverse=True)
            for i, d in enumerate(detections)
        }
        proposals: dict[int, list[int]] = {}
        for i, scores in candidates.items():
            if scores and scores[0][0] >= self.threshold:
                if len(scores) == 1 or scores[0][0] - scores[1][0] >= self.margin:
                    proposals.setdefault(scores[0][1], []).append(i)
        result = []
        for i, detection in enumerate(detections):
            scores = candidates[i]
            match = scores[0][1] if scores else None
            if match is None or proposals.get(match) != [i]:
                match = self.next_id
                self.next_id += 1
            self.tracks[match] = Track(detection.box, now)
            result.append((match, detection))
        return result
