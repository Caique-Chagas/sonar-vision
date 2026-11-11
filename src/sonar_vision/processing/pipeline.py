from dataclasses import dataclass

import cv2

from sonar_vision.core.config import Settings
from sonar_vision.core.contracts import OCRReading, VisionEvent
from sonar_vision.detection.yolo import Detector
from sonar_vision.ocr.paddle import OCREngine
from sonar_vision.processing.roi import Frame, extract_roi
from sonar_vision.speech.announcements import AnnouncementGate
from sonar_vision.tracking.iou import SignTracker
from sonar_vision.validation.normalizer import LineNormalizer
from sonar_vision.validation.routes import RouteMatcher
from sonar_vision.validation.temporal import Observation, TemporalValidator


@dataclass(frozen=True)
class Context:
    requested: str
    stop_id: str
    waiting_session_id: str
    source: str


class FrameProcessor:
    def __init__(
        self,
        detector: Detector,
        ocr: OCREngine,
        context: Context,
        settings: Settings,
        routes: set[str] | None = None,
    ):
        self.detector, self.ocr, self.context = detector, ocr, context
        self.tracker = SignTracker()
        # Requested full code is a parsing hint, never evidence that the route exists.
        self.normalizer = LineNormalizer((routes or set()) | {context.requested})
        self.temporal = TemporalValidator(settings)
        self.matcher = RouteMatcher(routes)
        self.gate = AnnouncementGate()
        self.detections_total = 0
        self.ocr_calls = 0

    def process(
        self, frame: Frame, index: int, timestamp: float
    ) -> tuple[Frame, list[VisionEvent]]:
        detections = [d for d in self.detector.detect(frame) if d.label == "destination_sign"]
        self.detections_total += len(detections)
        annotated = frame.copy()
        events = []
        self.temporal.prune(timestamp)
        for track_id, detection in self.tracker.update(detections, timestamp):
            self.ocr_calls += 1
            readings = self.ocr.read(extract_roi(frame, detection.box))
            # A single observation per sign/frame: competing codes must never become separate votes.
            normalized_readings = [(r, self.normalizer.normalize(r.text)) for r in readings]
            route_readings = [(r, n) for r, n in normalized_readings if n.candidates]
            if route_readings:
                text = " | ".join(r.text for r, _ in route_readings)
                confidence = min(r.confidence for r, _ in route_readings)
            else:
                text, confidence = " | ".join(r.text for r in readings), 0.0
            reading = OCRReading(text=text[:512], confidence=confidence)
            consensus = self.temporal.add(
                track_id,
                Observation(
                    index,
                    timestamp,
                    self.normalizer.normalize(reading.text),
                    detection.confidence,
                    reading.confidence,
                ),
            )
            match = self.matcher.match(consensus.route, self.context.requested)
            event = VisionEvent(
                stop_id=self.context.stop_id,
                waiting_session_id=self.context.waiting_session_id,
                track_id=track_id,
                route_requested=self.context.requested,
                route_detected=consensus.route,
                detection_confidence=consensus.detection_confidence,
                ocr_confidence=consensus.ocr_confidence,
                match_status=match.status,
                frames_considered=consensus.frames,
                supporting_frames=consensus.supporting,
                source=self.context.source,  # type: ignore[arg-type]
                simulated=self.context.source == "simulation",
                model_version=self.detector.model_version,
                status="CONFIRMED" if consensus.route else "UNCERTAIN",
                raw_texts=consensus.raw_texts,
                reason=f"{consensus.reason}. {match.reason}",
            )
            if self.gate.accept(event):
                events.append(event)
            b = detection.box
            cv2.rectangle(
                annotated, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 220, 220), 2
            )
            label = f"#{track_id} {consensus.route or 'INCERTO'}"
            cv2.putText(
                annotated,
                label,
                (int(b.x1), max(20, int(b.y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 220),
                2,
            )
        return annotated, events
