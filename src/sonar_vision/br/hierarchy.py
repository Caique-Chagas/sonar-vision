"""Opt-in bus-to-panel cascade; no production defaults or model weights change."""

import math
from typing import Any

from sonar_vision.core.contracts import Box, Detection
from sonar_vision.detection.yolo import Detector
from sonar_vision.processing.roi import Frame, validate_frame
from sonar_vision.tracking.iou import iou


class HierarchicalDetector:
    def __init__(self, bus: Detector, sign: Detector, *, margin: float = 0.05):
        if not math.isfinite(margin) or not 0 <= margin <= 0.15:
            raise ValueError("Invalid bus crop margin")
        self.bus, self.sign, self.margin = bus, sign, margin
        self.model_version = f"cascade:{bus.model_version}:{sign.model_version}"
        self.trace: list[dict[str, Any]] = []

    def detect(self, frame: Frame) -> list[Detection]:
        validate_frame(frame)
        height, width = frame.shape[:2]
        self.trace = []
        output = []
        buses = [b for b in self.bus.detect(frame) if b.label == "bus"]
        for bus in sorted(buses, key=lambda b: b.confidence, reverse=True)[:8]:
            b = bus.box
            dx, dy = (b.x2 - b.x1) * self.margin, (b.y2 - b.y1) * self.margin
            x1, y1 = max(0, math.floor(b.x1 - dx)), max(0, math.floor(b.y1 - dy))
            x2, y2 = min(width, math.ceil(b.x2 + dx)), min(height, math.ceil(b.y2 + dy))
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            local = self.sign.detect(frame[y1:y2, x1:x2].copy())
            self.trace.append(
                {
                    "bus_box": b.model_dump(),
                    "bus_confidence": bus.confidence,
                    "crop_xyxy": [x1, y1, x2, y2],
                    "signs": len(local),
                }
            )
            for prediction in local:
                if prediction.label != "destination_sign":
                    continue
                p = prediction.box
                ax, ay = max(x1, x1 + p.x1), max(y1, y1 + p.y1)
                bx, by = min(x2, x1 + p.x2), min(y2, y1 + p.y2)
                if ax < bx and ay < by:
                    output.append(
                        Detection(
                            box=Box(x1=ax, y1=ay, x2=bx, y2=by),
                            confidence=prediction.confidence,
                            label="destination_sign",
                        )
                    )
        kept: list[Detection] = []
        for prediction in sorted(output, key=lambda p: p.confidence, reverse=True):
            if all(iou(prediction.box, other.box) < 0.5 for other in kept):
                kept.append(prediction)
        return kept[:16]
