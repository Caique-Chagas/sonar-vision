"""Optional V4 recovery on actual bus crops; never extrapolates missing detections."""

from typing import Any

from sonar_vision.br.hierarchy import HierarchicalDetector
from sonar_vision.core.contracts import Box, Detection
from sonar_vision.detection.yolo import Detector
from sonar_vision.processing.roi import Frame
from sonar_vision.tracking.iou import iou


def belongs_to_panel_region(panel: Box, bus: Box) -> bool:
    """Conservative front-panel hypothesis, not a learned classification."""
    cx, cy = (panel.x1 + panel.x2) / 2, (panel.y1 + panel.y2) / 2
    area = (panel.x2 - panel.x1) * (panel.y2 - panel.y1)
    inside = max(0, min(panel.x2, bus.x2) - max(panel.x1, bus.x1)) * max(
        0, min(panel.y2, bus.y2) - max(panel.y1, bus.y1)
    )
    return (
        bus.x1 <= cx <= bus.x2
        and bus.y1 <= cy <= bus.y1 + 0.45 * (bus.y2 - bus.y1)
        and inside / area >= 0.85
        and (panel.x2 - panel.x1) / (panel.y2 - panel.y1) >= 2
    )


class CachedBuses:
    """Current-frame predictions supplied by the baseline; no annotation input."""

    model_version = "current-frame-cached-V3-bus"

    def __init__(self, buses: list[Detection]):
        self.buses = buses

    def detect(self, frame: Frame) -> list[Detection]:
        return self.buses


class PanelRecovery:
    def __init__(self, signs: Detector):
        self.signs = signs
        self.trace: list[dict[str, Any]] = []

    def detect(
        self, frame: Frame, buses: list[Detection], global_panels: list[Detection]
    ) -> list[Detection]:
        self.trace = []
        recovered: list[Detection] = []
        for bus in sorted(buses, key=lambda d: d.confidence, reverse=True)[:8]:
            if any(belongs_to_panel_region(p.box, bus.box) for p in global_panels):
                continue
            cascade = HierarchicalDetector(CachedBuses([bus]), self.signs, margin=0.05)
            proposals = cascade.detect(frame)
            accepted = []
            for panel in proposals:
                if belongs_to_panel_region(panel.box, bus.box) and all(
                    iou(panel.box, other.box) < 0.5 for other in global_panels + recovered
                ):
                    recovered.append(panel)
                    accepted.append(panel.model_dump())
            self.trace.append(dict(cascade=cascade.trace, accepted=accepted))
        return recovered
