"""Opt-in bounded left-code experiment; baseline texts and safety rules retained."""

from sonar_vision.br.ocr_optimization import OCRExperiment, preprocess
from sonar_vision.core.contracts import OCRReading
from sonar_vision.ocr.paddle import PaddleEngine
from sonar_vision.processing.roi import Frame, validate_frame
from sonar_vision.validation.normalizer import FULL_CODE


class LeftCodeOCR:
    """Additional recognition in a plausible code region, without expected-line input.

    Only independently observed complete suffix-bearing codes may supplement the
    baseline. Short/partial strings cannot be elevated by this heuristic.
    """

    def __init__(self, baseline: PaddleEngine) -> None:
        self.baseline = baseline

    def read(self, roi: Frame) -> list[OCRReading]:
        validate_frame(roi)
        readings = self.baseline.read(roi)
        height, width = roi.shape[:2]
        if width < 4 * height or height < 12:
            return readings
        crop = roi[:, : round(width * 0.55)].copy()
        crop = preprocess(crop, OCRExperiment(preprocessing="scale2"))
        additional = self.baseline.read(crop)
        readings.extend(
            r
            for r in additional
            if r.confidence >= 0.85 and FULL_CODE.fullmatch(r.text.strip().upper())
        )
        return readings
