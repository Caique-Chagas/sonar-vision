"""Opt-in OCR experiments. Production OCR, parsing and temporal defaults are unchanged."""

import re
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from sonar_vision.core.contracts import OCRReading
from sonar_vision.ocr.paddle import OCRFailure, PaddleEngine
from sonar_vision.processing.roi import Frame, validate_frame
from sonar_vision.validation.normalizer import LineNormalizer, NormalizedLine


@dataclass(frozen=True)
class OCRExperiment:
    preprocessing: str = "original"
    spatial: str = "none"
    max_width: int = 2560
    max_polygons: int = 4
    max_tokens: int = 8
    polygon_margin: float = 0.04
    additional_min_score: float = 0.85

    def __post_init__(self) -> None:
        if self.preprocessing not in {"original", "scale2", "scale3", "gray", "clahe", "adaptive"}:
            raise ValueError("Unknown experimental preprocessing")
        if self.spatial not in {"none", "redetect", "recognize"}:
            raise ValueError("Unknown spatial experiment")
        if not (
            64 <= self.max_width <= 2560
            and 1 <= self.max_polygons <= 4
            and 1 <= self.max_tokens <= 8
            and 0 <= self.polygon_margin <= 0.1
            and 0.7 <= self.additional_min_score <= 1
        ):
            raise ValueError("Experimental resource or confidence bounds exceeded")


def preprocess(roi: Frame, config: OCRExperiment) -> Frame:
    validate_frame(roi)
    output = roi.copy()
    if config.preprocessing.startswith("scale"):
        factor = min(
            float(config.preprocessing[-1]), config.max_width / roi.shape[1], 2160 / roi.shape[0]
        )
        output = np.asarray(
            cv2.resize(output, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC),
            dtype=np.uint8,
        )
    elif config.preprocessing != "original":
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        if config.preprocessing == "clahe":
            gray = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4)).apply(gray)
        elif config.preprocessing == "adaptive":
            gray = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9
            )
        output = np.asarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), dtype=np.uint8)
    validate_frame(output)
    return output


def token_crops(line: Frame, maximum: int) -> list[Frame]:
    """Split text using observed inter-word gaps; never fixed code coordinates or truth."""
    gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground = binary > 0
    if foreground.mean() > 0.5:
        foreground = ~foreground
    occupied = foreground.mean(axis=0) > 0.06
    minimum_gap = max(3, round(line.shape[0] * 0.22))
    runs = []
    start = 0
    for x in range(1, len(occupied)):
        if occupied[x] != occupied[x - 1]:
            if not occupied[x - 1] and x - start >= minimum_gap:
                runs.append((start, x))
            start = x
    cuts = [0] + [(a + b) // 2 for a, b in runs] + [line.shape[1]]
    return [
        line[:, a:b].copy()
        for a, b in zip(cuts, cuts[1:], strict=False)
        if 0.35 <= (b - a) / line.shape[0] <= 6
    ][:maximum]


class BrazilianSeparatorNormalizer(LineNormalizer):
    """Repair explicit separators, never missing digits, suffixes or O/I/S characters."""

    def normalize(self, text: str) -> NormalizedLine:
        parsed = re.sub(r"(?<=[0-9A-Z])\s*/\s*(?=\d{2}(?:\D|$))", "-", text.upper())
        # A separated dash before a destination word is not a route-code suffix.
        parsed = re.sub(r"(?<=\d)\s+[-–—]\s+(?=[A-Z]{2})", " ", parsed)
        value = super().normalize(parsed)
        return NormalizedLine(
            text, value.candidates, "BR_separators/" + value.method, value.ambiguous
        )


class ExperimentalOCR:
    def __init__(self, baseline: PaddleEngine, config: OCRExperiment, recognizer: Any = None):
        self.baseline, self.config, self.recognizer = baseline, config, recognizer
        self.trace: dict[str, Any] = {}

    def read(self, roi: Frame) -> list[OCRReading]:
        image = preprocess(roi, self.config)
        try:
            results = list(self.baseline.engine.predict(image))
            readings: list[OCRReading] = []
            polygons: list[Any] = []
            self.trace = {
                "input_shape": list(roi.shape),
                "processed_shape": list(image.shape),
                "native_results": [],
                "additional_reads": [],
            }
            for result in results:
                texts, scores = result["rec_texts"], result["rec_scores"]
                readings.extend(
                    OCRReading(text=str(t), confidence=float(s))
                    for t, s in zip(texts, scores, strict=True)
                    if str(t).strip()
                )
                polys = np.asarray(result.get("dt_polys", []))
                polygons.extend(polys)
                self.trace["native_results"].append(
                    {
                        "texts": list(texts),
                        "scores": [float(s) for s in scores],
                        "detected_polygons": polys.tolist(),
                        "recognized_polygons": np.asarray(result.get("rec_polys", [])).tolist(),
                    }
                )
            if self.config.spatial == "none":
                return readings
            for poly in polygons[: self.config.max_polygons]:
                points = np.asarray(poly)
                lo, hi = points.min(axis=0), points.max(axis=0)
                margin = (hi - lo) * self.config.polygon_margin
                x1, y1 = np.maximum(0, (lo - margin).astype(int))
                x2, y2 = np.minimum([image.shape[1], image.shape[0]], (hi + margin).astype(int))
                crop = image[y1:y2, x1:x2].copy()
                if not crop.size:
                    continue
                if self.config.spatial == "redetect":
                    extra = self.baseline.read(crop)
                else:
                    if self.recognizer is None:
                        raise ValueError("Explicit cached recognizer required")
                    crops = [crop] + token_crops(crop, self.config.max_tokens - 1)
                    extra = [
                        OCRReading(text=str(r["rec_text"]), confidence=float(r["rec_score"]))
                        for r in self.recognizer.predict(input=crops, batch_size=1)
                        if str(r["rec_text"]).strip()
                    ]
                self.trace["additional_reads"].append(
                    {
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "readings": [r.model_dump() for r in extra],
                    }
                )
                readings.extend(
                    r for r in extra if r.confidence >= self.config.additional_min_score
                )
            return readings
        except Exception as exc:
            raise OCRFailure("Experimental OCR failed; no confirmation produced") from exc
