"""Development-only perspective rectification from observed text polygons."""

from typing import Any

import cv2
import numpy as np

from sonar_vision.br.ocr_optimization import ExperimentalOCR, OCRExperiment, token_crops
from sonar_vision.core.contracts import OCRReading
from sonar_vision.ocr.paddle import PaddleEngine
from sonar_vision.processing.roi import Frame, validate_frame


def rectify_quad(image: Frame, polygon: Any) -> Frame | None:
    validate_frame(image)
    points = np.asarray(polygon, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return None
    if (
        (points < 0).any()
        or (points[:, 0] >= image.shape[1]).any()
        or (points[:, 1] >= image.shape[0]).any()
    ):
        return None
    center = points.mean(axis=0)
    order = np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    points = points[order]
    points = np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)
    if not cv2.isContourConvex(points) or cv2.contourArea(points) < 4:
        return None
    width = round(max(np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[3])))
    height = round(
        max(np.linalg.norm(points[3] - points[0]), np.linalg.norm(points[2] - points[1]))
    )
    if not (4 <= width <= 2560 and 4 <= height <= 512):
        return None
    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32
    )
    transform = cv2.getPerspectiveTransform(points, target)
    return np.asarray(cv2.warpPerspective(image, transform, (width, height)), dtype=np.uint8)


class RectifiedOCR:
    def __init__(self, engine: PaddleEngine, recognizer: Any):
        self.base = ExperimentalOCR(engine, OCRExperiment())
        self.recognizer = recognizer

    def read(self, roi: Frame) -> list[OCRReading]:
        readings = self.base.read(roi)
        polygons = [p for r in self.base.trace["native_results"] for p in r["detected_polygons"]]
        for polygon in polygons[:4]:
            crop = rectify_quad(roi, polygon)
            if crop is None:
                continue
            crops = [crop] + token_crops(crop, 7)
            for result in self.recognizer.predict(input=crops, batch_size=1):
                if str(result["rec_text"]).strip() and float(result["rec_score"]) >= 0.85:
                    readings.append(
                        OCRReading(
                            text=str(result["rec_text"]), confidence=float(result["rec_score"])
                        )
                    )
        return readings
