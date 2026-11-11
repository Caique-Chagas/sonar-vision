"""Explicit synthetic UI fixture. Never a training or computer vision benchmark."""

import cv2
import numpy as np

from sonar_vision.core.contracts import Box, Detection, OCRReading
from sonar_vision.processing.roi import Frame


class SimulationDetector:
    model_version = "SIMULATED-NO-TRAINED-MODEL"

    def detect(self, frame: Frame) -> list[Detection]:
        return [
            Detection(
                box=Box(x1=180, y1=155, x2=780, y2=245),
                confidence=0.9,
                label="destination_sign",
            )
        ]


class SimulationOCR:
    text = "971D-10"
    phase = 0

    def read(self, roi: Frame) -> list[OCRReading]:
        return [OCRReading(text=self.text, confidence=0.9)]


def simulation_frame(text: str) -> Frame:
    frame = np.full((540, 960, 3), (30, 25, 20), np.uint8)
    cv2.putText(
        frame,
        "SIMULACAO - SEM INFERENCIA REAL",
        (35, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 220, 255),
        2,
    )
    cv2.rectangle(frame, (140, 110), (820, 490), (115, 88, 50), -1)
    cv2.rectangle(frame, (180, 155), (780, 245), (0, 0, 0), -1)
    cv2.putText(frame, text, (220, 217), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 200, 255), 3)
    cv2.rectangle(frame, (180, 270), (780, 415), (60, 50, 35), -1)
    return frame
