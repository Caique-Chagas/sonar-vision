import cv2
import numpy as np
from numpy.typing import NDArray

from sonar_vision.core.contracts import Box

Frame = NDArray[np.uint8]


def validate_frame(frame: Frame) -> None:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError("Frame precisa ser numpy uint8")
    if frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
        raise ValueError("Frame precisa ter formato BGR H×W×3 não vazio")
    if frame.shape[0] * frame.shape[1] > 3840 * 2160:
        raise ValueError("Frame excede limite de 4K")


def extract_roi(frame: Frame, box: Box, mode: str = "original") -> Frame:
    validate_frame(frame)
    h, w = frame.shape[:2]
    x1, y1 = min(w, int(box.x1)), min(h, int(box.y1))
    x2, y2 = min(w, int(box.x2)), min(h, int(box.y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Bounding box está fora do frame")
    crop = frame[y1:y2, x1:x2].copy()
    scale = min(3.0, 1280 / crop.shape[1], max(1.0, 64 / crop.shape[0]))
    if scale != 1:
        crop = np.asarray(
            cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC),
            dtype=np.uint8,
        )
    if mode == "clahe":
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4)).apply(gray)
        crop = np.asarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), dtype=np.uint8)
    elif mode != "original":
        raise ValueError("Pré-processamento desconhecido")
    return crop
