"""Rights-gated, bounded offline screening; video pages are never decoder inputs."""

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np

from sonar_vision.processing.roi import Frame, validate_frame


@dataclass(frozen=True)
class MediaPermission:
    source_url: str
    author: str
    license: str
    evaluation_allowed: bool = False
    demonstration_allowed: bool = False
    training_allowed: bool = False

    def require_evaluation(self) -> None:
        parsed = urlsplit(self.source_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or not self.author.strip()
            or not self.license.strip()
            or not self.evaluation_allowed
        ):
            raise ValueError("Fonte sem origem/autoria/permissão de avaliação verificadas")


def exact_route_score(actual: str | None, expected: str | None) -> bool | None:
    """Missing independent ground truth is unscored, never counted as correct."""
    from sonar_vision.validation.normalizer import clean_code

    if expected is None or not expected.strip():
        return None
    return actual is not None and clean_code(actual) == clean_code(expected)


def normalize_frame(frame: Frame) -> Frame:
    # Large originals are retained; only the evaluation copy is reduced to <=4K.
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError("Frame corrompido")
    if frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
        raise ValueError("Frame corrompido")
    h, w = frame.shape[:2]
    factor = min(1.0, 3200 / w, 1800 / h)
    if factor < 1:
        frame = np.asarray(cv2.resize(frame, (int(w * factor), int(h * factor))), dtype=np.uint8)
    validate_frame(frame)
    return frame


def sample_video(
    path: Path,
    permission: MediaPermission,
    timestamps: list[float],
    max_frames: int = 120,
    max_seconds: float = 60,
) -> Iterator[tuple[float, Frame]]:
    permission.require_evaluation()
    if not 1 <= max_frames <= 120 or not 0 < max_seconds <= 180:
        raise ValueError("Limites de triagem inválidos")
    if len(timestamps) > max_frames or any(t < 0 for t in timestamps):
        raise ValueError("Amostra excede limites")
    if not path.is_file():
        raise ValueError("Arquivo de mídia indisponível")
    cap = cv2.VideoCapture(str(path))
    started = time.monotonic()
    try:
        if not cap.isOpened():
            raise ValueError("Vídeo indisponível ou corrompido")
        for timestamp in timestamps:
            if time.monotonic() - started >= max_seconds:
                raise TimeoutError("Orçamento de decodificação excedido")
            if not cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000):
                raise ValueError("Seek indisponível; não substituir por frame de outro instante")
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Frame indisponível em {timestamp}s")
            yield timestamp, normalize_frame(np.asarray(frame, dtype=np.uint8))
    finally:
        cap.release()
