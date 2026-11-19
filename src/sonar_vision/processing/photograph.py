"""New photograph inference, retaining actual detector ROIs and raw OCR only."""

import hashlib
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from sonar_vision.core.contracts import Box
from sonar_vision.processing.local_video import DemoFrameAnalyzer, LocalVideoError
from sonar_vision.processing.roi import Frame, extract_roi, validate_frame

MAX_IMAGE = 12 * 1024 * 1024


def validate_image_name(filename: str, mime: str, consent: str) -> None:
    suffix = Path(filename).suffix.lower()
    expected = "image/png" if suffix == ".png" else "image/jpeg"
    if (
        not filename
        or len(filename) > 180
        or any(c in filename for c in ["/", "\\"])
        or any(ord(c) < 32 for c in filename)
        or suffix not in {".jpg", ".jpeg", ".png"}
        or mime != expected
    ):
        raise LocalVideoError("Selecione uma fotografia JPEG ou PNG válida")
    if consent != "confirmed":
        raise LocalVideoError("Confirme a autorização para processar a fotografia")


def decode_photograph(path: Path) -> Frame:
    from PIL import Image

    if not 16 <= path.stat().st_size <= MAX_IMAGE:
        raise LocalVideoError("Fotografia vazia ou maior que12MiB")
    try:
        with Image.open(path) as image:
            w, h = image.size
            if image.format not in {"JPEG", "PNG"} or getattr(image, "n_frames", 1) != 1:
                raise LocalVideoError("Somente fotografias JPEG/PNG, sem animação")
            if not (2 <= w <= 4096 and 2 <= h <= 4096 and w * h <= 3840 * 2160):
                raise LocalVideoError("Fotografia excede4096px por lado ou8,29MP")
            image.verify()
        frame = cast(
            Frame,
            cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR),
        )
        validate_frame(frame)
        return frame
    except LocalVideoError:
        raise
    except Exception as exc:
        raise LocalVideoError("Fotografia corrompida ou não decodificável") from exc


def analyze_photograph(image: Frame, folder: Path, analyzer: DemoFrameAnalyzer) -> dict[str, Any]:
    if not analyzer.static:
        raise ValueError("Analisador deve identificar modalidade estática")
    annotated, row = analyzer.analyze(image)
    if not cv2.imwrite(str(folder / "photo-original.png"), image):
        raise LocalVideoError("Não foi possível salvar a fotografia")
    if not cv2.imwrite(str(folder / "photo-annotated.jpg"), annotated):
        raise LocalVideoError("Não foi possível salvar a inferência")
    for index, finding in enumerate(row["findings"]):
        roi = extract_roi(image, Box.model_validate(finding["box"]))
        if not cv2.imwrite(str(folder / f"roi-{index}.png"), roi):
            raise LocalVideoError("Não foi possível salvar a ROI prevista")
        finding.update(crop_id=index, crop_size=[roi.shape[1], roi.shape[0]])
    row.update(
        mode="static_photograph",
        model_version=analyzer.detector.model_version,
        detector_parameters={
            "confidence": 0.35,
            "imgsz": 640,
            "device": "cpu",
            "iou": 0.5,
            "rect": True,
            "batch": 1,
            "tiles": False,
            "roi_padding": 0,
        },
        OCR="Existing E3: PaddleOCR original + en_PP-OCRv5_mobile_rec spatial recognition",
        input_dimensions=[image.shape[1], image.shape[0]],
        source_sha256=hashlib.sha256((folder / "source-image").read_bytes()).hexdigest(),
        ground_truth_used=False,
        accuracy=None,
        temporal_confirmation=False,
    )
    return row
