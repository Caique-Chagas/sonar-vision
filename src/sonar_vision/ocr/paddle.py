from typing import Any, Protocol

from sonar_vision.core.contracts import OCRReading
from sonar_vision.processing.roi import Frame, validate_frame


class OCREngine(Protocol):
    def read(self, roi: Frame) -> list[OCRReading]: ...


class OCRUnavailable(RuntimeError):
    pass


class OCRFailure(RuntimeError):
    pass


class PaddleEngine:
    def __init__(self, engine: Any = None):
        if engine is None:
            try:
                from paddleocr import PaddleOCR

                engine = PaddleOCR(
                    lang="en",
                    ocr_version="PP-OCRv5",
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="en_PP-OCRv5_mobile_rec",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    device="cpu",
                )
            except Exception as exc:
                raise OCRUnavailable(
                    "PaddleOCR indisponível; instale o extra paddle e seus modelos"
                ) from exc
        self.engine = engine

    def read(self, roi: Frame) -> list[OCRReading]:
        validate_frame(roi)
        try:
            readings = []
            for result in self.engine.predict(roi):
                for text, score in zip(result["rec_texts"], result["rec_scores"], strict=True):
                    if str(text).strip():
                        readings.append(OCRReading(text=str(text), confidence=float(score)))
            return readings
        except Exception as exc:
            raise OCRFailure("Falha no reconhecimento OCR; nenhuma leitura confirmada") from exc
