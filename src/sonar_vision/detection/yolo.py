from pathlib import Path
from typing import Any, Protocol, cast

from sonar_vision.core.contracts import Box, Detection
from sonar_vision.processing.roi import Frame, validate_frame


class Detector(Protocol):
    model_version: str

    def detect(self, frame: Frame) -> list[Detection]: ...


class ModelUnavailable(RuntimeError):
    pass


class YoloSignDetector:
    def __init__(self, weights: Path, confidence: float = 0.35, model: Any = None):
        if not 0 <= confidence <= 1:
            raise ValueError("Confiança inválida")
        if model is None:
            if not weights.is_file():
                raise ModelUnavailable("Modelo customizado ausente. Consulte docs/TRAINING.md")
            try:
                import ultralytics

                model = cast(Any, ultralytics).YOLO(str(weights))
            except Exception as exc:
                raise ModelUnavailable("Falha ao carregar pesos ou dependência YOLO") from exc
        self.model = model
        self.confidence = confidence
        self.sign_classes = {
            int(i)
            for i, name in model.names.items()
            if str(name).lower() in {"destination_sign", "led-panel", "led_panel"}
        }
        if not self.sign_classes:
            raise ModelUnavailable(
                "Pesos não contêm a classe customizada destination_sign/LED-Panel"
            )
        from hashlib import sha256

        self.model_version = (
            f"yolo-sign-sha256:{sha256(weights.read_bytes()).hexdigest()}"
            if weights.is_file()
            else "injected-test-model"
        )

    def detect(self, frame: Frame) -> list[Detection]:
        validate_frame(frame)
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            classes=sorted(self.sign_classes),
            imgsz=640,
            verbose=False,
            save=False,
            max_det=16,
        )
        output = []
        h, w = frame.shape[:2]
        for result in results:
            if result.boxes is None:
                continue
            for coords, score, class_id in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
                strict=True,
            ):
                if int(class_id) not in self.sign_classes or score < self.confidence:
                    continue
                x1, y1, x2, y2 = coords
                x1, y1 = max(0, min(w, x1)), max(0, min(h, y1))
                x2, y2 = max(0, min(w, x2)), max(0, min(h, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                output.append(
                    Detection(
                        box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                        confidence=score,
                        label="destination_sign",
                    )
                )
        return output
