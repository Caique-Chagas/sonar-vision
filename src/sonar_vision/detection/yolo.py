from pathlib import Path
from typing import Any, Protocol, cast

from sonar_vision.core.contracts import Box, Detection
from sonar_vision.processing.roi import Frame, validate_frame


class Detector(Protocol):
    model_version: str

    def detect(self, frame: Frame) -> list[Detection]: ...


class ModelUnavailable(RuntimeError):
    pass


def _supports_native_tiles() -> bool:
    try:
        import ultralytics.cfg as cfg
    except ImportError:
        return False
    return "tile" in getattr(cfg, "DEFAULT_CFG_DICT", {})


def _native_tile_count(frame: Frame, size: int) -> int:
    from ultralytics.data.utils import tile_windows

    return len(tile_windows(frame.shape[:2], size))


class YoloSignDetector:
    def __init__(
        self,
        weights: Path,
        confidence: float = 0.35,
        model: Any = None,
        *,
        imgsz: int = 640,
        tile_size: int = 0,
    ):
        if not 0 <= confidence <= 1:
            raise ValueError("Confiança inválida")
        if type(imgsz) is not int or not 320 <= imgsz <= 1920 or imgsz % 32:
            raise ValueError("imgsz deve ser múltiplo de32 entre320 e1920")
        if type(tile_size) is not int or (
            tile_size != 0 and (not 320 <= tile_size <= 1280 or tile_size % 32)
        ):
            raise ValueError("tile_size deve ser0 ou múltiplo de32 entre320 e1280")
        if tile_size and not _supports_native_tiles():
            raise ModelUnavailable(
                "Ultralytics instalado não suporta tiles nativos; desative tiles"
            )
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
        self.imgsz, self.tile_size = imgsz, tile_size
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
        options: dict[str, Any] = {}
        if self.tile_size:
            if _native_tile_count(frame, self.tile_size) > 64:
                raise ValueError("Orçamento excedido: máximo64 tiles por imagem")
            options.update(tile=self.tile_size, batch=1)
        elif hasattr(getattr(getattr(self.model, "predictor", None), "args", None), "tile"):
            # Reset sticky native predict arguments on reuse; do not send new
            # kwargs to older Ultralytics versions that have no tile support.
            options["tile"] = False
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            classes=sorted(self.sign_classes),
            imgsz=self.imgsz,
            verbose=False,
            save=False,
            max_det=16,
            **options,
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
