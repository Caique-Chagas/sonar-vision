"""Two-class private V3 adapter. Bus overlays never enter OCR pixels."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image, ImageDraw

from sonar_vision.core.contracts import Box, Detection
from sonar_vision.desktop.render import ui_font
from sonar_vision.detection.yolo import Detector
from sonar_vision.processing.local_video import DemoFrameAnalyzer, create_demo_analyzer
from sonar_vision.processing.roi import Frame, validate_frame

ACTIVE = Path("artifacts/demo/sonar-video-v3/active.json")


def active_configuration() -> dict[str, Any] | None:
    if not ACTIVE.is_file():
        return None
    data: dict[str, Any] = json.loads(ACTIVE.read_text())
    if data.get("status") != "VALIDATED_PRIVATE" or data.get("rights_verified") is not False:
        raise ValueError("Perfil privadoV3 sem estado explícito")
    weight = Path(data["weights"])
    if (
        not weight.is_file()
        or hashlib.sha256(weight.read_bytes()).hexdigest() != data["weights_sha256"]
    ):
        raise ValueError("Pesos V3 ausentes ou alterados")
    if data.get("baseline_weights"):
        baseline = Path(data["baseline_weights"])
        if hashlib.sha256(baseline.read_bytes()).hexdigest() != data["baseline_sha256"]:
            raise ValueError("Pesos baseline alterados")
    for profile in data.get("profiles", {}).values():
        if (
            hashlib.sha256(Path(profile["video"]).read_bytes()).hexdigest()
            != profile["video_sha256"]
        ):
            raise ValueError("Vídeo privado alterado")
    return data


class VideoV3Detector:
    def __init__(self, weights: Path, imgsz: int = 960, model: Any = None):
        if type(imgsz) is not int or not 320 <= imgsz <= 1920 or imgsz % 32:
            raise ValueError("Resolução V3 inválida")
        if model is None:
            import importlib

            YOLO = importlib.import_module("ultralytics").YOLO

            if not weights.is_file():
                raise ValueError("PesosV3 ausentes;sem download/substituição")
            model = YOLO(str(weights))
        if model.names != {0: "bus", 1: "destination_sign"}:
            raise ValueError("V3 precisa de DUAS classes:bus0/destination_sign1")
        self.model, self.imgsz = model, imgsz
        self.model.overrides.update(device="cpu", iou=0.5, rect=True, batch=1, quantize=None)
        self.model_version = "video-v3-sha256:" + hashlib.sha256(weights.read_bytes()).hexdigest()

    def detect(self, frame: Frame) -> list[Detection]:
        validate_frame(frame)
        results = self.model.predict(
            frame, conf=0.35, imgsz=self.imgsz, classes=[0, 1], max_det=32, verbose=False
        )
        h, w = frame.shape[:2]
        output = []
        for result in results:
            for xy, score, cls in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
                strict=True,
            ):
                x1, y1, x2, y2 = xy
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    output.append(
                        Detection(
                            box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                            confidence=score,
                            label=self.model.names[int(cls)],
                        )
                    )
        return output


class FrameDetections:
    def __init__(self, rows: list[Detection], version: str):
        self.rows, self.model_version = rows, version

    def detect(self, frame: Frame) -> list[Detection]:
        return self.rows


class BusAndSignDetector:
    """Explicit experimental composition; never credits V1 panels to V3."""

    def __init__(self, buses: VideoV3Detector, signs: Detector):
        self.buses, self.signs = buses, signs
        self.model_version = f"composition:bus={buses.model_version};sign={signs.model_version}"

    def detect(self, frame: Frame) -> list[Detection]:
        start = time.perf_counter()
        buses = [d for d in self.buses.detect(frame) if d.label == "bus"]
        bus_end = time.perf_counter()
        signs = [d for d in self.signs.detect(frame) if d.label == "destination_sign"]
        self.last_timings = {
            "yolo_bus_seconds": bus_end - start,
            "yolo_sign_seconds": time.perf_counter() - bus_end,
        }
        return buses + signs


class VideoV3Analyzer:
    def __init__(self, detector: Detector, ocr: Any):
        self.detector, self.ocr = detector, ocr

    def analyze(self, frame: Frame) -> tuple[Frame, dict[str, Any]]:
        start = time.perf_counter()
        detections = self.detector.detect(frame)
        analyzer = DemoFrameAnalyzer(
            FrameDetections(detections, self.detector.model_version), self.ocr
        )
        annotated, row = analyzer.analyze(frame)
        buses = [d for d in detections if d.label == "bus"]
        scale = annotated.shape[1] / frame.shape[1]
        for bus in buses:
            b = bus.box
            p1, p2 = (
                (round(b.x1 * scale), round(b.y1 * scale)),
                (round(b.x2 * scale), round(b.y2 * scale)),
            )
            cv2.rectangle(annotated, p1, p2, (255, 140, 60), 2)
            cv2.putText(
                annotated,
                f"BUS {bus.confidence:.3f}",
                (p1[0], max(20, p1[1] - 5)),
                0,
                0.55,
                (255, 140, 60),
                1,
            )
        y = annotated.shape[0] - 140
        cv2.rectangle(annotated, (0, y), (annotated.shape[1], y + 38), (22, 27, 35), -1)
        image = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(image).text(
            (12, y + 12),
            "SONAR VIDEO V3 | VIDEO REAL | EXPERIMENTO LOCAL PRIVADO",
            font=ui_font(15),
            fill="white",
        )
        row["buses"] = [b.model_dump() for b in buses]
        row["rights_verified"] = False
        row["stage_timings"] = getattr(self.detector, "last_timings", {})
        row["inference_seconds"] = time.perf_counter() - start
        return cast(Frame, cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)), row


def make_analyzer(weights: Path, imgsz: int, bus_weights: Path | None = None) -> Any:
    if bus_weights is not None:
        original = create_demo_analyzer(weights, imgsz)
        return VideoV3Analyzer(
            BusAndSignDetector(VideoV3Detector(bus_weights, imgsz), original.detector),
            original.ocr,
        )
    config = active_configuration()
    if config and Path(config["weights"]).resolve() == weights.resolve():
        # Reuse the existing cached OCR initialization,replace only detector.
        original = create_demo_analyzer(weights, imgsz)
        return VideoV3Analyzer(
            VideoV3Detector(weights, imgsz, cast(Any, original.detector).model), original.ocr
        )
    return create_demo_analyzer(weights, imgsz)
