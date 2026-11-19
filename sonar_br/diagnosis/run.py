"""Paired diagnostic cascade on already-known V2 examples; not a new holdout."""

import json
from pathlib import Path
from typing import Any

import cv2
from paddleocr import TextRecognition
from ultralytics import YOLO

from sonar_br.ocr_experiments.run import case
from sonar_vision.br.hierarchy import HierarchicalDetector
from sonar_vision.br.ocr_optimization import ExperimentalOCR, OCRExperiment
from sonar_vision.core.contracts import Box, Detection
from sonar_vision.detection.yolo import YoloSignDetector
from sonar_vision.evaluation.final_photos import read_case, write_case
from sonar_vision.processing.roi import Frame
from sonar_vision.training.dataset import sha256

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "sonar_br/artifacts/v2-diagnosis"
REF = ROOT / "sonar_br/datasets/manifests/br-v2.review.json"


class GenericBusDetector:
    def __init__(self) -> None:
        self.model = YOLO(str(ROOT / "yolo11n.pt"))
        self.model_version = "COCO:" + sha256(ROOT / "yolo11n.pt")
        self.classes = [i for i, value in self.model.names.items() if value == "bus"]
        if not self.classes:
            raise ValueError("Generic model lacks bus class")

    def detect(self, frame: Frame) -> list[Detection]:
        predictions = self.model.predict(
            frame,
            classes=self.classes,
            conf=0.25,
            iou=0.5,
            imgsz=640,
            device="cpu",
            max_det=8,
            save=False,
            verbose=False,
        )
        output = []
        for result in predictions:
            for xyxy, score in zip(
                result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), strict=True
            ):
                output.append(
                    Detection(
                        box=Box(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3]),
                        confidence=score,
                        label="bus",
                    )
                )
        return output


def main() -> None:
    OUT.mkdir(exist_ok=True)
    files = [
        REF,
        ROOT / "yolo11n.pt",
        ROOT / "sonar_br/artifacts/train-v1/run/weights/best.pt",
        Path(__file__),
        ROOT / "src/sonar_vision/br/hierarchy.py",
    ]
    binding: dict[str, Any] = {
        "hashes": {str(p.relative_to(ROOT)): sha256(p) for p in files},
        "purpose": "diagnosis on known V2 test; never call independent validation",
        "bus": {"confidence": 0.25, "imgsz": 640, "max": 8, "margin": 0.05},
        "sign": {"confidence": 0.35, "imgsz": 640, "NMS": 0.5, "ROI_padding": 0},
        "OCR": "unchanged E3; CPU; same settings/normalizer/temporal as frozen V2",
        "order": "alternating baseline/cascade, reversed on odd photos",
    }
    frozen = OUT / "binding.json"
    if frozen.exists() and json.loads(frozen.read_text()) != binding:
        raise ValueError("Do not overwrite an executed experiment")
    frozen.write_text(json.dumps(binding, indent=2) + "\n")
    backend = YoloSignDetector(ROOT / "sonar_br/artifacts/train-v1/run/weights/best.pt")
    backend.model.overrides.update(device="cpu", iou=0.5, rect=True, quantize=None, batch=1)
    hierarchy = HierarchicalDetector(GenericBusDetector(), backend)
    baseline = ExperimentalOCR(
        PaddleEngine(),
        OCRExperiment(spatial="recognize"),
        TextRecognition(
            model_name="en_PP-OCRv5_mobile_rec",
            model_dir=str(Path.home() / ".paddlex/official_models/en_PP-OCRv5_mobile_rec"),
            device="cpu",
            enable_mkldnn=False,
            cpu_threads=2,
        ),
    )
    refs = [r for r in json.loads(REF.read_text())["records"] if r["split"] == "test"]
    for index, ref in enumerate(refs):
        ref = ref | {"scope": "known_V2_diagnostic_not_independent"}
        image = cv2.imread(ref["image"])
        for name in ["baseline", "hierarchy"] if index % 2 == 0 else ["hierarchy", "baseline"]:
            dest = OUT / "cases" / f"{name}-{ref['id']}.json"
            if read_case(dest, sha256(frozen)) is not None:
                continue
            row = case(image, ref, backend if name == "baseline" else hierarchy, baseline, False)
            row["bus_trace"] = hierarchy.trace if name == "hierarchy" else None
            write_case(dest, row, sha256(frozen))
            print(name, ref["id"], row["detection"], flush=True)


if __name__ == "__main__":
    from sonar_vision.ocr.paddle import PaddleEngine

    main()
