"""Frozen four-way comparison: new detection test and separate known code benchmark."""

import importlib.metadata
import json
import logging
import platform
from pathlib import Path
from typing import Any

import cv2
from paddleocr import TextRecognition

from sonar_br.ocr_experiments.run import case
from sonar_vision.br.evaluation import EvaluationProtocol, evaluate_model
from sonar_vision.br.ocr_optimization import ExperimentalOCR, OCRExperiment
from sonar_vision.core.config import Settings
from sonar_vision.detection.yolo import YoloSignDetector
from sonar_vision.evaluation.final_photos import read_case, write_case
from sonar_vision.ocr.paddle import PaddleEngine
from sonar_vision.training.dataset import sha256

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "sonar_br/datasets/manifests/br-v2.review.json"
OLD_REF = ROOT / "sonar_br/datasets/manifests/ocr-opt-v1.review.json"
FROZEN = ROOT / "sonar_br/configs/v2-comparison-frozen.json"
OUT = ROOT / "sonar_br/artifacts/v2-evaluation"


def binding() -> dict[str, Any]:
    protected = json.loads((ROOT / "sonar_br/ocr_experiments/candidate-frozen.json").read_text())[
        "protected"
    ]
    for name, digest in protected.items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Protected artifacts changed")
    return {
        "status": "frozen_before_any_V2_final_test_inference",
        "models": {
            "v1": "sonar_br/artifacts/train-v1/run/weights/best.pt",
            "v2": "sonar_br/artifacts/train-v2/run/weights/best.pt",
        },
        "weights_sha256": {
            v: sha256(ROOT / v)
            for v in [
                "sonar_br/artifacts/train-v1/run/weights/best.pt",
                "sonar_br/artifacts/train-v2/run/weights/best.pt",
            ]
        },
        "protected": protected,
        "reference_sha256": sha256(REF),
        "known_benchmark_sha256": sha256(OLD_REF),
        "source_hashes": {
            p: sha256(ROOT / p)
            for p in [
                "sonar_br/v2/evaluate.py",
                "sonar_br/ocr_experiments/run.py",
                "src/sonar_vision/br/ocr_optimization.py",
                "src/sonar_vision/br/code_region_ocr.py",
                "src/sonar_vision/validation/normalizer.py",
                "src/sonar_vision/processing/pipeline.py",
                "src/sonar_vision/br/lineage.py",
            ]
        },
        "detector_protocol": EvaluationProtocol().model_dump(),
        "settings": Settings(_env_file=None).model_dump(mode="json"),
        "OCRv2": {
            "configuration": {"spatial": "recognize"},
            "decision": (
                "Retain existing E3 experimental control; left ROI/scale did not improve DEV. "
                "No newly trained recognizer or new OCR superiority claim."
            ),
        },
        "OCR0": "PP-OCRv5_mobile_det + en_PP-OCRv5_mobile_rec, unmodified PaddleEngine CPU",
        "E3": OCRExperiment(spatial="recognize").__dict__,
        "OCR_artifacts": json.loads(
            (ROOT / "sonar_br/ocr_experiments/candidate-frozen.json").read_text()
        )["OCR_artifacts"],
        "variants": {
            "A": ["v1", "original"],
            "B": ["v1", "E3"],
            "C": ["v2", "original"],
            "D": ["v2", "E3"],
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in [
                "ultralytics",
                "torch",
                "paddleocr",
                "paddlepaddle",
                "opencv-python",
                "numpy",
            ]
        },
        "hardware": {
            "machine": platform.machine(),
            "processor": "Apple M1",
            "RAM_GiB": 8,
            "python": platform.python_version(),
            "inference": "CPU; training MPS",
        },
        "metrics_policy": (
            "Primary new test: detector/text, full-code accuracy undefined n=0. "
            "Secondary known OCR14: reused diagnostic regression, not new independent "
            "validation. No promotion based on it."
        ),
        "order": "A-B-C-D on even images, reversed on odd images; no cached detector timings",
        "roi_padding": 0,
        "temporal": (
            "Single photos only; final status UNCERTAIN or no event; no multi-frame claims"
        ),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    spec = binding()
    if FROZEN.exists() and json.loads(FROZEN.read_text()) != spec:
        raise ValueError("Frozen candidate changed; preserve final test")
    if not FROZEN.exists():
        FROZEN.write_text(json.dumps(spec, indent=2) + "\n")
    OUT.mkdir(exist_ok=True)
    digest = sha256(FROZEN)
    backends = {}
    for name, weights in spec["models"].items():
        backend = YoloSignDetector(ROOT / weights, confidence=0.35, imgsz=640)
        backend.model.overrides.update(
            device="cpu", iou=0.5, rect=True, quantize=None, batch=1, max_det=16
        )
        backends[name] = backend
    engine = PaddleEngine()
    cached = Path.home() / ".paddlex/official_models/en_PP-OCRv5_mobile_rec"
    rec = TextRecognition(
        model_name="en_PP-OCRv5_mobile_rec",
        model_dir=str(cached),
        device="cpu",
        enable_mkldnn=False,
        cpu_threads=2,
    )
    e3 = ExperimentalOCR(engine, OCRExperiment(spatial="recognize"), rec)
    new = [
        r | {"scope": "new_independent_detection_test"}
        for r in json.loads(REF.read_text())["records"]
        if r["split"] == "test"
    ]
    old = [
        r | {"scope": "known_historical_code_benchmark_not_new_test"}
        for r in json.loads(OLD_REF.read_text())["records"]
        if r["split"] == "holdout"
    ]
    for scope, refs in [("new_test", new), ("known_benchmark", old)]:
        for index, ref in enumerate(refs):
            image_path = Path(ref["image"])
            if not image_path.is_absolute():
                image_path = (REF.parent / image_path).resolve()
            if sha256(image_path) != ref["image_sha256"]:
                raise ValueError("Reference pixels changed")
            image = cv2.imread(str(image_path))
            order = list(spec["variants"])
            if index % 2:
                order.reverse()
            for name in order:
                model, ocr = spec["variants"][name]
                path = OUT / scope / f"{name}-{ref['id']}.json"
                row = read_case(path, digest)
                if row is None:
                    row = case(
                        image, ref, backends[model], engine if ocr == "original" else e3, False
                    )
                    row["variant"] = name
                    row["reference_visible_base_code"] = ref.get("visible_base_code")
                    row["full_code_denominator_eligible"] = bool(ref.get("expected_code"))
                    write_case(path, row, digest)
                print(scope, name, ref["id"], row["eligible_exact"], row["detection"], flush=True)
    data = ROOT / "sonar_br/datasets/prepared/br-v2/prepared/data.yaml"
    for name, weights in spec["models"].items():
        folder = OUT / f"AP-{name}"
        if not (folder / "detector.json").exists():
            evaluate_model(ROOT / weights, data, folder, EvaluationProtocol())


if __name__ == "__main__":
    main()
