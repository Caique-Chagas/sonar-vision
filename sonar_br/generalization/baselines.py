"""Frozen V1/V2 diagnostics on new DEV data; reserved209P is never an input."""

import importlib.metadata
import json
import time
from pathlib import Path

import cv2
from paddleocr import TextRecognition

from sonar_br.ocr_experiments.run import case
from sonar_vision.br.ocr_optimization import ExperimentalOCR, OCRExperiment
from sonar_vision.core.contracts import Box
from sonar_vision.detection.yolo import YoloSignDetector
from sonar_vision.evaluation.final_photos import read_case, write_case
from sonar_vision.ocr.paddle import PaddleEngine
from sonar_vision.processing.roi import extract_roi
from sonar_vision.training.dataset import sha256
from sonar_vision.validation.normalizer import LineNormalizer

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "sonar_br/datasets/manifests/generalization-20261116.review.json"
OUT = ROOT / "sonar_br/artifacts/generalization-20261116"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    refs = json.loads(REF.read_text())["records"]
    if any(r["id"] == "35046766" or r["split"] != "development_only_never_new_test" for r in refs):
        raise ValueError("Reserved/test images cannot enter DEV diagnostics")
    weights = {
        "V1": ROOT / "sonar_br/artifacts/train-v1/run/weights/best.pt",
        "V2": ROOT / "sonar_br/artifacts/train-v2/run/weights/best.pt",
    }
    design = {
        "reference": sha256(REF),
        "runner": sha256(Path(__file__)),
        "weights": {str(p.relative_to(ROOT)): sha256(p) for p in weights.values()},
        "OCR_source": sha256(ROOT / "src/sonar_vision/br/ocr_optimization.py"),
        "confidence": 0.35,
        "imgsz": 640,
        "NMS": 0.5,
        "ROI_margin": 0,
        "OCR": "E3 unchanged",
        "paddleocr": importlib.metadata.version("paddleocr"),
        "scope": "new DEV only, not independent test; no training or promotion",
        "reference_OCR": "manual module boxes, original and E3 independently; no GT given to OCR",
        "AP": "not computed: partial background annotations prevent exhaustive AP",
        "order": "V1-V2 on even photos, reversed on odd",
    }
    binding = OUT / "binding.json"
    if binding.exists() and json.loads(binding.read_text()) != design:
        raise ValueError("Executed diagnostic inputs changed; preserve receipts")
    binding.write_text(json.dumps(design, indent=2) + "\n")
    models = {}
    for name, path in weights.items():
        backend = YoloSignDetector(path, confidence=0.35, imgsz=640)
        backend.model.overrides.update(device="cpu", iou=0.5, rect=True, quantize=None, batch=1)
        models[name] = backend
    engine = PaddleEngine()
    rec = TextRecognition(
        model_name="en_PP-OCRv5_mobile_rec",
        model_dir=str(Path.home() / ".paddlex/official_models/en_PP-OCRv5_mobile_rec"),
        device="cpu",
        enable_mkldnn=False,
        cpu_threads=2,
    )
    e3 = ExperimentalOCR(engine, OCRExperiment(spatial="recognize"), rec)
    for index, ref in enumerate(refs):
        image = cv2.imread(ref["image"])
        if sha256(Path(ref["image"])) != ref["image_sha256"]:
            raise ValueError("Pixels changed")
        for name in ["V1", "V2"] if index % 2 == 0 else ["V2", "V1"]:
            dest = OUT / "cases" / f"{name}-{ref['id']}.json"
            if read_case(dest, sha256(binding)) is not None:
                continue
            row = case(image, ref, models[name], e3, False)
            row["variant"] = name + "+E3"
            write_case(dest, row, sha256(binding))
            print(name, ref["id"], row["detection"], row["eligible_exact"], flush=True)
        if not ref.get("expected_code"):
            continue
        for name, ocr in [("original", engine), ("E3", e3)]:
            dest = OUT / "reference-ocr" / f"{name}-{ref['id']}.json"
            if read_case(dest, sha256(binding)) is not None:
                continue
            roi = extract_roi(
                image,
                Box(**dict(zip(["x1", "y1", "x2", "y2"], ref["boxes"][0]["xyxy"], strict=True))),
            )
            start = time.perf_counter()
            readings = ocr.read(roi)
            ms = (time.perf_counter() - start) * 1000
            normalizer = LineNormalizer({ref["expected_code"]})
            codes = sorted(
                {
                    c
                    for r in readings
                    if r.confidence >= 0.7
                    for c in [normalizer.normalize(r.text).route]
                    if c
                }
            )
            row = {
                "id": ref["id"],
                "expected_code": ref["expected_code"],
                "scope": "manual_ROI_DEV_not_pipeline",
                "readings": [r.model_dump() for r in readings],
                "candidates": codes,
                "exact": ref["expected_code"] in codes,
                "wrong": [c for c in codes if c != ref["expected_code"]],
                "ms": ms,
            }
            write_case(dest, row, sha256(binding))


if __name__ == "__main__":
    main()
