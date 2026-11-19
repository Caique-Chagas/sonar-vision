"""Isolate OCR on verified DEV panel boxes; no final-test photographs are read."""

import json
import logging
import time
from pathlib import Path

import cv2
from paddleocr import TextRecognition

from sonar_vision.br.code_region_ocr import LeftCodeOCR
from sonar_vision.br.ocr_optimization import ExperimentalOCR, OCRExperiment
from sonar_vision.ocr.paddle import PaddleEngine
from sonar_vision.training.dataset import sha256
from sonar_vision.validation.normalizer import LineNormalizer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "sonar_br/artifacts/v2-ocr-dev"
REF = ROOT / "sonar_br/datasets/manifests/br-v2.review.json"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    OUT.mkdir(exist_ok=True)
    refs = [
        r
        for r in json.loads(REF.read_text())["records"]
        if r.get("index") is not None and r["split"] != "test" and r.get("expected_code")
    ]
    binding = {
        "reference": sha256(REF),
        "runner": sha256(Path(__file__)),
        "left_code": sha256(ROOT / "src/sonar_vision/br/code_region_ocr.py"),
    }
    file = OUT / "binding.json"
    if file.exists() and json.loads(file.read_text()) != binding:
        raise ValueError("Preserve executed DEV inputs")
    file.write_text(json.dumps(binding, indent=2) + "\n")
    engine = PaddleEngine()
    cached = Path.home() / ".paddlex/official_models/en_PP-OCRv5_mobile_rec"
    recognizer = TextRecognition(
        model_name="en_PP-OCRv5_mobile_rec",
        model_dir=str(cached),
        device="cpu",
        enable_mkldnn=False,
        cpu_threads=2,
    )
    variants = {
        "OCR0": engine,
        "OCR1_E3": ExperimentalOCR(engine, OCRExperiment(spatial="recognize"), recognizer),
        "OCR2_left": LeftCodeOCR(engine),
        "OCR3_scale2": ExperimentalOCR(engine, OCRExperiment(preprocessing="scale2")),
    }
    for ref in refs:
        image = cv2.imread(ref["image"])
        x1, y1, x2, y2 = [round(v) for v in ref["boxes"][0]["xyxy"]]
        roi = image[y1:y2, x1:x2].copy()
        for name, ocr in variants.items():
            path = OUT / f"{name}-{ref['id']}.json"
            if path.exists():
                continue
            start = time.perf_counter()
            readings = ocr.read(roi)
            elapsed = (time.perf_counter() - start) * 1000
            # Real catalog is not supplied. Requested line alone is shared with baseline.
            normalizer = LineNormalizer({ref["expected_code"]})
            codes = sorted(
                {
                    v
                    for r in readings
                    if r.confidence >= 0.7
                    for v in [normalizer.normalize(r.text).route]
                    if v
                }
            )
            row = {
                "id": ref["id"],
                "split": ref["split"],
                "expected_code": ref["expected_code"],
                "variant": name,
                "readings": [r.model_dump() for r in readings],
                "candidates": codes,
                "exact": ref["expected_code"] in codes,
                "wrong": [c for c in codes if c != ref["expected_code"]],
                "ms": elapsed,
                "scope": "manual_ROI_DEV_not_end_to_end",
            }
            path.write_text(json.dumps(row, indent=2) + "\n")
            print(name, ref["id"], ref["expected_code"], codes, flush=True)


if __name__ == "__main__":
    main()
