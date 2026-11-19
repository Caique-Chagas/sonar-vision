"""Test observed-polygon rectification on DEV only, preserving original/E3 receipts."""

import json
import time
from pathlib import Path

import cv2
from paddleocr import TextRecognition

from sonar_vision.br.ocr_rectification import RectifiedOCR
from sonar_vision.ocr.paddle import PaddleEngine
from sonar_vision.training.dataset import sha256
from sonar_vision.validation.normalizer import LineNormalizer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "sonar_br/artifacts/v2-diagnosis/ocr-dev"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ref = ROOT / "sonar_br/datasets/manifests/br-v2.review.json"
    binding = {
        str(p.relative_to(ROOT)): sha256(p)
        for p in [ref, Path(__file__), ROOT / "src/sonar_vision/br/ocr_rectification.py"]
    }
    saved = OUT / "binding.json"
    if saved.exists() and json.loads(saved.read_text()) != binding:
        raise ValueError("Preserve executed OCR experiment")
    saved.write_text(json.dumps(binding, indent=2) + "\n")
    rec = TextRecognition(
        model_name="en_PP-OCRv5_mobile_rec",
        model_dir=str(Path.home() / ".paddlex/official_models/en_PP-OCRv5_mobile_rec"),
        device="cpu",
        enable_mkldnn=False,
        cpu_threads=2,
    )
    engine = RectifiedOCR(PaddleEngine(), rec)
    for r in json.loads(ref.read_text())["records"]:
        if r["split"] == "test" or r.get("index") is None or not r.get("expected_code"):
            continue
        dest = OUT / (r["id"] + ".json")
        if dest.exists():
            continue
        im = cv2.imread(r["image"])
        x1, y1, x2, y2 = [round(v) for v in r["boxes"][0]["xyxy"]]
        start = time.perf_counter()
        readings = engine.read(im[y1:y2, x1:x2].copy())
        elapsed = (time.perf_counter() - start) * 1000
        normalizer = LineNormalizer({r["expected_code"]})
        codes = sorted(
            {
                code
                for item in readings
                if item.confidence >= 0.7
                for code in [normalizer.normalize(item.text).route]
                if code
            }
        )
        row = {
            "id": r["id"],
            "expected_code": r["expected_code"],
            "split": r["split"],
            "scope": "manual_ROI_DEV_not_end_to_end",
            "readings": [v.model_dump() for v in readings],
            "candidates": codes,
            "exact": r["expected_code"] in codes,
            "wrong": [c for c in codes if c != r["expected_code"]],
            "ms": elapsed,
        }
        dest.write_text(json.dumps(row, indent=2) + "\n")
        print(r["id"], r["expected_code"], codes, flush=True)


if __name__ == "__main__":
    main()
