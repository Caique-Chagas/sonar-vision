"""One curated small SPTrans panel, explicitly DEV/future-train only."""

import json
from pathlib import Path

import cv2
from paddleocr import TextRecognition

from sonar_br.final_correction.run import ROOT, evaluate
from sonar_vision.br.observations import code_visible_in_ocr
from sonar_vision.br.ocr_optimization import ExperimentalOCR, OCRExperiment
from sonar_vision.core.contracts import Box
from sonar_vision.detection.yolo import YoloSignDetector
from sonar_vision.evaluation.final_photos import read_case, write_case
from sonar_vision.ocr.paddle import PaddleEngine
from sonar_vision.processing.roi import extract_roi
from sonar_vision.training.dataset import sha256


def main() -> None:
    reference = ROOT / "sonar_br/datasets/manifests/final-correction-20261117.local.review.json"
    ref = next(r for r in json.loads(reference.read_text())["records"] if r["id"] == "170315291")
    folder = ROOT / "sonar_br/artifacts/final-correction-20261117/local"
    folder.mkdir(parents=True, exist_ok=True)
    variants = {
        "V1": ("train-v1", 0, 0),
        "V2": ("train-v2", 0, 0),
        "V1_T": ("train-v1", 640, 0),
        "V1_TM": ("train-v1", 640, 0.1),
    }
    paths = {
        name: ROOT / f"sonar_br/artifacts/{train}/run/weights/best.pt"
        for name, (train, _, _) in variants.items()
    }
    value = {
        "reference": sha256(reference),
        "runner": sha256(Path(__file__)),
        "evaluate_source": sha256(ROOT / "sonar_br/final_correction/run.py"),
        "variants": variants,
        "weights": {name: sha256(path) for name, path in paths.items()},
        "scope": "One local small panel DEV, not independent test; Capelinha train-group excluded",
        "parameters": "same frozen640/conf.35/NMS.5/E3; no additional tuning",
    }
    # JSON canonicalization preserves the frozen tuple/list representation.
    value = json.loads(json.dumps(value))
    binding = folder / "binding.json"
    if binding.exists() and json.loads(binding.read_text()) != value:
        raise ValueError("Local experiment changed")
    binding.write_text(json.dumps(value, indent=2) + "\n")
    image = cv2.imread(ref["image"])
    if sha256(Path(ref["image"])) != ref["image_sha256"]:
        raise ValueError("Local reference pixels changed")
    engine = PaddleEngine()
    recognizer = TextRecognition(
        model_name="en_PP-OCRv5_mobile_rec",
        model_dir=str(Path.home() / ".paddlex/official_models/en_PP-OCRv5_mobile_rec"),
        device="cpu",
        enable_mkldnn=False,
        cpu_threads=2,
    )
    e3 = ExperimentalOCR(engine, OCRExperiment(spatial="recognize"), recognizer)
    for name, (_, tile, margin) in variants.items():
        dest = folder / (name + ".json")
        if read_case(dest, sha256(binding)) is not None:
            continue
        model = YoloSignDetector(paths[name], confidence=0.35, imgsz=640, tile_size=tile)
        model.model.overrides.update(device="cpu", iou=0.5, rect=True, quantize=None, batch=1)
        row = evaluate(image, ref, model, e3, margin=margin)
        row["variant"] = name
        write_case(dest, row, sha256(binding))
        print(name, row["detection"], row["eligible_exact"], flush=True)
    box = Box(**dict(zip(["x1", "y1", "x2", "y2"], ref["boxes"][0]["xyxy"], strict=True)))
    for name, ocr in [("original", engine), ("E3", e3)]:
        dest = folder / ("manual-" + name + ".json")
        if read_case(dest, sha256(binding)) is not None:
            continue
        readings = ocr.read(extract_roi(image, box))
        row = {
            "id": ref["id"],
            "scope": "manual_ROI_diagnostic_only",
            "expected_code": ref["expected_code"],
            "readings": [r.model_dump() for r in readings],
            "exact": any(
                code_visible_in_ocr(r.text, ref["expected_code"]) and r.confidence >= 0.7
                for r in readings
            ),
        }
        write_case(dest, row, sha256(binding))
        print("manual", name, row["exact"], flush=True)


if __name__ == "__main__":
    main()
