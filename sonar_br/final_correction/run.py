"""Bounded DEV ablation of native tiles and ROI margin; no default changes."""

import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

import cv2

from sonar_br.scripts.final50_evaluate import boxes_for, predictions_for
from sonar_vision.br.observations import RecordedDetector, RecordedOCR
from sonar_vision.br.ocr_optimization import ExperimentalOCR, OCRExperiment
from sonar_vision.core.config import Settings
from sonar_vision.core.contracts import Box
from sonar_vision.detection.yolo import YoloSignDetector
from sonar_vision.evaluation.final_photos import read_case, write_case
from sonar_vision.evaluation.prospective import match_panels
from sonar_vision.processing.pipeline import Context, FrameProcessor
from sonar_vision.processing.roi import Frame
from sonar_vision.training.dataset import sha256

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "sonar_br/datasets/manifests/generalization-20261116.review.json"
OUT = ROOT / "sonar_br/artifacts/final-correction-20261117"
DESIGN = ROOT / "sonar_br/configs/final-correction-20261117.json"


def eligible(prediction: dict[str, Any]) -> bool:
    return (
        prediction["confidence"] >= 0.35
        and prediction["observation"]["ocr_confidence"] >= 0.7
        and max(prediction["reference_ious"], default=0) >= 0.5
    )


def evaluate(
    image: Frame, ref: dict[str, Any], backend: Any, engine: Any, *, margin: float
) -> dict[str, Any]:
    if ref["id"] == "35046766" or ref["split"] != "development_only_never_new_test":
        raise ValueError("Reserved/test inputs forbidden in DEV ablation")
    detector, ocr = RecordedDetector(backend), RecordedOCR(engine)
    processor = FrameProcessor(
        detector,
        ocr,
        Context(
            ref.get("expected_code") or "6030-10", "correction-static", ref["id"], "local_video"
        ),
        Settings(_env_file=None, roi_padding=margin),
    )
    start = time.perf_counter()
    _, events = processor.process(image, 0, 0.0)
    elapsed = (time.perf_counter() - start) * 1000
    if any(e.status != "UNCERTAIN" for e in events):
        raise ValueError("A single photograph cannot validate temporal consensus")
    predictions = predictions_for(detector, ocr, processor, ref)
    ignored = [Box(x1=b[0], y1=b[1], x2=b[2], y2=b[3]) for b in ref.get("ignore_regions", [])]
    code = ref.get("expected_code")
    return {
        "id": ref["id"],
        "scope": "same_known_DEV_not_independent_test",
        "expected_code": code,
        "reference_boxes": [b.model_dump() for b in boxes_for(ref)],
        "predictions": predictions,
        "detection": match_panels(predictions, boxes_for(ref), ignored),
        "eligible_exact": any(p["frame_code"] == code and eligible(p) for p in predictions)
        if code
        else None,
        "wrong_eligible_codes": [
            p["frame_code"]
            for p in predictions
            if code and p["frame_code"] and p["frame_code"] != code and eligible(p)
        ],
        "final_status": "UNCERTAIN",
        "events": [e.model_dump(mode="json") for e in events],
        "pipeline_ms": elapsed,
        "detector_ms": detector.wall_ms,
        "OCR_ms": sum(c["wall_ms"] for c in ocr.calls),
    }


def main() -> None:
    from paddleocr import TextRecognition

    from sonar_vision.ocr.paddle import PaddleEngine

    spec = json.loads(DESIGN.read_text())
    refs = json.loads(REF.read_text())["records"]
    if sha256(REF) != spec["reference_sha256"]:
        raise ValueError("Frozen DEV reference changed")
    weights = ROOT / spec["weights"]
    if sha256(weights) != spec["weights_sha256"]:
        raise ValueError("Frozen V1 changed")
    OUT.mkdir(parents=True, exist_ok=True)
    value = {
        "design": sha256(DESIGN),
        "runner": sha256(Path(__file__)),
        "reference": sha256(REF),
        "ultralytics": importlib.metadata.version("ultralytics"),
        "paddleocr": importlib.metadata.version("paddleocr"),
        "sources": {p: sha256(ROOT / p) for p in spec["source_files"]},
    }
    binding = OUT / "binding.json"
    if binding.exists() and json.loads(binding.read_text()) != value:
        raise ValueError("Ablation changed; preserve completed receipts")
    binding.write_text(json.dumps(value, indent=2) + "\n")
    model = YoloSignDetector(weights, confidence=0.35, imgsz=640, tile_size=640)
    model.model.overrides.update(device="cpu", iou=0.5, rect=True, quantize=None, batch=1)
    engine = PaddleEngine()
    recognizer = TextRecognition(
        model_name="en_PP-OCRv5_mobile_rec",
        model_dir=str(Path.home() / ".paddlex/official_models/en_PP-OCRv5_mobile_rec"),
        device="cpu",
        enable_mkldnn=False,
        cpu_threads=2,
    )
    e3 = ExperimentalOCR(engine, OCRExperiment(spatial="recognize"), recognizer)
    for index, ref in enumerate(refs):
        if sha256(Path(ref["image"])) != ref["image_sha256"]:
            raise ValueError("DEV pixels changed")
        image = cv2.imread(ref["image"])
        names = list(spec["variants"])
        if index % 2:
            names.reverse()
        for name in names:
            dest = OUT / "cases" / f"{name}-{ref['id']}.json"
            if read_case(dest, sha256(binding)) is not None:
                continue
            row = evaluate(image, ref, model, e3, margin=spec["variants"][name]["ROI_margin"])
            row["variant"] = name
            write_case(dest, row, sha256(binding))
            print(name, ref["id"], row["detection"], row["eligible_exact"], flush=True)


if __name__ == "__main__":
    main()
