"""Bounded opt-in experiments, saved case-by-case; no YOLO training or default changes."""

import argparse
import json
import logging
import resource
import time
from pathlib import Path
from typing import Any

import cv2

from sonar_br.scripts.final50_evaluate import boxes_for, predictions_for
from sonar_vision.br.observations import RecordedDetector, RecordedOCR
from sonar_vision.br.ocr_optimization import (
    BrazilianSeparatorNormalizer,
    ExperimentalOCR,
    OCRExperiment,
)
from sonar_vision.core.config import Settings
from sonar_vision.core.contracts import Box, Detection, OCRReading
from sonar_vision.detection.yolo import YoloSignDetector
from sonar_vision.evaluation.final_photos import read_case, write_case
from sonar_vision.evaluation.prospective import match_panels
from sonar_vision.ocr.paddle import PaddleEngine
from sonar_vision.processing.pipeline import Context, FrameProcessor
from sonar_vision.processing.roi import Frame
from sonar_vision.training.dataset import sha256

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "sonar_br/datasets/manifests/ocr-opt-v1.review.json"
FOLDER = ROOT / "sonar_br/ocr_experiments"
CACHE = ROOT / "sonar_br/artifacts/ocr-opt-v1"
LOGGER = logging.getLogger("sonar_vision.ocr_experiments")


class SavedDetector:
    def __init__(self, detections: list[Detection], version: str):
        self.detections, self.model_version = detections, version

    def detect(self, frame: Frame) -> list[Detection]:
        return self.detections


class SavedOCR:
    def __init__(self, calls: list[dict[str, Any]]):
        self.calls = iter(calls)

    def read(self, roi: Frame) -> list[OCRReading]:
        return [OCRReading.model_validate(r) for r in next(self.calls)["readings"]]


def case(
    image: Frame, ref: dict[str, Any], backend: Any, engine: Any, separators: bool
) -> dict[str, Any]:
    detector, ocr = RecordedDetector(backend), RecordedOCR(engine)
    processor = FrameProcessor(
        detector,
        ocr,
        Context(ref.get("expected_code") or "6030-10", "ocr-static", ref["id"], "local_video"),
        Settings(_env_file=None),
    )
    if separators:
        processor.normalizer = BrazilianSeparatorNormalizer({processor.context.requested})
    start = time.perf_counter()
    _, events = processor.process(image, 0, 0.0)
    elapsed = (time.perf_counter() - start) * 1000
    predictions = predictions_for(detector, ocr, processor, ref)
    if any(e.status != "UNCERTAIN" for e in events):
        raise ValueError("A photograph must not confirm a line")
    ignored = [
        Box(**dict(zip(["x1", "y1", "x2", "y2"], b, strict=True)))
        for b in ref.get("ignore_regions", [])
    ]
    return {
        "id": ref["id"],
        "split": ref["split"],
        "scope": ref["scope"],
        "expected_code": ref.get("expected_code"),
        "negative": ref["confirmed_negative"],
        "reference_boxes": [b.model_dump() for b in boxes_for(ref)],
        "predictions": predictions,
        "detection": match_panels(predictions, boxes_for(ref), ignored),
        "events": [e.model_dump(mode="json") for e in events],
        "final_code": None,
        "final_status": "UNCERTAIN",
        "raw_exact": any(p["raw_code_exact"] for p in predictions)
        if ref.get("expected_code")
        else None,
        "eligible_exact": any(
            p["frame_code"] == ref["expected_code"]
            and p["observation"]["ocr_confidence"] >= 0.7
            and p["confidence"] >= 0.35
            and p["reference_ious"][0] >= 0.5
            for p in predictions
        )
        if ref.get("expected_code")
        else None,
        "wrong_eligible_codes": [
            p["frame_code"]
            for p in predictions
            if p["frame_code"]
            and p["observation"]["ocr_confidence"] >= 0.7
            and ref.get("expected_code")
            and p["frame_code"] != ref["expected_code"]
            and p["reference_ious"][0] >= 0.5
        ],
        "negative_eligible_codes": [
            p["frame_code"]
            for p in predictions
            if ref["confirmed_negative"]
            and p["frame_code"]
            and p["observation"]["ocr_confidence"] >= 0.7
        ],
        "pipeline_ms": elapsed,
        "detector_ms": detector.wall_ms,
        "OCR_ms": sum(c["wall_ms"] for c in ocr.calls),
        "OCR_trace": getattr(engine, "trace", None),
    }


def verify_inputs(spec: dict[str, Any], refs: list[dict[str, Any]]) -> None:
    if sha256(REF) != spec["reference_sha256"]:
        raise ValueError("Frozen reference changed")
    for name, digest in spec["protected"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Protected weight/PDF/video changed")
    for r in refs:
        if not r["rights_verified"] or not r["rights_reviewed"]:
            raise ValueError("Rights review missing")
        if sha256((REF.parent / r["image"]).resolve()) != r["image_sha256"]:
            raise ValueError("Frozen pixels changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "holdout"], required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    reference = json.loads(REF.read_text())
    spec = json.loads((FOLDER / "design.json").read_text())
    refs = [r for r in reference["records"] if r["split"] == args.split]
    verify_inputs(spec, refs)
    if args.split == "holdout":
        candidate_path = FOLDER / "candidate-frozen.json"
        candidate = json.loads(candidate_path.read_text())
        if candidate["source_sha256"] != sha256(ROOT / "src/sonar_vision/br/ocr_optimization.py"):
            raise ValueError("Candidate code changed after freeze")
        variants = {"E0": spec["variants"]["E0"], "candidate": candidate["configuration"]}
    else:
        variants = spec["variants"]
    binding_value = {
        "reference": sha256(REF),
        "runner": sha256(Path(__file__)),
        "implementation": sha256(ROOT / "src/sonar_vision/br/ocr_optimization.py"),
        "design": sha256(FOLDER / "design.json"),
        "variants": variants,
    }
    binding_path = CACHE / f"{args.split}-binding.json"
    CACHE.mkdir(parents=True, exist_ok=True)
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding_value:
        raise ValueError("Existing experiment must be preserved; binding changed")
    binding_path.write_text(json.dumps(binding_value, indent=2) + "\n")
    digest = sha256(binding_path)
    backend = YoloSignDetector(
        ROOT / "sonar_br/artifacts/train-v1/run/weights/best.pt", confidence=0.35, imgsz=640
    )
    backend.model.overrides.update(device="cpu", iou=0.5, rect=True, quantize=None, batch=1)
    engine = PaddleEngine()
    recognizer = None
    rows = []
    for index, ref in enumerate(refs):
        image = cv2.imread(str((REF.parent / ref["image"]).resolve()))
        if image is None:
            raise ValueError("Frozen image cannot be decoded")
        baseline = None
        order = list(variants)
        if args.split == "holdout" and index % 2:
            order.reverse()
        for name in order:
            config = variants[name]
            path = CACHE / args.split / f"{name}-{ref['id']}.json"
            row = read_case(path, digest)
            if row is None:
                detector: Any = backend
                ocr: Any = engine
                if args.split == "dev" and baseline is not None:
                    detector = SavedDetector(
                        [
                            Detection(
                                box=Box(**p["box"]),
                                confidence=p["confidence"],
                                label="destination_sign",
                            )
                            for p in baseline["predictions"]
                        ],
                        backend.model_version,
                    )
                if name == "E4":
                    if baseline is None:
                        raise ValueError("Normalization experiment needs recorded baseline")
                    ocr = SavedOCR(baseline["predictions"])
                elif config["OCR"]:
                    if config["OCR"].get("spatial") == "recognize" and recognizer is None:
                        from paddleocr import TextRecognition

                        cached = Path.home() / ".paddlex/official_models/en_PP-OCRv5_mobile_rec"
                        if not cached.is_dir():
                            raise ValueError(
                                "Required cached recognizer absent; no implicit model upgrade"
                            )
                        recognizer = TextRecognition(
                            model_name="en_PP-OCRv5_mobile_rec", model_dir=str(cached), device="cpu"
                        )
                    ocr = ExperimentalOCR(engine, OCRExperiment(**config["OCR"]), recognizer)
                row = case(image, ref, detector, ocr, config["separators"])
                row.update(
                    variant=name,
                    first_photo=index == 0,
                    cached_detector=args.split == "dev" and baseline is not None,
                    cached_OCR=name == "E4",
                    max_resident_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                )
                write_case(path, row, digest)
                LOGGER.info("completed %s %s %s", args.split, name, ref["id"])
            if name == "E0":
                baseline = row
            rows.append(row)
    report = {
        "binding": binding_value,
        "records": rows,
        "temporal_validation": False,
        "new_YOLO_training": False,
        "baseline_normalizer_changed": False,
    }
    destination = ROOT / "sonar_br/reports" / f"ocr-opt-{args.split}.json"
    if destination.exists() and json.loads(destination.read_text()) != report:
        raise ValueError("Completed report differs; do not overwrite")
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
