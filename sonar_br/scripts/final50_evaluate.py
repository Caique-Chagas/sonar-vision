"""Frozen paired evaluation of new photos; no training, tuning or temporal claims."""

import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2

from sonar_vision.br.evaluation import EvaluationProtocol, evaluate_model
from sonar_vision.br.observations import (
    RecordedDetector,
    RecordedOCR,
    code_visible_in_ocr,
    observation_routes,
)
from sonar_vision.core.config import Settings
from sonar_vision.core.contracts import Box
from sonar_vision.core.logging import configure_logging
from sonar_vision.detection.yolo import YoloSignDetector
from sonar_vision.evaluation.final_photos import read_case, validate_final_reference, write_case
from sonar_vision.evaluation.prospective import match_panels
from sonar_vision.ocr.paddle import PaddleEngine
from sonar_vision.processing.pipeline import Context, FrameProcessor
from sonar_vision.processing.roi import Frame, extract_roi
from sonar_vision.tracking.iou import iou
from sonar_vision.training.dataset import sha256

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "sonar_br/datasets/manifests/final50.review.json"
PROTOCOL = ROOT / "sonar_br/configs/final50-protocol.json"
CACHE = ROOT / "sonar_br/artifacts/final50/cases"
REPORT = ROOT / "sonar_br/reports/final50-pipeline.json"
LOGGER = logging.getLogger("sonar_vision.final50")


def boxes_for(ref: dict[str, Any]) -> list[Box]:
    return [
        Box(**dict(zip(["x1", "y1", "x2", "y2"], b["xyxy"], strict=True))) for b in ref["boxes"]
    ]


def predictions_for(
    detector: RecordedDetector, ocr: RecordedOCR, processor: FrameProcessor, ref: dict[str, Any]
) -> list[dict[str, Any]]:
    truths = boxes_for(ref)
    observations = observation_routes(processor)
    predictions = []
    for detection, call, obs in zip(detector.detections, ocr.calls, observations, strict=True):
        text = " | ".join(r["text"] for r in call["readings"])
        normalized = obs["normalization"]
        route = (
            normalized["candidates"][0]
            if (len(normalized["candidates"]) == 1 and not normalized["ambiguous"])
            else None
        )
        predictions.append(
            {
                "box": detection.box.model_dump(),
                "confidence": detection.confidence,
                "reference_ious": [iou(detection.box, box) for box in truths],
                **call,
                "observation": obs,
                "frame_code": route,
                "raw_code_exact": code_visible_in_ocr(text, ref["expected_code"])
                if ref.get("expected_code")
                else None,
            }
        )
    return predictions


def evaluate_case(
    image: Frame, ref: dict[str, Any], backend: YoloSignDetector, engine: PaddleEngine
) -> dict[str, Any]:
    detector, ocr = RecordedDetector(backend), RecordedOCR(engine)
    processor = FrameProcessor(
        detector,
        ocr,
        Context(ref.get("expected_code") or "6030-10", "final50-static", ref["id"], "local_video"),
        Settings(_env_file=None, roi_padding=0),
    )
    start = time.perf_counter()
    _, events = processor.process(image, 0, 0.0)
    elapsed = (time.perf_counter() - start) * 1000
    if any(event.status == "CONFIRMED" for event in events):
        raise ValueError("Single photo must never confirm a line")
    predictions = predictions_for(detector, ocr, processor, ref)
    ignored = [
        Box(**dict(zip(["x1", "y1", "x2", "y2"], b, strict=True)))
        for b in ref.get("ignore_regions", [])
    ]
    return {
        "id": ref["id"],
        "city": ref["city"],
        "expected_code": ref.get("expected_code"),
        "code_location": ref.get("code_location"),
        "difficulty": ref["difficulty"],
        "actual_source": "licensed real photograph, not video",
        "reference_boxes": [box.model_dump() for box in boxes_for(ref)],
        "predictions": predictions,
        "detection": match_panels(predictions, boxes_for(ref), ignored),
        "final_code": None,
        "final_status": "UNCERTAIN",
        "events": [event.model_dump(mode="json") for event in events],
        "pipeline_wall_ms": elapsed,
        "detector_wall_ms": detector.wall_ms,
        "ocr_wall_ms": sum(call["wall_ms"] for call in ocr.calls),
    }


def freeze_binding() -> tuple[dict[str, Any], str]:
    value = {
        "protocol_sha256": sha256(PROTOCOL),
        "reference_sha256": sha256(REFERENCE),
        "runner_sha256": sha256(Path(__file__)),
        "validator_sha256": sha256(ROOT / "src/sonar_vision/evaluation/final_photos.py"),
        "environment_sha256": sha256(ROOT / "sonar_br/reports/final50-environment.json"),
        "AP_audit_sha256": sha256(
            ROOT / "sonar_br/datasets/prepared/final50-ap/prepared/audit.json"
        ),
        "order": "alternating Bangkok-BR / BR-Bangkok by reference index",
    }
    path = ROOT / "sonar_br/configs/final50-run-binding.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("Existing attempt differs from frozen inputs; preserve it")
    if not path.exists():
        path.write_text(json.dumps(value, indent=2) + "\n")
    return value, sha256(path)


def reference_ocr(image: Frame, ref: dict[str, Any], engine: PaddleEngine) -> dict[str, Any]:
    readings = []
    for box in boxes_for(ref):
        readings.extend(r.model_dump() for r in engine.read(extract_roi(image, box)))
    return {
        "id": ref["id"],
        "expected_code": ref["expected_code"],
        "code_location": ref["code_location"],
        "readings": readings,
        "exact_code": code_visible_in_ocr(
            " | ".join(r["text"] for r in readings), ref["expected_code"]
        ),
        "scope": "manual main-display ROI, not detector success; auxiliary plates not cropped",
    }


def evaluate_AP(spec: dict[str, Any]) -> None:
    report = ROOT / "sonar_br/reports/final50-AP.json"
    if report.exists():
        return
    data = ROOT / "sonar_br/datasets/prepared/final50-ap/prepared/data.yaml"
    protocol = EvaluationProtocol.model_validate(spec["AP_metrics"])
    cells = {}
    for name, weights in spec["weights"].items():
        output = ROOT / "sonar_br/artifacts/final50/AP" / name
        completed = output / "detector.json"
        cells[name] = (
            json.loads(completed.read_text())
            if completed.exists()
            else evaluate_model(ROOT / weights["path"], data, output, protocol)
        )
    report.write_text(
        json.dumps(
            {
                "cells": cells,
                "scope": "same39AP-eligible photos only; all50 use operating pipeline",
                "training_performed": False,
            },
            indent=2,
        )
        + "\n"
    )


def run_paired(
    refs: list[dict[str, Any]], spec: dict[str, Any], engine: PaddleEngine, binding: str
) -> list[dict[str, Any]]:
    detectors = {}
    cold = {name: True for name in spec["weights"]}
    for name, value in spec["weights"].items():
        inference = spec["inference"]
        backend = YoloSignDetector(
            ROOT / value["path"], confidence=inference["conf"], imgsz=inference["imgsz"]
        )
        backend.model.overrides.update(
            **{key: inference[key] for key in ["device", "iou", "rect", "quantize", "batch"]}
        )
        detectors[name] = backend
    rows = []
    for index, ref in enumerate(refs):
        image = cv2.imread(str((REFERENCE.parent / ref["image"]).resolve()))
        if image is None:
            raise ValueError("Frozen photo cannot be decoded")
        names = list(detectors) if index % 2 == 0 else list(reversed(detectors))
        for name in names:
            path = CACHE / f"{name}-{ref['id']}.json"
            row = read_case(path, binding)
            if row is None:
                row = evaluate_case(image, ref, detectors[name], engine)
                row.update(model=name, cold_first_detector_call=cold[name])
                cold[name] = False
                write_case(path, row, binding)
                LOGGER.info("photo_case_completed model=%s image=%s", name, ref["id"])
            rows.append(row)
    return rows


def main() -> None:
    configure_logging()
    prior = set(
        json.loads((ROOT / "artifacts/research/final50-20261115/catalog.json").read_text())[
            "prior_capture_groups"
        ]
    )
    refs = validate_final_reference(REFERENCE, PROTOCOL, ROOT, prior)["records"]
    spec = json.loads(PROTOCOL.read_text())
    value, binding = freeze_binding()
    if REPORT.exists():
        if json.loads(REPORT.read_text())["binding"] != value:
            raise ValueError("Completed report differs from frozen inputs")
        evaluate_AP(spec)
        return
    for name, files in spec["ocr_artifacts"].items():
        for filename, digest in files.items():
            cached = Path.home() / ".paddlex/official_models" / name / filename
            if sha256(cached) != digest:
                raise ValueError("Frozen OCR artifact changed")
    engine = PaddleEngine()
    rows = run_paired(refs, spec, engine, binding)
    manual = []
    for ref in refs:
        if not ref.get("expected_code"):
            continue
        path = CACHE / f"manual-{ref['id']}.json"
        row = read_case(path, binding)
        if row is None:
            image = cv2.imread(str((REFERENCE.parent / ref["image"]).resolve()))
            row = reference_ocr(image, ref, engine)
            write_case(path, row, binding)
        manual.append(row)
    REPORT.write_text(
        json.dumps(
            {
                "binding": value,
                "records": rows,
                "reference_ROI_OCR": manual,
                "temporal_validation": False,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    evaluate_AP(spec)


if __name__ == "__main__":
    main()
