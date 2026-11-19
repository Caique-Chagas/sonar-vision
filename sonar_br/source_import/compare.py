"""Frozen OCR diagnostics on five known DEV codes, with an explicitly empty test."""

import importlib.metadata
import json
import re
import time
from pathlib import Path
from statistics import median
from typing import Any

import cv2

from sonar_br.source_import.build import MANIFESTS, OUTPUT, ROOT
from sonar_vision.core.contracts import Box
from sonar_vision.evaluation.final_photos import read_case, write_case
from sonar_vision.processing.roi import extract_roi
from sonar_vision.training.dataset import sha256
from sonar_vision.training.recognition_dataset import verify_recognition
from sonar_vision.validation.normalizer import FULL_CODE

OUT = ROOT / "sonar_br/artifacts/source-import-20261117"


def select_code(readings: list[dict[str, Any]]) -> str | None:
    """Whole readings only; prefer complete suffix codes, abstain on conflicts."""
    full, regional = set(), set()
    for reading in readings:
        if reading["confidence"] < 0.7:
            continue
        text = reading["text"].strip()
        if FULL_CODE.fullmatch(text):
            full.add(text)
        elif re.fullmatch(r"[A-Z0-9]{2,6}", text) and any(c.isdigit() for c in text):
            regional.add(text)
    candidates = full or regional
    return next(iter(candidates)) if len(candidates) == 1 else None


def summarize(rows: list[dict[str, Any]], expected_ids: set[str]) -> dict[str, Any]:
    if {r["id"] for r in rows} != expected_ids or len(rows) != len(expected_ids):
        raise ValueError("All frozen examples required, including errors and no-ROI cases")
    return {
        "n": len(rows),
        "correct": sum(r["exact"] for r in rows),
        "exact_match_DEV": sum(r["exact"] for r in rows) / len(rows) if rows else None,
        "wrong_accepted": sum(
            r["prediction"] is not None and r["prediction"] != r["truth"] for r in rows
        ),
        "abstentions": sum(r["prediction"] is None for r in rows),
        "median_ms": median(r["ms"] for r in rows) if rows else None,
        "independent_test_exact_match": None,
    }


def main() -> None:
    from paddleocr import TextRecognition

    from sonar_vision.br.ocr_optimization import ExperimentalOCR, OCRExperiment
    from sonar_vision.ocr.paddle import PaddleEngine

    audit = verify_recognition(OUTPUT / "ocr")
    refs = audit["records"]
    if audit["counts"]["test"] != 0 or any(r["parent_id"] == "35046766" for r in refs):
        raise ValueError("This protocol contains only known DEV and no reserved source")
    protocol = ROOT / "sonar_br/configs/source-import-ocr-protocol.json"
    spec = json.loads(protocol.read_text())
    if sha256(OUTPUT / "ocr/audit.json") != spec["corpus"]:
        raise ValueError("Frozen corpus changed")
    for path, digest in spec["files"].items():
        if sha256(Path(path)) != digest:
            raise ValueError("Frozen model/source/receipt changed")
    OUT.mkdir(exist_ok=True)
    binding = OUT / "binding.json"
    value = {
        "protocol": sha256(protocol),
        "runner": sha256(Path(__file__)),
        "corpus": spec["corpus"],
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["paddleocr", "paddlepaddle", "numpy", "opencv-python"]
        },
    }
    if binding.exists() and json.loads(binding.read_text()) != value:
        raise ValueError("Preserve prior comparison")
    binding.write_text(json.dumps(value, indent=2) + "\n")
    model5 = Path.home() / ".paddlex/official_models/en_PP-OCRv5_mobile_rec"
    model4 = ROOT / "artifacts/research/source-import-20261117/models/en_PP-OCRv4_mobile_rec_infer"
    original = PaddleEngine()
    recognizers = {
        name: TextRecognition(
            model_name=name, model_dir=str(path), device="cpu", enable_mkldnn=False, cpu_threads=2
        )
        for name, path in [("en_PP-OCRv5_mobile_rec", model5), ("en_PP-OCRv4_mobile_rec", model4)]
    }
    e3 = ExperimentalOCR(
        original, OCRExperiment(spatial="recognize"), recognizers["en_PP-OCRv5_mobile_rec"]
    )
    variants = {"original": original, "E3": e3, **recognizers}
    old_probe = ROOT / "sonar_br/artifacts/generalization-20261116/code-probe"
    full_refs = json.loads((MANIFESTS / "generalization-20261116.review.json").read_text())[
        "records"
    ]
    full_refs += json.loads(
        (MANIFESTS / "final-correction-20261117.local.review.json").read_text()
    )["records"]
    all_rows = {name: [] for name in variants}
    auto = {"original": [], "E3": []}
    for ref in refs:
        id = ref["parent_id"]
        image = cv2.imread(str(OUTPUT / "ocr" / ref["image"]))
        for name, engine in variants.items():
            dest = OUT / "code-crops" / f"{name}-{id}.json"
            row = read_case(dest, sha256(binding))
            if row is None:
                reused = None
                if name == "en_PP-OCRv5_mobile_rec" and (old_probe / (id + ".json")).exists():
                    reused = read_case(
                        old_probe / (id + ".json"), sha256(old_probe / "binding.json")
                    )
                start = time.perf_counter()
                if reused:
                    readings, elapsed = reused["readings"], reused["ms"]
                elif name in recognizers:
                    readings = [
                        {"text": str(r["rec_text"]), "confidence": float(r["rec_score"])}
                        for r in engine.predict(input=image, batch_size=1)
                    ]
                    elapsed = (time.perf_counter() - start) * 1000
                else:
                    readings = [r.model_dump() for r in engine.read(image)]
                    elapsed = (time.perf_counter() - start) * 1000
                prediction = select_code(readings)
                row = {
                    "id": id,
                    "truth": ref["transcript"],
                    "prediction": prediction,
                    "readings": readings,
                    "exact": prediction == ref["transcript"],
                    "ms": elapsed,
                    "manual_reference_code_crop": True,
                    "reused_previous_V5_receipt": bool(reused),
                    "scope": "known_DEV_never_independent_test",
                }
                write_case(dest, row, sha256(binding))
                print(name, id, row["prediction"], row["exact"], flush=True)
            all_rows[name].append(row)
        full = next(r for r in full_refs if r["id"] == id)
        full_image = cv2.imread(full["image"])
        cached_path = ROOT / f"sonar_br/artifacts/generalization-20261116/cases/V1-{id}.json"
        if id == "170315291":
            cached_path = ROOT / "sonar_br/artifacts/final-correction-20261117/local/V1.json"
        saved = json.loads(cached_path.read_text())
        for name, engine in [("original", original), ("E3", e3)]:
            dest = OUT / "predicted-panels" / f"{name}-{id}.json"
            row = read_case(dest, sha256(binding))
            if row is None:
                start = time.perf_counter()
                reads = []
                support = []
                for pred in saved["predictions"]:
                    if name == "E3":
                        values = pred["readings"]
                    else:
                        values = [
                            r.model_dump()
                            for r in engine.read(extract_roi(full_image, Box(**pred["box"])))
                        ]
                    reads.extend(values)
                    support.append(
                        {
                            "box": pred["box"],
                            "IoU": pred["reference_ious"],
                            "YOLO_confidence": pred["confidence"],
                            "readings": values,
                        }
                    )
                prediction = select_code(reads)
                geometry_ok = any(
                    max(p["IoU"], default=0) >= 0.5 and select_code(p["readings"]) == prediction
                    for p in support
                )
                row = {
                    "id": id,
                    "truth": ref["transcript"],
                    "prediction": prediction,
                    "exact": prediction == ref["transcript"] and geometry_ok,
                    "ms": sum(p.get("wall_ms", 0) for p in saved["predictions"])
                    if name == "E3"
                    else ((time.perf_counter() - start) * 1000 if support else 0),
                    "detector_recorded_ms": saved["detector_ms"],
                    "latency_scope": (
                        "OCR only; E3 timing reused, detector timing stored separately, "
                        "no new end-to-end timing"
                    ),
                    "predicted_ROIs": support,
                    "has_ROI": bool(saved["predictions"]),
                    "IoU_is_evaluation_only": True,
                    "detector_and_E3_cached": True,
                    "status": "UNCERTAIN",
                    "scope": "known_DEV_no_new_detector_or_temporal_validation",
                }
                write_case(dest, row, sha256(binding))
            auto[name].append(row)
    ids = {r["parent_id"] for r in refs}
    result = {
        "code_crop_DEV": {n: summarize(rows, ids) for n, rows in all_rows.items()},
        "V1_predicted_panel_DEV": {n: summarize(rows, ids) for n, rows in auto.items()},
        "independent_test_n": 0,
        "independent_exact_match": None,
        "goal70_independent_demonstrated": False,
        "training": False,
        "no_promotion": True,
        "latency_warning": (
            "Four directV5 receipts and all E3 predicted ROI readings reused; "
            "E3 costs are recorded OCR-only durations; cached lookup times are not "
            "inference latency. Warmup included in new crop timings."
        ),
    }
    (ROOT / "sonar_br/reports/SOURCE_IMPORT_OCR.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
