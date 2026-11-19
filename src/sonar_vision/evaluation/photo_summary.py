"""Descriptive static-photo metrics from recorded inference, without model calls."""

from collections import Counter
from statistics import median
from typing import Any

from sonar_vision.br.observations import code_visible_in_ocr


def describe_case(row: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    expected = ref.get("expected_code")
    if row["expected_code"] != expected or row["id"] != ref["id"]:
        raise ValueError("Recorded case disagrees with frozen reference")
    if row["final_code"] is not None or any(
        event["status"] != "UNCERTAIN" for event in row["events"]
    ):
        raise ValueError("Single photographs cannot establish temporal confirmation")
    predictions = row["predictions"]
    raw = (
        any(
            code_visible_in_ocr(" | ".join(r["text"] for r in p["readings"]), expected)
            for p in predictions
        )
        if expected
        else None
    )
    eligible = [
        p
        for p in predictions
        if p["frame_code"] is not None
        and p["confidence"] >= 0.35
        and p["observation"]["ocr_confidence"] >= 0.7
    ]
    exact = any(p["frame_code"] == expected for p in eligible) if expected else None
    joint = (
        any(
            p["frame_code"] == expected and max(p["reference_ious"], default=0) >= 0.5
            for p in eligible
        )
        if expected
        else None
    )
    if not expected:
        cause = "no_reliable_code_truth_not_scored"
    elif ref["code_location"] == "auxiliary_front_placard":
        cause = "auxiliary_code_outside_main_panel_scope" if not raw else "raw_auxiliary_code"
    elif exact:
        cause = "eligible_static_code_temporal_evidence_missing"
    elif raw:
        cause = "raw_correct_normalization_or_confidence_rejection"
    elif not predictions:
        cause = "no_detected_ROI"
    elif not any(p["readings"] for p in predictions):
        cause = "OCR_empty_on_detected_ROI"
    else:
        cause = "OCR_incorrect_or_partial_on_detected_ROI"
    return {
        "id": row["id"],
        "model": row["model"],
        "city": ref["city"],
        "expected_code": expected,
        "code_location": ref.get("code_location"),
        "negative": ref["confirmed_negative"],
        "difficulty": ref["difficulty"],
        "ap_eligible": ref["ap_eligible"],
        "source": ref["source"],
        "reference_boxes": row["reference_boxes"],
        "predicted_boxes": [p["box"] for p in predictions],
        "YOLO_confidences": [p["confidence"] for p in predictions],
        "reference_ious": [p["reference_ious"] for p in predictions],
        "OCR_readings": [p["readings"] for p in predictions],
        "normalized_candidates": [p["frame_code"] for p in predictions],
        "raw_code_exact": raw,
        "eligible_code_exact": exact,
        "joint_panel_and_code_exact": joint,
        "eligible_wrong_candidates": [
            p["frame_code"] for p in eligible if expected and p["frame_code"] != expected
        ],
        "final_code": None,
        "final_status": "UNCERTAIN",
        "actual_events": len(row["events"]),
        "detection": row["detection"],
        "pipeline_ms": row["pipeline_wall_ms"],
        "detector_ms": row["detector_wall_ms"],
        "OCR_ms": row["ocr_wall_ms"],
        "cold_first": row["cold_first_detector_call"],
        "observed_failure_stage": cause,
        "image_quality_causes": "unproven; frozen difficulty is observational, not causal",
    }


def summarize_photos(
    recorded: list[dict[str, Any]], references: list[dict[str, Any]]
) -> dict[str, Any]:
    refs = {r["id"]: r for r in references}
    if not refs or len(refs) != len(references):
        raise ValueError("Nonempty unique reference required")
    rows: list[dict[str, Any]] = []
    for model in ["Bangkok", "BR"]:
        cases = [r for r in recorded if r["model"] == model]
        if len(cases) != len(refs) or {r["id"] for r in cases} != set(refs):
            raise ValueError("Paired models must receive the same unique photos")
        rows.extend(describe_case(r, refs[r["id"]]) for r in cases)
    if len(rows) != len(recorded):
        raise ValueError("Unexpected model in paired evaluation")
    output = {}
    for model in ["Bangkok", "BR"]:
        cases = [r for r in rows if r["model"] == model]
        coded = [r for r in cases if r["expected_code"]]
        warm = [r for r in cases if not r["cold_first"]]
        tp, fp, fn = (sum(r["detection"][k] for r in cases) for k in ["tp", "fp", "fn"])
        output[model] = {
            "images": len(cases),
            "panels": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "coded_images": len(coded),
            "raw_exact": sum(bool(r["raw_code_exact"]) for r in coded),
            "eligible_exact": sum(bool(r["eligible_code_exact"]) for r in coded),
            "joint_panel_code_exact": sum(bool(r["joint_panel_and_code_exact"]) for r in coded),
            "wrong_eligible_candidates": sum(len(r["eligible_wrong_candidates"]) for r in coded),
            "negative_images": sum(r["negative"] for r in cases),
            "negative_images_with_predictions": sum(
                r["negative"] and bool(r["predicted_boxes"]) for r in cases
            ),
            "negative_FP_boxes": sum(r["detection"]["fp"] for r in cases if r["negative"]),
            "ignored_unassessed": sum(r["detection"]["ignored_unassessed"] for r in cases),
            "actual_uncertain_events": sum(r["actual_events"] for r in cases),
            "photos_without_event": sum(r["actual_events"] == 0 for r in cases),
            "confirmed_events": 0,
            "warm_latency_n": len(warm),
            "pipeline_median_ms": median(r["pipeline_ms"] for r in warm) if warm else None,
            "detector_median_ms": median(r["detector_ms"] for r in warm) if warm else None,
            "failure_stages_coded": dict(Counter(r["observed_failure_stage"] for r in coded)),
            "by_code_location": {
                location: {
                    "n": sum(r["code_location"] == location for r in coded),
                    "raw_exact": sum(
                        bool(r["raw_code_exact"]) for r in coded if r["code_location"] == location
                    ),
                    "eligible_exact": sum(
                        bool(r["eligible_code_exact"])
                        for r in coded
                        if r["code_location"] == location
                    ),
                }
                for location in ["main_display", "auxiliary_front_placard"]
            },
        }
    return {
        "models": output,
        "cases": rows,
        "temporal_validation": False,
        "scope": "static descriptive comparison; no IID or encounter-success claim",
    }
