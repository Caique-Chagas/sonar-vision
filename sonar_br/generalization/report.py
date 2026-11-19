"""Aggregate acquired DEV references and frozen receipts; never run models or training."""

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from sonar_vision.br.curation import assess_targeted_coverage
from sonar_vision.evaluation.final_photos import read_case
from sonar_vision.training.dataset import sha256
from sonar_vision.validation.normalizer import FULL_CODE

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "sonar_br/datasets/manifests/generalization-20261116.review.json"
OUT = ROOT / "sonar_br/artifacts/generalization-20261116"
REPORTS = ROOT / "sonar_br/reports"


def main() -> None:
    refs = json.loads(REF.read_text())["records"]
    gate = assess_targeted_coverage(refs)
    summary: dict[str, Any] = {
        "scope": "new DEV diagnostics, not independent test or new model",
        "coverage_gate": gate,
        "new_detector_trained": False,
        "default_changed": False,
        "positives": sum(not r["confirmed_negative"] for r in refs),
        "negatives": sum(r["confirmed_negative"] for r in refs),
        "authors_positive": len({r["author"] for r in refs if not r["confirmed_negative"]}),
        "cities_positive": {},
        "variants": {},
    }
    for r in refs:
        if not r["confirmed_negative"]:
            city = r["city"]
            summary["cities_positive"][city] = summary["cities_positive"].get(city, 0) + 1
    lines = []
    for variant in ["V1", "V2"]:
        cases = [
            read_case(OUT / "cases" / f"{variant}-{r['id']}.json", sha256(OUT / "binding.json"))
            for r in refs
        ]
        if any(c is None for c in cases):
            raise ValueError("Incomplete baselines")
        rows = [c for c in cases if c is not None]
        coded = [r for r in rows if r["expected_code"]]
        tp = sum(r["detection"]["tp"] for r in rows)
        fp = sum(r["detection"]["fp"] for r in rows)
        summary["variants"][variant] = {
            "images": len(rows),
            "tp": tp,
            "fn": sum(r["detection"]["fn"] for r in rows),
            "fp": fp,
            "recall_primary_panels": tp / 12,
            "precision_assessed_regions": tp / (tp + fp),
            "code_denominator": len(coded),
            "eligible_exact": sum(bool(r["eligible_exact"]) for r in coded),
            "wrong_eligible_in_coded": sum(len(r["wrong_eligible_codes"]) for r in coded),
            "negative_codes": sum(len(r["negative_eligible_codes"]) for r in rows),
            "pipeline_median_ms": statistics.median(r["pipeline_ms"] for r in rows),
            "confirmed": sum(e["status"] != "UNCERTAIN" for r in rows for e in r["events"]),
            "small_tp": sum(
                r["detection"]["tp"]
                for r, ref in zip(rows, refs, strict=True)
                if ref.get("height640", 999) < 20
            ),
            "windshield_tp": sum(
                r["detection"]["tp"]
                for r, ref in zip(rows, refs, strict=True)
                if ref["windshield_integrated"]
            ),
        }
        for row, ref in zip(rows, refs, strict=True):
            compatible = bool(
                ref.get("expected_code") and FULL_CODE.fullmatch(ref["expected_code"])
            )
            lines.append(
                {
                    "variant": variant,
                    "id": ref["id"],
                    "city": ref["city"],
                    "scope": "DEV_only",
                    "GT": ref.get("expected_code"),
                    "API_route_format_compatible": compatible,
                    "gt_boxes": json.dumps(row["reference_boxes"]),
                    "predictions": json.dumps(row["predictions"], ensure_ascii=False),
                    "tp": row["detection"]["tp"],
                    "fn": row["detection"]["fn"],
                    "fp": row["detection"]["fp"],
                    "eligible_exact": row["eligible_exact"],
                    "status": row["final_status"],
                    "ms": row["pipeline_ms"],
                }
            )
    roi = {}
    for name in ["original", "E3"]:
        rows = [
            read_case(p, sha256(OUT / "binding.json"))
            for p in sorted((OUT / "reference-ocr").glob(name + "-*.json"))
        ]
        valid = [r for r in rows if r is not None]
        roi[name] = {
            "n": len(valid),
            "exact": sum(r["exact"] for r in valid),
            "wrong_cases": sum(bool(r["wrong"]) for r in valid),
            "exact_ids": [r["id"] for r in valid if r["exact"]],
        }
    summary["manual_full_panel_ROI"] = roi
    probe = OUT / "code-probe"
    rows = [
        read_case(p, sha256(probe / "binding.json"))
        for p in probe.glob("*.json")
        if p.name != "binding.json"
    ]
    valid = [r for r in rows if r is not None]
    summary["manual_CODE_crop_recognizer"] = {
        "n": len(valid),
        "exact": sum(r["exact"] for r in valid),
        "scope": "manual code crop, not automatic localization/pipeline",
    }
    summary["SPTrans_new_DEV"] = {
        "id": "69139074",
        "GT": "5109-10",
        "V1_auto_ROI": False,
        "V2_auto_ROI": False,
        "manual_ROI_original_and_E3_exact": True,
        "must_not_present_as_end_to_end_success": True,
    }
    summary["reserved_test"] = {
        "id": "35046766",
        "code": "209P-10",
        "inferred": False,
        "new_candidate_not_frozen": True,
    }
    summary["AP"] = (
        "Not computed:3frames lack exhaustive background labels; "
        "no new training/evaluation candidate"
    )
    (REPORTS / "GENERALIZATION_DATA.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    with (REPORTS / "generalization-cases.csv").open("w") as stream:
        w = csv.DictWriter(stream, fieldnames=list(lines[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(lines)
    with (REPORTS / "generalization-annotations.csv").open("w") as stream:
        fields = [
            "id",
            "group_id",
            "city",
            "license",
            "expected_code",
            "height640",
            "windshield_integrated",
            "exhaustive_annotation",
            "confirmed_negative",
            "source",
        ]
        w = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(refs)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
