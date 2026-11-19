"""Aggregate frozen receipts, without repeating inference or changing annotations."""

import csv
import json
from pathlib import Path
from statistics import median
from typing import Any

from sonar_br.final_correction.run import OUT, REF, ROOT
from sonar_vision.br.curation import assess_targeted_coverage
from sonar_vision.evaluation.final_photos import read_case
from sonar_vision.training.dataset import sha256


def load(folder: Path, name: str, binding: Path) -> dict[str, Any]:
    row = read_case(folder / name, sha256(binding))
    if row is None:
        raise ValueError("Missing diagnostic receipt: " + name)
    return row


def summarize(rows: list[dict[str, Any]], refs: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {k: sum(r["detection"][k] for r in rows) for k in ["tp", "fn", "fp"]}
    by_id = {r["id"]: r for r in rows}
    return {
        **counts,
        "recall": counts["tp"] / (counts["tp"] + counts["fn"]),
        "precision_assessed": counts["tp"] / (counts["tp"] + counts["fp"]),
        "images": len(rows),
        "eligible_exact_codes": sum(bool(r["eligible_exact"]) for r in rows),
        "coded_denominator": sum(bool(r.get("expected_code")) for r in rows),
        "raw_exact_cases": sum(any(p["raw_code_exact"] for p in r["predictions"]) for r in rows),
        "wrong_eligible_in_coded": sum(bool(r["wrong_eligible_codes"]) for r in rows),
        "negative_FP": sum(
            by_id[r["id"]]["detection"]["fp"] for r in refs if r["confirmed_negative"]
        ),
        "small_tp": sum(
            by_id[r["id"]]["detection"]["tp"] for r in refs if r.get("height640", 100) < 20
        ),
        "windshield_tp": sum(
            by_id[r["id"]]["detection"]["tp"] for r in refs if r.get("windshield_integrated")
        ),
        "pipeline_median_ms": median(r["pipeline_ms"] for r in rows),
        "confirmed_events": sum(
            e["status"] != "UNCERTAIN" for r in rows for e in r.get("events", [])
        ),
    }


def main() -> None:
    refs = json.loads(REF.read_text())["records"]
    old = ROOT / "sonar_br/artifacts/generalization-20261116"
    groups = {}
    for name, folder in [("V1", old), ("V2", old), ("V1_T", OUT), ("V1_TM", OUT)]:
        rows = [
            load(folder / "cases", f"{name}-{r['id']}.json", folder / "binding.json") for r in refs
        ]
        groups[name] = rows
    local = json.loads(
        (
            ROOT / "sonar_br/datasets/manifests/final-correction-20261117.local.review.json"
        ).read_text()
    )["records"]
    pool = refs + local
    coverage = assess_targeted_coverage(pool)
    if coverage["coverage_passed"]:
        raise ValueError("Dataset must remain blocked at this checkpoint")
    sub = OUT / "local"
    summary = {
        "scope": "Development only; no independent test or model promotion",
        "variants_same_14DEV": {n: summarize(rows, refs) for n, rows in groups.items()},
        "local_one_SPTrans_DEV": {n: load(sub, n + ".json", sub / "binding.json") for n in groups},
        "local_manual_ROI_only": {
            n: load(sub, "manual-" + n + ".json", sub / "binding.json") for n in ["original", "E3"]
        },
        "curated_pool": {
            "primary_positive_frames": 14,
            "primary_panels": 18,
            "negative_frames": 2,
            "distinct_positive_groups": 14,
            "new_capture_groups_this_phase": 1,
            "previously_train_negative_group_with_new_positive_labels": 1,
            "height640": [r["height640"] for r in refs if r.get("boxes")]
            + [v for r in local for v in r["panel_heights640"]],
            "gate": coverage,
            "reviewed_train_contexts": (
                "One, same parent170315291/group; not additional independent image"
            ),
        },
        "new_training": False,
        "default_changed": False,
        "reserved209P_inferred": False,
        "new_sequence_validation": False,
        "AP": "Not computed on incomplete full frames; thresholded operating metrics only",
        "latency": (
            "Old baseline receipts retained; different dates, not a paired speed benchmark"
        ),
    }
    dest = ROOT / "sonar_br/reports/FINAL_CORRECTION.json"
    dest.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    with dest.with_name("final-correction-cases.csv").open("w") as stream:
        fields = [
            "variant",
            "id",
            "GT",
            "reference_boxes",
            "predictions",
            "detection",
            "eligible_exact",
            "pipeline_ms",
            "final_status",
        ]
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        for name, rows in groups.items():
            for row in rows:
                writer.writerow(
                    {
                        "variant": name,
                        "id": row["id"],
                        "GT": row["expected_code"],
                        **{
                            k: json.dumps(row[k], ensure_ascii=False)
                            for k in ["reference_boxes", "predictions", "detection"]
                        },
                        **{k: row[k] for k in ["eligible_exact", "pipeline_ms", "final_status"]},
                    }
                )
    print(json.dumps({n: summarize(rows, refs) for n, rows in groups.items()}, indent=2))


if __name__ == "__main__":
    main()
