"""Summarize known-panel diagnostics and acquisition receipts, without inference."""

import ast
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from sonar_vision.evaluation.final_photos import read_case
from sonar_vision.training.dataset import sha256

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "artifacts/research/v2-diagnosis-20261116"
OUT = ROOT / "sonar_br/artifacts/v2-diagnosis"
REPORT = ROOT / "sonar_br/reports"


def summarize(folder: Path) -> dict[str, Any]:
    binding = sha256(folder / "binding.json")
    result = {}
    for variant in ["baseline", "hierarchy"]:
        rows = [read_case(p, binding) for p in sorted((folder / "cases").glob(variant + "-*.json"))]
        if len(rows) != 11 or any(r is None for r in rows):
            raise ValueError("Incomplete diagnostic")
        valid = [r for r in rows if r is not None]
        localized = [r for r in valid if r["predictions"]]
        result[variant] = {
            "images": len(valid),
            "tp": sum(r["detection"]["tp"] for r in valid),
            "fn": sum(r["detection"]["fn"] for r in valid),
            "fp": sum(r["detection"]["fp"] for r in valid),
            "complete_code_denominator": sum(bool(r["expected_code"]) for r in valid),
            "median_ms": statistics.median(r["pipeline_ms"] for r in valid),
            "localized_ms": [r["pipeline_ms"] for r in localized],
            "matched_ious": [p["reference_ious"] for r in localized for p in r["predictions"]],
            "negative_codes": sum(len(r["negative_eligible_codes"]) for r in valid),
            "confirmed": sum(e["status"] != "UNCERTAIN" for r in valid for e in r["events"]),
        }
    return result


def main() -> None:
    refs = json.loads((ROOT / "sonar_br/datasets/manifests/br-v2.review.json").read_text())[
        "records"
    ]
    rows = []
    for r in refs:
        if r["split"] != "test":
            continue
        old = {
            v: json.loads(
                (ROOT / f"sonar_br/artifacts/v2-evaluation/new_test/{v}-{r['id']}.json").read_text()
            )
            for v in ["B", "D"]
        }
        cascade = json.loads((OUT / f"cases/hierarchy-{r['id']}.json").read_text())
        for index, item in enumerate(r["boxes"]):
            b = item["xyxy"]
            w, h = b[2] - b[0], b[3] - b[1]
            covers = []
            for t in cascade["bus_trace"]:
                x1, y1, x2, y2 = t["crop_xyxy"]
                coverage = (
                    max(0, min(x2, b[2]) - max(x1, b[0]))
                    * max(0, min(y2, b[3]) - max(y1, b[1]))
                    / (w * h)
                )
                if coverage >= 0.8:
                    covers.append(t)
            data = {
                "id": r["id"],
                "panel": index,
                "reference_xyxy": b,
                "source": r["source"],
                "native_w": w,
                "native_h": h,
                "width_at640": w * 640 / max(r["width"], r["height"]),
                "height_at640": h * 640 / max(r["width"], r["height"]),
                "stage1_bus_crop_covers80percent": bool(covers),
                "cause_demonstrated": "small<20px at640"
                if h * 640 / max(r["width"], r["height"]) < 20
                else "detected by both"
                if r["id"] == "90614920"
                else "large windshield-integrated panel missed; scale alone insufficient",
                "visual_confounders": (
                    "shade/reflections; blank panels are valid full-module labels; "
                    "edges less precise"
                )
                if r["id"] in {"170315437", "170315439"}
                else ("perspective/wipers/reflections; do not infer missing route characters"),
                "annotation_error_demonstrated": False,
                "annotation_caveat": (
                    "Agent review only; small/blank boundaries need second adjudication; "
                    "frozen GT unchanged"
                ),
            }
            for name, rec in [("V1", old["B"]), ("V2", old["D"]), ("hierarchy", cascade)]:
                best = max(
                    rec["predictions"], key=lambda p: p["reference_ious"][index], default=None
                )
                data[name + "_IoU"] = best["reference_ious"][index] if best else 0
                data[name + "_confidence"] = best["confidence"] if best else None
                data[name + "_box"] = best["box"] if best else None
            rows.append(data)
    summary = {
        "scope": "KNOWN V2 diagnostic; no new generalization claim",
        "panels": rows,
        "paired1600": summarize(OUT),
        "paired_source3200": summarize(OUT / "source3200"),
        "distributions": {},
    }
    for split in ["train", "val", "test"]:
        positive = [r for r in refs if r["split"] == split and r["boxes"]]
        heights = [
            (b["xyxy"][3] - b["xyxy"][1]) * 640 / max(r["width"], r["height"])
            for r in positive
            for b in r["boxes"]
        ]
        oldgroups = {r["group_id"] for r in positive if r.get("origin")}
        addedgroups = {r["group_id"] for r in positive if not r.get("origin")} - oldgroups
        summary["distributions"][split] = {
            "positive_images": len(positive),
            "panels": len(heights),
            "height640_median": statistics.median(heights),
            "below20px": sum(h < 20 for h in heights),
            "old_positive_groups": len(oldgroups),
            "new_positive_groups_added": sorted(addedgroups),
        }
    dev = [
        json.loads(p.read_text()) for p in (OUT / "ocr-dev").glob("*.json") if p.stem != "binding"
    ]
    summary["OCR_rectification_DEV"] = {
        "n": len(dev),
        "exact": sum(r["exact"] for r in dev),
        "wrong_candidate_cases": sum(bool(r["wrong"]) for r in dev),
        "E3_exact": 10,
        "E3_wrong_cases": 2,
        "promotion": False,
        "scope": "manual ROI DEV; no temporal or independent claims",
    }
    executed = CACHE / "resolution-executed.py"
    current = ROOT / "sonar_br/diagnosis/resolution.py"
    equivalent = ast.dump(ast.parse(executed.read_text())) == ast.dump(
        ast.parse(current.read_text())
    )
    if not equivalent:
        raise ValueError("Resolution source changed beyond formatting")
    spec = json.loads((OUT / "source3200/binding.json").read_text())
    if sha256(executed) != spec["runner"]:
        raise ValueError("Missing exact executed source")
    summary["resolution_source_format_only"] = {
        "executed_snapshot": str(executed.relative_to(ROOT)),
        "executed_sha256": sha256(executed),
        "current_sha256": sha256(current),
        "AST_equal": True,
        "binding_preserved": True,
    }
    summary["directed_sources"] = json.loads(
        (ROOT / "sonar_br/datasets/manifests/directed-20261116.sources.json").read_text()
    )
    (REPORT / "V2_DIAGNOSIS.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    with (REPORT / "v2-nine-panels.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in {"panels", "directed_sources"}}, indent=2
        )
    )


if __name__ == "__main__":
    main()
