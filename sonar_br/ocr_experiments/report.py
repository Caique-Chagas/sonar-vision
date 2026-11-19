"""Generate tables and authentic annotated figures from immutable recorded results."""

import csv
import json
import math
import textwrap
from statistics import median
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from sonar_br.ocr_experiments.run import REF, ROOT  # noqa: E402
from sonar_br.ocr_experiments.select import aggregate  # noqa: E402

REPORTS = ROOT / "sonar_br/reports"


def wilson(k: int, n: int) -> list[float]:
    z = 1.96
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    delta = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [center - delta, center + delta]


def summarize() -> dict[str, Any]:
    data = json.loads((REPORTS / "ocr-opt-holdout.json").read_text())
    summary = {}
    for name in ["E0", "candidate"]:
        rows = [r for r in data["records"] if r["variant"] == name]
        values = aggregate(rows)
        located = [r for r in rows if not r["first_photo"] and r["predictions"]]
        values.update(
            images=len(rows),
            panels=sum(r["detection"]["tp"] + r["detection"]["fn"] for r in rows),
            panel_tp=sum(r["detection"]["tp"] for r in rows),
            panel_fp=sum(r["detection"]["fp"] for r in rows),
            panel_fn=sum(r["detection"]["fn"] for r in rows),
            negatives=sum(r["negative"] for r in rows),
            actual_uncertain_events=sum(len(r["events"]) for r in rows),
            photos_without_event=sum(not r["events"] for r in rows),
            located_warm_n=len(located),
            located_pipeline_median_ms=median(r["pipeline_ms"] for r in located),
            located_OCR_median_ms=median(r["OCR_ms"] for r in located),
            process_peak_resident_bytes=max(r["max_resident_bytes"] for r in rows),
            illustrative_Wilson95=wilson(values["eligible_exact"], values["codes"]),
        )
        summary[name] = values
    return {
        "holdout": summary,
        "classification": "MELHORIA PARCIAL",
        "target70_reached": False,
        "temporal_confirmation": False,
        "production_promotion": False,
        "uncertainty": (
            "Wilson is illustrative under independent Bernoulli assumption; "
            "correlated convenience sample is not a population guarantee"
        ),
    }


def export_rows() -> None:
    reference = {r["id"]: r for r in json.loads(REF.read_text())["records"]}
    rows = []
    for split in ["dev", "holdout"]:
        data = json.loads((REPORTS / f"ocr-opt-{split}.json").read_text())
        for r in data["records"]:
            ref = reference[r["id"]]
            if not ref["expected_code"]:
                cause = (
                    "negative_no_panel" if ref["confirmed_negative"] else "no_reliable_code_truth"
                )
            elif not r["predictions"]:
                cause = "detection_failed_no_ROI"
            elif r["eligible_exact"]:
                cause = "eligible_static_code_no_temporal_evidence"
            elif r["wrong_eligible_codes"]:
                cause = "incorrect_static_candidate_not_temporal_confirmation"
            elif r["raw_exact"]:
                cause = "normalization_conflict_or_confidence_geometry_gate"
            else:
                cause = "OCR_text_or_segmentation_failure_inspect_preserved_transcript"
            rows.append(
                {
                    "split": split,
                    "variant": r["variant"],
                    "scope": r["scope"],
                    "actual_source": "licensed real photograph; not video or temporal evidence",
                    "image": r["id"],
                    "source": ref["source"],
                    "author": ref["author"],
                    "license": ref["license"],
                    "capture_group": ref["group_id"],
                    "code_truth": ref.get("expected_code"),
                    "reference_boxes": r["reference_boxes"],
                    "predictions": r["predictions"],
                    "YOLO_confidences": [p["confidence"] for p in r["predictions"]],
                    "reference_ious": [p["reference_ious"] for p in r["predictions"]],
                    "raw_OCR_text": " | ".join(
                        t["text"] for p in r["predictions"] for t in p["readings"]
                    ),
                    "raw_OCR_scores": [
                        t["confidence"] for p in r["predictions"] for t in p["readings"]
                    ],
                    "normalized_codes": [p["frame_code"] for p in r["predictions"]],
                    "raw_exact": r["raw_exact"],
                    "eligible_exact": r["eligible_exact"],
                    "wrong_eligible_codes": r["wrong_eligible_codes"],
                    "final_code": None,
                    "final_status": "UNCERTAIN",
                    "actual_events": len(r["events"]),
                    "failure_stage": cause,
                    "pipeline_ms": r["pipeline_ms"],
                    "OCR_ms": r["OCR_ms"],
                    "detector_ms": r["detector_ms"],
                    "cached_detector": r["cached_detector"],
                    "cached_OCR": r["cached_OCR"],
                }
            )
    with (REPORTS / "ocr-opt-cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                    for k, v in row.items()
                }
            )


def evidence() -> None:
    refs = {r["id"]: r for r in json.loads(REF.read_text())["records"]}
    dev = json.loads((REPORTS / "ocr-opt-dev.json").read_text())["records"]
    holdout = json.loads((REPORTS / "ocr-opt-holdout.json").read_text())["records"]
    selected = [
        ("75775509", dev, "E3", "DEV retrospectivo"),
        ("75605720", holdout, "candidate", "HOLDOUT novo"),
        ("170315304", holdout, "candidate", "HOLDOUT novo; detecção falhou"),
    ]
    figure, axes = plt.subplots(3, 2, figsize=(12, 11))
    credits = []
    for index, (identifier, rows, right, scope) in enumerate(selected):
        ref = refs[identifier]
        image = Image.open((REF.parent / ref["image"]).resolve())
        x1, y1, x2, y2 = ref["boxes"][0]["xyxy"]
        for col, name in enumerate(["E0", right]):
            row = next(r for r in rows if r["id"] == identifier and r["variant"] == name)
            axis = axes[index, col]
            axis.imshow(image)
            axis.set_xlim(max(0, x1 - 35), min(image.width, x2 + 35))
            axis.set_ylim(min(image.height, y2 + 30), max(0, y1 - 30))
            axis.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="#ffd100",
                    linestyle="--",
                    linewidth=1.5,
                )
            )
            for p in row["predictions"]:
                box = p["box"]
                axis.add_patch(
                    Rectangle(
                        (box["x1"], box["y1"]),
                        box["x2"] - box["x1"],
                        box["y2"] - box["y1"],
                        fill=False,
                        edgecolor="#00d5ff",
                        linewidth=2,
                    )
                )
            texts = " | ".join(t["text"] for p in row["predictions"] for t in p["readings"])
            readings = "\n".join(textwrap.wrap(texts or "sem ROI; OCR não executado", 64))
            label = "Baseline" if name == "E0" else "E3 experimental"
            title = (
                f"{scope} | {label} | {identifier}\nReferência {ref['expected_code']} | {readings}"
            )
            axis.set_title(title, fontsize=9)
            axis.axis("off")
        credits.append(
            f"- {scope}, ID{identifier}: [{ref['title']}]({ref['source']}), "
            f"{ref['author']}, [{ref['license']}]({ref['license_url']})."
        )
    figure.suptitle("Inferência gravada: ganho em DEV, ganho independente e limite do detector")
    figure.text(
        0.5,
        0.014,
        "Zoom manual apenas para visualização; inferência recebeu fotografias completas. "
        "Amarelo: referência; azul: YOLO real.\n"
        "Todos os eventos são UNCERTAIN; nenhum zoom manual substituiu a ROI automática.\n"
        "Rafael Delazari CC BY 2.0; MacGyver Santista CC0. "
        "Alterações: recortes, caixas e legendas.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.96))
    figure.subplots_adjust(hspace=0.6)
    folder = REPORTS / "evidence"
    figure.savefig(folder / "ocr-opt-real-evidence.jpg", dpi=145, pil_kwargs={"quality": 88})
    plt.close(figure)
    (folder / "OCR_OPT_CREDITS.md").write_text(
        "# Créditos OCR optimization\n\nDerivado composto CC BY 2.0; fotografias reais. "
        "Sem geração/simulação de imagem. Recortes/caixas/legendas acrescentados.\n\n"
        + "\n".join(credits)
        + "\n\nO gráfico numérico é próprio. Fotos não são AGPL; "
        "nenhum dataset bruto foi publicado.\n"
    )
    figure, axis = plt.subplots(figsize=(8, 4.2))
    metrics = [
        aggregate([r for r in data if r["variant"] == variant])
        for data, variant in [(dev, "E0"), (dev, "E3"), (holdout, "E0"), (holdout, "candidate")]
    ]
    axis.bar(
        [0, 1, 3, 4],
        [m["eligible_exact"] / m["codes"] for m in metrics],
        color=["#71838c", "#007d96"] * 2,
    )
    axis.axhline(0.7, color="#b23b3b", linestyle="--", label="Meta experimental 70%")
    axis.set_xticks([0, 1, 3, 4], ["E0 DEV", "E3 DEV", "E0 holdout", "E3 holdout"])
    axis.set_ylim(0, 1)
    for x, metric in zip([0, 1, 3, 4], metrics, strict=True):
        k, n = metric["eligible_exact"], metric["codes"]
        axis.text(x, k / n + 0.025, f"{k}/{n}", ha="center")
    axis.set_title("Código completo elegível — sem confirmação temporal")
    axis.legend(loc="upper right")
    figure.text(
        0.5,
        0.02,
        "DEV inclui 17 códigos históricos + 2 novos. Holdout: 7 códigos novos; amostra pequena.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(folder / "ocr-opt-comparison.png", dpi=145)
    plt.close(figure)


def main() -> None:
    result = summarize()
    (REPORTS / "ocr-opt-summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    export_rows()
    evidence()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
