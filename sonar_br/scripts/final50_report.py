"""Summarize and draw saved inference only; never run YOLO or OCR."""

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from sonar_vision.evaluation.photo_summary import summarize_photos  # noqa: E402
from sonar_vision.training.dataset import sha256  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "sonar_br/reports"
REFERENCE = ROOT / "sonar_br/datasets/manifests/final50.review.json"
FIGURES = REPORTS / "evidence"


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def figures(summary: dict[str, Any], refs: list[dict[str, Any]], native: dict[str, Any]) -> None:
    FIGURES.mkdir(exist_ok=True)
    models = summary["models"]
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for axis, key, denominator, title in zip(
        axes,
        ["tp", "raw_exact", "eligible_exact"],
        [40, 17, 17],
        [
            "Painel inteiro: IoU ≥ 0,5",
            "Código completo no OCR bruto",
            "Candidato estático elegível",
        ],
        strict=True,
    ):
        values = [models[m][key] for m in ["Bangkok", "BR"]]
        axis.bar(["Bangkok", "BR"], values, color=["#747f88", "#007b95"])
        axis.set_ylim(0, denominator * 1.15)
        axis.set_title(title, fontsize=10)
        for index, value in enumerate(values):
            axis.text(index, value + denominator * 0.03, f"{value}/{denominator}", ha="center")
    figure.suptitle("50 fotos reais: 40 painéis, 10 negativas; códigos: subconjunto pós-QA de 17")
    figure.text(
        0.5,
        0.015,
        "QA não cega excluiu 5 gabaritos dos 22 originais; ver errata. AP: 39 fotos / 32 painéis.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.92))
    figure.savefig(FIGURES / "final50-metrics.png", dpi=145)
    plt.close(figure)

    by_id = {r["id"]: r for r in refs}
    rows = {(r["model"], r["id"]): r for r in native["records"]}
    selected = ["75779868", "75775509", "75775113"]
    figure, axes = plt.subplots(3, 2, figsize=(12, 11))
    credits = []
    for index, identifier in enumerate(selected):
        ref = by_id[identifier]
        if ref["license"] != "CC BY 2.0" or "Rafael Delazari" not in ref["author"]:
            raise ValueError("Evidence derivative requires verified individual CC BY credit")
        image = Image.open((REFERENCE.parent / ref["image"]).resolve())
        if sha256((REFERENCE.parent / ref["image"]).resolve()) != ref["image_sha256"]:
            raise ValueError("Evidence pixels changed")
        x1, y1, x2, y2 = ref["boxes"][0]["xyxy"]
        for col, model in enumerate(["Bangkok", "BR"]):
            axis, row = axes[index, col], rows[model, identifier]
            axis.imshow(image)
            axis.set_xlim(max(0, x1 - 45), min(image.width, x2 + 45))
            axis.set_ylim(min(image.height, y2 + 45), max(0, y1 - 55))
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
                        edgecolor="#00dbff",
                        linewidth=2,
                    )
                )
            text = " | ".join(r["text"] for p in row["predictions"] for r in p["readings"])
            details = (
                "sem caixa"
                if not row["predictions"]
                else "; ".join(
                    f"YOLO {p['confidence']:.3f}, IoU {max(p['reference_ious'], default=0):.3f}, "
                    f"OCR {max((r['confidence'] for r in p['readings']), default=0):.3f}"
                    for p in row["predictions"]
                )
            )
            axis.set_title(
                f"{model} | ID {identifier} | referência {ref['expected_code']}\n"
                f"{details}\nOCR: {text or 'nenhum'}",
                fontsize=9,
            )
            axis.axis("off")
        credits.append(
            f"- ID {identifier}: [{ref['title']}]({ref['source']}), "
            f"{ref['author']}, [{ref['license']}]({ref['license_url']})."
        )
    figure.suptitle("Caixas e OCR gravados na inferência real — fotografias estáticas", fontsize=13)
    figure.text(
        0.5,
        0.012,
        "Amarelo tracejado: referência manual. Azul: previsão real. Eventos UNCERTAIN.\n"
        "Rafael Delazari / Wikimedia Commons, CC BY 2.0. Recortes, caixas e legendas adicionados.\n"
        "Exemplos ilustrativos após a avaliação completa; não representam a taxa global de acerto.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.96))
    figure.subplots_adjust(hspace=0.65)
    figure.savefig(FIGURES / "final50-paired-real.jpg", dpi=145, pil_kwargs={"quality": 88})
    plt.close(figure)
    (FIGURES / "FINAL50_CREDITS.md").write_text(
        "# Créditos da evidência final50\n\n"
        "`final50-paired-real.jpg`: derivado CC BY 2.0; fotos reais, sem geração visual por IA. "
        "Alterações: recortes, caixas e legendas sobre resultados persistidos.\n\n"
        + "\n".join(credits)
        + "\n\n"
        "`final50-metrics.png`: gráfico próprio calculado dos 100 casos; código AGPL-3.0. "
        "As licenças das fotografias são individuais, não AGPL. "
        "Nenhuma foto bruta ou dataset foi incluído no Git.\n"
    )


def main() -> None:
    reference = json.loads(REFERENCE.read_text())
    native_path = REPORTS / "final50-pipeline.json"
    native = json.loads(native_path.read_text())
    binding = native["binding"]
    for key, path in [
        ("reference_sha256", REFERENCE),
        ("protocol_sha256", ROOT / "sonar_br/configs/final50-protocol.json"),
        ("runner_sha256", ROOT / "sonar_br/scripts/final50_evaluate.py"),
    ]:
        if sha256(path) != binding[key]:
            raise ValueError("Frozen evaluation binding changed")
    summary = summarize_photos(native["records"], reference["records"])
    errata = json.loads((REPORTS / "final50-reference-errata.json").read_text())
    if errata["reference_sha256"] != sha256(REFERENCE):
        raise ValueError("Reference QA addendum differs from original binding")
    excluded = {r["id"] for r in errata["affected_records"]}
    summary["posthoc_code_sensitivity"] = {}
    for model in ["Bangkok", "BR"]:
        cases = [
            r
            for r in summary["cases"]
            if r["model"] == model and r["expected_code"] and r["id"] not in excluded
        ]
        summary["posthoc_code_sensitivity"][model] = {
            "codes": len(cases),
            "raw_exact": sum(bool(r["raw_code_exact"]) for r in cases),
            "eligible_exact": sum(bool(r["eligible_code_exact"]) for r in cases),
            "scope": "nonblind QA subset, not a replacement independent test",
        }
    for row in summary["cases"]:
        row["failure_stage_against_nominal_frozen_GT"] = row["observed_failure_stage"]
        if row["id"] in excluded:
            row["observed_failure_stage"] = "reference_QA_issue_not_attributed_to_OCR"
        row["code_truth_posthoc_QA"] = (
            "excluded_reference_issue"
            if row["id"] in excluded
            else ("retained" if row["expected_code"] else "no_code_truth")
        )
    for model in summary["models"].values():
        model["failure_stages_nominal_frozen_GT"] = model.pop("failure_stages_coded")
    summary["native_pipeline_sha256"] = sha256(native_path)
    summary["reference_sha256"] = sha256(REFERENCE)
    summary["manual_main_ROI"] = {
        loc: {
            "n": sum(r["code_location"] == loc for r in native["reference_ROI_OCR"]),
            "raw_exact": sum(
                r["exact_code"] for r in native["reference_ROI_OCR"] if r["code_location"] == loc
            ),
        }
        for loc in ["main_display", "auxiliary_front_placard"]
    }
    summary["city_counts"] = dict(Counter(r["city"] for r in reference["records"]))
    summary["by_city"] = {}
    for city in summary["city_counts"]:
        summary["by_city"][city] = {}
        for model in ["Bangkok", "BR"]:
            rows = [r for r in summary["cases"] if r["city"] == city and r["model"] == model]
            summary["by_city"][city][model] = {
                "images": len(rows),
                "panels": sum(r["detection"]["tp"] + r["detection"]["fn"] for r in rows),
                "tp": sum(r["detection"]["tp"] for r in rows),
                "coded": sum(bool(r["expected_code"]) for r in rows),
                "raw_exact": sum(bool(r["raw_code_exact"]) for r in rows),
            }
    save_json(REPORTS / "final50-summary.json", summary)
    with (REPORTS / "final50-results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["cases"][0]), lineterminator="\n")
        writer.writeheader()
        for row in summary["cases"]:
            writer.writerow(
                {
                    k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                    for k, v in row.items()
                }
            )
    figures(summary, reference["records"], native)
    print(json.dumps(summary["models"], indent=2))


if __name__ == "__main__":
    main()
