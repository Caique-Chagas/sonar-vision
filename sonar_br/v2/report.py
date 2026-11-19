"""Aggregate frozen receipts without rerunning inference or changing references."""

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from sonar_vision.evaluation.final_photos import read_case
from sonar_vision.training.dataset import sha256

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "sonar_br/artifacts/v2-evaluation"
REPORTS = ROOT / "sonar_br/reports"


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    codes = [r for r in rows if r["expected_code"]]
    localized = [r for r in rows if r["predictions"]]
    events = [e for r in rows for e in r["events"]]
    return {
        "images": len(rows),
        "panels": sum(len(r["reference_boxes"]) for r in rows),
        "negative_images": sum(r["negative"] for r in rows),
        "tp": sum(r["detection"]["tp"] for r in rows),
        "fn": sum(r["detection"]["fn"] for r in rows),
        "fp": sum(r["detection"]["fp"] for r in rows),
        "complete_code_denominator": len(codes),
        "raw_exact": sum(bool(r["raw_exact"]) for r in codes),
        "eligible_exact": sum(bool(r["eligible_exact"]) for r in codes),
        "complete_code_accuracy": sum(bool(r["eligible_exact"]) for r in codes) / len(codes)
        if codes
        else None,
        "wrong_eligible": sum(len(r["wrong_eligible_codes"]) for r in rows),
        "negative_eligible": sum(len(r["negative_eligible_codes"]) for r in rows),
        "confirmed_events": sum(e["status"] != "UNCERTAIN" for e in events),
        "uncertain_events": len(events),
        "photos_without_event": sum(not r["events"] for r in rows),
        "pipeline_median_ms": statistics.median(r["pipeline_ms"] for r in rows),
        "localized_median_ms": statistics.median(r["pipeline_ms"] for r in localized)
        if localized
        else None,
        "localized_latency_n": len(localized),
        "median_max_iou": statistics.median(
            max(p["reference_ious"], default=0) for r in rows for p in r["predictions"]
        )
        if localized
        else None,
    }


def render_example(ref: dict[str, Any], row: dict[str, Any], path: Path) -> None:
    file = Path(ref["image"])
    if not file.is_absolute():
        file = (ROOT / "sonar_br/datasets/manifests" / file).resolve()
    im = Image.open(file).convert("RGB")
    im.thumbnail((1280, 950))
    factor = im.width / ref["width"]
    draw = ImageDraw.Draw(im)
    for item in ref["boxes"]:
        draw.rectangle([v * factor for v in item["xyxy"]], outline="lime", width=3)
    for pred in row["predictions"]:
        draw.rectangle(
            [pred["box"][k] * factor for k in ["x1", "y1", "x2", "y2"]], outline="red", width=3
        )
    canvas = Image.new("RGB", (1280, im.height + 210), "white")
    canvas.paste(im, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 17)
    texts = " | ".join(r["text"] for p in row["predictions"] for r in p["readings"])
    caption = [
        "SONAR BR V2 + E3 / FOTO REAL ESTATICA / status final UNCERTAIN",
        "VERDE = referencia manual; VERMELHO = inferencia real. Nenhuma caixa = detector falhou.",
        "OCR bruto: " + (texts[:125] or "sem leitura / nenhuma ROI"),
        "Codigo completo de referencia: " + str(ref.get("expected_code") or "NAO ESTABELECIDO"),
        ref["author"] + " / " + ref["license"] + " / Commons pageID " + ref["id"],
        ref["source"][:148],
        (
            "Referencia por agente, sem segunda revisao humana. "
            "Benchmark historico identificado no relatorio."
        ),
    ]
    for n, text in enumerate(caption):
        draw.text((10, im.height + 8 + n * 27), text, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    digest = sha256(ROOT / "sonar_br/configs/v2-comparison-frozen.json")
    summary: dict[str, Any] = {"binding_sha256": digest, "no_default_promotion": True}
    flat = []
    for scope in ["new_test", "known_benchmark"]:
        summary[scope] = {}
        for variant in "ABCD":
            rows = []
            for path in sorted((OUT / scope).glob(variant + "-*.json")):
                row = read_case(path, digest)
                if row is None:
                    raise ValueError("Missing case")
                rows.append(row)
                flat.append(
                    {
                        "scope": scope,
                        "variant": variant,
                        "id": row["id"],
                        "expected_code": row["expected_code"],
                        "raw_exact": row["raw_exact"],
                        "eligible_exact": row["eligible_exact"],
                        "tp": row["detection"]["tp"],
                        "fn": row["detection"]["fn"],
                        "fp": row["detection"]["fp"],
                        "pipeline_ms": row["pipeline_ms"],
                        "final_status": row["final_status"],
                        "reference_boxes": json.dumps(row["reference_boxes"]),
                        "predicted_boxes": json.dumps([p["box"] for p in row["predictions"]]),
                        "detection_scores": json.dumps(
                            [p["confidence"] for p in row["predictions"]]
                        ),
                        "OCR": json.dumps(
                            [p["readings"] for p in row["predictions"]], ensure_ascii=False
                        ),
                        "IoUs": json.dumps([p["reference_ious"] for p in row["predictions"]]),
                        "frame_codes": json.dumps([p["frame_code"] for p in row["predictions"]]),
                        "failure": "no code ground truth; assess detection only"
                        if not row["expected_code"]
                        else "detector missing ROI"
                        if not row["predictions"]
                        else "exact frame candidate; no temporal confirmation"
                        if row["eligible_exact"]
                        else "OCR/normalization needs diagnosis; inspect raw receipt",
                    }
                )
            expected = 11 if scope == "new_test" else 14
            if len(rows) != expected:
                raise ValueError("Incomplete comparison")
            summary[scope][variant] = aggregate(rows)
    for name in ["v1", "v2"]:
        summary["AP_" + name] = json.loads((OUT / f"AP-{name}/detector.json").read_text())
    dev = {}
    for variant in ["OCR0", "OCR1_E3", "OCR2_left", "OCR3_scale2"]:
        rows = [
            json.loads(p.read_text())
            for p in (ROOT / "sonar_br/artifacts/v2-ocr-dev").glob(variant + "-*.json")
        ]
        dev[variant] = {
            "n": len(rows),
            "exact": sum(r["exact"] for r in rows),
            "wrong": sum(bool(r["wrong"]) for r in rows),
            "median_ms": statistics.median(r["ms"] for r in rows),
        }
    summary["OCR_DEV"] = dev
    summary["partial_base_erratum"] = (
        "New test id90614920: preliminary visual base475H has first-glyph4/8 ambiguity "
        "under LED sampling. Never scored as a full code; frozen reference preserved. "
        "Do not use OCR875H/75H as new ground truth."
    )
    (REPORTS / "v2-comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (REPORTS / "v2-cases.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)
    refs = {}
    for name in ["br-v2.review.json", "ocr-opt-v1.review.json"]:
        refs.update(
            {
                r["id"]: r
                for r in json.loads((ROOT / "sonar_br/datasets/manifests" / name).read_text())[
                    "records"
                ]
            }
        )
    for scope, source_id in [
        ("new_test", "90614920"),
        ("new_test", "194550876"),
        ("known_benchmark", "75775492"),
        ("known_benchmark", "75605720"),
    ]:
        row = read_case(OUT / scope / f"D-{source_id}.json", digest)
        if row is None:
            raise ValueError("Missing rendered receipt")
        render_example(refs[source_id], row, REPORTS / "figures/v2" / f"{scope}-{source_id}.jpg")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(list("ABCD"), [summary["new_test"][v]["tp"] for v in "ABCD"])
    axes[0].set(
        title="Novo teste: paineis detectados (IoU >= 0,5)",
        ylim=(0, 9),
        ylabel="Acertos / 9 paineis",
    )
    axes[1].bar(list("ABCD"), [summary["known_benchmark"][v]["eligible_exact"] for v in "ABCD"])
    axes[1].set(
        title="Benchmark HISTORICO conhecido: codigos",
        ylim=(0, 7),
        ylabel="Candidatos exatos / 7; sem confirmacao temporal",
    )
    fig.suptitle("A=v1+original; B=v1+E3; C=v2+original; D=v2+E3")
    fig.tight_layout()
    fig.savefig(REPORTS / "figures/v2/comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
