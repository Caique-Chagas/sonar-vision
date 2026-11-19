"""Render persisted test results; no inference, training or OCR is rerun."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    reference_path = ROOT / "sonar_br/datasets/manifests/commons-br-v2.review.json"
    refs = {r["id"]: r for r in json.loads(reference_path.read_text())["records"]}
    results = json.loads((ROOT / "sonar_br/reports/pipeline-test.json").read_text())
    rows = {r["id"]: r for r in results["records"] if r["model"] == "BR"}
    figure, axes = plt.subplots(2, 2, figsize=(13, 6.5), width_ratios=[1.5, 1])
    for index, identifier in enumerate(["75681290", "75777357"]):
        ref, row = refs[identifier], rows[identifier]
        assert ref["license"] == "CC BY 2.0" and "Rafael Delazari" in ref["author"]
        prediction = row["predictions"][0]
        image = Image.open((reference_path.parent / ref["image"]).resolve())
        x1, y1, x2, y2 = ref["boxes"][0]["xyxy"]
        crop_x, crop_y = max(0, x1 - 40), max(0, y1 - 55)
        axes[index, 0].imshow(image)
        axes[index, 0].set_xlim(crop_x, min(image.width, x2 + 45))
        axes[index, 0].set_ylim(min(image.height, y2 + 35), crop_y)
        box = prediction["box"]
        axes[index, 0].add_patch(
            Rectangle(
                (box["x1"], box["y1"]),
                box["x2"] - box["x1"],
                box["y2"] - box["y1"],
                fill=False,
                edgecolor="#00c8ff",
                linewidth=2,
            )
        )
        axes[index, 0].set_title(f"Foto real de teste {identifier}: caixa prevista pelo SONAR BR")
        axes[index, 0].axis("off")
        texts = " | ".join(r["text"] for r in prediction["readings"])
        score = prediction["readings"][0]["confidence"]
        axes[index, 1].text(
            0,
            0.85,
            f"Gabarito visual: {ref['expected_code']}\n\n"
            f"OCR real: {texts}\n"
            f"YOLO: {prediction['confidence']:.3f}  |  OCR: {score:.3f}\n"
            f"IoU: {max(prediction['reference_ious']):.3f}\n\n"
            "Evento: UNCERTAIN — uma observação.\n"
            "Fotografia estática; não é validação temporal.",
            va="top",
            fontsize=11,
            linespacing=1.6,
        )
        axes[index, 1].axis("off")
    figure.suptitle("SONAR VISION BR — acerto e erro preservados no teste reservado", fontsize=15)
    figure.text(
        0.5,
        0.02,
        "Fotografias: Rafael Delazari / Wikimedia Commons, CC BY 2.0. "
        "Alterações: recortes, caixa prevista e legendas.\n"
        "Teste completo: 6 fotos / 5 painéis / 4 códigos; 3/4 leituras estáticas exatas, "
        "0 confirmações temporais. Créditos/URLs em evidence/CREDITS.md.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.11, 1, 0.93))
    destination = ROOT / "sonar_br/reports/evidence/BR-static-test-v1.png"
    destination.parent.mkdir(exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
