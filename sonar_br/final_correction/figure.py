"""Scientific evidence figure: real photograph, manual reference explicitly labelled."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from sonar_br.final_correction.run import OUT, ROOT


def main() -> None:
    ref = json.loads(
        (
            ROOT / "sonar_br/datasets/manifests/final-correction-20261117.local.review.json"
        ).read_text()
    )["records"][0]
    for name in ["V1", "V2", "V1_T", "V1_TM"]:
        if json.loads((OUT / "local" / (name + ".json")).read_text())["predictions"]:
            raise ValueError("Figure must reflect the actual absence of automatic detections")
    manual = json.loads((OUT / "local/manual-E3.json").read_text())
    readings = manual["readings"]
    best = max(
        (r for r in readings if r["text"] == ref["expected_code"]), key=lambda r: r["confidence"]
    )
    image = Image.open(ref["image"])
    x1, y1, x2, y2 = ref["boxes"][0]["xyxy"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.4, 1]})
    axes[0].imshow(image)
    axes[0].add_patch(
        Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2, linestyle="--")
    )
    axes[0].set_title(
        "Fotografia estática — desenvolvimento\nCaixa vermelha: referência MANUAL", fontsize=13
    )
    axes[1].imshow(image.crop((x1, y1, x2, y2)), interpolation="nearest")
    axes[1].set_title(
        "OCR na ROI manual\nNão é reconhecimento automático ponta a ponta", fontsize=12
    )
    for axis in axes:
        axis.axis("off")
    fig.suptitle("SPTrans 5701-10: falha de detecção, leitura condicionada ao recorte", fontsize=16)
    fig.text(
        0.54,
        0.21,
        (
            f"E3: {best['text']} · score {best['confidence']:.3f}\n"
            "YOLO V1 / V2 / T / TM: nenhuma ROI automática\n"
            "Decisão: UNCERTAIN · sem sequência temporal"
        ),
        fontsize=12,
    )
    fig.text(
        0.04,
        0.04,
        (
            "Foto: MacGyver Santista · 02/07/2025 · CC0 · Commons page170315291\n"
            "Modificações: recorte/ampliação e marcação de referência. "
            "Nenhum pixel de texto foi sintetizado."
        ),
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.25, 1, 0.93))
    path = ROOT / "artifacts/research/final-correction-20261117/SPTrans-manual-diagnosis.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
