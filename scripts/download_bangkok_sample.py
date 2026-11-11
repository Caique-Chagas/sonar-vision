"""Download a bounded, licensed subset via immutable public GitHub URLs. No secrets."""

import argparse
import csv
import hashlib
import json
import math
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sonar_vision.training.dataset import origin_family, sha256, validate_labels

COMMIT = "cacdb66543150261916da59eaa6ce34d89f6839e"
REPO = "https://raw.githubusercontent.com/llsrwsaint225/Bangkok-Bus-Dataset-YOLOV8/"
SOURCE = "https://universe.roboflow.com/saint70239/bangkok-bus-dataset"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--output", type=Path, default=Path("data/bangkok-pilot-raw"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 3 <= args.count <= 200:
        parser.error("Amostra deve conter entre 3 e 200 famílias; não baixa o dataset inteiro")
    if args.output.exists() and not args.resume:
        parser.error("Diretório já existe; preserve os dados e use outro --output")
    with urllib.request.urlopen(
        f"https://api.github.com/repos/llsrwsaint225/Bangkok-Bus-Dataset-YOLOV8/git/trees/{COMMIT}?recursive=1",
        timeout=30,
    ) as response:
        tree = json.load(response)
    families: dict[str, list[str]] = {}
    for item in tree["tree"]:
        if item["path"].startswith("train/images/") and item["path"].endswith(".jpg"):
            families.setdefault(origin_family(Path(item["path"]).name), []).append(item["path"])
    selected = random.Random(42).sample(sorted(families), min(args.count, len(families)))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "original-labels").mkdir(exist_ok=True)
    blob_hashes = {item["path"]: item["sha"] for item in tree["tree"] if item["type"] == "blob"}
    for filename in ("README.dataset.txt", "README.roboflow.txt", "data.yaml"):
        with urllib.request.urlopen(f"{REPO}{COMMIT}/{filename}", timeout=30) as response:
            (args.output / filename).write_bytes(response.read(100_000))
    if "CC BY 4.0" not in (args.output / "README.dataset.txt").read_text():
        raise ValueError("A declaração de licença mudou; interrompendo download")

    def download(family: str) -> dict[str, str]:
        path = sorted(families[family])[0]  # Exactly one published augmentation per source photo.
        name = Path(path).name
        image = args.output / name
        label = args.output / f"{image.stem}.txt"
        original_label = args.output / "original-labels" / label.name
        for remote, target, cap in (
            (path, image, 2_000_000),
            (f"train/labels/{image.stem}.txt", original_label, 100_000),
        ):
            if target.exists():
                content = target.read_bytes()
                digest = hashlib.sha1(
                    f"blob {len(content)}\0".encode() + content, usedforsecurity=False
                ).hexdigest()
                if digest == blob_hashes[remote]:
                    continue
            with urllib.request.urlopen(f"{REPO}{COMMIT}/{remote}", timeout=30) as response:
                content = response.read(cap + 1)
            if len(content) > cap:
                raise ValueError("Arquivo excede limite da amostra")
            target.write_bytes(content)
        converted = []
        for line in original_label.read_text().splitlines():
            parts = line.split()
            if not parts or parts[0] != "1":
                continue
            values = [float(v) for v in parts[1:]]
            if len(values) == 4:
                converted.append(values)
            elif len(values) >= 6 and len(values) % 2 == 0:
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                    raise ValueError("Polígono fora da imagem; precisa de revisão")
                xs, ys = values[0::2], values[1::2]
                converted.append(
                    [
                        (min(xs) + max(xs)) / 2,
                        (min(ys) + max(ys)) / 2,
                        max(xs) - min(xs),
                        max(ys) - min(ys),
                    ]
                )
            else:
                raise ValueError("Anotação desconhecida; precisa de revisão")
        # Class 1 is the actual route-number display. Exclude fleet numbers and Thai destinations.
        label.write_text(
            "\n".join("0 " + " ".join(str(v) for v in values) for values in converted) + "\n"
        )
        validate_labels(label, 1)
        return {
            "image": name,
            "label": label.name,
            "group_id": family,
            "source": SOURCE,
            "license": "CC BY 4.0",
        }

    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(download, selected))
    with (args.output / "manifest.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "repository": REPO,
                "commit": COMMIT,
                "count": len(rows),
                "seed": 42,
                "class_mapping": {"Bus Line Number (1)": "destination_sign (0)"},
                "limitation": (
                    "One augmentation per source photo; video/vehicle identity unavailable"
                ),
                "manifest_sha256": sha256(args.output / "manifest.csv"),
            },
            indent=2,
        )
    )
    print(f"{len(rows)} imagens e labels: {args.output / 'manifest.csv'}")


if __name__ == "__main__":
    main()
