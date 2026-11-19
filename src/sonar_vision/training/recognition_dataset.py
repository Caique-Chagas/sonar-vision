"""Reviewed PaddleOCR corpora with explicit provenance and independent-test gates."""

import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import cv2

from sonar_vision.br.curation import canonical_author
from sonar_vision.processing.roi import Frame, validate_frame
from sonar_vision.training.dataset import dhash, sha256


@dataclass(frozen=True)
class RecognitionSample:
    image: Path
    transcript: str
    split: Literal["train", "val", "test"]
    parent_id: str
    group: str
    author: str
    source: str
    license: str
    image_sha256: str
    rights_verified: bool
    manual_verified: bool
    scope: Literal["known_development", "new_unseen", "historical_test", "reserved"]
    vehicle: str | None = None


def export_recognition(samples: list[RecognitionSample], output: Path) -> dict[str, Any]:
    """Export labels, retaining an empty test as empty; never authorize training."""
    if output.exists() or not samples:
        raise ValueError("Preserve existing version; nonempty reviewed corpus required")
    identities: dict[tuple[str, str], str] = {}
    digests: set[str] = set()
    perceptual: list[tuple[str, int]] = []
    train_codes = {s.transcript for s in samples if s.split == "train"}
    for sample in samples:
        if sample.rights_verified is not True or sample.manual_verified is not True:
            raise ValueError("Rights and pixel-verified transcription required")
        if sample.parent_id == "35046766" or sample.scope in {"reserved", "historical_test"}:
            raise ValueError("Reserved/historical evaluation source forbidden")
        if sample.split == "test" and sample.scope != "new_unseen":
            raise ValueError("Known development cannot become independent test")
        if sample.split == "test" and sample.transcript in train_codes:
            raise ValueError("Independent code benchmark requires unseen test codes")
        if (
            sample.split not in {"train", "val", "test"}
            or not sample.group
            or not sample.author
            or not sample.parent_id
            or not sample.license
            or not sample.source.startswith("https://")
        ):
            raise ValueError("Explicit source, author, capture and split required")
        if not re.fullmatch(r"[A-Z0-9]{2,8}(?:-[A-Z0-9]{1,3})?", sample.transcript) or not any(
            c.isdigit() for c in sample.transcript
        ):
            raise ValueError(
                "Exact code transcript required; no tab/newline or inferred characters"
            )
        keys = [
            ("parent", sample.parent_id),
            ("capture", sample.group),
            ("author", canonical_author(sample.author)),
        ]
        if sample.vehicle:
            keys.append(("vehicle", sample.vehicle))
        for key in keys:
            if key in identities and identities[key] != sample.split:
                raise ValueError("Parent/capture/author/vehicle crosses splits")
            identities[key] = sample.split
        digest = sha256(sample.image)
        if digest != sample.image_sha256 or digest in digests:
            raise ValueError("Changed or duplicate crop")
        digests.add(digest)
        image = cv2.imread(str(sample.image))
        validate_frame(cast(Frame, image))
        fingerprint = dhash(image)
        if any(
            split != sample.split and (fingerprint ^ h).bit_count() <= 3 for split, h in perceptual
        ):
            raise ValueError("Near-duplicate crop crosses splits")
        perceptual.append((sample.split, fingerprint))
    output.mkdir(parents=True)
    (output / "images").mkdir()
    lines: dict[str, list[str]] = {name: [] for name in ["train", "val", "test"]}
    rows = []
    for index, sample in enumerate(samples):
        relative = f"images/{index:04d}{sample.image.suffix.lower()}"
        shutil.copy2(sample.image, output / relative)
        lines[sample.split].append(relative + "\t" + sample.transcript)
        rows.append(
            {
                "image": relative,
                "sha256": sample.image_sha256,
                "transcript": sample.transcript,
                "split": sample.split,
                "parent_id": sample.parent_id,
                "group": sample.group,
                "author": sample.author,
                "vehicle": sample.vehicle,
                "source": sample.source,
                "license": sample.license,
                "scope": sample.scope,
            }
        )
    for split, entries in lines.items():
        (output / f"rec_gt_{split}.txt").write_text(
            "\n".join(entries) + ("\n" if entries else ""), encoding="utf-8"
        )
    counts = {name: len(entries) for name, entries in lines.items()}
    report = {
        "counts": counts,
        "training_authorized": False,
        "target30_test_samples_reached": counts["test"] >= 30,
        "test_capture_groups": len({s.group for s in samples if s.split == "test"}),
        "test_distinct_codes": len({s.transcript for s in samples if s.split == "test"}),
        "independent_exact_match": None,
        "vehicle_identity_unknown": sum(s.vehicle is None for s in samples),
        "unique_transcripts": len({s.transcript for s in samples}),
        "label_hashes": {name: sha256(output / f"rec_gt_{name}.txt") for name in lines},
        "scope_counts": dict(Counter(s.scope for s in samples)),
        "records": rows,
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def verify_recognition(output: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads((output / "audit.json").read_text())
    for split, digest in report["label_hashes"].items():
        if sha256(output / f"rec_gt_{split}.txt") != digest:
            raise ValueError("Recognition labels changed")
    for sample in report["records"]:
        if sha256(output / sample["image"]) != sample["sha256"]:
            raise ValueError("Recognition pixels changed")
    return report
