"""Acquire only explicitly visually accepted originals; record identity and byte hashes."""

import hashlib
import json
from pathlib import Path

import httpx
from PIL import Image

from sonar_vision.br.curation import BoundedSourceClient, CollectionStopped

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "artifacts/research/generalization-20261116"
ACCEPTED = {
    "69139074": "SPTrans5109-10 small frontal windshield LED",
    "53751365": "SPTrans large windshield-integrated module, destination only",
    "22865593": "Curitiba front module BOQUEIRAO, display technology to verify",
    "189553067": "Porto Alegre F04 full main LED, occluded lower body",
    "80062761": "Porto Alegre distant front sign; tone-mapped scene, no code truth",
    "47259073": "Porto Alegre green bus frontal small LED, code943 to verify",
    "64802621": "Recife frontal small LED, code650 to verify; surrounding text hard negatives",
}


def main(accepted: dict[str, str] | None = None, prefix: str = "") -> None:
    selected = [
        r
        for r in json.loads((CACHE / (prefix + "preview-receipt.json")).read_text())["records"]
        if r["id"] in (accepted or ACCEPTED)
    ]
    folder = ROOT / "sonar_br/datasets/raw/generalization-20261116"
    folder.mkdir(exist_ok=True)
    old = json.loads(
        (
            ROOT / "artifacts/research/v2-diagnosis-20261116/existing-image-inventory.json"
        ).read_text()
    )
    old_hashes = {r["sha256"] for r in old["files"]}
    client = httpx.Client(headers={"User-Agent": "SonarVision-Academic/0.1"}, timeout=45)
    source = BoundedSourceClient(CACHE / "network", client, max_requests=64)
    for r in selected:
        r["visual_evidence"] = (accepted or ACCEPTED)[r["id"]]
        r["independent_human_review"] = False
        r["rights_evidence"] = (
            "Own-work declaration OR named original Flickr with license review; "
            "visual credit consistency reviewed; retain attribution/license/modification history"
        )
        r["split"] = "curation_unassigned_NO_INFERENCE"
        file = folder / (r["id"] + ".jpg")
        if not file.exists():
            if r["original_bytes"] > 12_000_000:
                r["status"] = "not acquired: exceeds per-file original budget"
                continue
            try:
                body = source.fetch(r["original_url"], max_bytes=12_000_000)
            except CollectionStopped:
                break
            if hashlib.sha256(body).hexdigest() in old_hashes:
                r["status"] = "not saved: identical existing bytes"
                continue
            with file.open("xb") as stream:
                stream.write(body)
        with Image.open(file) as im:
            r["actual_wh"] = list(im.size)
            im.verify()
        r["image"] = str(file.relative_to(ROOT))
        r["image_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
        r["bytes"] = file.stat().st_size
        r["status"] = "original acquired after visual preview; geometry/annotation review pending"
        print(r["id"], r["group_id"], r["bytes"], flush=True)
    (CACHE / (prefix + "acquisition.json")).write_text(
        json.dumps({"records": selected}, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
