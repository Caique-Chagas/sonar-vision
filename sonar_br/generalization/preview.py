"""Fetch a small explicit shortlist of licensed previews; stop on source restrictions."""

import hashlib
import json
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

from sonar_vision.br.curation import BoundedSourceClient, CollectionStopped

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "artifacts/research/generalization-20261116"
SHORTLIST = [
    "69139074",
    "53751365",
    "175933592",
    "22913004",
    "22865593",
    "189553067",
    "80062761",
    "47259073",
    "86415548",
    "64802621",
    "25952697",
]


def main(shortlist: list[str] | None = None, prefix: str = "") -> None:
    refs = json.loads((CACHE / (prefix + "new-candidates.json")).read_text())["records"]
    selected = [r for r in refs if r["id"] in (shortlist or SHORTLIST)]
    client = httpx.Client(headers={"User-Agent": "SonarVision-Academic/0.1"}, timeout=40)
    source = BoundedSourceClient(CACHE / "network", client, max_requests=64)
    for r in selected:
        file = CACHE / (r["id"] + "-preview.jpg")
        if not file.exists():
            try:
                body = source.fetch(r["preview_url"], max_bytes=2_500_000)
            except CollectionStopped:
                break
            file.write_bytes(body)
        with Image.open(file) as im:
            im.verify()
        r["preview_path"] = str(file.relative_to(ROOT))
        r["preview_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
        r["visual_status"] = "pending agent visual review; NOT training annotation"
    (CACHE / (prefix + "preview-receipt.json")).write_text(
        json.dumps({"records": selected}, ensure_ascii=False, indent=2) + "\n"
    )
    for batch in range((len(selected) + 5) // 6):
        sheet = Image.new("RGB", (1600, 1350), "white")
        draw = ImageDraw.Draw(sheet)
        for n, r in enumerate(selected[batch * 6 : batch * 6 + 6]):
            if not r.get("preview_path"):
                continue
            im = Image.open(ROOT / r["preview_path"])
            im.thumbnail((790, 390))
            x = n % 2 * 800
            y = n // 2 * 450
            sheet.paste(im, (x, y))
            draw.text((x, y + 395), r["id"] + " " + r["group_id"][:62], fill="black")
            draw.text((x, y + 416), r["title"][:95], fill="black")
        sheet.save(CACHE / f"{prefix}preview-sheet-{batch}.jpg")
    print("previews obtained", sum("preview_path" in r for r in selected))


if __name__ == "__main__":
    main()
