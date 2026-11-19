"""Cached category discovery; metadata first, never bulk image downloads."""

import json
import re
from pathlib import Path
from urllib.parse import urlencode

import httpx

from sonar_vision.br.curation import BoundedSourceClient, CollectionStopped

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "artifacts/research/generalization-20261116"
CATEGORIES = [
    "Articulated buses in São Paulo",
    "Bi-articulated buses in São Paulo",
    "Trolleybuses in São Paulo (state)",
    "Buses in Belo Horizonte",
    "Buses in Curitiba",
    "Buses in Porto Alegre",
    "Buses in Recife",
    "Buses in Salvador",
]


def clean(text: str) -> str:
    return re.sub("<[^>]*>", "", text).strip()


def main(categories: list[str] | None = None, prefix: str = "") -> None:
    base = json.loads((CACHE / "cache-candidates.json").read_text())
    known = set(base["known_ids"])
    groups = set(base["prior_groups"])
    groups.add("Wilfredor|2014-08-24")
    client = httpx.Client(headers={"User-Agent": "SonarVision-Academic/0.1"}, timeout=40)
    source = BoundedSourceClient(CACHE / "network", client, max_requests=64)
    candidates = {}
    for index, category in enumerate(categories or CATEGORIES):
        file = CACHE / f"{prefix}category-{index}.json"
        if file.exists():
            body = json.loads(file.read_text())
        else:
            params = {
                "action": "query",
                "format": "json",
                "generator": "categorymembers",
                "gcmtitle": "Category:" + category,
                "gcmnamespace": 6,
                "gcmtype": "file",
                "gcmlimit": 40,
                "prop": "imageinfo",
                "iiprop": "url|size|extmetadata",
                "iiurlwidth": 960,
            }
            try:
                body = json.loads(
                    source.fetch(
                        "https://commons.wikimedia.org/w/api.php?" + urlencode(params),
                        max_bytes=2_500_000,
                    )
                )
            except CollectionStopped:
                break
            file.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n")
        for page in body.get("query", {}).get("pages", {}).values():
            identity = str(page["pageid"])
            if identity in known or page.get("ns") != 6 or not page.get("imageinfo"):
                continue
            info = page["imageinfo"][0]
            meta = info.get("extmetadata", {})
            author = clean(meta.get("Artist", {}).get("value", ""))
            if "Rafael Delazari" in author:
                author = "Rafael Delazari"
            date = clean(meta.get("DateTimeOriginal", {}).get("value", ""))
            match = re.search(r"\d{4}-\d{2}-\d{2}", date)
            if not match:
                continue
            group = author + "|" + match.group()
            if group in groups:
                continue
            license = meta.get("LicenseShortName", {}).get("value", "")
            credit = clean(meta.get("Credit", {}).get("value", ""))
            cats = meta.get("Categories", {}).get("value", "")
            if not author or license not in {
                "CC0",
                "CC BY 2.0",
                "CC BY-SA 2.0",
                "CC BY-SA 3.0",
                "CC BY-SA 4.0",
                "CC BY 4.0",
            }:
                continue
            if not ("Own work" in credit or "Flickr images reviewed" in cats):
                continue
            if not info.get("thumburl") or info["width"] < 800:
                continue
            candidates[identity] = {
                "id": identity,
                "title": page["title"],
                "source": info["descriptionurl"],
                "category_source": category,
                "author": author,
                "date": date,
                "group_id": group,
                "license": license,
                "license_url": meta.get("LicenseUrl", {}).get("value", ""),
                "credit": credit,
                "categories": cats,
                "description": clean(meta.get("ImageDescription", {}).get("value", "")),
                "preview_url": info["thumburl"],
                "original_url": info["url"],
                "original_wh": [info["width"], info["height"]],
                "original_bytes": info["size"],
                "rights_status": (
                    "candidate chain verified in metadata; "
                    "visual/watermark consistency review required"
                ),
            }
    (CACHE / (prefix + "new-candidates.json")).write_text(
        json.dumps({"records": list(candidates.values())}, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        "metadata candidates",
        len(candidates),
        "groups",
        len({r["group_id"] for r in candidates.values()}),
    )
    for r in candidates.values():
        print(r["id"], r["group_id"], r["title"], r["category_source"])


if __name__ == "__main__":
    main()
