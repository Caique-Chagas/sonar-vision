"""Export reviewed V2 data once; never derive test labels from a detector."""

import json
from pathlib import Path

from sonar_vision.br.dataset import audit_training_readiness, prepare_reviewed_dataset
from sonar_vision.br.lineage import validate_lineage

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "sonar_br/datasets/manifests/br-v2.review.json"
OUTPUT = ROOT / "sonar_br/datasets/prepared/br-v2"


def main() -> None:
    audit = json.loads((ROOT / "reports/br-v2-inbox-audit.json").read_text())
    current = {r["id"]: r for r in audit["records"]}
    review = json.loads(REVIEW.read_text())
    prior = []
    manifests = [
        "sonar_br/datasets/manifests/commons-br-v2.review.json",
        "sonar_br/datasets/manifests/final50.review.json",
        "sonar_br/datasets/manifests/ocr-opt-v1.review.json",
        "reports/datasets/profile-D-independent-20261115.review.json",
        "reports/datasets/commons-br-panel-v1.review.json",
    ]
    for name in manifests:
        for source in json.loads((ROOT / name).read_text())["records"]:
            r = dict(source)
            author = "Rafael Delazari" if "Rafael Delazari" in r["author"] else r["author"]
            date = r.get("date") or current.get(r["id"], {}).get("date")
            r["group_id"] = author + "|" + (date[:10] if date else "UNKNOWN")
            r["split"] = r.get("split", "test")
            prior.append(r)
    lineage = validate_lineage(review["records"], prior)
    reserved = {r["image_sha256"] for r in prior if r["split"] not in {"train", "val"}}
    if not OUTPUT.exists():
        prepare_reviewed_dataset(REVIEW, OUTPUT, reserved)
    readiness = audit_training_readiness(OUTPUT / "prepared/data.yaml", reserved)
    result = {"lineage": lineage, "readiness": readiness, "reserved_hashes": sorted(reserved)}
    (ROOT / "reports/br-v2-readiness.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result | {"reserved_hashes": len(reserved)}, indent=2))


if __name__ == "__main__":
    main()
