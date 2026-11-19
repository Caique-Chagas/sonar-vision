"""Rebind absolute audit paths on a new checkout without changing images/labels/splits."""

import argparse
import json
from pathlib import Path

from sonar_vision.br.dataset import audit_training_readiness, reserved_test_hashes
from sonar_vision.br.training import TrainingPlan, check_initialization
from sonar_vision.training.dataset import sha256, verify_dataset

ROOT = Path(__file__).resolve().parents[2]
INDEX_FIELDS = ["sha256", "label_sha256", "group_id", "split", "boxes"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Local plan exists; preserve it")
    plan = TrainingPlan.model_validate_json(
        (ROOT / "sonar_br/configs/training-plan-v1.json").read_text()
    )
    audit = verify_dataset(plan.data)
    expected = json.loads((ROOT / "sonar_br/reports/prepared-index.json").read_text())["records"]
    actual = [{key: record[key] for key in INDEX_FIELDS} for record in audit["records"]]
    if actual != expected:
        raise ValueError("Restored dataset differs in pixels, labels, groups or splits")
    reserved = reserved_test_hashes(
        [
            ROOT / "reports/datasets/commons-br-panel-v1.review.json",
            ROOT / "reports/datasets/profile-D-independent-20261115.review.json",
        ]
    )
    readiness = audit_training_readiness(plan.data, reserved)
    if not readiness["training_eligible"]:
        raise ValueError("Restored dataset is not eligible")
    local_plan = plan.model_copy(
        update={
            "data_audit_sha256": sha256(plan.data.parent / "audit.json"),
        }
    )
    check_initialization(local_plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(local_plan.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
