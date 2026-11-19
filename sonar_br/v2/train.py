"""Bounded generic V2 training with frozen inputs and shared resume support."""

import argparse
import json
from pathlib import Path

from sonar_vision.br.training import TrainingPlan, run_training
from sonar_vision.training.dataset import sha256

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "sonar_br/configs/training-plan-v2.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not PLAN.exists():
        baseline = json.loads((ROOT / "sonar_br/configs/training-plan-v1.json").read_text())
        baseline.update(
            data="sonar_br/datasets/prepared/br-v2/prepared/data.yaml",
            config="sonar_br/configs/train-v2.yaml",
            data_audit_sha256=sha256(ROOT / "sonar_br/datasets/prepared/br-v2/prepared/audit.json"),
            config_sha256=sha256(ROOT / "sonar_br/configs/train-v2.yaml"),
        )
        PLAN.write_text(json.dumps(baseline, indent=2) + "\n")
    plan = TrainingPlan.model_validate_json(PLAN.read_text())
    reserved = set(
        json.loads((ROOT / "reports/br-v2-readiness.json").read_text())["reserved_hashes"]
    )
    v1 = ROOT / "sonar_br/artifacts/train-v1/run/weights/best.pt"
    if sha256(v1) != "5978a1571bd3ee03b3ddd48625917e64db8db4bcd8ab53a4eb22d18fe5db412a":
        raise ValueError("Protected BRv1 changed")
    out = ROOT / ("sonar_br/artifacts/smoke-v2" if args.smoke else "sonar_br/artifacts/train-v2")
    result = run_training(plan, out, reserved, smoke=args.smoke, resume=args.resume)
    print(result)
    if sha256(v1) != "5978a1571bd3ee03b3ddd48625917e64db8db4bcd8ab53a4eb22d18fe5db412a":
        raise RuntimeError("Protected BRv1 changed")


if __name__ == "__main__":
    main()
