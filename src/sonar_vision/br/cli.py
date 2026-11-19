"""Local experiment commands; no API changes or automatic model promotion."""

import argparse
import json
import logging
from pathlib import Path

from sonar_vision.br.dataset import (
    audit_training_readiness,
    prepare_reviewed_dataset,
    reserved_test_hashes,
)
from sonar_vision.br.evaluation import EvaluationProtocol, compare_same_dataset, evaluate_model
from sonar_vision.br.training import TrainingPlan, run_training

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reserved", type=Path, action="append", default=[])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["prepare", "audit", "train", "evaluate", "compare"]:
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name in {"audit", "evaluate"}:
            command.add_argument("--data", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--review", type=Path, required=True)
        if name == "train":
            command.add_argument("--plan", type=Path, required=True)
            command.add_argument("--smoke", action="store_true")
            command.add_argument("--resume", type=Path)
        if name == "evaluate":
            command.add_argument("--weights", type=Path, required=True)
            command.add_argument("--protocol", type=Path, required=True)
            command.add_argument("--split", choices=["val", "test"], default="test")
        if name == "compare":
            command.add_argument("--left", type=Path, required=True)
            command.add_argument("--right", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    if args.command in {"prepare", "audit", "train"} and not args.reserved:
        parser.error("--reserved prior frozen manifests are required")
    reserved = reserved_test_hashes(args.reserved)
    if args.command in {"audit", "compare"} and args.output.exists():
        parser.error("Output exists; preserve previous evidence")
    if args.command == "prepare":
        prepare_reviewed_dataset(args.review, args.output, reserved)
    elif args.command == "audit":
        result = audit_training_readiness(args.data, reserved)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    elif args.command == "train":
        plan = TrainingPlan.model_validate_json(args.plan.read_text())
        run_training(plan, args.output, reserved, smoke=args.smoke, resume=args.resume)
    elif args.command == "evaluate":
        protocol = EvaluationProtocol.model_validate_json(args.protocol.read_text())
        evaluate_model(args.weights, args.data, args.output, protocol, split=args.split)
    else:
        result = compare_same_dataset(
            json.loads(args.left.read_text()), json.loads(args.right.read_text())
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    LOGGER.info("experiment_command_completed command=%s output=%s", args.command, args.output)


if __name__ == "__main__":
    main()
