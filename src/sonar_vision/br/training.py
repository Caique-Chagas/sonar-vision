"""Generic-initialized training with explicit provenance and protected Bangkok weights."""

import json
from pathlib import Path
from typing import Any, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field

from sonar_vision.br.dataset import audit_training_readiness
from sonar_vision.training.dataset import sha256
from sonar_vision.training.runner import train


class TrainingPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    generic_weights: Path
    generic_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    protected_weights: Path
    protected_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    data: Path
    data_audit_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    config: Path
    config_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class Trainer(Protocol):
    def __call__(
        self,
        data: Path,
        config: Path,
        output: Path,
        *,
        device: str | None = None,
        smoke: bool = False,
        resume: Path | None = None,
    ) -> Path: ...


def check_initialization(plan: TrainingPlan) -> None:
    if plan.generic_weights.resolve() == plan.protected_weights.resolve():
        raise ValueError("Brazil initialization must differ from protected Bangkok weights")
    checks = [
        (plan.generic_weights, plan.generic_sha256),
        (plan.protected_weights, plan.protected_sha256),
        (plan.data.parent / "audit.json", plan.data_audit_sha256),
        (plan.config, plan.config_sha256),
    ]
    if any(sha256(path) != expected for path, expected in checks):
        raise ValueError("Frozen training input changed")
    config = yaml.safe_load(plan.config.read_text())
    if Path(config["model"]).resolve() != plan.generic_weights.resolve():
        raise ValueError("Training config does not initialize the frozen generic model")
    if plan.generic_sha256 == plan.protected_sha256:
        raise ValueError("Specialized checkpoint cannot initialize SONAR BR")


def run_training(
    plan: TrainingPlan,
    output: Path,
    reserved_hashes: set[str],
    *,
    smoke: bool = False,
    resume: Path | None = None,
    trainer: Trainer = train,
) -> Path:
    check_initialization(plan)
    readiness = audit_training_readiness(plan.data, reserved_hashes)
    if not readiness["training_eligible"]:
        raise ValueError("Dataset not eligible: " + "; ".join(readiness["reasons"]))
    if resume is not None:
        previous = output / "sonar-br-plan.json"
        if not previous.is_file() or json.loads(previous.read_text()) != plan.model_dump(
            mode="json"
        ):
            raise ValueError("Resume requires the same saved training plan")
        if not resume.resolve().is_relative_to(output.resolve()):
            raise ValueError("Resume checkpoint must belong to this run")
    elif output.exists():
        raise ValueError("Existing experiment must be resumed, never overwritten")
    try:
        result = trainer(plan.data, plan.config, output, smoke=smoke, resume=resume)
    finally:
        # The shared trainer records INTERRUPTED_OR_FAILED and optimizer checkpoints.
        # Preserve the plan even on native failures, then propagate the exception.
        if output.is_dir():
            receipt: dict[str, Any] = plan.model_dump(mode="json")
            (output / "sonar-br-plan.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if sha256(plan.protected_weights) != plan.protected_sha256:
            raise RuntimeError("Protected Bangkok weights changed during training")
    return result
