import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any, cast

import yaml

from sonar_vision.training.dataset import sha256, verify_dataset


def hardware() -> dict[str, Any]:
    import torch

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "ultralytics": importlib.metadata.version("ultralytics"),
    }


def train(
    data: Path,
    config: Path,
    output: Path,
    *,
    device: str | None = None,
    smoke: bool = False,
    resume: Path | None = None,
) -> Path:
    import ultralytics

    audit = verify_dataset(data)
    hw = hardware()
    device = device or ("0" if hw["cuda"] else "mps" if hw["mps"] else "cpu")
    if device == "cpu" and not smoke:
        raise ValueError("Treino completo em CPU desativado; use GPU/MPS ou smoke test")
    if output.exists() and resume is None:
        raise ValueError("Diretório de execução já existe; use --resume ou novo --output")
    args = yaml.safe_load(config.read_text())
    model_name = args.pop("model")
    if smoke:
        args.update(epochs=1, imgsz=320, batch=2, workers=0, plots=False)
    if device == "mps":
        args.update(batch=min(args.get("batch", 4), 4), workers=0, amp=False)
    output.mkdir(parents=True, exist_ok=True)
    evidence = {
        "purpose": "smoke_only" if smoke else "custom_training",
        "status": "STARTED",
        "dataset_audit_sha256": sha256(data.parent / "audit.json"),
        "split_assurance": audit["split_assurance"],
        "hardware": hw,
        "device": device,
        "args": args,
        "initialization": model_name,
        "resume_checkpoint": str(resume) if resume else None,
    }
    evidence_path = output / "evidence.json"
    if resume and evidence_path.exists():
        previous = json.loads(evidence_path.read_text())
        if previous["dataset_audit_sha256"] != evidence["dataset_audit_sha256"]:
            raise ValueError("Retomada exige a mesma versão auditada do dataset")
        evidence["previous_attempt"] = previous
    evidence_path.write_text(json.dumps(evidence, indent=2))
    try:
        if resume:
            # Ultralytics restores optimizer, epochs and original training arguments.
            model = cast(Any, ultralytics).YOLO(str(resume.resolve(strict=True)))
            model.train(resume=True, device=device)
        else:
            model = cast(Any, ultralytics).YOLO(model_name)
            model.train(
                data=str(data.resolve()),
                device=device,
                project=str(output.resolve()),
                name="run",
                exist_ok=False,
                **args,
            )
        run_dir = Path(model.trainer.save_dir)
        best = run_dir / "weights/best.pt"
        evidence.update(
            status="TRAINED",
            run_dir=str(run_dir),
            best=str(best),
            best_sha256=sha256(best),
            last=str(run_dir / "weights/last.pt"),
        )
        evidence_path.write_text(json.dumps(evidence, indent=2))
        return best
    except BaseException as exc:
        evidence.update(status="INTERRUPTED_OR_FAILED", error_category=type(exc).__name__)
        evidence_path.write_text(json.dumps(evidence, indent=2))
        raise


def evaluate_detector(
    weights: Path, data: Path, output: Path, device: str = "cpu"
) -> dict[str, Any]:
    import ultralytics

    audit = verify_dataset(data)
    model = cast(Any, ultralytics).YOLO(str(weights.resolve(strict=True)))
    metrics = model.val(
        data=str(data.resolve()),
        split="test",
        device=device,
        imgsz=640,
        batch=1,
        workers=0,
        project=str(output),
        name="detector_test",
    )
    result = {
        "scope": "detector_test",
        "samples": audit["counts"]["test"],
        "split_assurance": audit["split_assurance"],
        "model_sha256": sha256(weights),
        "dataset_audit_sha256": sha256(data.parent / "audit.json"),
        "hardware": hardware(),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "speed_ms": metrics.speed,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "detector_metrics.json").write_text(json.dumps(result, indent=2))
    return result
