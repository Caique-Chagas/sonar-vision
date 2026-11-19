import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from sonar_vision.br.dataset import audit_training_readiness, prepare_reviewed_dataset
from sonar_vision.br.evaluation import EvaluationProtocol, compare_same_dataset
from sonar_vision.br.training import TrainingPlan, check_initialization, run_training
from sonar_vision.training.dataset import sha256


@pytest.fixture
def br_dataset(tmp_path):
    records = []
    for index in range(30):
        image = tmp_path / f"capture-{index}.png"
        cv2.imwrite(
            str(image), np.random.default_rng(index).integers(0, 255, (48, 100, 3), np.uint8)
        )
        records.append(
            {
                "image": image.name,
                "image_sha256": sha256(image),
                "reviewed": True,
                "visual_evidence": "fixture, not real vision evidence",
                "rights_verified": True,
                "rights_evidence": "test fixture",
                "author": "test",
                "license": "CC0",
                "source": "https://example.org/test",
                "group_id": str(index),
                "group_evidence": "separate fixture",
                "split": "train" if index < 20 else "val" if index < 25 else "test",
                "boxes": [
                    {"xyxy": [10, 10, 90, 25], "approved": True, "boundary_evidence": "fixture"}
                ],
            }
        )
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "geometry_policy": "whole_main_front_destination_display",
                "records": records,
                "reviewer": "fixture",
            }
        )
    )
    prepared = tmp_path / "version1"
    prepare_reviewed_dataset(review, prepared, set())
    return prepared / "prepared/data.yaml", review


@pytest.fixture
def br_plan(br_dataset, tmp_path):
    data, _ = br_dataset
    generic = tmp_path / "generic.pt"
    generic.write_bytes(b"generic fixture")
    protected = tmp_path / "bangkok.pt"
    protected.write_bytes(b"protected fixture")
    config = tmp_path / "train.yaml"
    config.write_text(yaml.safe_dump({"model": str(generic), "epochs": 2}))
    return TrainingPlan(
        generic_weights=generic,
        generic_sha256=sha256(generic),
        protected_weights=protected,
        protected_sha256=sha256(protected),
        data=data,
        data_audit_sha256=sha256(data.parent / "audit.json"),
        config=config,
        config_sha256=sha256(config),
    )


def test_eligible_reviewed_dataset(br_dataset):
    data, _ = br_dataset
    assert audit_training_readiness(data, set())["training_eligible"]


def test_reserved_image_rejected_before_export(br_dataset, tmp_path):
    _, review = br_dataset
    digest = json.loads(review.read_text())["records"][0]["image_sha256"]
    with pytest.raises(ValueError, match="reserved"):
        prepare_reviewed_dataset(review, tmp_path / "forbidden", {digest})
    assert not (tmp_path / "forbidden").exists()


def test_changed_review_rejected(br_dataset):
    data, review = br_dataset
    review.write_text(review.read_text() + "\n")
    with pytest.raises(ValueError, match="review"):
        audit_training_readiness(data, set())


def test_prior_test_cannot_enter_train(br_dataset):
    data, review = br_dataset
    digest = json.loads(review.read_text())["records"][0]["image_sha256"]
    with pytest.raises(ValueError, match="reserved"):
        audit_training_readiness(data, {digest})


def test_negative_does_not_count_as_positive_group(br_dataset):
    data, _ = br_dataset
    audit_path = data.parent / "audit.json"
    audit = json.loads(audit_path.read_text())
    for record in audit["records"]:
        if record["split"] == "train":
            image = data.parent / record["prepared_image"]
            label = image.parent.parent / "labels" / (image.stem + ".txt")
            label.write_text("")
            record.update(boxes=0, label_sha256=sha256(label))
    audit_path.write_text(json.dumps(audit))
    assert not audit_training_readiness(data, set())["training_eligible"]


@pytest.mark.parametrize("field", ["generic_weights", "protected_weights", "config"])
def test_changed_frozen_input_rejected(br_plan, field):
    getattr(br_plan, field).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        check_initialization(br_plan)


def test_specialized_initialization_rejected(br_plan):
    invalid = br_plan.model_copy(
        update={
            "generic_weights": br_plan.protected_weights,
            "generic_sha256": br_plan.protected_sha256,
        }
    )
    with pytest.raises(ValueError, match="differ"):
        check_initialization(invalid)


def test_training_uses_shared_adapter_and_preserves_original(br_plan, tmp_path):
    calls = []

    def trainer(data, config, output, **kwargs):
        calls.append((data, config, kwargs))
        output.mkdir()
        result = output / "best.pt"
        result.write_bytes(b"fixture result")
        return result

    output = tmp_path / "run"
    result = run_training(br_plan, output, set(), smoke=True, trainer=trainer)
    assert result.exists() and calls[0][2]["smoke"] is True
    assert sha256(br_plan.protected_weights) == br_plan.protected_sha256
    with pytest.raises(ValueError, match="Existing"):
        run_training(br_plan, output, set(), trainer=trainer)


def test_failed_training_preserves_plan_and_propagates(br_plan, tmp_path):
    def trainer(data, config, output, **kwargs):
        output.mkdir()
        raise RuntimeError("backend failed")

    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="backend failed"):
        run_training(br_plan, output, set(), trainer=trainer)
    assert (output / "sonar-br-plan.json").exists()


def test_resume_outside_run_rejected(br_plan, tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "sonar-br-plan.json").write_text(br_plan.model_dump_json())
    with pytest.raises(ValueError, match="belong"):
        run_training(br_plan, output, set(), resume=br_plan.protected_weights)


def test_resume_requires_original_plan(br_plan, tmp_path):
    with pytest.raises(ValueError, match="same saved"):
        run_training(br_plan, tmp_path / "run", set(), resume=Path("missing.pt"))


def test_comparison_requires_same_protocol_and_data():
    left = {
        "model_sha256": "a",
        "dataset_audit_sha256": "data",
        "split": "test",
        "protocol": EvaluationProtocol().model_dump(),
        "images": 5,
        "panels": 4,
        "precision": 0.1,
        "recall": 0.2,
        "map50": 0.3,
        "map50_95": 0.1,
    }
    right = left | {"model_sha256": "b", "map50": 0.4}
    assert compare_same_dataset(left, right)["right_minus_left"]["map50"] == pytest.approx(0.1)
    for key, value in [
        ("dataset_audit_sha256", "other"),
        ("split", "val"),
        ("images", 6),
        ("protocol", {}),
    ]:
        with pytest.raises(ValueError, match="Incompatible"):
            compare_same_dataset(left, right | {key: value})


def test_cli_accepts_repeated_reserved_flags(br_dataset, tmp_path, monkeypatch):
    from sonar_vision.br.cli import main

    data, _ = br_dataset
    empty = tmp_path / "prior.json"
    empty.write_text('{"records": []}')
    output = tmp_path / "readiness.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "sonar-br",
            "--reserved",
            str(empty),
            "--reserved",
            str(empty),
            "audit",
            "--data",
            str(data),
            "--output",
            str(output),
        ],
    )
    main()
    assert json.loads(output.read_text())["training_eligible"]


def test_cli_refuses_missing_reservations(tmp_path, monkeypatch):
    from sonar_vision.br.cli import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "sonar-br",
            "audit",
            "--data",
            "missing.yaml",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(SystemExit):
        main()


@pytest.mark.parametrize(
    "text,expected,match",
    [
        ("8677/10 JD LIBANO", "8677-10", True),
        ("8677-10", "8677-10", True),
        ("679 GROTAO", "679", True),
        ("1679", "679", False),
        ("8677-11", "8677-10", False),
        ("BATERIA", "679", False),
    ],
)
def test_raw_code_metric_does_not_invent_characters(text, expected, match):
    from sonar_vision.br.observations import code_visible_in_ocr

    assert code_visible_in_ocr(text, expected) is match


def test_instrumentation_preserves_pipeline_contract(frame, sign, settings):
    from sonar_vision.br.observations import RecordedDetector, RecordedOCR, observation_routes
    from sonar_vision.core.contracts import OCRReading
    from sonar_vision.processing.pipeline import Context, FrameProcessor

    class Backend:
        model_version = "fixture"

        def detect(self, frame):
            return [sign]

        def read(self, roi):
            return [OCRReading(text="679", confidence=0.99)]

    detector, ocr = RecordedDetector(Backend()), RecordedOCR(Backend())
    processor = FrameProcessor(
        detector, ocr, Context("679", "test", "test", "local_video"), settings
    )
    _, events = processor.process(frame, 0, 0.0)
    assert detector.model_version == "fixture"
    assert len(ocr.calls) == 1 and detector.detections == [sign]
    assert events[0].status == "UNCERTAIN"
    assert observation_routes(processor)[0]["normalization"]["candidates"] == ("679",)
