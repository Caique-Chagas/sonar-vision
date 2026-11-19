from typing import Any

import numpy as np
import pytest

from sonar_vision.br.ocr_optimization import (
    BrazilianSeparatorNormalizer,
    ExperimentalOCR,
    OCRExperiment,
    preprocess,
    token_crops,
)
from sonar_vision.core.config import Settings
from sonar_vision.core.contracts import OCRReading
from sonar_vision.ocr.paddle import OCRFailure, PaddleEngine
from sonar_vision.validation.temporal import Observation, TemporalValidator


@pytest.mark.parametrize(
    "text,expected",
    [
        ("01 - MINA CAUE", "01"),
        ("1 - CASSIAS", "1"),
        ("2290 CENTRO", "2290"),
        ("8677/10 JD LIBANO", "8677-10"),
        ("6913 - 10", "6913-10"),
        ("X09 CENTRO", "X09"),
        ("AAA 217A/10 CENTRO", "217A-10"),
        ("1111/10", "1111-10"),
        ("A123-10", "A123-10"),
    ],
)
def test_separator_preserves_complete_explicit_code(text: str, expected: str) -> None:
    result = BrazilianSeparatorNormalizer({expected}).normalize(text)
    assert result.route == expected
    assert result.raw == text


@pytest.mark.parametrize("text", ["290", "01 - 10", "8677", "O1", "I", "S5", "", "DESTINO"])
def test_never_complete_or_substitute_characters(text: str) -> None:
    result = BrazilianSeparatorNormalizer({"2290", "01", "1", "55", "8677-10"}).normalize(text)
    assert result.route not in {"2290", "01", "1", "55", "8677-10"}


def test_different_suffix_not_repaired_to_requested() -> None:
    result = BrazilianSeparatorNormalizer({"6913-10"}).normalize("6913/11")
    assert result.route == "6913-11"


def test_single_static_observation_never_confirms() -> None:
    normalized = BrazilianSeparatorNormalizer({"01"}).normalize("01 - MINA CAUE")
    consensus = TemporalValidator(Settings(_env_file=None)).add(
        1, Observation(0, 0.0, normalized, 0.99, 0.99)
    )
    assert consensus.route is None


@pytest.mark.parametrize("mode", ["original", "scale2", "scale3", "gray", "clahe", "adaptive"])
def test_transform_does_not_mutate_original(mode: str) -> None:
    frame = np.full((64, 256, 3), 90, dtype=np.uint8)
    result = preprocess(frame, OCRExperiment(preprocessing=mode))
    assert result.dtype == np.uint8 and result.ndim == 3
    assert np.all(frame == 90)
    assert not np.shares_memory(frame, result)


def test_invalid_config_and_corrupt_frame_refused() -> None:
    with pytest.raises(ValueError):
        OCRExperiment(max_tokens=100)
    with pytest.raises(ValueError):
        preprocess(np.zeros((2, 3), dtype=np.uint8), OCRExperiment())


class FakeNative:
    def predict(self, image: Any) -> list[dict[str, Any]]:
        return [{"rec_texts": ["01 - MINA CAUE"], "rec_scores": [0.98], "dt_polys": []}]


def test_native_trace_preserved_and_empty_panel_bounded() -> None:
    adapter = ExperimentalOCR(PaddleEngine(FakeNative()), OCRExperiment())
    assert adapter.read(np.zeros((64, 256, 3), np.uint8)) == [
        OCRReading(text="01 - MINA CAUE", confidence=0.98)
    ]
    assert adapter.trace["native_results"][0]["texts"] == ["01 - MINA CAUE"]
    assert len(token_crops(np.zeros((64, 256, 3), np.uint8), 3)) <= 3


def test_backend_failure_cannot_become_reading() -> None:
    adapter = ExperimentalOCR(PaddleEngine(object()), OCRExperiment())
    with pytest.raises(OCRFailure):
        adapter.read(np.zeros((64, 256, 3), np.uint8))
