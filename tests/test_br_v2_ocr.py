import numpy as np
import pytest

from sonar_vision.br.code_region_ocr import LeftCodeOCR
from sonar_vision.core.contracts import OCRReading
from sonar_vision.validation.normalizer import LineNormalizer


class OCR:
    def __init__(self, extra):
        self.calls = 0
        self.extra = extra

    def read(self, image):
        self.calls += 1
        return [OCRReading(text="DESTINO", confidence=0.99)] if self.calls == 1 else self.extra


@pytest.mark.parametrize("text", ["290", "675R", "675R-1", "675R-10 DESTINO", "0677/10"])
def test_extra_partial_or_unsafe_text_does_not_become_code(text):
    engine = OCR([OCRReading(text=text, confidence=0.99)])
    values = LeftCodeOCR(engine).read(np.zeros((40, 400, 3), np.uint8))
    assert [v.text for v in values] == ["DESTINO"]
    assert engine.calls == 2


def test_complete_observed_code_retained_without_truth_or_catalog():
    engine = OCR([OCRReading(text="675R-10", confidence=0.99)])
    assert LeftCodeOCR(engine).read(np.zeros((40, 400, 3), np.uint8))[-1].text == "675R-10"


def test_low_confidence_extra_rejected():
    engine = OCR([OCRReading(text="675R-10", confidence=0.84)])
    assert len(LeftCodeOCR(engine).read(np.zeros((40, 400, 3), np.uint8))) == 1


def test_small_non_panel_roi_has_no_extra_calls():
    engine = OCR([])
    LeftCodeOCR(engine).read(np.zeros((40, 60, 3), np.uint8))
    assert engine.calls == 1


def test_short_city_code_requires_explicit_catalog_and_does_not_supply_missing_digit():
    assert LineNormalizer({"01"}).normalize("01").route == "01"
    assert LineNormalizer({"2290"}).normalize("290").route is None
    assert LineNormalizer({"8677-10"}).normalize("0677/10").route is None
    assert LineNormalizer({"475H-10"}).normalize("475H").route is None
