"""Tests for recognition.paddle_recognizer — construction/validation only,
no model loading (that's covered by the live pipeline smoke test, not unit
tests, since it requires multi-GB model downloads)."""

import pytest

from recognition.paddle_recognizer import PaddleOcrRecognizer, LANG_CODES


def test_valid_scripts_construct_without_loading_model():
    for script in ("english", "tamil", "hindi"):
        recognizer = PaddleOcrRecognizer(script)
        assert recognizer.script == script
        assert recognizer.lang_code == LANG_CODES[script]
        assert recognizer._ocr is None  # lazy -- not loaded yet


def test_invalid_script_raises_valueerror():
    with pytest.raises(ValueError):
        PaddleOcrRecognizer("klingon")
