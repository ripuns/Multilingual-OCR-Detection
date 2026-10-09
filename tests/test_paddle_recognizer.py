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


def test_sort_reading_order_orders_lines_then_left_to_right():
    from recognition.paddle_recognizer import sort_reading_order

    def r(text, x1, y1, x2, y2):
        return {"index": -1, "bbox": [x1, y1, x2, y2], "text": text}

    # 'sutra' slightly higher than 'kshar' on the same line, listed first.
    regions = [
        r("sutra", 400, 95, 600, 205),
        r("kshar", 100, 100, 350, 200),
        r("title", 50, 10, 650, 60),
        r("footer", 100, 300, 500, 350),
    ]
    ordered = sort_reading_order(regions)
    assert [x["text"] for x in ordered] == ["title", "kshar", "sutra", "footer"]
    assert [x["index"] for x in ordered] == [0, 1, 2, 3]
