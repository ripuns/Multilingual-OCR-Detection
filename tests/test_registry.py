"""Tests for recognition.registry — route registration/lookup, no model loading."""

import pytest

from recognition import registry


def test_register_and_get_route():
    registry.register("test_dummy_route", "some/model-id")
    assert registry.get_route("test_dummy_route") == "some/model-id"


def test_get_route_missing_raises_keyerror():
    with pytest.raises(KeyError):
        registry.get_route("does_not_exist_route")


def test_all_three_scripts_registered_on_import():
    import recognition.paddle_recognizer  # noqa: F401 -- side effect: registers all 3 scripts

    routes = registry.registered_routes()
    assert "english" in routes
    assert "tamil" in routes
    assert "hindi" in routes


def test_registered_routes_identify_paddleocr_with_correct_lang_code():
    import recognition.paddle_recognizer  # noqa: F401

    assert registry.get_route("english") == "PaddleOCR PP-OCRv5 (en)"
    assert registry.get_route("tamil") == "PaddleOCR PP-OCRv5 (ta)"
    assert registry.get_route("hindi") == "PaddleOCR PP-OCRv5 (hi)"
