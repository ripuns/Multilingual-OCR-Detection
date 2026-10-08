"""Tests for recognition.registry — route registration/lookup, no model loading."""

import pytest

from recognition import registry


def test_register_and_get_route():
    registry.register("test_dummy_route", "some/model-id")
    assert registry.get_route("test_dummy_route") == "some/model-id"


def test_get_route_missing_raises_keyerror():
    with pytest.raises(KeyError):
        registry.get_route("does_not_exist_route")


def test_registered_routes_includes_default_routes():
    import recognition.trocr_recognizer  # noqa: F401 -- side effect: registers default routes

    routes = registry.registered_routes()
    assert "printed" in routes
    assert "handwritten" in routes


def test_default_routes_point_to_expected_models():
    import recognition.trocr_recognizer  # noqa: F401

    assert registry.get_route("printed") == "microsoft/trocr-large-printed"
    assert registry.get_route("handwritten") == "microsoft/trocr-large-handwritten"


def test_tamil_route_registered_on_import():
    import recognition.ocr_tamil_recognizer  # noqa: F401 -- side effect: registers "tamil"

    assert "tamil" in registry.registered_routes()
    assert "ocr_tamil" in registry.get_route("tamil")


def test_hindi_route_registered_on_import():
    import recognition.easyocr_hindi_recognizer  # noqa: F401 -- side effect: registers "hindi"

    assert "hindi" in registry.registered_routes()
    assert "easyocr" in registry.get_route("hindi")
