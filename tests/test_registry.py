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
