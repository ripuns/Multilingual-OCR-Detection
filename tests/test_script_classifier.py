import json

import pytest

from recognition import script_classifier as sc
from recognition.script_classifier import ScriptClassifier, extract_features, pick_script_calibrated


def reg(text, score=0.9):
    return {"index": 0, "bbox": [0, 0, 1, 1], "text": text, "score": score, "route": "fake"}


def test_feature_vector_matches_declared_names_and_is_length_invariant():
    short = {"english": [reg("ab")], "tamil": [reg("ab")], "hindi": [reg("ab")]}
    long = {s: [reg("ab" * 30)] for s in sc.SCRIPTS}
    assert len(extract_features(short)) == len(sc.FEATURE_NAMES)
    assert extract_features(short) == pytest.approx(extract_features(long))


def test_empty_engine_sets_empty_flag():
    f = dict(zip(sc.FEATURE_NAMES, extract_features({"english": [reg("abc")], "tamil": [], "hindi": []})))
    assert f["tamil_empty"] == 1.0 and f["english_empty"] == 0.0


def test_no_text_at_all_returns_none():
    clf = ScriptClassifier([0] * 15, [1] * 15, [[0] * 15] * 3, [0, 0, 0])
    picked, info = pick_script_calibrated({"english": [], "tamil": [], "hindi": []}, clf)
    assert picked is None and info["probabilities"] is None


def test_out_of_script_garbage_is_still_classified():
    clf = ScriptClassifier([0] * 15, [1] * 15, [[0] * 15] * 3, [0, 0, 0])
    picked, info = pick_script_calibrated({"english": [reg("123")], "tamil": [], "hindi": []}, clf)
    assert picked is not None and info["share"] == 0.0 and info["probabilities"] is not None


def test_probabilities_sum_to_one_and_pick_argmax():
    w = [[0.0] * 15 for _ in range(3)]
    idx = sc.FEATURE_NAMES.index("tamil_share")
    w[1][idx] = 5.0
    clf = ScriptClassifier([0] * 15, [1] * 15, w, [0, 0, 0])
    picked, info = pick_script_calibrated({"english": [reg("ab", 0.3)], "tamil": [reg("வணக்கம்")], "hindi": []}, clf)
    assert picked == "tamil"
    assert sum(info["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_shipped_weights_match_feature_set():
    d = json.load(open(sc.WEIGHTS_PATH, encoding="utf-8"))
    assert d["features"] == sc.FEATURE_NAMES and d["classes"] == list(sc.SCRIPTS)
    ScriptClassifier.load()


@pytest.mark.parametrize("variant,path", [("base", sc.WEIGHTS_PATH), ("ft", sc.WEIGHTS_PATH_FT), ("ft2", sc.WEIGHTS_PATH_FT2)])
def test_every_weight_variant_loads_and_resolves_to_its_file(variant, path):
    d = json.load(open(path, encoding="utf-8"))
    assert d["features"] == sc.FEATURE_NAMES and d["classes"] == list(sc.SCRIPTS)
    sc._DEFAULT.pop(variant, None)
    assert sc.load_default(variant) is not None
