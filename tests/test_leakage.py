from recognition.leakage import is_latin_leak, suppress_leakage


def r(text, score, i=0):
    return {"index": i, "bbox": [0, 0, 10, 10], "text": text, "score": score}


def test_low_confidence_latin_from_indic_engine_is_leakage():
    assert is_latin_leak(r("BODA", 0.5), "tamil")
    assert is_latin_leak(r("Dutnt", 0.4), "hindi")


def test_confident_latin_or_indic_text_is_kept():
    assert not is_latin_leak(r("Parlophone", 0.95), "tamil")
    assert not is_latin_leak(r("வணக்கம்", 0.3), "tamil")
    assert not is_latin_leak(r("टाटा", 0.3), "hindi")


def test_mixed_region_with_own_script_is_kept():
    assert not is_latin_leak(r("அmi", 0.4), "tamil")


def test_english_engine_is_never_suppressed():
    assert not is_latin_leak(r("xq", 0.1), "english")


def test_suppress_returns_kept_and_dropped_and_renumbers():
    regions = [r("BODA", 0.4, 0), r("வணக்கம்", 0.9, 1), r("zz", 0.3, 2), r("எடுக்கலாம்", 0.8, 3)]
    kept, dropped = suppress_leakage(regions, "tamil")
    assert [x["text"] for x in kept] == ["வணக்கம்", "எடுக்கலாம்"]
    assert [x["index"] for x in kept] == [0, 1]
    assert [x["text"] for x in dropped] == ["BODA", "zz"]
