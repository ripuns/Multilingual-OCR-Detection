from recognition.script_detector import pick_script, script_mass


def region(text, score):
    return {"text": text, "score": score}


def test_script_mass_counts_only_in_block_characters():
    # 'abc' Latin (3) + digits/space (ignored)
    assert script_mass("english", [region("abc 123", 1.0)]) == 3
    assert script_mass("tamil", [region("abc 123", 1.0)]) == 0


def test_script_mass_is_confidence_weighted():
    assert script_mass("english", [region("abcd", 0.5)]) == 2.0


def test_script_mass_recognizes_tamil_and_devanagari_blocks():
    assert script_mass("tamil", [region("வணக்கம்", 1.0)]) == len("வணக்கம்")
    assert script_mass("hindi", [region("नमस्ते", 1.0)]) == len("नमस्ते")
    assert script_mass("hindi", [region("வணக்கம்", 1.0)]) == 0


def test_pick_script_chooses_script_with_most_in_block_mass():
    results = {
        "english": [region("xq", 0.2)],
        "tamil": [region("வணக்கம்", 0.9)],
        "hindi": [],
    }
    script, info = pick_script(results)
    assert script == "tamil"
    assert info["share"] > 0.9


def test_pick_script_ignores_out_of_block_garbage_from_wrong_engine():
    # Hindi engine emits Latin letters on an English image: contributes 0 to hindi.
    results = {
        "english": [region("Hello world", 0.99)],
        "tamil": [],
        "hindi": [region("Hello", 0.8)],
    }
    script, _ = pick_script(results)
    assert script == "english"


def test_pick_script_returns_none_when_no_in_block_text():
    results = {"english": [region("123 !!", 0.9)], "tamil": [], "hindi": []}
    script, info = pick_script(results)
    assert script is None
    assert info["share"] == 0.0


def test_pick_script_handles_empty_results():
    script, _ = pick_script({"english": [], "tamil": [], "hindi": []})
    assert script is None
