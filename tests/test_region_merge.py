from recognition.region_merge import line_bands, merge_by_line
from recognition.script_detector import pick_script


def reg(text, y1, y2, x1=0, x2=100, score=0.9):
    return {"index": 0, "bbox": [x1, y1, x2, y2], "text": text, "score": score, "route": "fake"}


def test_line_bands_group_overlapping_boxes_from_different_engines():
    by = {"english": [reg("a", 0, 20), reg("b", 100, 120)], "tamil": [reg("c", 2, 22), reg("d", 98, 118)], "hindi": []}
    assert line_bands(by) == [[0, 22], [98, 120]]


def test_each_line_gets_its_own_script():
    by = {
        "english": [reg("hello world", 0, 20), reg("xq", 100, 120, score=0.2)],
        "tamil": [reg("hello world", 0, 20), reg("வணக்கம்", 100, 120)],
        "hindi": [reg("hello", 0, 20), reg("zz", 100, 120, score=0.2)],
    }
    regions, det = merge_by_line(by, pick_script)
    assert det["mode"] == "mixed"
    assert [b["script"] for b in det["bands"]] == ["english", "tamil"]
    assert [(x["text"], x["script"]) for x in regions] == [("hello world", "english"), ("வணக்கம்", "tamil")]
    assert [x["index"] for x in regions] == [0, 1]


def test_band_without_in_script_text_falls_back_to_english_and_flags_it():
    by = {"english": [reg("123", 0, 20)], "tamil": [], "hindi": []}
    regions, det = merge_by_line(by, pick_script)
    assert det["fallback"] is True
    assert regions[0]["script"] == "english"
