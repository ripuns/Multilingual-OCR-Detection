"""Tests for main.recognize() orchestration using fake recognizers (no models)."""

import main


class FakeRecognizer:
    def __init__(self, regions):
        self.regions = regions
        self.calls = 0

    def detect_and_recognize(self, image_path):
        self.calls += 1
        return [dict(r) for r in self.regions]


def region(text, score=0.9):
    return {"index": 0, "bbox": [0, 0, 1, 1], "text": text, "score": score, "route": "fake"}


def install(monkeypatch, per_script):
    fakes = {s: FakeRecognizer(regions) for s, regions in per_script.items()}
    monkeypatch.setattr(main, "_get_recognizer", lambda script: fakes[script])
    return fakes


def test_auto_picks_tamil_and_reuses_its_regions(monkeypatch):
    fakes = install(monkeypatch, {
        "english": [region("xq", 0.2)],
        "tamil": [region("வணக்கம்", 0.9)],
        "hindi": [],
    })
    regions, detection = main.recognize("img.png", "auto")
    assert detection["mode"] == "auto"
    assert detection["script"] == "tamil"
    assert detection["fallback"] is False
    assert regions[0]["text"] == "வணக்கம்"
    assert all(f.calls == 1 for f in fakes.values())  # each engine ran exactly once


def test_auto_falls_back_to_english_when_no_in_script_text(monkeypatch):
    install(monkeypatch, {"english": [region("123", 0.9)], "tamil": [], "hindi": []})
    regions, detection = main.recognize("img.png", "auto")
    assert detection["script"] == main.FALLBACK_SCRIPT
    assert detection["fallback"] is True


def test_manual_runs_only_the_chosen_engine(monkeypatch):
    fakes = install(monkeypatch, {
        "english": [region("hello")],
        "tamil": [region("வணக்கம்")],
        "hindi": [region("नमस्ते")],
    })
    regions, detection = main.recognize("img.png", "hindi")
    assert detection == {"mode": "manual", "script": "hindi", "fallback": False, "share": None, "masses": None}
    assert regions[0]["text"] == "नमस्ते"
    assert fakes["hindi"].calls == 1
    assert fakes["english"].calls == 0 and fakes["tamil"].calls == 0


def test_auto_detection_reports_masses_and_share(monkeypatch):
    install(monkeypatch, {"english": [region("Hello world", 1.0)], "tamil": [], "hindi": []})
    _, detection = main.recognize("img.png", "auto")
    assert detection["script"] == "english"
    assert detection["masses"]["english"] == 10.0  # letters in "Helloworld"
    assert detection["share"] == 1.0
