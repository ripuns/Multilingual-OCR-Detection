# recognition/

## What
Detection + recognition for the three supported languages, and the logic that chooses
the language automatically.

## Why
V1 used a different library, interface and (for Hindi) a different detection path per
language. V2 uses one PaddleOCR pipeline per language behind one method, and adds a
script identifier so users need not pick a language.

## How
- `registry.py` — `register(name, label)`, `get_route(name)` (raises `KeyError` listing
  known routes), `registered_routes()`.
- `paddle_recognizer.py` — `PaddleOcrRecognizer(script)` for `english`/`tamil`/`hindi`
  (`lang` codes `en`/`ta`/`hi`). Lazy: the PaddleOCR object is built on `warm_up()` or the
  first `detect_and_recognize(image_path)` call. Returns
  `[{index, bbox:[x1,y1,x2,y2], text, score, route}]`, sorted into reading order by
  `sort_reading_order` (lines top-to-bottom, left-to-right within a line, `index` re-numbered).
  PaddleOCR is created with document
  orientation/unwarping/textline-orientation disabled and `enable_mkldnn=False` (oneDNN crash
  workaround). `ENGINE_LABELS` holds the human-readable engine names registered in the
  registry; under `paddleocr==3.7.0` English resolves to PP-OCRv6 and Tamil/Hindi to PP-OCRv5.
- `script_detector.py` — pure functions: `script_mass(script, regions)` =
  sum(score x count of characters inside the script's Unicode block);
  `pick_script(results_by_script)` -> `(winner | None, {"masses", "share"})`: argmax of mass, with an
  **Indic override** (`INDIC_SCRIPTS`, `INDIC_MIN_SHARE = 0.25`): Tamil/Hindi win once they hold >= 25%
  of the total mass, because the Indic engines also read Latin words while the English engine garbles
  Indic text.
  Blocks: Tamil U+0B80-0BFF, Devanagari U+0900-097F, Latin letters A-Z/a-z.

## Structure
- `registry.py`
- `paddle_recognizer.py` — registers the three routes as an import-time side effect.
- `script_detector.py`

## Summary
`main.py` calls `PaddleOcrRecognizer.detect_and_recognize` for one engine (manual mode) or
for all three followed by `pick_script` (auto mode). See `docs/architecture.md` Section 4
for the algorithm and its limits, and `docs/research_contribution.md` Part 3 for measured
accuracy. The V1 recognizers (TrOCR, ocr_tamil, EasyOCR) live in `legacy_v1/recognition/`.
