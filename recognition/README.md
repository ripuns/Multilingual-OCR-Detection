# recognition/

## What
Runs detection + recognition for all 3 supported languages (English, Tamil,
Hindi) through a single backend, selected by the registry.

## Why
V1 (preserved in `legacy_v1/`) used three separate libraries with three
different interfaces (TrOCR via `transformers`, `ocr_tamil` via its own
package, EasyOCR via its own package), and needed a second recognizer
contract once it turned out EAST couldn't detect Devanagari at all — see
`docs/research_contribution.md`. V2 replaces all of that with one engine,
PaddleOCR PP-OCRv5, which has its own per-language detection and recognition
models and handles all 3 languages through the same API.

## How
- `registry.py` — unchanged pattern from V1: `register(name, model_id)` /
  `get_route(name)` / `registered_routes()`.
- `paddle_recognizer.py` — `PaddleOcrRecognizer(script)` registers all 3
  routes (`english` -> `en`, `tamil` -> `ta`, `hindi` -> `hi`) at import time.
  `detect_and_recognize(image_path)` is the only method — PaddleOCR owns
  detection internally, so there's no separate crop-based contract anymore
  (V1 needed two: crop-based for EAST-compatible scripts, self-detecting for
  Hindi). Returns a list of `{index, bbox, text, score, route}` per detected
  region, including PaddleOCR's own recognition confidence score.
  `enable_mkldnn=False` works around a PaddlePaddle/oneDNN crash
  (`NotImplementedError` on `ConvertPirAttribute2RuntimeAttribute`) observed
  on this project's CPU during evaluation.

## Structure
- `registry.py` — `register()` / `get_route()` / `registered_routes()`.
- `paddle_recognizer.py` — `PaddleOcrRecognizer`, the only recognizer in
  production; registers all 3 language routes as an import-time side effect.

## Summary
V1 went from a hardcoded dict, to a registry with lazy loading, to a registry
supporting two different recognizer contracts (crop-based vs self-detecting)
once Hindi revealed EAST itself didn't transfer to every script. V2 replaces
all of that complexity with one engine that never needed a separate
detection stage to begin with. The real cost of that simplicity is
accuracy on real handwriting, not architecture — see
`docs/research_contribution.md` for the measured numbers (24.1%-71.1% CER
depending on script, real datasets) and `docs/limitations.md` for what that
means in practice.
