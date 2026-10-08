# recognition/

## What
Runs text recognition on a crop, routing to the correct backend by label.
Two backends exist today: TrOCR (English printed/handwritten) and `ocr_tamil`
(Tamil).

## Why
Different scripts/styles need different recognition models, and those models
may have entirely different APIs (TrOCR via `transformers`, `ocr_tamil` via its
own package). The registry lets `main.py` stay agnostic to which backend a
label maps to.

## How
- `registry.py` — a route registry independent of any specific recognizer
  class: `register(name, model_id)` adds a route, `get_route(name)` looks one
  up (raises `KeyError` with the list of known routes if missing),
  `registered_routes()` lists what's registered. Adding a route is a
  `register()` call; it does not require editing `main.py`.
- `trocr_recognizer.py` — `TrOCRRecognizer(device="auto")` registers the two
  default English routes (`printed` -> `microsoft/trocr-large-printed`,
  `handwritten` -> `microsoft/trocr-large-handwritten`) at import time, then
  loads a route's processor/model **lazily** on first use of that label
  (`recognize(image, label)`), caching it for subsequent calls.
- `ocr_tamil_recognizer.py` — `OcrTamilRecognizer` registers the `tamil` route
  and wraps the `ocr_tamil` PyPI package (CRAFT detection + PARSEQ
  recognition), which takes an image *path*, not a PIL image — the crop is
  written to a temp file internally and removed after the call. Lazily
  constructs the underlying `ocr_tamil.ocr.OCR()` instance on first use.
- Both recognizer classes implement the same contract: `recognize(image,
  label) -> (text, route)`, where `route` is a descriptive string identifying
  what actually produced the text — this is what `main.py` writes into
  `output/ocr_results.json`'s `route` field for traceability.

`main.py`'s `--script {english,tamil}` picks which recognizer (and whether the
printed/handwritten classifier runs at all) for an entire run — see root
README's Configuration section. There is no automatic script detection; see
`docs/limitations.md`.

## Structure
- `registry.py` — `register()` / `get_route()` / `registered_routes()`.
- `trocr_recognizer.py` — `TrOCRRecognizer`, the English registry-backed
  recognizer; registers its two routes as an import-time side effect.
- `ocr_tamil_recognizer.py` — `OcrTamilRecognizer`, the Tamil recognizer;
  registers the `tamil` route as an import-time side effect.

## Summary
Started as a hardcoded dict (both English models loaded unconditionally in
`__init__`); now a small registry supporting lazy loading and genuinely
different backend implementations per route — proven by adding Tamil in one
evening with zero changes to `main.py`'s detection/grouping/classification
stages. License/scope verification for any newly registered model is a
prerequisite documented in `docs/dataset_and_license_inventory.md`, not
enforced by this module.
