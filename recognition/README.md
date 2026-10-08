# recognition/

## What
Runs text recognition, routing to the correct backend by language/script.
Three backends exist today: TrOCR (English printed/handwritten), `ocr_tamil`
(Tamil, EAST-cropped), and EasyOCR (Hindi, self-detecting — see Why below).

## Why
Different scripts/styles need different recognition models, and sometimes
different *detection* too: EAST (used for English and Tamil) produces zero
bounding boxes for Devanagari text at any confidence threshold tested, down to
0.05 — not low-confidence boxes that got filtered, an empty result at every
threshold. See `docs/research_contribution.md` for the full measurement. So
Hindi can't just plug a new recognizer into the existing EAST-crop pipeline;
it needs its own detection stage entirely.

## How
Two distinct contracts exist, because of the Devanagari finding above:

- **Crop-based recognizers** (`recognize(image, label) -> (text, route)`):
  take an already-detected, already-cropped region from EAST. Used by English
  and Tamil.
- **Self-detecting recognizers** (`detect_and_recognize(image) ->
  [{bbox, text, route}, ...]`): take the *whole* image and own their own
  detection. Used by Hindi, since EAST doesn't work for it.

`main.py` picks which contract to use per `--script` value via two small
dicts (`CROP_BASED_RECOGNIZERS`, `SELF_DETECTING_RECOGNIZERS`) — English and
Tamil still run EAST detection + grouping + clamping exactly as before; Hindi
skips all three and calls the self-detecting recognizer directly on the full
image.

- `registry.py` — a route registry independent of any specific recognizer
  class: `register(name, model_id)` adds a route, `get_route(name)` looks one
  up (raises `KeyError` with the list of known routes if missing),
  `registered_routes()` lists what's registered.
- `trocr_recognizer.py` — `TrOCRRecognizer(device="auto")` registers the two
  default English routes (`printed` -> `microsoft/trocr-large-printed`,
  `handwritten` -> `microsoft/trocr-large-handwritten`) at import time, then
  loads a route's processor/model **lazily** on first use.
- `ocr_tamil_recognizer.py` — `OcrTamilRecognizer` registers the `tamil`
  route, wraps the `ocr_tamil` PyPI package (CRAFT+PARSEQ), which takes an
  image *path* — the crop is written to a temp file internally and removed
  after the call.
- `easyocr_hindi_recognizer.py` — `EasyOcrHindiRecognizer` registers the
  `hindi` route, wraps `easyocr.Reader(['hi','en'])`. Implements
  `detect_and_recognize()`, not `recognize()` — it does its own CRAFT-based
  detection on the full image and returns every detected region's bbox/text.

## Structure
- `registry.py` — `register()` / `get_route()` / `registered_routes()`.
- `trocr_recognizer.py` — `TrOCRRecognizer`, English, crop-based.
- `ocr_tamil_recognizer.py` — `OcrTamilRecognizer`, Tamil, crop-based.
- `easyocr_hindi_recognizer.py` — `EasyOcrHindiRecognizer`, Hindi, self-detecting.

## Summary
Started as a hardcoded dict (both English models loaded unconditionally in
`__init__`); grew into a registry supporting lazy loading and genuinely
different backends per route; then had to grow a second contract
(self-detecting) once Hindi revealed that EAST itself — not just the
recognizer — doesn't transfer to every script. That escalation is itself part
of the project's measured finding, not incidental plumbing — see
`docs/research_contribution.md`. License/scope verification for any newly
registered model is documented in `docs/dataset_and_license_inventory.md`, not
enforced by this module.
