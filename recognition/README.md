# recognition/

## What
Runs text recognition (TrOCR) on a classified crop, routing to the correct
model by the classifier's label.

## Why
Different TrOCR checkpoints are trained for different text styles (printed
vs. handwritten). The classifier's label needs to select the right model
without the recognizer hardcoding that mapping.

## How
- `registry.py` — a route registry independent of any specific recognizer
  class: `register(name, model_id)` adds a route, `get_route(name)` looks one
  up (raises `KeyError` with the list of known routes if missing),
  `registered_routes()` lists what's registered. Adding a language/script
  route is a `register()` call; it does not require editing `main.py` or
  `TrOCRRecognizer`.
- `trocr_recognizer.py` — `TrOCRRecognizer(device="auto")` registers the two
  default routes (`printed` -> `microsoft/trocr-large-printed`, `handwritten`
  -> `microsoft/trocr-large-handwritten`) at import time, then loads a
  route's processor/model **lazily** on first use of that label
  (`recognize(image, label)`), caching it for subsequent calls. Only the
  routes actually exercised in a run get loaded into memory. `recognize()`
  returns `(text, route)`, where `route` is the model id the label resolved
  to — this is what `main.py` writes into `output/ocr_results.json`'s
  `route` field for traceability.

## Structure
- `registry.py` — `register()` / `get_route()` / `registered_routes()`.
- `trocr_recognizer.py` — `TrOCRRecognizer`, the registry-backed recognizer;
  registers the two default routes as an import-time side effect.

## Summary
Replaces a previously hardcoded dict (both models loaded unconditionally in
`__init__`) with a small registry plus lazy per-route loading. License/scope
verification for any newly registered model is a prerequisite documented in
`docs/dataset_and_license_inventory.md`, not enforced by this module.
