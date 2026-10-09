# tests/

## What
The production test suite (pytest). `legacy_v1/tests/` holds the V1-era tests and runs
alongside it.

## Why
To protect the behavior that can be verified without large models: route registration,
recognizer construction/validation, and the script-identification scoring rule.

## How
```
pytest tests/ legacy_v1/tests/ -q        # expected: 41 passed (no model downloads, no network)
```

No test loads PaddleOCR or EAST. Model-dependent behavior is checked by the evaluation
scripts in `experiments/runs/` and by live requests against the web app.

## Structure (production, 20 tests)
- `test_registry.py` (4) — register/lookup, missing-route `KeyError`, all three scripts
  registered when `recognition.paddle_recognizer` is imported, and the exact engine labels
  (`PaddleOCR PP-OCRv6 (en)`, `PaddleOCR PP-OCRv5 (ta)`, `PaddleOCR PP-OCRv5 (hi)`).
- `test_paddle_recognizer.py` (3) — valid scripts construct lazily (model not loaded);
  an unsupported script raises `ValueError`; `sort_reading_order` groups lines, orders
  left-to-right and re-numbers `index`.
- `test_script_detector.py` (9) — in-script glyph counting (Latin/Tamil/Devanagari blocks),
  confidence weighting, argmax selection, out-of-block garbage ignored, the Indic override
  (Hindi wins on a mixed Hindi+English poster; English kept when the Indic share is small), and
  the "no evidence" cases (`None`, share 0).
- `test_main_recognize.py` (4) — `main.recognize` with fake recognizers: auto picks the right
  script and reuses its regions (each engine runs once), English fallback is flagged, manual
  runs only the chosen engine, masses/share are reported.

## Structure (legacy, 21 tests, `legacy_v1/tests/`)
- `test_grouping.py` (8) — grouping regression/adversarial cases; the left-drift cases
  fail on the pre-fix code.
- `test_boxes.py` (12) — `clamp_box` and `pad_and_clamp_box`.
- `test_east_detector.py` (1) — missing image raises `FileNotFoundError` (constructs the
  detector via `__new__` to avoid loading the model file).

## UI tests
The web UI has its own Playwright-based test in `webapp/ui_tests/` (mock API + 32 checks); it is not collected by
`pytest` because it needs a browser. See `webapp/README.md`.

## Summary
41 tests, all passing at the time of writing. Not covered by unit tests: file/crop writing in
`run_pipeline`, the FastAPI endpoints, and the frontend.
