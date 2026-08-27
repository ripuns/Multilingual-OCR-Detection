# tests/

## What
Automated test suite for the OCR pipeline, run with `pytest`.

## Why
The project had no tests prior to this suite, so fixes (e.g. the grouping
min/max bug) could not be verified or protected against regression.

## How
Run from the repository root with the project's virtual environment:

```
ocr_env/Scripts/python -m pytest tests/ -v
```

(`pytest` is listed in `requirements.txt` as a dev-only dependency.)

## Structure
- `test_grouping.py` — unit/regression/adversarial tests for
  [`../grouping/text_grouping.py`](../grouping/text_grouping.py), including
  the min/max bounding-box bug fix.
- `test_boxes.py` — unit tests for
  [`../boxes.py`](../boxes.py)'s `clamp_box()` (clamping and invalid/empty-box
  rejection).
- `test_registry.py` — unit tests for
  [`../recognition/registry.py`](../recognition/registry.py) (route
  register/lookup, miss behavior, default routes).
- `test_east_detector.py` — failure-path test for
  [`../detection/east_detector.py`](../detection/east_detector.py) (missing
  image raises `FileNotFoundError`); constructs the detector via
  `__new__` to skip loading the EAST model file.

## Summary
Grows incrementally alongside each fix/feature per the project's atomic
implementation rule — one test file is added or extended per logical change,
not written all at once.
