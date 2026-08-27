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

(`pytest` must be installed in `ocr_env` — it is a dev-only dependency, not
yet listed in `requirements.txt`.)

## Structure
- `test_grouping.py` — unit/regression/adversarial tests for
  [`../grouping/text_grouping.py`](../grouping/text_grouping.py), including
  the min/max bounding-box bug fix.

## Summary
Grows incrementally alongside each fix/feature per the project's atomic
implementation rule — one test file is added or extended per logical change,
not written all at once.
