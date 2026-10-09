# legacy_v1/

## What
The complete original OCR pipeline: EAST text detector, a geometric
box-grouping heuristic, a printed/handwritten heuristic classifier, and three
separate recognition backends (TrOCR for English, `ocr_tamil` for Tamil,
EasyOCR for Hindi).

## Why
This code is **not the production system** — it was replaced by the unified
PaddleOCR-based pipeline at the repo root (`main.py`,
`recognition/paddle_recognizer.py`). It's kept here, not deleted, because
`docs/research_contribution.md`'s central comparison (V1's ad-hoc
multi-backend architecture vs. V2's unified engine) needs V1 to remain
runnable and reproducible. Deleting it would turn a verifiable before/after
comparison into an unverifiable claim.

## How
Fully self-contained: every internal import inside `legacy_v1/` is prefixed
`legacy_v1.` (e.g. `from legacy_v1.detection.east_detector import
EASTDetector`), including its own copy of `recognition/registry.py`, so it
never shares state with the production `recognition/registry.py` at the repo
root. Run it from the repo root:

```
python -m legacy_v1.main --script tamil --input path/to/image.png
```

It needs its own, much heavier dependency set — `pip install -r
requirements-legacy.txt` (PyTorch, transformers, `ocr_tamil`, EasyOCR) — not
installed by default with the production `requirements.txt`.

## Structure
- `main.py` — the original pipeline orchestrator (EAST -> grouping -> clamp
  -> classify -> recognize, with a second self-detecting path for Hindi).
- `config.py` / `config.yaml` — V1's config (detection thresholds, grouping
  tolerances, per-script crop padding).
- `boxes.py` — `clamp_box()` / `pad_and_clamp_box()`.
- `detection/`, `grouping/`, `classification/` — EAST wrapper, the
  min/max grouping fix, the heuristic classifier.
- `recognition/` — `trocr_recognizer.py`, `ocr_tamil_recognizer.py`,
  `easyocr_hindi_recognizer.py`, and this module's own `registry.py`.
- `tests/` — the V1-era regression tests (`test_boxes.py`,
  `test_east_detector.py`, `test_grouping.py`), runnable alongside the
  production test suite: `pytest tests/ legacy_v1/tests/`.

## Summary
A frozen snapshot, not a maintained codebase. Do not add new features here;
new work belongs in the production pipeline at the repo root. See
`docs/research_contribution.md` for what this snapshot is evidence of, and
`docs/limitations.md` for why it should not be used going forward.
