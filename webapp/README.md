# webapp/

## What
A FastAPI backend + single-page web UI wrapping `main.py`'s pipeline for a
live, interactive OCR demo.

## Why
Lets a judge/reviewer upload a real image and see results immediately,
without using the CLI, and gives the research findings a visible home instead
of being buried in `docs/`.

## How
- `app.py` — FastAPI app. Pre-warms all 3 language models at startup
  (`lifespan`) so the first real request isn't slow. `/api/ocr` (POST,
  multipart file + `script` field) runs the pipeline and returns structured
  results with a `crop_url` per region. `/api/crop/{run_id}/{n}.png` serves
  the cropped region images (path-validated against a strict regex).
  `/api/research` returns the V1-vs-V2 research findings as JSON, duplicated
  from `docs/research_contribution.md` (not parsed from the markdown) so the
  frontend has a stable shape to render.
- `static/index.html` — single-file frontend (no build step, no framework).
  Two toggle states: **User mode** (clean text + Copy/Share) and **Dev mode**
  (bounding boxes, crops, per-region confidence score). A **Research** tab
  renders `/api/research`'s data as readable tables/bars.
- `uploads/` — per-request working directory (`{run_id}/input.*`,
  `cropped/*.png`, `ocr_results.*`), gitignored, not meant to persist.

## Structure
- `app.py`
- `static/index.html`

## Summary
Deliberately thin: no auth, no database, no deployment config — a demo
surface for `main.py`'s pipeline, not a separate product. Run with:
`pip install -r requirements-webapp.txt && uvicorn webapp.app:app --host 127.0.0.1 --port 8000`.
