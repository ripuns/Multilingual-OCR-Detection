# webapp/

## What
A FastAPI backend and a single-file web UI around `main.run_pipeline`, for live
demonstration.

## Why
Lets a reviewer upload an image and see results without the CLI, and gives the research
findings a visible home (Research tab).

## How
Run: `pip install -r requirements-webapp.txt` then
`uvicorn webapp.app:app --host 127.0.0.1 --port 8000`, open `http://127.0.0.1:8000`.

- `app.py` — FastAPI app.
  - `lifespan` startup calls `main.warm_up()` (loads English, Tamil and Hindi engines) so the
    first request is fast; `/api/health` exposes the warm state.
  - `POST /api/ocr` (multipart `file` + `script` in `auto|english|tamil|hindi`, default `auto`)
    saves the upload to `uploads/{run_id}/input.<ext>` (extension whitelisted; the client
    filename is never used), runs the pipeline and returns `{run_id, requested_script, script,
    detection, results[], elapsed_seconds, was_warm}`; each result has a `crop_url`.
  - `GET /api/crop/{run_id}/{n}.png` — strict regex validation on both parameters.
  - `GET /api/research` — hand-maintained JSON copy of the research findings for the UI.
  - Static files served from `static/`.
- `static/index.html` — no build step. Script buttons (**Auto-detect** default, English, Tamil,
  Hindi), **User mode** (combined recognized text, Copy/Share) and **Dev mode** (per-region
  crops, confidence pill, engine label), a **detected-script chip** after auto runs (or a
  warning when no readable text was found), and a **Research** tab that renders
  `/api/research` (V1 failure table, real-data CER bars, vendor benchmarks, caveats).
- `uploads/` — per-request working directory (gitignored, never cleaned automatically).

## Structure
- `app.py`
- `static/index.html`
- `uploads/` (runtime)

## Summary
Deliberately thin: no authentication, database, rate limiting, upload-size limit or
deployment configuration; CORS is `*`. Full API reference: `docs/architecture.md` Section 3.1.
