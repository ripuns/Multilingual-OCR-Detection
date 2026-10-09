# webapp/

## What
A FastAPI backend and a single-file web UI around `main.run_pipeline`.

## Why
Lets a reviewer upload an image and read the result without the CLI, and gives the research
findings and their caveats a visible home (Research tab).

## How
Run: `pip install -r requirements-webapp.txt` then
`uvicorn webapp.app:app --host 127.0.0.1 --port 8000`, open `http://127.0.0.1:8000`.

- `app.py` - FastAPI app.
  - `lifespan` startup calls `main.warm_up()` (loads English, Tamil and Hindi engines); `/api/health`
    exposes the warm state, which the UI shows as a status pill.
  - `POST /api/ocr` (multipart `file` + `script` in `auto|english|tamil|hindi`, default `auto`) saves
    the upload to `uploads/{run_id}/input.<ext>` (extension whitelisted; the client filename is never
    used), runs the pipeline and returns `{run_id, requested_script, script, detection, results[],
    elapsed_seconds, was_warm}`; each result has a `crop_url`.
  - `GET /api/crop/{run_id}/{n}.png` - strict regex validation on both parameters.
  - `GET /api/sample` - the fixed sample page, for the UI's "Try sample" button.
  - `GET /api/research` - hand-maintained JSON copy of the research findings for the UI.
  - Static files served from `static/`.
- `static/index.html` - one dependency-free file (no build, no external fonts/scripts; works
  offline; responsive to ~390 px; light/dark). Upload by drag-drop, click, paste or sample; **Auto-detect**
  default plus English/Tamil/Hindi; loading state with timer and Cancel; friendly error states;
  **Simple** view (text, Copy / Download .txt / Share, low-confidence marking, "Read as <language>"
  re-run) and **Developer** view (boxes drawn on the image with hover linking, evidence bars, region
  table with crops and confidence pills, raw JSON); **Research** tab. Full behaviour description:
  `docs/architecture.md` Section 3.3 and the root README Section 5.5.
- `ui_tests/` - `mock_server.py` (serves the real UI with canned API responses; the 16-region result is
  the recorded real output for the sample page) and `ui_test.py` (Playwright driving an installed
  Edge/Chrome; 32 checks). Not part of `pytest`. Setup: `pip install playwright`; run
  `python webapp/ui_tests/mock_server.py` in one terminal and `python webapp/ui_tests/ui_test.py` in
  another (needs `experiments/runs/tamil_pilot/images/sample_0.png` for the Tamil step; set `SHOTS_DIR`
  to choose the screenshot folder, `UI_URL` to point at another server).
- `uploads/` - per-request working directory (gitignored, never cleaned automatically).

## Structure
- `app.py`
- `static/index.html`
- `ui_tests/mock_server.py`, `ui_tests/ui_test.py`
- `uploads/` (runtime)

## Summary
Deliberately thin: no authentication, database, rate limiting, upload-size limit (the UI caps files at
15 MB client-side only) or deployment configuration; CORS is `*`. Full API reference:
`docs/architecture.md` Section 3.1.
