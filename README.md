# Multilingual OCR Detection System

An end-to-end OCR system supporting English, Tamil, and Hindi, built on
PaddleOCR PP-OCRv5. Includes a FastAPI web demo (`webapp/`) and a full
research writeup comparing this production architecture against an earlier,
now-retired pipeline (`legacy_v1/`) — see **Research Findings** below.

---

## Overview

Given an image and a `--script` choice (`english`, `tamil`, or `hindi`), the
pipeline detects and recognizes every text region and returns structured
results (bounding box, text, confidence score).

**Honest accuracy, measured on real handwriting, not a marketing number:**

| Script | Real dataset | n | CER | Exact match |
|---|---|---|---|---|
| English | `Teklia/IAM-line` | 30 | 24.1% | 0/30 |
| Hindi | `IIIT-INDIC-HW-WORDS-Hindi` | 30 | 47.7% | 2/30 |
| Tamil | `IIIT-INDIC-HW-WORDS-Tamil` | 30 | 71.1% | 1/30 |

These numbers are *lower* than PaddleOCR's own official per-language
benchmarks (en 85.25%, ta 94.2%, Devanagari 84.96%, on PaddleOCR's own test
sets) — and that gap, measured honestly rather than hidden, is this project's
main research finding. Full detail in `docs/research_contribution.md`.

---

## Architecture

```
Input Image
   -> [--script selects language]
   -> [PaddleOCR PP-OCRv5: detection + recognition, one call]
   -> Structured results (bbox, text, confidence) per region
   -> JSON + plain-text output
```

This replaced a more complex V1 pipeline (EAST detector -> geometric grouping
-> printed/handwritten classifier -> per-language recognizer, with Hindi
needing an entirely separate detection backend because EAST produced zero
detections for Devanagari). V1 is preserved, not deleted, in `legacy_v1/` —
see `legacy_v1/README.md` and `docs/research_contribution.md` for why keeping
it mattered: the V1-vs-V2 comparison is the project's research evidence, and
deleting V1 would make that comparison unverifiable.

---

## Project Structure

```
├── main.py                        # CLI entry point
├── config.py / config.yaml        # script/paths/logging config
├── recognition/
│   ├── registry.py                # route registration pattern
│   └── paddle_recognizer.py       # the only recognizer in production
├── webapp/                        # FastAPI backend + web UI
│   ├── app.py
│   └── static/index.html
├── legacy_v1/                     # preserved V1 architecture (not production)
│   └── README.md                  # explains what it is and why it's kept
├── tests/                         # production test suite
├── legacy_v1/tests/               # V1 regression tests (still passing)
├── experiments/runs/              # evaluation data + CER reports (gitignored)
├── docs/                          # audit, plan, research writeup (gitignored)
├── models/frozen_east_text_detection.pb   # only needed by legacy_v1/
└── input/images/sample.png
```

---

## Installation

```
git clone https://github.com/ripuns/Multilingual-OCR-Detection.git
cd Multilingual-OCR-Detection
python -m venv ocr_env
ocr_env\Scripts\activate          # Windows
source ocr_env/bin/activate       # Linux/Mac

pip install -r requirements.txt           # production app
pip install -r requirements-dev.txt       # pytest, datasets (for tests/eval)
pip install -r requirements-webapp.txt    # FastAPI, uvicorn (for the demo)
```

`requirements-legacy.txt` is only needed to run `legacy_v1/` (adds a full
PyTorch/transformers install) — not needed for the production app.

PaddleOCR downloads its model weights automatically on first use per
language (cached locally after that).

---

## Usage

### CLI

```
python main.py --script english
python main.py --input path/to/image.png --script tamil --output-dir out/
```

- `--input` — path to the input image (default: `input/images/sample.png`)
- `--output-dir` — where results/crops are written (default: `output/`)
- `--config` — path to the config file (default: `config.yaml`)
- `--script` — `english` (default), `tamil`, or `hindi`. An explicit per-run
  choice, not automatic script detection — see `docs/limitations.md`.

### Web demo

```
pip install -r requirements-webapp.txt
uvicorn webapp.app:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000`. All 3 languages' models are pre-warmed at
startup so the first real request isn't slow. The UI has:
- **User mode** — clean recognized-text view with Copy/Share.
- **Dev mode** — full detail: bounding boxes, crops, per-region confidence.
- **Research Findings tab** — the V1-vs-V2 comparison and real CER numbers,
  rendered from `/api/research`.

---

## Output

`output/ocr_results.json` — one entry per detected region:
`index`, `bbox` (`[x1, y1, x2, y2]`), `text`, `score` (PaddleOCR's own
recognition confidence), `route` (which model produced it).
`output/ocr_results.txt` — plain recognized text, one region per line.
`output/cropped/` — cropped region images.

---

## Research Findings

The short version: this project went through a real engineering process —
build a pipeline, find real bugs, fix them, measure the fix, realize the
architecture itself was the bigger problem, replace it, and then discover
(by actually testing against real handwriting instead of clean synthetic
samples) that the new, better architecture still has a large, honestly-
measured accuracy gap against real-world data. Both halves are documented:

- `docs/research_contribution.md` — the full V1 (severity-gradient diagnosis)
  and V2 (real-dataset evaluation, synthetic-vs-real gap) writeup.
- `docs/limitations.md` — what not to claim, read before presenting any
  number from this project.
- `docs/dataset_and_license_inventory.md` — exactly what data/models were
  used, their license status, and what's `REQUIRES_REVIEW`.
- `docs/patent_ip_assessment.md` — why this is a paper-track contribution,
  not a patent claim (researched and explicitly decided against).

---

## Limitations

- No automatic script detection — `--script` is an explicit per-run choice.
- Real-world CER (24-71% depending on script) is substantially higher than
  PaddleOCR's own official benchmarks — do not quote the official numbers as
  this project's accuracy.
- n=30 real samples per non-English language — enough for a real signal and
  a genuine difficulty ordering, not a statistically robust benchmark.
- `legacy_v1/` requires a separate, much heavier dependency install and is
  not maintained going forward.

Full detail in `docs/limitations.md`.

---

## Acknowledgements

- PaddleOCR / PaddlePaddle (Baidu), Apache 2.0
- `Teklia/IAM-line` (Hugging Face mirror of the IAM Handwriting Database)
- `c3rl/IIIT-INDIC-HW-WORDS-{Tamil,Hindi}` (CVIT, IIIT Hyderabad — Gongidi & Jawahar)
- V1 architecture: OpenCV (EAST), HuggingFace Transformers, Microsoft TrOCR,
  `ocr_tamil` (GnanaPrasath, MIT), EasyOCR (JaidedAI, Apache 2.0)

---

## Author

Ripun
IT Engineering, VIT Vellore

---

## Note

This project is an academic/portfolio demonstration. Accuracy numbers above
are real and reproducible (`experiments/runs/`) but reflect a small-scale
evaluation, not a production-certified benchmark — see `docs/limitations.md`
before relying on them for any decision.
