# Multilingual OCR — English, Tamil, Hindi, with automatic script detection

An OCR system that reads English, Tamil and Hindi (Devanagari) text from images and,
by default, **detects the script automatically** from the image. It is built on
PaddleOCR, ships with a CLI and a FastAPI web demo, and comes with an unusually
honest evaluation: accuracy is measured on **real third-party handwriting datasets**,
and the project's earlier architecture (V1) is preserved so the move to the current
one (V2) is reproducible.

> **How to read this document.** This README is deliberately complete and
> self-contained: `docs/` and `experiments/` are gitignored (local only), so every key
> fact, number, diagram and caveat needed to write a full technical report is repeated
> here. Section 17 is a chapter-by-chapter guide for writing a 30-40 page report from
> this file alone. Everything stated here is implemented or measured; anything not
> done is listed explicitly under Limitations and Future Work.

---

## Table of contents

1. [At a glance](#1-at-a-glance)
2. [Problem, motivation, objectives, scope](#2-problem-motivation-objectives-scope)
3. [Contributions and how strong each claim is](#3-contributions-and-how-strong-each-claim-is)
4. [System architecture (V2, production)](#4-system-architecture-v2-production)
5. [Detailed design](#5-detailed-design)
6. [Automatic script identification](#6-automatic-script-identification)
7. [The previous architecture (V1, `legacy_v1/`)](#7-the-previous-architecture-v1-legacy_v1)
8. [Implementation reference](#8-implementation-reference)
9. [Evaluation methodology](#9-evaluation-methodology)
10. [Results](#10-results)
11. [Analysis and discussion](#11-analysis-and-discussion)
12. [Testing](#12-testing)
13. [Installation and usage](#13-installation-and-usage)
14. [Related work and prior-art findings](#14-related-work-and-prior-art-findings)
15. [Limitations and threats to validity](#15-limitations-and-threats-to-validity)
16. [Project history and future work](#16-project-history-and-future-work)
17. [Guide for writing the technical report](#17-guide-for-writing-the-technical-report)
18. [Repository map, documentation index, glossary, acknowledgements](#18-repository-map-documentation-index-glossary-acknowledgements)

---

## 1. At a glance

| Item | Value |
|---|---|
| Languages / scripts | English (Latin), Tamil, Hindi (Devanagari) |
| Recognition engine | PaddleOCR (`paddleocr==3.7.0`, `paddlepaddle==3.3.1`): English -> PP-OCRv6 medium det+rec; Tamil -> PP-OCRv5 server det + `ta_PP-OCRv5_mobile_rec`; Hindi -> PP-OCRv5 server det + `devanagari_PP-OCRv5_mobile_rec`; the Tamil and Hindi recognisers ship with head-only fine-tuned versions (Sections 6.8-6.9) |
| Language selection | **Automatic by default** (`auto`, one script per image), **`mixed`** (script chosen per text line), manual override `english` / `tamil` / `hindi`. The automatic choice uses a small **calibrated classifier** (probabilities), with the earlier glyph-mass rule kept as a fallback |
| Interfaces | CLI (`main.py`), HTTP API + web UI (`webapp/`) |
| Web UI | Single-file, dependency-free, light/dark, responsive, redesigned (landing hero, three-step explainer, probability strip): drag-drop/paste/"Try sample" upload, Auto-detect default, **Simple** view (text, Copy/Download/Share, low-confidence marking, "Read as <language>" re-run) and **Developer** view (boxes drawn on the image, evidence bars, regions table, raw JSON), Research tab |
| Outputs | `ocr_results.json`, `ocr_results.txt`, `ocr_detection.json`, `cropped/*.png` |
| Runtime | CPU-only, single process; all three models warm-loaded at web-server start |
| Measured real-data accuracy (200 fresh images each) | English 14.0% CER (sentence lines), Hindi **16.1%** and Tamil **16.4%** (words) with the head-adapted recognisers (stock: 41.9% and 73.5%) — **not comparable across languages**; the Hindi and Tamil gains are adaptation to one handwriting dataset family, see Sections 10.8 and 10.12 |
| Script identification accuracy | **93.2% on 600 fresh images** (95% CI 91.0-95.0; English 196/200, Tamil 179/200, Hindi 184/200) with the calibrated classifier fitted for the adapted engines, against 55.5% for the earlier hand-written rule and 83.3% for taking the engine with the most in-script text; 91.0% with stock engines and stock weights; 83.3% on 60 short English words; 56/60 lines of synthetic mixed pages. See Sections 10.5 and 10.9-10.14 |
| Tests | 65 passing (44 production + 21 legacy), no models/network needed; plus the Playwright UI test (38 checks) |
| Language / platform | Python 3.11.4, Windows 11 (developed and measured on AMD Ryzen 5 5625U, 15.3 GB RAM, no GPU) |
| Codebase size | ~190 lines `main.py`, ~110 `paddle_recognizer.py`, ~50 `script_detector.py`, ~100 `script_classifier.py`, ~60 `region_merge.py`, ~30 `leakage.py`, ~250 `webapp/app.py`, ~990 `webapp/static/index.html` |
| Previous architecture | `legacy_v1/` (EAST + grouping + classifier + TrOCR / ocr_tamil / EasyOCR), preserved, runnable, tested |
| Author | Ripun, IT Engineering, VIT Vellore |

---

## 2. Problem, motivation, objectives, scope

### 2.1 Problem

Most OCR tools require the user to say which language the document is in, and
open-source OCR for Indic scripts (Tamil, Devanagari) is far less mature and less
evaluated than for English. A usable multilingual tool should (a) read the supported
scripts, (b) not demand a language choice, and (c) be honest about how accurate it
really is on real handwriting, which is much harder than clean printed text.

### 2.2 Project origin

The project began (March 2026) as an English prototype — EAST text detector, a
geometric word-grouping step, a printed/handwritten heuristic and TrOCR — that
worked on one sample image. It was then audited, repaired, extended to Tamil and
Hindi, evaluated, and finally re-architected.

### 2.3 Objectives

1. Fix correctness bugs found in the prototype and protect them with tests.
2. Support three languages through a clean, extensible architecture.
3. Measure accuracy on **real** data with a clearly defined metric.
4. Remove the need for the user to choose a language (automatic script detection).
5. Provide a presentable web demo and complete documentation.
6. Extract any honest, defensible research findings (journal-oriented), while
   explicitly not overclaiming novelty.

### 2.4 Non-goals / scope limits

Not a state-of-the-art recognizer (it wraps an existing engine); not a patentable
invention (investigated and dropped); not a deployed service (demo-grade web app); not
page-layout analysis, tables, or mixed-script per-region reading; not GPU-optimized.

---

## 3. Contributions and how strong each claim is

The contributions are **engineering and evaluation findings**, not a new algorithm.

| # | Contribution | Evidence | Strength |
|---|---|---|---|
| C1 | A grouping bug in the prototype silently clipped OCR input; fixing it recovered clipped text | 19 of 32 recognized lines gained text (none lost); regression tests fail before the fix and pass after | Solid for that pipeline/image; qualitative (no ground truth for the page) |
| C2 | In V1 the EAST detector failed differently per script: grouping bug (English), under-sized boxes (Tamil), zero boxes (Hindi) | English: C1; Tamil: 5 synthetic images; Hindi: 1 synthetic image at 2 thresholds | Suggestive; small and synthetic |
| C3 | Symmetric box padding is a cheap partial mitigation for the Tamil under-sizing (CER 46.3% -> 25.9% at 20 px; worse at 30 px) | 5 synthetic images | Suggestive (n=5) |
| C4 | Replacing three per-language backends with one PaddleOCR engine per language removed the detection failures and simplified the system | 16 coherent regions vs 32 fragments on the sample page; simpler code and one API | Solid as engineering; not a controlled accuracy comparison |
| C5 | On real handwriting the engines still err a lot: English 24.1%, Hindi 47.7%, Tamil 71.1% CER | 30 real images each | Real signal, small n; cross-language comparison invalid (different units) |
| C6 | A clean synthetic pilot made accuracy look far better than on real data (Hindi 7.1% vs 47.7%; Tamil 25.9% vs 71.1%) | V1 pilots vs V2 real-data runs | Supports an evaluation-practice caution only; engine and data both changed |
| C7 | "Script leakage": the Tamil engine outputs Latin letters on 17/30 real images (CER 0.91 on those vs 0.46 on the rest); Hindi 2/30 | per-sample analysis of C5 outputs | Solid for this sample; mechanism not isolated |
| C8 | Automatic script identification from recognizer outputs (argmax + Indic override), evaluated on 150 real images | 77.3% overall (English 96%, Hindi 86%, Tamil 50%); the Tamil misses are recognizer failures (no Tamil characters emitted) | Real signal, small n; works for English/Hindi, unreliable for handwritten Tamil; current rule's held-out figure not clean |
| C9 | A small logistic-regression selector over length-invariant engine-output statistics (confidences, in-script and Latin fractions, mass shares) replaces the hand-set 25% rule and returns probabilities | 120 fresh images never used for design: 93.3% vs 74.2% (rule) and 55.8% (argmax); 89.7% cross-validated on the earlier 150 with the fine-tuned Tamil model, 91.4% with the stock model | Solid improvement on this data; n = 120, English scored on lines vs Indic words (a possible confound), no confidence intervals |
| C10 | Fine-tuning only the SVTR neck and CTC head of `ta_PP-OCRv5_mobile_rec` (frozen backbone, cached features, 6,000 words, CPU) cut Tamil CER from 71.1% to 13.9% on the original 30 images (83.9% to 24.9% on 40 fresh images) and removed the Latin leakage (0/30 and 0/40) | `experiments/finetune/` outputs; stock vs fine-tuned on identical images | Real and large for this handwriting source; evaluation words come from the same dataset as the training words (different rows, probably shared writers), so generalization to other Tamil handwriting is **not** shown |
| C11 | Per-line script identification for mixed-script pages | 20 stitched pages (60 lines): 59/60 lines correct with the calibrated selector (32/60 with the hand rule on the fine-tuned model; 20/60 for one script per page) | Synthetic pages built from word/line crops, single script per line; not a real mixed-script document benchmark |
| C12 | Withholding low-confidence Latin output from the Indic engines | flags 96% of leaked regions and 6% of genuine Latin ones; CER barely moves (Tamil 71.1% to 70.4%, Hindi 47.7% to 47.2%) | Weak as an accuracy measure; useful as a presentation safeguard; moot for the fine-tuned Tamil model (no leakage) |
| C13 | Head-only adaptation of `devanagari_PP-OCRv5_mobile_rec` (same recipe as C10; 6,000 words, 30 epochs, CPU) cut Hindi CER from 41.9% to 16.1% on 200 fresh words (42.6% to 16.0% on the first 40) and removed Latin output (20/200 to 0/200) | `experiments/finetune/`, `experiments/fresh_large/analysis.json` | Real and large for this handwriting source; same one-dataset-family caveat as C10; single run |
| C14 | Adapting the recognisers changes what the selector must see: with both recognisers adapted the earlier weights fail (38/120) and the hand rule scores 55.5% on 600 images; refitting on 550 images gives 93.2% [91.0, 95.0] (rule 55.5%, argmax 83.3%; paired exact tests p < 1e-11). A first fit that had only long English lines scored 58.3% on short English words; adding 100 short training words gave 83.3%, equal to argmax | `experiments/fresh/analyze_set.py`, `ablation_large.py`, `significance_ablation.py` | Solid for these engines and these images; two refits were made after seeing test results (disclosed in 10.12 and 15); probabilities are unvalidated outside handwriting |
| C15 | Against EasyOCR 1.7.2 and Tesseract 5.4 (default settings, true script given) the adapted recognisers have far lower CER on handwriting (Hindi 16.1% vs 68.3% and 83.2%; Tamil 16.4% vs 85.2% for Tesseract; English lines 14.0% vs 57.3% and 40.9%); on printed words Tesseract is on par for Hindi (37.2% vs 35.2%) and behind for Tamil (28.9% vs 7.6%) | `experiments/compare/` | Real signal; competitors untuned, EasyOCR's Tamil model does not load, the adapted models were trained on the test dataset family, n = 50-200 |

**Explicitly not claimed:** a new algorithm; that V2 is more accurate than V1 on the
same data (V1 was never run on the real datasets); that script difficulty orders as
English < Hindi < Tamil (the English test set uses long lines, the others isolated
words); that the Tamil fine-tune generalizes beyond the handwriting source it was trained on;
that results generalize beyond the small samples; novelty of script
identification (an established problem).

---

## 4. System architecture (V2, production)

### 4.1 Context diagram

```mermaid
flowchart LR
    U[User / Reviewer] -->|image + optional language| W[Web UI<br/>webapp/static/index.html]
    U -->|CLI args| C[main.py CLI]
    W -->|POST /api/ocr| A[FastAPI app<br/>webapp/app.py]
    A --> P[run_pipeline<br/>main.py]
    C --> P
    P --> R[PaddleOcrRecognizer x3<br/>english / tamil / hindi]
    P --> S[script_detector.pick_script<br/>auto mode only]
    R --> M[(PaddleOCR weights<br/>~/.paddlex/official_models)]
    P --> O[(output files<br/>json / txt / detection / crops)]
    A -->|JSON| W
```

Plain text: User -> Web UI -> `POST /api/ocr` -> FastAPI -> `run_pipeline()` (also
reachable from the CLI) -> one or three `PaddleOcrRecognizer` engines (+
`pick_script` in auto mode) -> output files and JSON -> back to the UI. One process,
no database, queue, auth or external service at runtime (the only network use is the
one-time model weight download).

### 4.2 Components

| Layer | File | Responsibility |
|---|---|---|
| Entry points | `main.py` (CLI), `webapp/app.py` (HTTP) | parse input, call `run_pipeline`, return results |
| Orchestration | `main.py`: `recognize()`, `run_pipeline()` | manual vs auto mode, crops, output files |
| Recognition | `recognition/paddle_recognizer.py` | one `PaddleOcrRecognizer(script)` wraps one PaddleOCR detection+recognition pipeline |
| Script identification | `recognition/script_detector.py` | pure functions: in-script glyph mass, argmax with an Indic override at >= 25% share (the rule) |
| Calibrated selection | `recognition/script_classifier.py` + `script_classifier_weights*.json` | 15 length-invariant features of the three engines' outputs -> multinomial logistic regression -> class probabilities; two weight files (stock Tamil model / fine-tuned Tamil model) |
| Per-line mode | `recognition/region_merge.py` | groups all engines' detections into text-line bands and picks the script of each band |
| Leakage handling | `recognition/leakage.py` | withholds low-confidence Latin-only output of the Tamil/Hindi engines |
| Registry | `recognition/registry.py` | name -> engine label map |
| Configuration | `config.py`, `config.yaml` | `script`, `selector`, `suppress_leakage`, `models`, `paths`, `logging` with defaults merge |
| Frontend | `webapp/static/index.html` | single-file UI: upload, language chooser, simple/developer views, overlay, research tab |

### 4.3 PaddleOcrRecognizer

- One instance per script; lazy: the PaddleOCR object is created on `warm_up()` or the
  first `detect_and_recognize(image_path)` call (so constructing it in tests is free).
- Created with `use_doc_orientation_classify=False`, `use_doc_unwarping=False`,
  `use_textline_orientation=False` (avoids loading extra document-preprocessing models)
  and `enable_mkldnn=False` (workaround for a PaddlePaddle/oneDNN crash, [Section 8.4](#84-environment-and-platform-notes)).
- Returns `[{index, bbox:[x1,y1,x2,y2], text, score, route}]`, sorted into reading order by `sort_reading_order`: regions are grouped into lines (two regions share a line when their vertical centres differ by at most half the first region's box height), lines are ordered top-to-bottom, regions left-to-right within a line, and `index` is re-numbered. This fixes PaddleOCR emitting a word before its left neighbour on the same line (e.g. `सूत्र` before `क्षार`).
- Which model family each language gets is chosen **inside PaddleOCR**, verified from
  its source and model-creation logs: English -> `PP-OCRv6_medium_det/rec`; Tamil ->
  `PP-OCRv5_server_det` + `ta_PP-OCRv5_mobile_rec`; Hindi -> `PP-OCRv5_server_det` +
  `devanagari_PP-OCRv5_mobile_rec`. The `route` label on each region
  (`PaddleOCR PP-OCRv6 (en)`, `PaddleOCR PP-OCRv5 (ta)`, `PaddleOCR PP-OCRv5 (hi)`)
  comes from `ENGINE_LABELS` and must be re-verified if the `paddleocr` pin changes.

### 4.4 Process-lifetime cache and warm-up

`main.py` keeps `_RECOGNIZER_CACHE {script -> PaddleOcrRecognizer}`. `warm_up()` loads
all three. The FastAPI `lifespan` hook calls it at startup so the first request is not
slow; `/api/health` reports `warm: {english, tamil, hindi, done}`.

---

## 5. Detailed design

### 5.1 Request workflow

```mermaid
sequenceDiagram
    participant B as Browser
    participant F as FastAPI (app.py)
    participant M as main.run_pipeline
    participant E as PaddleOcrRecognizer(s)
    participant D as script_detector

    Note over F,E: startup: lifespan -> warm_up() loads english, tamil, hindi
    B->>F: POST /api/ocr (file, script = auto|english|tamil|hindi)
    F->>F: validate script; save upload as webapp/uploads/{run_id}/input.<ext>
    F->>M: run_pipeline(path, run_dir, config)
    alt script == auto
        M->>E: detect_and_recognize(path)  english
        M->>E: detect_and_recognize(path)  tamil
        M->>E: detect_and_recognize(path)  hindi
        M->>D: pick_script({english, tamil, hindi})
        D-->>M: (winner | None, masses, share)
        M->>M: winner's regions are the result (fallback: english, flagged)
    else manual
        M->>E: detect_and_recognize(path)  chosen script
    end
    M->>M: crop each bbox -> cropped/{index}.png; write json/txt files
    M-->>F: {results, detection}
    F-->>B: JSON {run_id, requested_script, script, detection, results[], elapsed_seconds, was_warm}
    B->>F: GET /api/crop/{run_id}/{index}.png  (thumbnails in dev mode)
```

In words: the browser uploads an image and a script choice; the server stores it under
a random 12-hex id; `run_pipeline` runs one engine (manual) or all three (auto);
in auto mode the winning script's regions are reused as the result (no second
recognition pass); crops and files are written; JSON is returned.

### 5.2 Data model

Region (`ocr_results.json` is a list of these; `crop_url` is added only by the web API):

| Field | Type | Meaning |
|---|---|---|
| `index` | int | position in reading order: regions are sorted into lines top-to-bottom, then left-to-right within a line (`sort_reading_order`), and `index` is re-numbered after sorting |
| `bbox` | `[x1,y1,x2,y2]` ints | axis-aligned box in original image pixels |
| `text` | string | recognized text of the region |
| `score` | float 0-1 | PaddleOCR recognition confidence (4 decimals) |
| `route` | string | engine label that produced it |
| `script` | string | script the region was read as (`english`/`tamil`/`hindi`) |
| `crop_url` | string | (API only) `/api/crop/{run_id}/{index}.png` |

Detection info (`ocr_detection.json`, and `detection` in the API response):

| Field | Meaning |
|---|---|
| `mode` | `auto` or `manual` |
| `script` | script finally used |
| `fallback` | `true` if auto found no in-script text anywhere and defaulted to English |
| `share` | winner's fraction of total glyph mass (auto only; `null` in manual) |
| `masses` | per-script confidence-weighted glyph mass (auto only; `null` in manual) |

### 5.3 HTTP API

| Method & path | Purpose |
|---|---|
| `GET /api/health` | `{"status":"ok","warm":{english,tamil,hindi,done}}` |
| `GET /api/research` | static research-findings JSON used by the Research tab |
| `GET /api/sample` | the fixed sample page (`input/images/sample.png`), used by the UI's "Try sample" button |
| `POST /api/ocr` | multipart: `file` (image), `script` (`auto` default / `mixed` / `english` / `tamil` / `hindi`) |
| `GET /api/crop/{run_id}/{n}.png` | cropped region image; `run_id` must match `[a-f0-9]{12}` and filename `\d+\.png`, else HTTP 400 |
| `GET /` | static frontend |

The `detection` object depends on the mode: `auto` adds `probabilities` (per-script, when the calibrated selector is used) next to `masses`/`share`; `mixed` returns `bands` (per text line: `y` range, `script`, `probabilities`, `fallback`) and each region carries its own `script` and `band`; a manual Tamil/Hindi run with `suppress_leakage` adds `suppressed` (withheld regions with text, score and box).

Errors: unsupported `script` -> 400; unreadable image -> 400 (message from
`FileNotFoundError`); other pipeline exception -> 500 `"Pipeline error: ..."`.
Security details: the client filename is never used in a path (only a whitelisted
extension is kept); crop URLs are regex-validated (a traversal attempt returned 404);
CORS is `*`; there is **no** authentication, rate limiting or upload size limit.

### 5.4 Configuration and CLI

`config.yaml` (merged over built-in defaults by `config.py`):

```yaml
script: auto            # auto | mixed | english | tamil | hindi
selector: calibrated    # calibrated (learned) | rule (glyph-mass argmax + Indic override)
suppress_leakage: true  # withhold Latin garbage from the Tamil/Hindi engines
models:
  tamil: models/ta_PP-OCRv5_mobile_rec_ft   # fine-tuned Tamil recogniser; remove to use the stock model
  hindi: models/devanagari_PP-OCRv5_mobile_rec_ft   # fine-tuned Hindi recogniser; remove to use the stock model
paths:
  input: input/images/sample.png
  output_dir: output
logging:
  level: INFO
```

CLI flags override config: `--config`, `--input`, `--output-dir`, `--script`
(`auto|mixed|english|tamil|hindi`). In `auto` the CLI logs the detected script and its share.
`models.tamil` and `models.hindi` are ignored (with a warning) when their directories do not exist, in which case the
stock models are used. The selector weights are paired automatically with the configured recognisers: neither
-> `base`; Tamil only -> `ft`; Tamil and Hindi -> `ft2` (a Hindi-only configuration falls back to `base`, which was not fitted
for it). **The weights must stay paired** with the recognisers they were fitted on: the classifier learns each engine's output
statistics, and the `rule` selector fails with the adapted models (they read other scripts as their own).

### 5.5 Web UI behavior

Single file (`webapp/static/index.html`), no build step, no external fonts/scripts (works
offline), responsive down to ~390 px, light and dark themes (system default, saved in
`localStorage` when available). Screenshots below are committed in `webapp/screenshots/` (real models, October 2026 design); more are in `docs/figures/` (local) — see Section 17.1.

| Home | Auto-detected English page |
|---|---|
| ![Home](webapp/screenshots/home.png) | ![English result](webapp/screenshots/result-english-auto.png) |

| Tamil word (auto, with probability strip) | Dark theme |
|---|---|
| ![Tamil result](webapp/screenshots/result-tamil-auto.png) | ![Dark result](webapp/screenshots/result-dark.png) |

| Research tab | Phone width |
|---|---|
| ![Research tab](webapp/screenshots/research-tab.png) | ![Mobile](webapp/screenshots/mobile.png) |

- **Top bar:** brand, two tabs (Recognize, Research & accuracy), a server-status pill fed by
  `/api/health` polling ("Loading language models…" until all three are warm, then "Models ready",
  or "Server unreachable"), theme toggle.
- **Step 1 — Image:** drag-and-drop, click to browse, paste (Ctrl+V) or **Try sample** (loads
  `/api/sample`); PNG/JPG/BMP up to 15 MB, validated client-side; preview with size and
  dimensions and a Remove button.
- **Step 2 — Language:** **Auto-detect** (default, "recommended") or English / Tamil / Hindi, with a
  one-line hint per choice (auto: about 3x slower and least reliable for handwritten Tamil).
  **Extract text** or Ctrl+Enter runs it.
- **States:** empty; loading (spinner, elapsed timer, Cancel via `AbortController`, text that
  explains why auto-detect takes longer); error (distinguishes "server unreachable" from
  server-reported errors, with Try again); result.
- **Result header:** chip with the detected script (auto), the chosen language (manual), or a
  warning when no readable text was found; **Simple / Developer** switch.
- **Simple view:** recognized text with regions on one line joined by spaces (same rule as the
  server's `sort_reading_order`), correct font and `lang` attribute for Tamil/Devanagari; stats
  (lines, mean confidence, seconds); **Copy**, **Download .txt** (UTF-8 with BOM) and **Share**
  (Web Share API, else clipboard). Lines with a region below 60% confidence get a dotted
  underline plus a notice. After an auto run: "Not the right language? Read as ..." buttons re-run
  the same image in manual mode.
- **Developer view:** metadata (requested/used script, regions, elapsed, warm flag, run id,
  engine); for auto runs, per-script evidence bars with the note that this is a heuristic;
  the image with every region box drawn as an SVG overlay (green >= 0.85, amber >= 0.60, red
  below) with hover linking between boxes and table rows; a regions table with crop thumbnails
  and confidence pills; collapsible raw JSON with Copy JSON.
- **Mixed page, probabilities, hidden garbage:** a fifth choice **Mixed page (per line)** runs all three models and chooses the script of every text line (chip "Per-line scripts: ..."; the Developer view lists the script and probability per line). After an auto run the chip shows the calibrated probability ("Detected script: Tamil · 73% likely") and the Developer view prints all three probabilities with the caveat that they come from a small classifier. A manual Tamil/Hindi run that withheld low-confidence Latin garbage shows a notice with the count.
- **Honesty by design:** the UI does not show the winner `share` as a confidence (it is 1.0 whenever
  only one engine emits in-script characters, including when that winner is wrong, Section 10.5).
- **Research tab:** renders `/api/research`: real-handwriting CER bars (with a note that the
  datasets differ and are not comparable, and the earlier synthetic-pilot bars), the
  script-leakage table, the script-identification table, the V1 failure table, vendor
  benchmark context with model names, and the "what this does NOT claim" list.
- **Design (October 2026 redesign):** landing hero with three script cards, numbered input steps, a "How Lekhak reads a page" section, a probability strip above the text in auto mode (English / Tamil / Hindi probabilities from the calibrated classifier), and a refreshed Research tab with stat tiles and the comparison tables. Element IDs and behaviour are unchanged; the 38 UI checks pass.
- **Robustness/accessibility:** keyboard-operable dropzone and radio group, visible focus rings,
  `aria-live` results, `prefers-reduced-motion` respected; all dynamic text is inserted via
  `textContent` (no HTML injection from OCR output).

---

## 6. Automatic script identification

### 6.1 Idea

A recognizer shown text in a script it was not built for mostly emits low-confidence
garbage or characters from another script. So every engine reads the image, and the
engine whose output contains the most **confident text inside its own Unicode block**
wins. No extra model is trained or downloaded; the signal comes from the recognizers
themselves.

| Script | Characters counted |
|---|---|
| Tamil | U+0B80 - U+0BFF |
| Hindi | Devanagari U+0900 - U+097F |
| English | ASCII letters `A-Z`, `a-z` |

Digits, spaces and punctuation are ignored (script-neutral).

### 6.2 Algorithm

```
for each script s with recognized regions (text_i, score_i):
    mass(s) = sum_i  score_i * |{ c in text_i : c in block(s) }|
total   = sum_s mass(s)
winner  = argmax_s mass(s)
# Indic override (added after the first version, see 6.4):
best_indic = argmax over {tamil, hindi} of mass
if mass(best_indic) / total >= 0.25:  winner = best_indic
share   = mass(winner) / total                      # how concentrated the evidence is
if total == 0:  winner = None  ->  pipeline uses "english", fallback = true
result  = winner's regions (reused; no second recognition pass)
```

```mermaid
flowchart TD
    I[Image] --> E1[English engine] & E2[Tamil engine] & E3[Hindi engine]
    E1 --> M1[mass_en = sum score x #Latin letters]
    E2 --> M2[mass_ta = sum score x #Tamil chars]
    E3 --> M3[mass_hi = sum score x #Devanagari chars]
    M1 & M2 & M3 --> T{total mass > 0 ?}
    T -- no --> FB[fallback: english, fallback=true]
    T -- yes --> A[winner = argmax mass]
    A --> O{best Indic script holds >= 25% of total mass?}
    O -- yes --> W[winner = that Indic script]
    O -- no --> W2[keep argmax winner]
    W --> R[reuse winner's regions as the result]
    W2 --> R
    FB --> R
```

Implementation: `recognition/script_detector.py` (`script_mass`, `pick_script`,
constants `INDIC_SCRIPTS = ("tamil", "hindi")` and `INDIC_MIN_SHARE = 0.25`) — pure
functions, unit-tested without any model; orchestration in `main.recognize()`.
`share` reported for the winner is `mass(winner) / total`, so with the override it can be
below 0.5 (an Indic winner chosen at 25-50% share).

### 6.3 Known properties and limits

> The properties below describe the glyph-mass **rule**. Since the second round of work (Sections 6.5-6.8) the default selector is the calibrated classifier, `mixed` mode chooses a script per line, and Tamil is read by a fine-tuned model; each limit below is annotated where that changed it.

- **Cost:** about 3x the compute of manual mode (three engines). Manual mode runs one.
- **One script per image:** mixed-script documents get a single script and one engine's
  reading. The Indic engines also read Latin words (so a Hindi page with English words
  is read in full by the Hindi engine), but the English engine garbles Indic text, so an
  English-script winner on such a page loses the Indic text. There is no per-region
  script identification.
- **Script leakage weakens the signal** (Section 10.4): on hard handwritten Tamil the Tamil
  engine often emits Latin letters, which add to the *English* engine's mass rather than
  Tamil's, so such Tamil images can be identified as English. The measured accuracy is in
  Section 10.5.
- **The Indic override has a cost:** if an Indic engine produces >= 25% of the mass from
  garbage characters on an image that is really English, the image is misrouted to the Indic
  script. How often this happens on the evaluation images is measured in Section 10.5.
- **Blank or symbol-only images:** no evidence -> English fallback, flagged.
- **Only three scripts;** other languages in the same scripts are not distinguished
  (Marathi/Nepali in Devanagari would be reported as `hindi`).
- It is **not** a novelty claim: script identification for OCR is an established problem.

### 6.4 How the rule evolved

1. **First version (argmax):** the script with the largest glyph mass wins. Unit-tested and
   evaluated as the baseline.
2. **Indic override (author's later commit `ec77275`):** auto-detect chose English on a real
   Hindi poster containing English words, because the English engine turns Devanagari into
   Latin garbage that inflates English mass while the Hindi engine reads both scripts. The
   override lets an Indic script win once it holds >= 25% of the total mass. The 25%
   threshold is a hand-chosen constant, not fitted on the evaluation data (as far as the
   record shows); two unit tests cover it (Indic wins on a mixed poster; English kept when the
   Indic share is small).
3. **Evaluation reports both rules** (Section 10.5) because the second rule was introduced
   before the evaluation ran, so its held-out figure is not a clean unseen-data estimate.

### 6.5 Calibrated script selection (C9)

The hand-written rule has one hand-chosen constant and, as Section 10.5 shows, collapses when the
engines change. `recognition/script_classifier.py` replaces it with a multinomial logistic
regression (L2-regularized, `scipy` L-BFGS) over 15 features, five per engine: confidence-weighted
mean score, fraction of the engine's output inside its own script, fraction of Latin letters, an
"empty output" flag, and its share of the total glyph mass. Features are deliberately
**length-invariant** (no character counts): the evaluation pairs ~61-character English lines with
~6-10-character Indic words, so counts would let the model detect line length instead of script.
The selector returns per-script probabilities. It returns `None` (English fallback, flagged) only
when no engine produced any text at all; out-of-script garbage (a Tamil word read as Latin
letters) is itself evidence and is classified.

Training and honest evaluation (`experiments/script_classifier/train_classifier.py`): 10 repeats of
stratified 5-fold cross-validation over the 150 labelled outputs of E7, with the L2 strength chosen
by an inner 4-fold search inside every training fold; the two rules are scored on the same images.
The shipped weights are fitted on all 150 images. Because the weights depend on how the engines
behave, there are two files: `script_classifier_weights.json` (stock Tamil model) and
`script_classifier_weights_ft.json` (fine-tuned Tamil model); `main.py` pairs them automatically
(Section 5.4). A third file, `script_classifier_weights_ft2.json`, is for the Tamil- and Hindi-adapted pair: it is fitted on 550 images (the 150 above, 300 extra images and 100 short handwritten English words) with cross-validation accuracy 90.2% (sd 0.4 over repeats), ECE 0.014, log loss 0.284. Results: Sections 10.9 and 10.12.

### 6.6 Per-line (mixed-script) mode (C11)

`script: mixed` runs the three engines on the whole page, merges the vertical extents of all their
detections into horizontal **bands** (text lines; two extents merge when their overlap is at least
30% of the smaller height), applies the selector to the outputs falling inside each band, and takes the
band's regions from the winning engine (each region keeps its `script` and `band`). A single line
that itself mixes scripts still receives one script, and the cost equals auto mode. Evaluation on
stitched pages: Section 10.10.

### 6.7 Leakage suppression (C12)

`recognition/leakage.py`: a region of the Tamil or Hindi engine that contains Latin letters, none
of its own script, and a score below 0.7 is withheld and reported (`detection.suppressed`). On the
cached outputs the 0.7 cut flagged 96% of leaked regions and 6% of genuine Latin ones (14 of 233
Latin regions read by Indic engines on the English images). The threshold was read off those
outputs, not tuned on a separate set. In auto/mixed mode leakage is instead resolved by comparing
engines. Effect on CER: Section 10.11.

### 6.8 Fine-tuned Tamil recogniser (C10)

The stock `ta_PP-OCRv5_mobile_rec` is the project's weakest component (71.1% CER; leakage on 17/30
images). Full fine-tuning through PaddleX/PaddleOCR ran at about 0.4 samples/s on the 6-core laptop
CPU (roughly 4 hours per epoch of 6,000 words) and PaddleX has no registered training entry for
the Tamil model, so the backbone (PPLCNetV3) was **frozen**, its features for 6,000 training words
(rows 1000-6999 of `c3rl/IIIT-INDIC-HW-WORDS-Tamil`; the evaluation images are rows 0-49 and the
validation words rows 500-799) were computed once (about one hour) and cached, and only the CTC
branch (SVTR neck and CTC head) was trained with CTC loss for 25 epochs (Adam, cosine learning
rate 5e-4, batch 32; `experiments/finetune/head_finetune.py`). No image augmentation was possible
on cached features. The best checkpoint (epoch 25) was exported with PaddleOCR's `export_model.py`
into `models/ta_PP-OCRv5_mobile_rec_ft/` (detector unchanged; loaded through
`text_recognition_model_dir`). Validation CER (300 words) went 68.9% -> 6.7% and exact match 2.7% ->
61%. Pipeline results: Section 10.8; method and commands: `docs/tamil_finetune.md`.

Side effects to know about: the fine-tuned engine was trained on Tamil only, so it renders any
script as Tamil characters; this makes the Tamil mass non-zero on English and Hindi images and breaks
the hand-written selection rule (Section 10.9), which is why the classifier is retrained for it.
Tamil words the detector fails to find still come out empty.

### 6.9 Fine-tuned Hindi recogniser (C13)

The same recipe as 6.8 applied to `devanagari_PP-OCRv5_mobile_rec`: backbone frozen, features cached for 6,000 training words (rows 2000-7999 of `c3rl/IIIT-INDIC-HW-WORDS-Hindi`; validation rows 500-799; the evaluation rows 0-49, 1000-1059, 1100-1299 and 1300-1399 are untouched), SVTR neck and CTC head trained for 30 epochs (about 1.5 minutes each, best validation checkpoint at epoch 28), exported to `models/devanagari_PP-OCRv5_mobile_rec_ft/` (7.8 MB) and loaded through config key `models.hindi`. Validation CER (300 words) went 41.9% -> 15.9% and exact match 11.7% -> 45.7%. Results: Section 10.12; method and commands: `docs/hindi_finetune.md`.

Side effect: on English lines the adapted Hindi head produces long, confident Devanagari strings (score about 0.8) where the English engine reads almost nothing, which is why the selector was refitted for the adapted pair (Section 10.12).

---

## 7. The previous architecture (V1, `legacy_v1/`)

```mermaid
flowchart TD
    IMG[Image] --> SW{--script}
    SW -- english / tamil --> EAST[EASTDetector<br/>OpenCV DNN + frozen EAST .pb]
    EAST --> GRP[group_text<br/>sort by y1, merge by v_tol / h_gap]
    GRP --> PAD[pad_and_clamp_box<br/>per-script padding, tamil = 20 px]
    PAD --> CROP[crop region]
    CROP --> CLS{script}
    CLS -- english --> HC[TextClassifier<br/>gray std > 70 => handwritten]
    HC --> TR[TrOCRRecognizer<br/>trocr-large-printed / -handwritten]
    CLS -- tamil --> OT[OcrTamilRecognizer<br/>CRAFT + PARSEQ]
    SW -- hindi --> EZ[EasyOcrHindiRecognizer<br/>EasyOCR CRAFT + CRNN, owns detection]
    TR --> OUT[regions: index, bbox, label, route, text]
    OT --> OUT
    EZ --> OUT
```

Key facts: EAST decoded score/geometry maps with NMS (`min_confidence` 0.3, overlap
0.3); grouping used `v_tol = 0.5 x avg box height`, `h_gap = 1.5 x avg box height`
(after the fix, groups expand all four edges and are returned in `(y1,x1)` order);
the classifier was a grayscale-std threshold (70), never validated; Hindi needed a
second recognizer contract ("self-detecting", takes the whole image) because EAST
returned no boxes for Devanagari. V1 has its own registry copy (so importing both
generations in one process cannot clash), config, tests (21) and heavy dependencies
(`requirements-legacy.txt`: PyTorch, transformers, `ocr_tamil`, EasyOCR) plus the EAST
weights file (`models/frozen_east_text_detection.pb`, gitignored, license unconfirmed).
Run from the repo root: `python -m legacy_v1.main --script <s>`.

**Why it was replaced:** three languages needed three backends and a second contract;
it needed per-language geometric patches (grouping fix, padding); it was slow (126.8 s
for the 32-region sample); and its detector could not handle Devanagari at all. It is
kept (not deleted) so the V1-vs-V2 comparison is reproducible.

---

## 8. Implementation reference

### 8.1 Files

| Path | Role |
|---|---|
| `main.py` | CLI + orchestration (`recognize`, `run_pipeline`, `warm_up`) |
| `config.py`, `config.yaml` | configuration loader and defaults |
| `recognition/registry.py` | route/label registry |
| `recognition/paddle_recognizer.py` | PaddleOCR wrapper per language |
| `recognition/script_detector.py` | script identification scoring (rule) |
| `recognition/script_classifier.py`, `script_classifier_weights.json`, `script_classifier_weights_ft.json` | calibrated selector and its two weight files |
| `recognition/region_merge.py` | per-line (mixed-script) merge |
| `recognition/leakage.py` | Latin-leakage suppression |
| `models/ta_PP-OCRv5_mobile_rec_ft/` | fine-tuned Tamil recognition model (PaddleOCR inference format, 7.8 MB) |
| `webapp/app.py`, `webapp/static/index.html` | web API and UI |
| `webapp/ui_tests/` | `mock_server.py` (canned API, real static UI) and `ui_test.py` (Playwright, 38 checks); not part of `pytest` |
| `docs/figures/` | UI screenshots for the report (local-only; see its README) |
| `tests/` | 39 production tests |
| `legacy_v1/` | V1 code, tests, config, README |
| `requirements*.txt` | `requirements.txt` (production), `-dev` (pytest, datasets), `-webapp` (FastAPI, uvicorn), `-legacy` (V1 stack) |
| `experiments/runs/` | evaluation scripts, data and reports (gitignored) |
| `experiments/fresh/`, `script_classifier/`, `leakage/`, `finetune/` | fresh-image evaluation, classifier training and cross-validation, leakage evaluation, Tamil fine-tuning scripts and reports (gitignored) |
| `docs/` | research and project documents (gitignored) |
| `input/images/sample.png` | sample English exam page used throughout |
| `models/frozen_east_text_detection.pb` | V1 only, gitignored |

### 8.2 Dependencies (production `requirements.txt`)

`numpy==2.3.5`, `opencv-python==4.13.0.92`, `pillow==12.1.1`, `PyYAML==6.0.2`,
`paddlepaddle==3.3.1`, `paddleocr==3.7.0` (pulls `paddlex==3.7.2`). Web:
`fastapi==0.143.0`, `uvicorn==0.54.0`, `python-multipart==0.0.32`. Dev: `pytest==9.1.1`,
`datasets==5.1.0`.

### 8.3 Output files

`ocr_results.json` (regions), `ocr_results.txt` (one region text per line),
`ocr_detection.json` (detection info), `cropped/{index}.png` (bbox crops, clamped to the image).

### 8.4 Environment and platform notes

| Topic | Detail |
|---|---|
| oneDNN crash | With defaults PaddlePaddle 3.3.1 raised `NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]` during detection on the development CPU; `enable_mkldnn=False` avoids it (other machines may not need it and may be faster with it on) |
| Windows file lock | `pip install` fails with `[WinError 5] Access is denied: ...cv2.pyd` while the server is running (module loaded) — stop the server first |
| Console encoding | Printing Tamil/Devanagari to a cp1252 console raises `UnicodeEncodeError` in ad-hoc scripts; the pipeline writes UTF-8 and reconfigures stdout |
| First run | PaddleOCR downloads weights per model into `~/.paddlex/official_models/` |
| Long-running jobs | The 150-image script-ID run takes a long time on CPU and a run was once lost when its process was interrupted; the runner is therefore incremental and resumable (`raw_outputs.jsonl`) |

---

## 9. Evaluation methodology

Full detail: `docs/experiment_protocol.md`. Summary:

### 9.1 Metric

**Character Error Rate (CER)** = Levenshtein distance over Unicode code points divided
by ground-truth length; overall CER = sum of distances / sum of lengths. It can exceed
1.0. Code points (not grapheme clusters) are used, so a visible Tamil/Devanagari
character spanning several code points can cost several edits; no Unicode
normalization or whitespace normalization is applied. **Exact match** = string equality.
No word error rate was computed.

### 9.2 Datasets (all `REQUIRES_REVIEW`, none redistributed)

| Folder | Source | Unit | Used |
|---|---|---|---|
| `english_real/` | Hugging Face `Teklia/IAM-line`, split `test` (mirror of the IAM Handwriting Database; mirror MIT, IAM has its own research terms) | handwritten sentence **lines**, mean 60.9 chars | first 30 |
| `tamil_real/` | `c3rl/IIIT-INDIC-HW-WORDS-Tamil`, `train` (CVIT, IIIT Hyderabad; Gongidi & Jawahar; no license stated on the mirror) | handwritten **words**, mean 9.6 chars | first 30 |
| `hindi_real/` | `c3rl/IIIT-INDIC-HW-WORDS-Hindi`, `train` | handwritten **words**, mean 6.4 chars | first 30 |
| `script_id/` | same three sources | as above | first 50 each (150); items 0-29 are byte-identical to the `*_real` images ("design"), 30-49 are "held-out" |
| `tamil_pilot/`, `hindi_pilot/` | self-generated: 5 short phrases each, `PIL.ImageDraw` + Windows `Nirmala.ttc`, white background | synthetic | 5 each |
| `input/images/sample.png` | pre-existing | one English exam page | 1 |

Samples are the first rows of each dataset stream, in order — not randomly sampled.
**Comparability caveat:** English is scored on long lines, Tamil/Hindi on short words,
from different datasets, writers and capture conditions.

### 9.3 Experiments

| ID | What | Data | Output |
|---|---|---|---|
| E1 | V1 grouping bug ablation (qualitative text diff) | `sample.png` | `experiments/runs/day1_baseline/` |
| E2 | EAST on Devanagari (box count at two thresholds) | 1 synthetic image | — |
| E3 | Tamil box-padding sweep (V1) | 5 synthetic | in-process sweep + pipeline re-run `tamil_pilot/cer_report.json` |
| E4 | Hindi pilot (V1, EasyOCR) | 5 synthetic | `hindi_pilot/cer_report.json` |
| E5 | V2 real-data CER | 30 real x 3 | `{english,tamil,hindi}_real/cer_report.json` via `run_paddleocr_eval.py` |
| E6 | Script-leakage analysis | E5 outputs | derived |
| E7 | Script identification (design / held-out) | 150 real | `script_id/raw_outputs.jsonl`, `script_id_report.json` |
| E8 | Latency | sample + word images | single timed runs |
| E9 | V1 vs V2 qualitative on `sample.png` | 1 page | — |
| E10 | Tamil recogniser fine-tuning: CER, exact match and Latin leakage, stock vs fine-tuned | 30 earlier images (rows 0-29) + 40 fresh (rows 7000-7039) | `experiments/finetune/cer_report.json` |
| E11 | Selector comparison: argmax vs rule vs calibrated, both Tamil models | 120 fresh images (40 per language) | `experiments/fresh/eval_images_orig.json`, `eval_ft.json` |
| E12 | Cross-validation of the classifier | the 150 images of E7 | `script_id/classifier_cv_report_*.json` |
| E13 | Per-line script identification | 20 stitched pages, 60 lines | `experiments/fresh/eval_orig.json`, `eval_ft.json` |
| E14 | Leakage suppression | 30 + 30 earlier images, 40 + 40 fresh | `experiments/leakage/leakage_report.json`, E11 reports |

### 9.4 Controls and gaps

Controlled: one machine, fixed models, deterministic inference, fixed dataset order,
the same CER code for all languages, and a held-out split for script identification. The
first (argmax) rule was fixed before any of the evaluation data was scored and has no fitted
parameters; the current rule adds a hand-chosen 25% Indic-override threshold that was
introduced before the evaluation ran, so both rules are reported and the current rule's
held-out figure is not a clean unseen-data estimate. E11-E14 use **fresh images** pulled after the
learned components were built (English: IAM-line rows 50-109; Hindi: rows 1000-1059; Tamil: rows
7000-7059, beyond the 6,000 fine-tuning words); the classifier never saw them, but one design choice
(letting the classifier decide when out-of-script text is present, which matches how it is trained and
cross-validated) was made after seeing the first fresh result, so the fresh figure is slightly
optimistic. Not done: random sampling, repeated
runs, confidence intervals, significance tests, grapheme-level CER, WER, comparison
with other OCR engines, V1 on the real datasets.

---

## 10. Results

### 10.1 V1 results (legacy, synthetic / single page)

**English grouping fix (E1), `sample.png`:** 19 of 32 lines changed, all gained text:

| Before | After |
|---|---|
| `SUITABLE.` | `JUSTLY WAY THEY ARE SUITABLE.` |
| `SAFETY SYSTEM` | `FACTORY SAFETY SYSTEM` |
| `8051` | `TO AN 8051` |
| `(INT1)` | `INTERRUPT 1 (INT1)` |
| `P2` | `ITO PORT P2` |
| `BUTTON` | `EMERGENCY BUTTON` |

`MICROCONTROLLE` and a cut-off parenthetical were unchanged (different cause).

**Tamil synthetic pilot (E3):** before padding all 5 outputs were truncated
(e.g. `வணக்கம்` -> `வணக்க`); CER 46.3%, 0/5 exact. In-process padding sweep (values
from the run output; not saved as a file; concatenated all regions per image):

| Padding (px) | 0 | 5 | 10 | 15 | 16 | 18 | 20 | 22 | 24 | 26 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CER | 0.500 | 0.426 | 0.352 | 0.352 | 0.315 | 0.296 | 0.296 | 0.333 | 0.315 | 0.296 | 0.370 |

Through the real pipeline (first region only) at 20 px: **CER 25.9%, 1/5 exact**
(sample 0 `வணக்கம்` exact; `கல்வி அறிவு` -> `கல்விஅறிவு`; `இது ஒரு சோதனை` -> `ததன`, CER 0.77,
still a severe failure). Absolute values differ slightly between the two procedures; the
trend (improves to ~18-26 px, worse at 30 px) is the finding. n=5, one font.

**Hindi synthetic pilot (E4):** CER 7.1%, exact 3/5 (EasyOCR). **E2:** EAST returned 0
boxes at confidence 0.3 and 0.05 on a synthetic Devanagari image (1143 non-white pixels);
EasyOCR's detector read it correctly (`नमस्ते`, 0.90).

### 10.2 V2 qualitative on the sample page (E9)

16 coherent regions (V1: 32 fragments). Examples: `Smoke sensor connected to External
Interrupt 0 (INT0)` complete (V1: cut at `0 (`); `microcontroller:` (V1: `MICROCONTROLLE`);
`Emergency push button connected to External Interrupt 1 (INT1)`; confidences mostly 0.98-1.0
(one low: a stray `1`, 0.55).

### 10.3 V2 real-data CER (E5)

| Script | Dataset | Unit | n | Mean GT chars | **CER** | Exact | Median CER | Samples CER<=0.25 | Samples CER>=1.0 | Empty predictions |
|---|---|---|---|---|---|---|---|---|---|---|
| English | IAM-line | line | 30 | 60.9 | **24.1%** | 0/30 | 0.146 | 23 | 2 | 2 |
| Hindi | IIIT-INDIC-HW-WORDS-Hindi | word | 30 | 6.4 | **47.7%** | 2/30 | 0.333 | 12 | 8 | 0 |
| Tamil | IIIT-INDIC-HW-WORDS-Tamil | word | 30 | 9.6 | **71.1%** | 1/30 | 0.909 | 4 | 12 | 0 |

Examples (ground truth -> prediction, per-sample CER):

| Script | Ground truth | Prediction | CER |
|---|---|---|---|
| English | `He is also a director of a couple of garages . And he finds time as well to be a lyric` | same text but `garages.` | 0.012 |
| English | `the switch because of the topicality of African` | `the swith because ...` | 0.021 |
| English | `She plays the possessive mother of a man whose hobby revolves` | `` (empty) | 1.0 |
| Hindi | `टाटा`, `बेचने` | exact | 0.0 |
| Hindi | `केंद्रों` -> `केंदट्रों`; `फायदेमंद` -> `फापदेमंद` | small substitutions | 0.125 |
| Hindi | `डीजल` -> `Dutnt` | Latin garbage | 1.25 |
| Tamil | `எடுக்கலாம்` | exact | 0.0 |
| Tamil | `இந்தியர்களுக்கு` -> `இந்தியர்கமுக்கு` | one glyph wrong | 0.067 |
| Tamil | `பேச்சைக்` -> `BODA`; `ஏளனம்` -> `Jaostlo` | Latin garbage | 1.0 / 1.4 |

### 10.4 Script leakage (E6)

| Engine | Non-empty predictions | With Latin letters | CER on those | CER on the rest |
|---|---|---|---|---|
| Tamil | 30 | **17** (57%) | 0.912 | 0.461 (13 predictions; 9 fully in Tamil script) |
| Hindi | 30 | 2 | 1.167 | 0.455 (28 predictions; 25 fully in Devanagari) |

Examples: Tamil `nigeggn`, `uoog்`, `BODA`, `Bnonbe`, `TளியwT`, `நரTய`, `அmi`; Hindi `tt`, `Dutnt`.
The English engine instead returned empty strings for 2/30 lines. Interpretation: the
multilingual recognizers include Latin characters in their output vocabulary and fall
back to them on hard input (inferred from outputs, not from the dictionary files).

### 10.5 Automatic script identification accuracy

Both rules were scored on the same cached engine outputs for 150 real images (50 per
language). "Design" = the first 30 images per language (examined while the first rule was built);
"held-out" = items 30-49 per language. The `argmax` rule was fixed before any of this data was
scored; the `current` rule (argmax + Indic override at >= 25% of glyph mass) was introduced before
the evaluation ran, so **its held-out figure is not a clean unseen-data estimate**.

| Rule | Design (n=90) | Held-out (n=60) | All (n=150) |
|---|---|---|---|
| `argmax` (first rule) | 62.2% (56/90) | 70.0% (42/60) | 65.3% (98/150) |
| `current` (+ Indic override) | 76.7% (69/90) | 78.3% (47/60) | **77.3% (116/150)** |

Per true script, all 150 images:

| True script | `argmax` | `current` |
|---|---|---|
| English | 48/50 (96%) | 48/50 (96%) |
| Hindi | 32/50 (64%) | 43/50 (86%) |
| Tamil | 18/50 (36%) | 25/50 (50%) |

Confusion matrix of the `current` rule (all 150; rows = true script, columns = predicted):

| True \ predicted | english | tamil | hindi | none |
|---|---|---|---|---|
| english | 48 | 0 | 0 | 2 |
| tamil | 18 | 25 | 2 | 5 |
| hindi | 6 | 0 | 43 | 1 |

Held-out only (n=60, `current`): English 20/20, Hindi 17/20 (3 -> english), Tamil 10/20 (6 -> english, 4 -> none).

What the numbers show:

- **English is reliable (96%).** The two misses are the two images on which the English engine
  itself returned an empty string (no evidence -> `none`, which the pipeline turns into a flagged
  English fallback).
- **The Indic override helped without a measured downside here:** +12 points overall (Hindi 64% -> 86%,
  Tamil 36% -> 50%), and **0 of the 50 English images were misrouted to an Indic script**.
- **Tamil is unreliable (50%) and the cause is the recognizer, not the scoring rule.** Of the 25
  misidentified Tamil images, in 21 the Tamil engine's output contained **no Tamil characters at all**
  (22 contained Latin letters, i.e. script leakage) and in 3 the Tamil share was below 25%; of the 7
  misidentified Hindi images, 5 had no Devanagari at all. When the matching engine cannot read the
  script, no rule built on the engines' outputs can recognize it.
- **`share` is not a usable confidence.** The median winner share is 1.0 because the other engines
  often emit no in-script characters, which also happens when the winner is wrong (an English winner
  on a Tamil image typically has share 1.0). The UI therefore does not present it as certainty.
- **Practical guidance:** use auto-detect for English and Hindi; when the text is known to be
  handwritten Tamil, choose Tamil manually (it is also 2-3x faster).

Latency (medians over the 150 images, which mix sentence lines and word images, CPU-only): English
engine 2.95 s, Tamil engine 5.16 s, Hindi engine 5.11 s per call, **13.27 s per image for all three**
(auto mode) vs about 3-5 s for one engine (manual mode). Two images stalled for about 36 minutes and
about 3 hours (the laptop pausing); medians are used so those two do not distort the figures.
Sample size is 50 images per language; intervals are wide (e.g. Tamil 25/50 is 50% +/- about 14 points).


### 10.6 Latency (CPU, warm models, single measurements)

| Case | Time |
|---|---|
| V2 English, 16 regions, manual | 14.5 s |
| V2 Tamil, one real word image, manual | 6.4 s |
| V2 Hindi, one real word image, manual | 2.5 s |
| V1 English, 32 regions | 126.8 s (TrOCR-large, ~4 s/region) |
| V1 Tamil / Hindi, 1-region synthetic images | ~0.4 s |
| V2 auto mode | 13.3 s per image (median over 150 images; sum of the three engines: 3.0 + 5.2 + 5.1 s) |

### 10.7 Vendor-reported accuracy (context only)

PaddleOCR's own published recognition accuracies: `en_PP-OCRv5_mobile_rec` 85.25%,
`ta_PP-OCRv5_mobile_rec` 94.2%, `devanagari_PP-OCRv5_mobile_rec` 84.96%, each on the
vendor's own test sets. The English figure belongs to a PP-OCRv5 model; this project's
English pipeline uses PP-OCRv6, so it is not the model in use. These are not comparable to
the real-handwriting results above and the large gap was not diagnosed.

### 10.8 Fine-tuned Tamil recogniser (C10)

CER = sum of Levenshtein distances / sum of reference lengths (code points); prediction = the
engine's regions joined by spaces; same detector, only the recognition model differs.

| Image set | Model | CER | Exact | Predictions with Latin letters |
|---|---|---|---|---|
| Earlier 30 images (rows 0-29) | stock | 71.1% | 1/30 | 17/30 |
| | fine-tuned | **13.9%** | 13/30 | 0/30 |
| Fresh 40 images (rows 7000-7039) | stock | 83.9% | 1/40 | 25/40 |
| | fine-tuned | **24.9%** | 19/40 | 0/40 |
| Validation, 300 words (rows 500-799) | stock head | 68.9% | 2.7% | — |
| | fine-tuned head | 6.7% | 61% | — |

Caveats: all words come from one dataset (IIIT-INDIC-HW-WORDS-Tamil); the evaluation rows are
disjoint from the training rows but probably share writers, so this is adaptation to that handwriting
source, not evidence for Tamil handwriting in general (no other Tamil set was tried). The validation
figure was used for checkpoint selection; the other two sets were not. Hindi (47.7% on the earlier
images, 42.6% on the fresh ones) and English are unchanged.

### 10.9 Script identification: calibrated classifier vs rules (C9)

**Fresh images** (120, 40 per language, never examined; the cleanest figure in this project):

| Tamil model | Selector | Overall | English | Tamil | Hindi |
|---|---|---|---|---|---|
| stock | argmax (first rule) | 55.8% (67/120) | 39/40 | 10/40 | 18/40 |
| stock | rule (argmax + Indic override) | 74.2% (89/120) | 39/40 | 13/40 | 37/40 |
| stock | **calibrated classifier** | **93.3% (112/120)** | 39/40 | 36/40 | 37/40 |
| fine-tuned | argmax | 60.8% (73/120) | 31/40 | 33/40 | 9/40 |
| fine-tuned | rule | 42.5% (51/120) | 0/40 | 35/40 | 16/40 |
| fine-tuned | **calibrated classifier** | **93.3% (112/120)** | 36/40 | 38/40 | 38/40 |

**Cross-validation on the earlier 150 images** (10 repeats of 5-fold, inner search for the L2 strength):

| Tamil model | Classifier accuracy (mean, sd over repeats) | Per script (English / Tamil / Hindi) | Log loss | Brier | Calibration error (ECE) | argmax / rule on the same images |
|---|---|---|---|---|---|---|
| stock | 91.4% (1.4) | 47.1/50, 45.1/50, 44.9/50 | 0.335 | 0.143 | 0.028 | 65.3% / 77.3% |
| fine-tuned | 89.7% (1.2) | 45.7/50, 45.5/50, 43.3/50 | 0.443 | 0.178 | 0.041 | 71.3% / 47.3% |

What it shows: the learned selector fixes the weak point of the rule (Tamil: 13/40 -> 36/40 on fresh
images with the stock model) and keeps working when the engines change, whereas the rule breaks
because the fine-tuned Tamil engine emits Tamil characters for other scripts too. Caveats: n = 120
and 150, no confidence intervals; English images are sentence lines and Indic images are single
words, so English may be easier to separate than short English words would be (the features avoid
counts, but confidences can still differ by unit); the weights are fitted on only 150 images, and
probabilities on image types other than these (printed text, photos) are unvalidated. The
two fresh-image rows marked "calibrated" share the 112/120 total by coincidence.

### 10.10 Per-line script identification on stitched pages (C11)

20 pages, each stacking one English line, one Tamil word and one Hindi word in random order (fresh
images 40-59 of each language, white padding); 60 ground-truth lines.

| Tamil model | Selector | Lines correct | Pages fully correct |
|---|---|---|---|
| stock | **calibrated, per line** | **58/60** | 18/20 |
| stock | rule, per line | 49/60 | 9/20 |
| stock | one script for the whole page (auto) | 20/60 | — |
| fine-tuned | **calibrated, per line** | **59/60** | 19/20 |
| fine-tuned | rule, per line | 32/60 | 0/20 |
| fine-tuned | one script for the whole page (auto) | 20/60 | — |

The pages are synthetic (crops of the same handwriting datasets), each line contains one script, and
no real mixed-script document was tested. A mixed page took about 70 s on this CPU (large image, three
engines).

### 10.11 Leakage suppression (C12)

| Image set | Language | CER without | CER with suppression | Regions withheld |
|---|---|---|---|---|
| Earlier 30 images | Tamil (stock) | 71.1% | 70.4% | 10 |
| | Hindi | 47.7% | 47.2% | 2 |
| Fresh 40 images | Tamil (stock) | 83.9% | 82.6% | — |
| | Hindi | 42.6% | 41.8% | — |

A withheld region scores like an empty prediction, which is only slightly better than a wrong Latin
string, so the CER effect is small; the value is that garbage text is not shown to the user (UI notice).
With the fine-tuned Tamil model there is nothing to withhold for Tamil (0/30 and 0/40 Latin outputs).

### 10.12 Hindi recogniser, the 600-image test and the refitted selector (C13, C14)

Test rows are disjoint from every training, validation and fitting row (`docs/data_splits.md`). CER as in 10.8; intervals are 95% bootstrap over images (10,000 resamples, seed 0).

| Image set | Language | Model | CER | Exact | Latin letters | Empty |
|---|---|---|---|---|---|---|
| Fresh 40 (Hindi rows 1000-1039) | Hindi | stock | 42.6% [34.6, 50.7] | 2/40 | 6/40 | 1/40 |
|  |  | head-adapted | 16.0% [9.8, 22.9] | 20/40 | 0/40 | 1/40 |
| Large 200 (rows 1100-1299) | Hindi | stock | 41.9% [37.7, 46.2] | 25/200 | 20/200 | 3/200 |
|  |  | head-adapted | 16.1% [13.4, 19.0] | 96/200 | 0/200 | 1/200 |
| Large 200 (rows 7100-7299) | Tamil | stock | 73.5% [68.6, 78.4] | 5/200 | 107/200 | 10/200 |
|  |  | head-adapted | 16.4% [12.6, 20.7] | 97/200 | 0/200 | 12/200 |

Paired CER drops: Hindi 25.8 points (CI 21.8-29.8; 137 images improved, 11 worse, 52 unchanged); Tamil 57.1 points (CI 51.9-62.5; 178 improved, 4 worse, 18 unchanged). Caveat as for C10: one dataset family, no writer IDs, one training run.

**Script identification on the 600 fresh images** (200 per script; an image on which every engine returned nothing counts as an error):

| Engines | Selector | Overall [95% CI] | English | Tamil | Hindi |
|---|---|---|---|---|---|
| stock (stock weights) | argmax | 66.8% [63.0, 70.5] | 198/200 | 69/200 | 134/200 |
|  | rule | 77.2% [73.8, 80.5] | 198/200 | 84/200 | 181/200 |
|  | **calibrated classifier** | 91.0% [88.7, 93.2] | 198/200 | 166/200 | 182/200 |
| adapted (ft2 weights) | argmax | 83.3% [80.3, 86.3] | 187/200 | 142/200 | 171/200 |
|  | rule | 55.5% [51.5, 59.5] | 5/200 | 143/200 | 185/200 |
|  | **calibrated classifier** | 93.2% [91.0, 95.0] | 196/200 | 179/200 | 184/200 |

Exact McNemar tests (calibrated-only correct vs other-only correct): stock calibrated vs rule 92 vs 9, p = 1.8e-18, vs argmax 154 vs 9, p = 3.2e-35; adapted calibrated vs rule 236 vs 10, p = 3.4e-57, vs argmax 68 vs 9, p = 2.4e-12.

**Earlier 120 fresh images** with the adapted pair: calibrated 87.5% [80.8, 93.3] (English 34/40, Tamil 34/40, Hindi 37/40), rule 52.5%, argmax 72.5%. A cluster of English lines (fresh lines 11-19, IAM-line test rows 61-69) is read by the English engine as almost nothing while the adapted Hindi head produces confident Devanagari, and most of them are sent to Hindi (6 of the 40 English lines are wrong with the final weights); the 200 English lines of the large set (rows 110-309) do not show this (196/200 correct), so the effect is localised, not general. **Mixed pages** (20 stitched pages, 60 lines): calibrated 56/60 lines and 16/20 pages fully correct; rule 34/60 and 0/20; one script per page 20/60.

**Refit history and post-hoc disclosure.** The ft2 weights were fitted three times: on 150 images (120-image score 102/120), on 450 images after 300 extra images were added (104/120; 600-image score 94.7%), and on 550 images after 100 short handwritten English words (IAM words, train split) were added because the selector scored only 58.3% on the 60 held-out short English words (test split). The last refit lowered the 600-image score from 94.7% to 93.2% and raised the English-word score to 83.3%. Each refit followed seeing a test result, so the 120-, 600- and 60-image figures are not clean one-shot estimates. Design settings (L2 strength) were always chosen by inner cross-validation on the fitting images only.


### 10.13 Ablations and comparison with other open-source OCR (C14, C15)

**Feature ablation** (retrain on the fitting images with groups removed, score on the 600 fresh images; classifier argmax, so an image with no engine output is not counted as an error here):

| Features | Stock engines (fit 150) | Adapted engines (fit 550) |
|---|---|---|
| all 15 | 546/600 | 559/600 |
| drop confidence | 500/600 | 506/600 |
| drop in-script fraction | 556/600 | 560/600 |
| drop Latin fraction | 555/600 | 555/600 |
| drop empty flag | 536/600 | 558/600 |
| drop mass share | 552/600 | 560/600 |
| confidence only | 493/600 | 538/600 |
| in-script fraction only | 517/600 | 310/600 |
| Latin fraction only | 478/600 | 318/600 |
| empty flag only | 213/600 | 206/600 |
| mass share only | 500/600 | 474/600 |

Confidence is the only group whose removal clearly hurts in both settings (stock 546 -> 500, adapted 559 -> 506 of 600). With the stock engines, dropping the in-script fraction, Latin fraction or mass share even raised the score slightly (+10, +9, +6 images) and dropping the empty flag cost 10; with the adapted engines, dropping any group other than confidence changed the score by 1-4 images. No single group does the work; confidence alone gets 538/600 with the adapted engines. (The earlier 150-fit ablation scored on the 120 images, `experiments/fresh/significance_ablation.json`, agrees.)

**Comparison with EasyOCR 1.7.2 and Tesseract 5.4** (the true script is given to every system, so this compares recognition only; default settings; Tesseract with `--psm 7`; Tesseract `tam`/`hin` data from `tessdata_fast`; EasyOCR hi+en reader, because its Tamil model fails to load in 1.7.2; CER of the whole line or word):

| Set | Script (n) | Ours, stock | Ours, adapted | EasyOCR | Tesseract |
|---|---|---|---|---|---|
| Handwriting, large | English (200) | 14.0% [11.3, 17.1] | 14.0% [11.3, 17.1] | 57.3% [55.0, 59.6] | 40.9% [38.2, 43.7] |
| Handwriting, large | Hindi (200) | 41.9% [37.8, 46.1] | 16.1% [13.4, 19.0] | 68.3% [64.1, 72.5] | 83.2% [78.5, 87.9] |
| Handwriting, large | Tamil (200) | 73.5% [68.5, 78.2] | 16.4% [12.6, 20.6] | n/a | 85.2% [81.3, 88.9] |
| Handwriting, first 40 | English (40) | 33.2% [19.1, 46.9] | 33.2% [19.0, 46.6] | 58.5% [52.3, 64.0] | 45.9% [38.1, 53.4] |
| Handwriting, first 40 | Hindi (40) | 42.6% [34.7, 50.6] | 16.0% [9.8, 22.9] | 69.6% [60.0, 79.4] | 84.0% [74.7, 92.7] |
| Handwriting, first 40 | Tamil (40) | 83.9% [74.6, 92.2] | 24.9% [13.8, 37.9] | n/a | 92.7% [85.6, 99.8] |
| Short English words | English (60) | 15.2% [8.5, 23.8] | 15.2% [8.6, 24.1] | 73.0% [61.7, 83.9] | 83.5% [71.3, 95.9] |
| Printed words (Mozhi) | Hindi (50) | 53.8% [40.3, 68.6] | 35.2% [23.7, 49.0] | 46.2% [33.8, 59.7] | 37.2% [23.7, 53.2] |
| Printed words (Mozhi) | Tamil (50) | 34.6% [25.9, 44.7] | 7.6% [4.7, 10.9] | n/a | 28.9% [20.2, 39.3] |

Script identification baselines on the same images: Tesseract's own script detection (OSD, `--psm 0`) returned an answer on only 57 of 600 large-set images and on none of the 60 words or 100 printed words, so it is not a usable baseline for crops this small; taking the script with the most characters in Tesseract's eng+tam+hin output gives 42.7% [38.8, 46.7] on the large set against 93.2% [91.2, 95.2] for the calibrated selector on adapted engines.

Reading the comparison: on handwriting the adapted recognisers are far ahead of both open-source systems, but those systems were run untuned, and the adapted recognisers were trained on handwriting from the same dataset family as the Hindi and Tamil test words, so the gap partly reflects domain match. On printed words the picture changes: Tesseract is on par for Hindi (overlapping intervals, n = 50) and is behind only for Tamil.


### 10.14 Other domains: short English words and printed words

**60 short handwritten English words** (IAM words, test split first 60): selector accuracy with the adapted pair 83.3% [73.3, 91.7] after the 550-image refit (58.3% with the 450-image fit; 28.3% [16.7, 40.0] with the stock weights, which had only seen long English lines). On these words the hand rule scores 23.3% and argmax 83.3% (the same 50/60 as the calibrated selector; the two disagree on 10 images, 5 each way), so the learned selector has **no advantage over argmax on short English words**. English CER 15.2% [8.6, 24.1] (37/60 exact) against EasyOCR 73.0% and Tesseract 83.5%.

**Printed words** (Mozhi test set, 50 random Tamil and 50 random Hindi words, seed 0; the handwriting-trained weights are applied to a different domain): selector accuracy 82.0% [74.0, 89.0] with adapted engines (Tamil 47/50, Hindi 35/50) and 69.0% [60.0, 78.0] with stock engines, against 58.0% [49.0, 68.0] for Tesseract's multi-language output. CER with the adapted recognisers: Tamil 7.6% [4.7, 10.9] (stock 34.6%, Tesseract 28.9%), Hindi 35.2% [23.7, 49.0] (stock 53.8%, EasyOCR 46.2%, Tesseract 37.2%). The stock Hindi engine returned an empty output for 27 of the 50 printed Hindi words (cause not investigated).


### 10.15 Real mixed-script photographs (Wikimedia Commons)

61 photographs of signboards, plaques and banknotes downloaded from Wikimedia Commons categories for bilingual and trilingual signs in India (`experiments/fresh/pull_mixed_real.py`; 65 downloaded, 4 rejected as unreadable, wrong script or unconfirmed). Labels are **per-image script presence** (English 61, Tamil 39, Hindi 36 of 61 images), written by visual inspection of every image, not by a Tamil or Hindi reader, and not per line (no pixel annotation was made); the absent-script labels are the weaker ones (small logos or fine print can contain another script). A script counts as present when at least 3 characters of that script are read, for every system. Licences and authors: `experiments/mixed_real/sources.csv`.

| System | English precision / recall | Tamil precision / recall | Hindi precision / recall | Exact script set recovered [95% CI] |
|---|---|---|---|---|
| This project, mixed mode (per-line script, ft2 weights) | 100% / 92% | 91% / 77% | 86% / 83% | 57.4% [44.3, 68.9] |
| This project, one script per image (auto) | 100% / 69% | 100% / 26% | 100% / 25% | 0.0% [0.0, 0.0] |
| EasyOCR 1.7.2 (hi+en reader; no Tamil) | 100% / 100% | n/a / 0% | 60% / 97% | 36.1% [24.6, 49.2] |
| Tesseract 5.4, eng+tam+hin, --psm 7 (one line) | 100% / 8% | n/a / 0% | 50% / 3% | 0.0% [0.0, 0.0] |
| Tesseract 5.4, eng+tam+hin, --psm 11 (sparse text) | 100% / 100% | 64% / 100% | 60% / 81% | 27.9% [16.4, 39.3] |

Reading the table: mixed mode is the only system with both high precision and useful recall, and it recovers the exact script set of 57.4% of the photos (a single script per image can never do so on mixed signs). EasyOCR and Tesseract read the text on these photos but report scripts that are not there: EasyOCR (no Tamil model) reports Hindi in 23 images that have none, and Tesseract in sparse-text mode reports Tamil in 22 and Hindi in 19 images that have none, so their exact-set rates are lower; Tesseract with one-line segmentation (`--psm 7`) is a poor fit for scene photos and is shown only for completeness. Caveats: small n (61 photographs, several from the same locations), presence labels from visual inspection, a fixed 3-character threshold, default competitor settings, and no per-line accuracy on real photographs, so the mixed-page line accuracy of Section 10.12 remains a synthetic result.


### 10.16 Supplementary numbers (confusion matrices, row ranges, calibration, references)

**Confusion matrix on the 600 fresh images** (calibrated selector; rows = true script, columns = chosen script; "none" = every engine returned nothing, so no script could be chosen):

| Adapted engines (ft2 weights) | English | Tamil | Hindi | none |
|---|---|---|---|---|
| English (200) | **196** | 2 | 2 | 0 |
| Tamil (200) | 3 | **179** | 6 | 12 |
| Hindi (200) | 5 | 10 | **184** | 1 |

| Stock engines (stock weights) | English | Tamil | Hindi | none |
|---|---|---|---|---|
| English (200) | **198** | 2 | 0 | 0 |
| Tamil (200) | 14 | **166** | 17 | 3 |
| Hindi (200) | 4 | 13 | **182** | 1 |

**Row ranges.** The 300 extra fitting images ("Fit-2"): English = Teklia/IAM-line **train** split rows 0-99, Hindi = IIIT-INDIC-HW-WORDS-Hindi train rows 1300-1399, Tamil = IIIT-INDIC-HW-WORDS-Tamil train rows 7300-7399. The 100 short English training words: `priyank-m/IAM_words_text_recognition` **train** split rows 0-99 (the 60 test words are the first 60 rows of its **test** split). Hindi rows 1300-1399 were used only as selector fitting images; they are not part of the Hindi recogniser's fine-tuning (rows 2000-7999), validation (rows 500-799) or any test set (rows 0-49, 1000-1059, 1100-1299).

**Calibration of the final selector** (ft2 weights, 550 fitting images, 10 repeats of stratified 5-fold cross-validation): accuracy 90.2% (sd 0.4), log loss 0.284, **Brier score 0.143**, expected calibration error 0.014. (Earlier fits: 150 images with stock engines, Brier 0.143, ECE 0.028; with the Tamil-adapted engine, Brier 0.178, ECE 0.041; the 450-image ft2 fit, Brier 0.106, ECE 0.008, was superseded.)

**Printed words (Mozhi, 50 Tamil + 50 Hindi), script accuracy of every selector:** stock engines: argmax 51.0% [41.0, 61.0], rule 57.0% [47.0, 67.0], calibrated 69.0% [60.0, 78.0]; adapted engines: argmax 66.0% [57.0, 75.0], rule 68.0% [59.0, 77.0], calibrated 82.0% [74.0, 89.0].

**Leakage suppression on the 40 fresh Tamil and 40 fresh Hindi words** (stock engines, threshold 0.7; counts of regions): Tamil 20 of 47 regions withheld (CER 83.9% -> 82.6%); Hindi 1 of 42 regions withheld (CER 42.6% -> 41.8%). With the adapted recognisers nothing is withheld because no output contains Latin letters.

**References for the data** (verify before submission): Mozhi = Mathew, Mondal and Jawahar, "Towards Deployable OCR models for Indic languages", arXiv:2205.06740, submitted 13 May 2022 (v2 18 December 2024), presented at ICPR 2024 per the arXiv comments; dataset page `https://cvit.iiit.ac.in/usodi/tdocrmil.php`. IAM database = Marti and Bunke, "The IAM-database: an English sentence database for offline handwriting recognition", International Journal on Document Analysis and Recognition 5(1), 2002 (from memory, not re-checked); the Hugging Face copy `priyank-m/IAM_words_text_recognition` was created on 7 September 2022, states no licence, and the IAM terms (non-commercial research) apply. IIIT-INDIC-HW-WORDS = Gongidi and Jawahar, ICDAR 2021.

---

## 11. Analysis and discussion

1. **Architecture.** Moving from a per-language patchwork to one engine per language
   removed an entire class of failures (detector incompatibility, grouping) and
   collapsed three code paths into one. The price is dependence on PaddleOCR's internal
   model selection and a heavier engine than a trivial detector.
2. **Accuracy on real handwriting is the main weakness.** Even with a strong modern
   engine, Tamil word recognition is mostly wrong (median CER 0.91) and Hindi is
   mixed (median 0.33); English sentence lines are comparatively good (median 0.15).
   Because units differ (lines vs words) and datasets differ, **these three numbers must
   not be read as a ranking of script difficulty**.
3. **Script leakage** explains part of the Tamil errors: predictions that contain Latin
   letters have twice the CER of those that do not. It also couples recognition quality
   to script identification: a Tamil image on which the Tamil engine emits Latin letters
   adds mass to the English engine.
4. **Synthetic vs real.** The V1 synthetic pilots suggested 7.1% (Hindi) and 25.9% (Tamil)
   CER; real handwriting gave 47.7% and 71.1% on V2. Since both engine and data changed,
   the safe lesson is about practice: validate on a real third-party dataset before
   quoting accuracy. (Informally, V2 also read the clean synthetic Hindi images well, which
   suggests data difficulty rather than the engine swap drives the gap; not saved as a result.)
5. **Vendor benchmarks do not transfer.** Official figures are for specific mobile
   models on the vendor's data; the project's English model is a different generation.
6. **Auto-detect trades compute for convenience** (about 3x) and is bounded by the
   recognizers' own failure modes; manual mode is the fast path when the language is
   known. See Section 10.5 for the measured accuracy.
7. **The second round of work moved the weakest numbers, with caveats.** Fine-tuning the
   Tamil head (Section 6.8) cut Tamil CER from 71.1% to 13.9% and removed the leakage, but it
   adapts the model to one handwriting source and makes the engine read every script as Tamil,
   which broke the hand-written selection rule; a small learned selector (Section 6.5) restored
   and improved script identification (93.3% on 120 fresh images). The pattern is worth stating
   plainly: a component-level gain can invalidate a heuristic downstream, and the learned
   selector must be paired with the exact model it was trained on. The remaining uncertainty is
   generalization: all gains are measured on the same few datasets.
8. **Adaptation of one component moved another (second and third rounds).** Adapting the Hindi head as well as the Tamil head again broke the selector fitted for the Tamil-adapted engine (38/120) and the hand-written rule (55.5% on 600 images, English 5/200). Refitting restored 93.2% [91.0, 95.0] on 600 images, and a failure on short English words (58.3%) exposed that the earlier fit had only seen long English lines; adding 100 short words fixed it (83.3%) at a cost of 1.5 points on the large set. On handwriting the adapted recognisers are far ahead of untuned EasyOCR and Tesseract, while on printed Hindi Tesseract is on par.
9. **Honest framing for a paper:** a case study of script-dependent failure modes and
   their remediation, a real-data measurement with a quantified failure mode, an
   evaluation-practice caution, and a tested script-ID baseline. A stronger paper needs
   larger real datasets, word-level units for all languages, grapheme-level metrics,
   same-engine controls and comparison with other engines.

---

## 12. Testing

`pytest tests/ legacy_v1/tests/ -q` -> **65 passed**, no model downloads or network.

| File | Tests | Covers |
|---|---|---|
| `tests/test_registry.py` | 4 | register/lookup, missing route, three scripts registered, exact engine labels |
| `tests/test_paddle_recognizer.py` | 3 | lazy construction (model not loaded), invalid script -> `ValueError`, `sort_reading_order` (line grouping, left-to-right order, re-numbered `index`) |
| `tests/test_script_detector.py` | 9 | Latin/Tamil/Devanagari glyph counting, confidence weighting, argmax, out-of-block garbage ignored, no-evidence cases, Indic override on a mixed Hindi+English poster, English kept when the Indic share is small |
| `tests/test_main_recognize.py` | 11 | `main.recognize` orchestration with fake recognizers: auto picks the right script and reuses its regions, English fallback is flagged, manual runs only the chosen engine, masses/share reported, mixed mode chooses per line, leakage suppression only when asked, calibrated selector reports probabilities, fine-tuned model directories honoured/ignored (Tamil and Hindi), classifier weights paired with the configured models (`base`, `ft`, `ft2`) |
| `tests/test_script_classifier.py` | 9 | feature vector matches its names and is length-invariant, empty-engine flag, `None` only when there is no text at all, out-of-script garbage still classified, probabilities sum to 1, shipped weights match the feature set, all three weight variants (`base`, `ft`, `ft2`) load |
| `tests/test_leakage.py` | 5 | Latin garbage from Indic engines flagged, confident/own-script text kept, English never suppressed, renumbering |
| `tests/test_region_merge.py` | 3 | line bands across engines, per-line scripts, English fallback flagged |
| `legacy_v1/tests/test_grouping.py` | 8 | grouping regression/adversarial cases (the left-drift cases fail on pre-fix code) |
| `legacy_v1/tests/test_boxes.py` | 12 | `clamp_box`, `pad_and_clamp_box` |
| `legacy_v1/tests/test_east_detector.py` | 1 | missing image -> `FileNotFoundError` |

Not unit-tested: file/crop writing in `run_pipeline`, FastAPI endpoints. These were verified by live
requests (all three languages in auto and manual mode, crops, health, research and sample endpoints,
invalid script, traversal attempt).

**UI tests (outside `pytest`):** `webapp/ui_tests/ui_test.py` drives an installed Edge/Chrome through
Playwright against `webapp/ui_tests/mock_server.py` and checks 38 behaviours: status pill, sample
load, loading state, detected chip, line rendering, low-confidence marking, re-run buttons,
overlay/row hover linking, clipboard copy, download, calibrated-probability chip, mixed-page mode and its per-line list,
hidden-garbage notice, theme switch, research tab, Tamil rendering
(`lang="ta"`), cancel, server-error and network-failure states, no horizontal overflow at 390 px
wide, no console errors. All pass. The UI was also driven against the real server with real
models (no console errors; auto-detect chips correct for the sample page and a Hindi word). Setup:
`pip install playwright` (uses the installed browser), then `python webapp/ui_tests/mock_server.py`
and `python webapp/ui_tests/ui_test.py`. Not tested: Firefox/Safari, screen readers.

---

## 13. Installation and usage

```
git clone https://github.com/ripuns/Multilingual-OCR-Detection.git
cd Multilingual-OCR-Detection
python -m venv ocr_env
ocr_env\Scripts\activate                 # Linux/macOS: source ocr_env/bin/activate
pip install -r requirements.txt          # production
pip install -r requirements-dev.txt      # pytest, datasets
pip install -r requirements-webapp.txt   # FastAPI, uvicorn
```

CLI:

```
python main.py                                        # auto-detect script on input/images/sample.png
python main.py --script tamil --input path\to\img.png # force one language (faster)
python main.py --script mixed --input page.png        # script chosen per text line
python main.py --input img.png --output-dir out
```

Web demo: `uvicorn webapp.app:app --host 127.0.0.1 --port 8000`, open
`http://127.0.0.1:8000` (models warm-load at start; `GET /api/health` shows readiness).

Reproduce the evaluation (detail in `docs/reproducibility.md`):

```
python experiments/runs/pull_datasets.py                 # real images from Hugging Face
python experiments/runs/run_paddleocr_eval.py all        # E5: CER reports
python experiments/runs/script_id/run_engines.py         # E7: 3 engines x 150 images (slow, resumable)
python experiments/runs/script_id/evaluate.py            # E7: script-ID accuracy
```

Learned components (detail in `docs/tamil_finetune.md` and `docs/reproducibility.md`):

```
python experiments/fresh/pull_fresh.py                   # fresh evaluation images (English/Hindi/Tamil)
python experiments/fresh/build_pages.py                  # 20 stitched mixed-script pages
python experiments/fresh/run_engines.py experiments/fresh/manifest.json experiments/fresh/raw_images_orig.jsonl --use image_level
python experiments/fresh/eval_fresh.py experiments/fresh/raw_images_orig.jsonl experiments/fresh/raw_pages_orig.jsonl experiments/fresh/eval_orig.json
python experiments/script_classifier/train_classifier.py experiments/runs/script_id/raw_outputs_ft.jsonl --write --out=recognition/script_classifier_weights_ft.json
```

The Tamil fine-tune itself needs a separate training environment (`paddlepaddle` + `paddlex` +
the PaddleOCR repository cloned by `paddlex --install PaddleOCR`) and hours of CPU time; see
`docs/tamil_finetune.md`. The exported model is shipped in `models/ta_PP-OCRv5_mobile_rec_ft/`.

Legacy V1: `pip install -r requirements-legacy.txt`, place the EAST weights in `models/`,
`python -m legacy_v1.main --script english`.

---

## 14. Related work and prior-art findings

Gathered by web search during development and read as search-result summaries (titles,
abstracts, snippets), **not** as full papers; treat as pointers, not a literature review.

- **Modular vs end-to-end OCR.** Confidence-routed modular pipelines are described as
  standard in production; cascading error in modular pipelines is a named problem that
  recent work (e.g. Qianfan-OCR, arXiv 2603.13398) addresses by going end-to-end. Six open
  VLM-OCR models appeared in Oct 2025 (Nanonets OCR2-3B, PaddleOCR-VL-0.9B, DeepSeek-OCR-3B,
  Chandra-OCR-8B, OlmOCR-2-7B, LightOnOCR-1B); PP-OCRv6 (arXiv 2606.13108) reports small
  models rivaling billion-parameter VLMs.
- **Text-line grouping.** Heuristic and learned approaches both established (e.g. ReLaText,
  *Pattern Recognition*); patents exist on clustering/consolidating OCR boxes (US
  2020/0210743; USPTO 11909934).
- **Low-resource script OCR.** Active, less saturated: SynthOCR-Gen (arXiv 2601.16113), Nayana
  OCR (ACL 2025), GlotOCR Bench (arXiv 2604.12978), "Deciphering the Underserved" (arXiv
  2412.16119), Sinhala/Tamil zero-shot study (arXiv 2507.18264); Kashmiri reported unsupported
  by Tesseract/TrOCR/PaddleOCR.
- **Script identification.** Established for OCR (e.g. arXiv 1708.04671), hence not claimed novel.
- **Datasets.** IIIT-INDIC-HW-WORDS (Gongidi & Jawahar, ICDAR 2021; 872K words, 8 Indic
  scripts per its project page); IAM Handwriting Database (Marti & Bunke); uTHCD Tamil
  characters (arXiv 2103.07676).
- **Crowded, therefore dropped as contribution angles:** document-forgery detection
  (DocForge-Bench, AIForge-Doc), print steganography patents, exam/answer-sheet grading
  (CHECK-MAT, "When VLMs 'Fix' Students", commercial Indian multilingual graders), national
  credential infrastructure (NAD/DigiLocker).
- **Patent decision:** none pursued; rationale in `docs/patent_ip_assessment.md`.

---

## 15. Limitations and threats to validity

(Full list: `docs/limitations.md`, `docs/experiment_protocol.md` Section 5.)

1. n = 30 per language (150 for script-ID); first-rows sample; no confidence intervals.
2. English lines vs Tamil/Hindi words, different datasets: cross-language CER not comparable.
3. Code-point CER over-penalizes Indic graphemes; no WER.
4. V1 (synthetic, 5 images) vs V2 (real, 30 images) mixes engine and data changes.
5. V1 was never evaluated on the real datasets.
6. Script leakage and empty outputs inflate error and are not detected or corrected.
7. Script-ID design split overlaps the CER images; only the held-out split is unbiased.
8. Auto mode: ~3x compute, one script per image, three scripts only.
9. EAST/Devanagari failure was shown on one synthetic image/font; real Devanagari photos untested.
10. Vendor benchmarks are for different models/data; English label corrected to PP-OCRv6.
11. Demo hardening is minimal (no auth, rate limit, upload cap, cleanup); CORS `*`.
12. CPU-only, synchronous, single process; latency is single-run on one laptop.
13. Datasets and the EAST weights are `REQUIRES_REVIEW` for licensing.
14. Single author, single machine; no independent replication.
15. The Tamil fine-tune is evaluated on words from the dataset it was trained on (different rows, probably shared writers); no other Tamil handwriting was tested, and a frozen backbone limits what it can learn.
16. The calibrated selector is fitted on 150 images and checked on 120; its probabilities are unvalidated outside these image types, English is scored on lines vs Indic words, and its weights must be paired with the Tamil model they were trained for.
17. Per-line mode was tested only on synthetic stacked pages with one script per line; mixed lines are not handled.
18. The fine-tuned Tamil engine reads any script as Tamil; Tamil words the detector misses still produce no output.
19. Neither adaptation is evaluated on independent handwriting: no free labelled Tamil word or line set from other writers was found, three web-sourced Tamil photos could not be transcribed reliably and were not used, and the datasets give no writer IDs.
20. The selector was refitted for the adapted engines three times (150, 450, 550 images), the last after it failed on short English words; the 120-, 600- and 60-image sets had been seen before that change, so their figures are not clean one-shot estimates (Section 10.12).
21. On short English words the selector is no better than argmax (50/60 each); on printed words (a different domain) it reaches 82%, and its probabilities are unvalidated outside handwriting.
22. Results depend on which rows are tested: English line CER is 14.0% on rows 110-309 but 33.2% on rows 50-89, and a run of consecutive English lines was misrouted; consecutive rows share writers.
23. The comparison with EasyOCR and Tesseract uses default settings and one text line per image; EasyOCR's Tamil model does not load in 1.7.2, and Tesseract's script detection fails on word crops. No comparison with large vision-language OCR models was made.
24. Mixed-script pages are stitched crops; the real mixed-script test (Wikimedia Commons photos, Section 10.15) is small and labelled by script presence from visual inspection, not by a native reader and not per line.
25. One training run per language, no seed variation; the English recogniser was not adapted.

---

## 16. Project history and future work

### 16.1 Timeline (from Git history)

| Date | Milestone | Commits (examples) |
|---|---|---|
| 2026-03-26/27 | Prototype; Chinese TrOCR route removed; README | `7f11b45`, `a2d69dd`, `38299d6` |
| 2026-08-26 | Audit of the prototype; EAST missing-image handling | `42e599e`, `58b5b43` |
| 2026-08-27 | Grouping fix + tests; CLI/UTF-8 logging; pinned deps; config; registry; box clamping; JSON output | `e7ff323`, `6a1cbc8`, `57e4d62`, `aed56d7`, `4205eb2`, `02596e1`, `5b8aca8`, `7636214` |
| 2026-10-09 | Tamil route (`ocr_tamil`); Hindi route (EasyOCR, self-detecting contract); padding + warm-up | `bdcfbaa`, `a1d61d2`, `bb47ce0` |
| 2026-10-09 | V2: PaddleOCR rewrite, V1 moved to `legacy_v1/` | `e90636d` |
| 2026-10-09 | Automatic script identification (`d8f8290`); Indic-override rule + reading-order sort (`ec77275`); documentation overhaul | `d8f8290`, `ec77275` |
| 2026-10-11 | Calibrated selector, per-line mode, leakage suppression, fine-tuned Tamil recogniser, fresh-image evaluation (working tree; commit pending) | — |
| 2026-10-12 | Fine-tuned Hindi recogniser, selector refit for the adapted pair (`ft2`), 600-image test, ablations, EasyOCR/Tesseract comparison, short-word and printed-word sets, real mixed-script photos, UI redesign (working tree; commit pending) | — |

Decision log with rationale, dead ends and corrections: `docs/project_history.md`.

### 16.2 Corrections made during documentation (kept for transparency)

1. English engine is PP-OCRv6, not PP-OCRv5 (code, tests, docs fixed; vendor English figure is for another model).
2. The "script-difficulty ordering" was withdrawn (line-vs-word confound).
3. Latin "script leakage" was discovered and quantified.
4. "Synthetic overestimates real" restated as an evaluation-practice caution (engine and data both changed).
5. The script-ID evaluation runner was made incremental/resumable after a killed process lost an hour of work.
6. The script-identification rule changed after it was first documented (Indic override, Section 6.4); the evaluation therefore reports both rules, and the "rule frozen before evaluation" statement was narrowed to the first rule.
7. Latency reporting uses medians because two runs of the evaluation stalled for 36 minutes and ~3 hours (the laptop pausing), which would have made a mean meaningless.

### 16.3 Future work

- Per-region (word-level) script identification, so that one line mixing scripts is handled.
- Evaluate the Tamil and Hindi adaptations on independent handwriting from other writers (none was available); unfreeze the backbone with augmentation if GPU time becomes available; apply the recipe to Telugu (the IIIT-HW-Telugu word data was downloaded but not used).
- Validate the classifier's probabilities on more printed text and photographs, add short-word and printed-word training data, and add a rejection threshold for "not one of the three scripts".
- Larger, word-level, multi-dataset evaluation for all three languages; grapheme-level CER and WER; confidence intervals; V1 on the same real data; same-engine synthetic-vs-real control.
- Isolate why vendor Tamil accuracy (94.2%) and the real-handwriting result (71.1% CER) differ.
- More scripts/languages (a `LANG_CODES` + label entry plus evaluation); the most open research direction identified is OCR for very low-resource scripts.
- Productionization: auth, upload limits, cleanup, async/batching, GPU path, caching of the per-engine results (a mixed page took ~70 s).

---

## 17. Guide for writing the technical report

Target: **30-40 pages**. Suggested chapters, page budget, and where to take the content.
Figures use the Mermaid diagrams in this README (render them, or redraw); tables can be
copied directly.

| # | Chapter | Pages | Source in this README (and docs) |
|---|---|---|---|
| 1 | Abstract and introduction | 2 | Sections 1-3; motivation Section 2 |
| 2 | Background and related work | 3-4 | Section 14; glossary Section 18.4; `docs/project_history.md` Phase 2 |
| 3 | Requirements and design decisions | 2-3 | Section 2; design-decision table `docs/architecture.md` Section 8 |
| 4 | System architecture | 5-6 | Section 4 (context diagram, components, PaddleOcrRecognizer, cache) |
| 5 | Detailed design and workflows | 5-6 | Section 5 (sequence diagram, data model, API, config, UI) |
| 6 | Automatic script identification | 3-4 | Section 6 (idea, formula, flowchart, limits) + 10.5 results |
| 7 | The legacy architecture and migration | 3 | Section 7 (V1 flowchart, why replaced) |
| 8 | Implementation | 2 | Section 8 (files, dependencies, environment issues incl. oneDNN) |
| 9 | Evaluation methodology | 3 | Section 9 (metric definition, datasets, experiments table, controls) |
| 10 | Results | 4-5 | Section 10 (V1 results, V2 qualitative, CER table, leakage, script-ID, latency) |
| 11 | Discussion, limitations, threats to validity | 3 | Sections 11 and 15 |
| 12 | Testing | 1 | Section 12 |
| 13 | Conclusion and future work | 1-2 | Sections 3, 16.3 |
| A | Appendices | 2-3 | per-sample examples; full file map; reproduction commands; API examples |

### 17.1 Figures to include (data is in this README)

1. System context diagram (4.1). 2. Component diagram (4.2 table as a block diagram).
3. Request sequence diagram (5.1). 4. Script-ID flowchart (6.2). 5. V1 pipeline flowchart (7).
6. Bar chart of V2 real-data CER by language with unit labels (10.3) and, beside it, the V1
synthetic-pilot values (7.1% Hindi, 25.9% Tamil) — caption must state units differ.
7. Line chart of Tamil CER vs padding (10.1 table). 8. Latin-leakage grouped bars (10.4).
9. Script-ID confusion matrix (10.5). 10. Latency bar chart V1 vs V2 (10.6).
11. UI screenshots — already captured in `docs/figures/` (local; see `docs/figures/README.md` for what each shows and
which are real-model vs mock-API captures): `ui_real_01_english_auto_simple.png`, `ui_real_02_english_auto_developer.png`
(boxes + evidence bars), `ui_real_03_hindi_auto_simple.png`, `ui_real_04_tamil_manual_simple.png`,
`ui_real_05_research_light.png`, plus mock-API captures of the empty/loading/error states, the dark theme and the phone
layout (`ui_01` ... `ui_11`).

### 17.2 Tables to include

Contribution/strength table (3); experiments table (9.3); datasets table (9.2); CER table
(10.3); examples (10.3); leakage (10.4); script-ID results (10.5); latency (10.6); test
inventory (12); limitations (15).

### 17.3 Things the report must NOT claim (common pitfalls)

- That the project invented an OCR algorithm or script-identification method.
- That English < Hindi < Tamil is a difficulty ordering of the scripts.
- That synthetic data "overestimates" accuracy as a clean, same-engine fact.
- That V2 beats V1 on identical data (never measured), or the vendor benchmark numbers as
  this project's accuracy (and the English one is for a different model).
- That the system is production-ready: it is a CPU-only demo with ~24-71% real-handwriting CER.
- That mixed-script documents are handled.

### 17.4 Quick facts for captions and prose

3 languages; 3 PaddleOCR engines (English PP-OCRv6, Tamil/Hindi PP-OCRv5); auto detection
by confidence-weighted in-script glyph mass; 150 real images for script-ID (30 design + 20
held-out per language) plus 120 fresh images and 20 mixed pages; 90 real images for CER (30 per language) plus 600 fresh images (200 per language) for the adapted recognisers and the selector; 65 tests; CPU-only AMD
Ryzen 5 5625U; V1 English 126.8 s vs V2 English 14.5 s on the same sample page (V1 32
fragments, V2 16 lines).

---

## 18. Repository map, documentation index, glossary, acknowledgements

### 18.1 Repository map

```
.
├── main.py                       CLI + orchestration
├── config.py / config.yaml       configuration
├── recognition/
│   ├── registry.py               label registry
│   ├── paddle_recognizer.py      PaddleOCR wrapper (3 languages)
│   ├── script_detector.py        script identification scoring (rule)
│   ├── script_classifier.py      calibrated selector (+ weights *.json: stock, fine-tuned Tamil, fine-tuned Tamil + Hindi)
│   ├── region_merge.py           per-line mixed-script mode
│   ├── leakage.py                Latin-leakage suppression
│   └── README.md
├── webapp/
│   ├── app.py                    FastAPI backend
│   ├── static/index.html         single-file UI
│   ├── ui_tests/                 mock API + Playwright UI test (not part of pytest)
│   └── README.md
├── tests/                        44 production tests (+ README.md)
├── legacy_v1/                    V1 (EAST, grouping, classifier, TrOCR/ocr_tamil/EasyOCR) + 21 tests + README.md
├── input/images/sample.png       sample English page
├── models/                       fine-tuned Tamil and Hindi recognition models (*_rec_ft/); EAST weights (V1 only, gitignored)
├── docs/                         local-only research docs (see 18.2); docs/figures/ = UI screenshots
├── experiments/                  local-only evaluation data/scripts/reports (+ README.md)
└── requirements*.txt             production / dev / webapp / legacy
```

### 18.2 Documentation index (`docs/`, local-only)

| File | Content |
|---|---|
| `architecture.md` | both architectures, diagrams, API, algorithm, runtime notes, design decisions |
| `experiment_protocol.md` | metric definitions, datasets, every experiment, threats to validity |
| `research_contribution.md` | claims with strength ratings, V1/V2 results, interpretation |
| `limitations.md` | what not to claim |
| `reproducibility.md` | environment, commands, regeneration table |
| `project_history.md` | timeline, decisions, prior-art findings, corrections |
| `dataset_and_license_inventory.md` | models/data provenance and license status |
| `patent_ip_assessment.md` | why no patent |
| `changelog.md` | notable changes |
| `tamil_finetune.md`, `hindi_finetune.md` | head-only adaptation method and results |
| `data_splits.md` | row ranges of every train / validation / fitting / test set and overlap checks |
| `paper_draft.md`, `session_context_round3.md` | paper draft and the latest session handover |
| `project_audit.md`, `implementation_plan.md`, `session_context_summary.md` | historical (superseded) |

### 18.3 Licensing and attribution

PaddleOCR / PaddlePaddle (Baidu), Apache 2.0. Datasets: `Teklia/IAM-line` (mirror of the IAM
Handwriting Database), `c3rl/IIIT-INDIC-HW-WORDS-{Tamil,Hindi}` (CVIT, IIIT Hyderabad;
Gongidi & Jawahar), `priyank-m/IAM_words_text_recognition`, Mozhi (printed Indic words, IIIT Hyderabad) and Wikimedia Commons photographs (licences and authors in `experiments/mixed_real/sources.csv`) — all `REQUIRES_REVIEW`, none redistributed. V1: OpenCV (EAST),
Hugging Face Transformers + Microsoft TrOCR (MIT), `ocr_tamil` (GnanaPrasath, MIT),
EasyOCR (JaidedAI, Apache 2.0). The repository itself has no LICENSE file yet.

### 18.4 Glossary

| Term | Meaning |
|---|---|
| OCR | optical character recognition |
| Script vs language | script = writing system (Latin, Tamil, Devanagari); language = Hindi, English, ... |
| Detection / recognition | finding text regions vs reading the characters in a region |
| EAST | "Efficient and Accurate Scene Text" detector (V1) |
| CRAFT, PARSEQ, CRNN | detection/recognition network families used by `ocr_tamil` / EasyOCR |
| TrOCR | transformer OCR model (V1 English) |
| PP-OCR (v5/v6) | PaddleOCR's detection+recognition model families (V2) |
| NMS | non-maximum suppression; removes overlapping duplicate boxes |
| oneDNN / MKL-DNN | CPU acceleration library used by PaddlePaddle (disabled here) |
| CER | character error rate (Levenshtein / reference length) |
| Levenshtein distance | minimum single-character insertions, deletions, substitutions |
| Code point / grapheme | Unicode unit vs user-perceived character (Indic graphemes span several code points) |
| Virama, conjunct | Indic marks that join consonants into one visual cluster |
| Shirorekha | Devanagari horizontal headline stroke |
| Script leakage | a recognizer for script X emitting Latin letters when it fails |
| Glyph mass | confidence-weighted count of in-script characters (script-ID score) |
| Share | winner's fraction of total glyph mass |
| Manual / auto mode | user-chosen language vs automatic script identification |
| Design / held-out split | images used while designing the script-ID rule vs images not examined beforehand |
| Warm-up | loading all models at startup so requests are fast |
| V1 / V2 | previous (`legacy_v1/`) / current (repo root) architecture |

### 18.5 Author

Ripun — IT Engineering, VIT Vellore.

### 18.6 Note

Academic/portfolio project. The numbers here are real and reproducible
(`experiments/runs/`) but come from small-scale evaluations on one machine; read the
limitations before relying on any of them.
