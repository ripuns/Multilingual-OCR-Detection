# Multilingual OCR Detection System

A modular Optical Character Recognition (OCR) pipeline that detects, groups, classifies, and recognizes text using a multiplexed architecture inspired by research literature.

---

## Overview

This project implements an end-to-end OCR system supporting three languages/scripts, selected per run with `--script`:

- **English** — EAST detection, heuristic printed/handwritten classification, TrOCR recognition
- **Tamil** — EAST detection, `ocr_tamil` (CRAFT+PARSEQ) recognition
- **Hindi** — EasyOCR's own detection + recognition (EAST produces zero detections for Devanagari — see below)

All three are pilot-scale additions on top of the original English pipeline —
see `docs/limitations.md` and `docs/research_contribution.md` before relying on
Tamil/Hindi accuracy. The actual contribution of this project is not "it
supports 3 languages" by itself, but what extending it to 3 languages revealed:
**a measured severity gradient in how a modular pipeline's detector fails when
applied to a script it wasn't built for** — English had a recoverable grouping
bug (now fixed), Tamil's detector produces boxes that are systematically
under-sized (CER 46.3% on a small pilot), and Hindi's detector (EAST) produces
*no boxes at all*, requiring a full detection-backend swap (after which CER
drops to 7.1%). Full writeup in `docs/research_contribution.md`.

---

## Architecture

Input Image
   ↓
[EAST Text Detection]
   ↓
[Word Bounding Boxes]
   ↓
[Text Grouping]
   ↓
[Sentence Bounding Boxes]
   ↓
[Classifier]
   ↓
[Multiplexer]
   ↓
[TrOCR Models]
   ↓
[Recognized Text Output]

---

## Key Features

- EAST-based text detection using OpenCV  
- Sentence-level grouping via heuristic clustering  
- Lightweight text classification  
- Multiplexed routing mechanism  
- Transformer-based OCR (TrOCR)  
- Modular and extensible design  

---

## Tech Stack

| Component | Technology |
|----------|-----------|
| Detection | OpenCV (EAST) |
| Recognition | TrOCR (HuggingFace Transformers) |
| Backend | PyTorch |
| Image Processing | OpenCV, PIL |

---

## Project Structure

ocr_project/

├── main.py  
├── config.py  
├── config.yaml  
├── detection/  
│   └── east_detector.py  
├── grouping/  
│   └── text_grouping.py  
├── classification/  
│   └── classifier.py  
├── recognition/  
│   ├── registry.py  
│   ├── trocr_recognizer.py  
│   ├── ocr_tamil_recognizer.py  
│   └── easyocr_hindi_recognizer.py  
├── models/  
│   └── frozen_east_text_detection.pb  
├── input/images/  
├── output/  
│   ├── cropped/  
│   └── ocr_results.txt  
├── tests/  

---

## Installation

### 1. Clone repository

git clone https://github.com/ripuns/Multilingual-OCR-Detection.git  
cd Multilingual-OCR-Detection  

### 2. Create virtual environment

python -m venv ocr_env  

Activate:

Windows:  
ocr_env\Scripts\activate  

Linux/Mac:  
source ocr_env/bin/activate  

### 3. Install dependencies

pip install -r requirements.txt  

Versions are pinned in `requirements.txt` for reproducibility (includes `pytest`,
needed to run `tests/`).

---

## Model Setup

Download the EAST model from:

https://github.com/argman/EAST/releases

Place the file in:

models/frozen_east_text_detection.pb  

---

## Usage

python main.py  

Default input:

input/images/sample.png  

Optional flags:

python main.py --input path/to/image.png --output-dir path/to/output --config path/to/config.yaml --script tamil  

- `--input` — path to the input image (overrides `config.yaml`'s `paths.input`)
- `--output-dir` — directory for cropped regions and results (overrides `config.yaml`'s `paths.output_dir`)
- `--config` — path to the config file (default: `config.yaml`)
- `--script` — `english` (default), `tamil`, or `hindi`. This is an explicit
  choice per run, not automatic script detection — see `docs/limitations.md`.

---

## Configuration

Runtime settings live in `config.yaml` at the repo root (loaded by `config.py`;
missing keys fall back to built-in defaults, missing file falls back entirely
to defaults):

```yaml
detection:
  min_confidence: 0.3       # EAST score threshold
  nms_overlap_thresh: 0.3   # EAST non-max-suppression overlap threshold
grouping:
  v_tol_multiplier: 0.5     # vertical tolerance, x average box height
  h_gap_multiplier: 1.5     # horizontal gap tolerance, x average box height
device: auto                 # auto | cpu | cuda
script: english               # english | tamil | hindi
paths:
  input: input/images/sample.png
  output_dir: output
logging:
  level: INFO
```

`--input`/`--output-dir` CLI flags take priority over `config.yaml` when both
are given.

---

## Output

- Cropped text regions: output/cropped/  
- Recognized text (plain): output/ocr_results.txt  
- Recognized text (structured): output/ocr_results.json — one entry per
  region: `index`, `bbox` (`[x1, y1, x2, y2]`), `label`, `route` (the
  resolved model id), `text`.

---

## Example Output

[0] printed → Hello World  
[1] printed → OCR Pipeline  

---

## Design Insight

This project reflects a simplified implementation of multiplexed OCR systems:

| Concept | Implementation |
|--------|--------------|
| Multiple recognition heads | Multiple TrOCR models |
| Routing mechanism | Heuristic classifier |
| Multiplexer | Conditional model selection |
| End-to-end pipeline | Modular design |

---

## Limitations

- Heuristic classifier (not learned), English path only
- No rotation handling
- No automatic script detection — `--script` is an explicit per-run choice
- Tamil and Hindi routes are small (n=5), synthetic pilots — see
  `docs/limitations.md` and `docs/research_contribution.md` for the measured
  accuracy and its caveats
- Hindi uses a different detector (EasyOCR) than English/Tamil (EAST) — not
  directly comparable as "the same pipeline, different language"
- Performance drops on very small text regions

---

## Future Work

- Replace classifier with CNN or CLIP-based model
- Automatic script detection (currently manual via `--script`)
- Larger, real (non-synthetic) evaluation sets for Tamil and Hindi; isolate
  why EAST produces zero detections for Devanagari specifically (font
  rendering artifact vs. a more fundamental mismatch — see
  `docs/research_contribution.md`)
- Additional scripts/languages beyond English, Tamil, and Hindi
- Improve grouping with clustering algorithms
- Handle rotated and curved text
- Deploy as API (FastAPI)
- Enable real-time OCR

---

## Resume Description

Developed a modular OCR pipeline using EAST for detection and TrOCR for recognition, implementing a multiplexed architecture that dynamically routes inputs across specialized models.

---

## Author

Ripun  
IT Engineering, VIT Vellore  

---

## Acknowledgements

- OpenCV  
- HuggingFace Transformers  
- PyTorch  
- Microsoft TrOCR  
- `ocr_tamil` (CRAFT + PARSEQ) by GnanaPrasath, MIT licensed  
- EasyOCR (JaidedAI), Apache 2.0 licensed  

---

## Note

This project is a prototype intended for academic demonstration, system design exploration, and understanding of modern OCR architectures.
