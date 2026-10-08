# Multilingual OCR Detection System

A modular Optical Character Recognition (OCR) pipeline that detects, groups, classifies, and recognizes text using a multiplexed architecture inspired by research literature.

---

## Overview

This project implements an end-to-end OCR system that:

- Detects text regions using the EAST text detector  
- Groups word-level detections into sentence-level regions  
- Classifies text type (printed / handwritten) for the English path  
- Routes inputs dynamically using a registry-backed multiplexer  
- Recognizes English text via TrOCR, and Tamil text via a separate `ocr_tamil`-backed route, selected per run with `--script`  

The system is designed as a modular pipeline and reflects concepts from multiplexed OCR architectures. Tamil support is a small, pilot-scale addition — see `docs/limitations.md` and `docs/research_contribution.md` before relying on its accuracy.

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
│   └── ocr_tamil_recognizer.py  
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
- `--script` — `english` (default) or `tamil`. This is an explicit choice per run,
  not automatic script detection — see `docs/limitations.md`.

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
script: english               # english | tamil
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
- Tamil route is a small (n=5), synthetic pilot — see `docs/limitations.md` and
  `docs/research_contribution.md` for the measured accuracy and its caveats
- Performance drops on very small text regions

---

## Future Work

- Replace classifier with CNN or CLIP-based model
- Automatic script detection (currently manual via `--script`)
- Larger, real (non-synthetic) Tamil evaluation set; investigate whether EAST's
  Latin-script training is the cause of the Tamil pilot's clipping pattern
  (see `docs/research_contribution.md`)
- Additional scripts/languages beyond English and Tamil
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

---

## Note

This project is a prototype intended for academic demonstration, system design exploration, and understanding of modern OCR architectures.
