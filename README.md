# Multilingual OCR Detection System

A modular Optical Character Recognition (OCR) pipeline that detects, groups, classifies, and recognizes text using a multiplexed architecture inspired by research literature.

---

## Overview

This project implements an end-to-end OCR system that:

- Detects text regions using the EAST text detector  
- Groups word-level detections into sentence-level regions  
- Classifies text type (printed / handwritten)  
- Routes inputs dynamically using a multiplexer  
- Recognizes text using multiple TrOCR models  

The system is designed as a modular pipeline and reflects concepts from multiplexed OCR architectures.

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
├── detection/  
│   └── east_detector.py  
├── grouping/  
│   └── text_grouping.py  
├── classification/  
│   └── classifier.py  
├── recognition/  
│   └── trocr_recognizer.py  
├── models/  
│   └── frozen_east_text_detection.pb  
├── input/images/  
├── output/  
│   ├── cropped/  
│   └── ocr_results.txt  

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

pip install numpy opencv-python imutils pillow torch torchvision torchaudio transformers accelerate tqdm  

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

python main.py --input path/to/image.png --output-dir path/to/output  

- `--input` — path to the input image (default: `input/images/sample.png`)
- `--output-dir` — directory for cropped regions and results (default: `output`)

---

## Output

- Cropped text regions: output/cropped/  
- Recognized text: output/ocr_results.txt  

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

- Heuristic classifier (not learned)  
- No rotation handling  
- Limited multilingual capability  
- Performance drops on very small text regions  

---

## Future Work

- Replace classifier with CNN or CLIP-based model  
- Add multilingual support  
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

---

## Note

This project is a prototype intended for academic demonstration, system design exploration, and understanding of modern OCR architectures.
