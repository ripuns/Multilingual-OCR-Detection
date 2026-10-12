import os
import re
import shutil
import sys
import time
import uuid
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import load_config
from main import run_pipeline, warm_up

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

ALLOWED_SCRIPTS = {"auto", "mixed", "english", "tamil", "hindi"}

_WARM = {"english": False, "tamil": False, "hindi": False, "done": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Warming up PaddleOCR for all 3 languages (English, Tamil, Hindi)...")
    start = time.monotonic()
    warm_up(load_config("config.yaml"))
    elapsed = time.monotonic() - start
    for script in ("english", "tamil", "hindi"):
        _WARM[script] = True
    _WARM["done"] = True
    print(f"Warm-up complete in {elapsed:.1f}s. All 3 languages ready.")
    yield


app = FastAPI(title="Multilingual OCR", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "warm": _WARM}


# Static research findings for the demo's "Research" view. Mirrors
# docs/research_contribution.md's V1-vs-V2 comparison -- duplicated here
# (not parsed from the doc) so the frontend has a stable, simple JSON shape
# independent of markdown formatting.
RESEARCH_FINDINGS = {
    "claim": (
        "This project produced linked engineering and evaluation findings, not a new OCR algorithm. "
        "One PaddleOCR engine per language reads English, Tamil and Hindi. On real handwriting the stock Tamil and Hindi "
        "engines make many errors, and the stock Tamil engine often answers in Latin letters. Fine-tuning only the sequence head of "
        "each recogniser on isolated handwritten words (a CPU-only recipe) cut Tamil CER from 73.5% to 16.4% and Hindi CER from "
        "41.9% to 16.1% on 200 fresh words each, but the words come from one dataset family, so this shows adaptation to that "
        "handwriting source, not general Tamil or Hindi OCR. A small learned classifier over the three engines' output statistics "
        "chooses the script far better than a hand-written rule (93.2% vs 55.5% on 600 fresh images with the adapted engines), "
        "and the adapted engines changed the engines' behaviour enough that the classifier had to be refitted for them."
    ),
    "tiles": [
        {"value": "93.2%", "label": "script identification on 600 fresh images (95% CI 91.0-95.0)"},
        {"value": "16.4%", "label": "Tamil character error rate after adaptation (stock 73.5%), 200 words"},
        {"value": "16.1%", "label": "Hindi character error rate after adaptation (stock 41.9%), 200 words"},
        {"value": "56/60", "label": "lines of synthetic mixed pages given the right script (rule: 34/60)"},
    ],
    "v1_gradient": [
        {
            "script": "English",
            "detector": "EAST (legacy_v1)",
            "detection_succeeds": "Yes",
            "failure_mode": "Grouping bug clipped text (fixed)",
            "metric_value": "19/32 lines recovered clipped text after fix (qualitative)",
            "mitigated": "Fixed at the source",
        },
        {
            "script": "Tamil",
            "detector": "EAST (legacy_v1)",
            "detection_succeeds": "Yes, but boxes too small",
            "failure_mode": "Recognized text truncated",
            "metric_value": "46.3% -> 25.9% CER with 20px box padding (5 synthetic images)",
            "mitigated": "Partially, via box padding",
        },
        {
            "script": "Hindi",
            "detector": "EAST (legacy_v1)",
            "detection_succeeds": "No -- zero boxes (1 synthetic image, 2 thresholds)",
            "failure_mode": "Nothing detected; backend swapped",
            "metric_value": "7.1% CER via EasyOCR's own detector (5 synthetic images)",
            "mitigated": "Fixed by swapping detectors",
        },
    ],
    "v2_real_data": [
        {"script": "English", "dataset": "Teklia/IAM-line (test rows 110-309)", "unit": "sentence line", "n": 200, "cer": 0.1402, "exact_match": "7/200", "synthetic_cer": None},
        {"script": "Hindi", "dataset": "IIIT-INDIC-HW-WORDS-Hindi (rows 1100-1299)", "unit": "word", "n": 200, "cer": 0.161, "exact_match": "96/200", "synthetic_cer": 0.071, "stock_cer": 0.4191, "note": "head-adapted Hindi recogniser"},
        {"script": "Tamil", "dataset": "IIIT-INDIC-HW-WORDS-Tamil (rows 7100-7299)", "unit": "word", "n": 200, "cer": 0.164, "exact_match": "97/200", "synthetic_cer": 0.259, "stock_cer": 0.7353, "note": "head-adapted Tamil recogniser"},
    ],
    "leakage_after_finetune": "After head adaptation 0/200 Tamil and 0/200 Hindi predictions contained Latin letters on the 600-image test set (stock: 107/200 Tamil, 20/200 Hindi).",
    "leakage": [
        {"script": "Tamil", "with_latin": "17/30", "cer_with": 0.912, "cer_without": 0.461},
        {"script": "Hindi", "with_latin": "2/30", "cer_with": 1.167, "cer_without": 0.455},
    ],
    "script_id": {
        "rows": [
            {"split": "Large test set: 600 fresh images, adapted engines", "n": 600, "accuracy": 0.9317, "per_script": "English 196/200 / Tamil 179/200 / Hindi 184/200 (95% CI 91.0-95.0; rule 55.5%, argmax 83.3%)"},
            {"split": "Large test set: 600 fresh images, stock engines (stock weights)", "n": 600, "accuracy": 0.91, "per_script": "English 198/200 / Tamil 166/200 / Hindi 182/200 (rule 77.2%, argmax 66.8%)"},
            {"split": "Earlier fresh set: 120 images, adapted engines", "n": 120, "accuracy": 0.875, "per_script": "English 34/40 / Tamil 34/40 / Hindi 37/40 (a cluster of English lines is misrouted to Hindi)"},
            {"split": "Cross-validation on the 550 fitting images", "n": 550, "accuracy": 0.9018, "per_script": "English 232/250 / Tamil 128.4/150 / Hindi 135.6/150 (mean over 10 repeats)"},
            {"split": "Mixed pages, each line scored (20 stitched pages)", "n": 60, "accuracy": 0.9333, "per_script": "56/60 lines; 16/20 pages fully correct (rule: 34/60)"},
            {"split": "Short handwritten English words (IAM words)", "n": 60, "accuracy": 0.8333, "per_script": "50/60; 58.3% before 100 short training words were added to the fitting set, and equal to argmax (50/60)"},
            {"split": "Printed words (Mozhi Tamil and Hindi)", "n": 100, "accuracy": 0.82, "per_script": "Tamil 47/50 / Hindi 35/50 (domain shift: fitted on handwriting)"},
        ],
        "note": "Calibrated classifier (small logistic regression on the three engines' output statistics) paired with the engine set it was fitted on. With the adapted Tamil and Hindi engines the earlier hand-written rule fails (it labels almost every English image as Tamil or Hindi) because those engines read other scripts as their own, so the classifier is required. Tesseract's script detection answered on only 57 of 600 images, so it is not a usable baseline for word crops. English is scored on lines in the large set and on short words in a separate set; the selector's lead over argmax disappears on short English words.",
    },
    "comparisons": [
        {
            "title": "Comparison with other open-source OCR (character error rate, 200 fresh handwriting images per script)",
            "headers": ["System", "English lines", "Hindi words", "Tamil words"],
            "rows": [
                ["This project, adapted recognisers", "14.0%", "16.1%", "16.4%"],
                ["This project, stock recognisers", "14.0%", "41.9%", "73.5%"],
                ["EasyOCR 1.7.2 (hi+en reader; Tamil model fails to load)", "57.3%", "68.3%", "n/a"],
                ["Tesseract 5.4 (eng / hin / tam, one line per image)", "40.9%", "83.2%", "85.2%"],
            ],
            "note": "The true script is given to every system, so this compares recognition only. Default settings, no tuning of the competitors; the adapted recognisers were trained on handwriting from the same dataset family as the Hindi and Tamil test words.",
        },
        {
            "title": "Printed words (Mozhi test set, 50 random words per script)",
            "headers": ["System", "Hindi CER", "Tamil CER"],
            "rows": [
                ["This project, adapted recognisers", "35.2%", "7.6%"],
                ["This project, stock recognisers", "53.8%", "34.6%"],
                ["EasyOCR 1.7.2", "46.2%", "n/a"],
                ["Tesseract 5.4", "37.2%", "28.9%"],
            ],
            "note": "Printed words are a different domain from the handwriting the adapted recognisers were trained on. On Hindi the intervals overlap widely (n=50), so no ranking is claimed.",
        },
        {
            "title": "Real mixed-script photographs (61 Wikimedia Commons signs; precision / recall of script presence)",
            "headers": ["System", "English", "Tamil", "Hindi", "Exact script set"],
            "rows": [
                ["This project, mixed mode (per-line script, ft2 weights)", "100% / 92%", "91% / 77%", "86% / 83%", "57.4% [44.3, 68.9]"],
                ["This project, one script per image (auto)", "100% / 69%", "100% / 26%", "100% / 25%", "0.0% [0.0, 0.0]"],
                ["EasyOCR 1.7.2 (hi+en reader; no Tamil)", "100% / 100%", "n/a / 0%", "60% / 97%", "36.1% [24.6, 49.2]"],
                ["Tesseract 5.4, eng+tam+hin, --psm 7 (one line)", "100% / 8%", "n/a / 0%", "50% / 3%", "0.0% [0.0, 0.0]"],
                ["Tesseract 5.4, eng+tam+hin, --psm 11 (sparse text)", "100% / 100%", "64% / 100%", "60% / 81%", "27.9% [16.4, 39.3]"]
            ],
            "note": "Labels are per-image script presence from visual inspection, not per line; a script counts when at least 3 of its characters are read. Competitors report scripts that are not present, which lowers their exact-set rate. Small n; several photos share locations.",
        },
    ],
    "vendor_benchmarks": {
        "english": {"value": 0.8525, "model": "en_PP-OCRv5_mobile_rec (this project's English pipeline uses PP-OCRv6, so this is NOT the model in use)"},
        "tamil": {"value": 0.942, "model": "ta_PP-OCRv5_mobile_rec"},
        "hindi": {"value": 0.8496, "model": "devanagari_PP-OCRv5_mobile_rec"},
    },
    "caveats": [
        "The three CERs are NOT comparable across languages: English is scored on sentence lines and Tamil/Hindi on isolated words, from different datasets and writers. No claim about which script is harder is supported.",
        "The Tamil and Hindi gains come from one dataset family (IIIT-INDIC-HW-WORDS). Test rows differ from training rows but may share writers; the datasets give no writer IDs. No independent Tamil or Hindi handwriting set was found, so generalisation is NOT shown.",
        "The selector was refitted for the adapted engines, then once more after short English words were added when it failed on them (58.3% -> 83.3%); the 600-image set, the 120-image set and the English words were all seen before that last change, so those numbers are not a clean one-shot estimate.",
        "The mixed pages are synthetic (one script per stitched line). A small set of real Wikimedia Commons photos was labelled by script presence only, by visual inspection, not by a Tamil or Hindi reader.",
        "n is 40-200 per script, intervals are bootstrap over images, there is a single training run per language, and CER counts Unicode code points, which penalises Tamil and Devanagari more than a grapheme-level metric would.",
        "Competitors were run with default settings and one text line per image; EasyOCR's Tamil model could not be loaded, and Tesseract's script detection failed on most word crops.",
        "Vendor benchmark figures were measured by PaddleOCR on its own test sets with specific PP-OCRv5 mobile models; they are context, not a comparison point.",
        "V1 was never run on the real datasets, so V2 is not claimed to be more accurate than V1 on the same data.",
        "A patent claim was investigated and dropped. Script identification is an established problem; this implementation is a usability feature, not a novelty claim.",
    ],
}


@app.get("/api/research")
def research():
    return RESEARCH_FINDINGS


@app.post("/api/ocr")
async def ocr(file: UploadFile = File(...), script: str = Form("auto")):
    if script not in ALLOWED_SCRIPTS:
        raise HTTPException(status_code=400, detail=f"script must be one of {sorted(ALLOWED_SCRIPTS)}")

    run_id = uuid.uuid4().hex[:12]
    run_dir = os.path.join(UPLOADS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in {".png", ".jpg", ".jpeg", ".bmp"}:
        ext = ".png"
    input_path = os.path.join(run_dir, f"input{ext}")
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    config = load_config("config.yaml")
    config["script"] = script

    was_warm = _WARM["done"] if script in ("auto", "mixed") else _WARM.get(script, False)
    start = time.monotonic()
    try:
        outcome = run_pipeline(input_path, run_dir, config)
        results, detection = outcome["results"], outcome["detection"]
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")
    elapsed = time.monotonic() - start

    for r in results:
        r["crop_url"] = f"/api/crop/{run_id}/{r['index']}.png"

    return JSONResponse({
        "run_id": run_id,
        "requested_script": script,
        "script": detection["script"],
        "detection": detection,
        "results": results,
        "elapsed_seconds": round(elapsed, 2),
        "was_warm": was_warm,
    })


SAMPLE_IMAGE = os.path.join(os.path.dirname(BASE_DIR), "input", "images", "sample.png")


@app.get("/api/sample")
def sample():
    """The repository's fixed sample page, so the UI can offer a one-click demo."""
    if not os.path.isfile(SAMPLE_IMAGE):
        raise HTTPException(status_code=404, detail="sample image not available")
    return FileResponse(SAMPLE_IMAGE, media_type="image/png")


@app.get("/api/crop/{run_id}/{filename}")
def get_crop(run_id: str, filename: str):
    if not re.fullmatch(r"[a-f0-9]{12}", run_id) or not re.fullmatch(r"\d+\.png", filename):
        raise HTTPException(status_code=400, detail="invalid run_id or filename")

    path = os.path.join(UPLOADS_DIR, run_id, "cropped", filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="crop not found")
    return FileResponse(path)


app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")
