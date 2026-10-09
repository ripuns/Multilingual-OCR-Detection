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

ALLOWED_SCRIPTS = {"auto", "english", "tamil", "hindi"}

_WARM = {"english": False, "tamil": False, "hindi": False, "done": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Warming up PaddleOCR for all 3 languages (English, Tamil, Hindi)...")
    start = time.monotonic()
    warm_up()
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
        "This project produced linked engineering findings, not a new algorithm. "
        "V1: a modular OCR pipeline's detector failed differently depending on the "
        "script it was extended to (a grouping bug for English, under-sized boxes "
        "for Tamil, no boxes at all for Hindi). V2: one PaddleOCR engine per language "
        "removed those detection failures, but on real third-party handwriting the "
        "engines still make many errors, notably Latin-letter 'script leakage' by the "
        "Tamil engine. A clean synthetic pilot had made accuracy look far better than "
        "it is on real data."
    ),
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
        {"script": "English", "dataset": "Teklia/IAM-line", "unit": "sentence line", "mean_len": 60.9, "n": 30, "cer": 0.241, "exact_match": "0/30", "synthetic_cer": None},
        {"script": "Hindi", "dataset": "IIIT-INDIC-HW-WORDS-Hindi", "unit": "word", "mean_len": 6.4, "n": 30, "cer": 0.477, "exact_match": "2/30", "synthetic_cer": 0.071},
        {"script": "Tamil", "dataset": "IIIT-INDIC-HW-WORDS-Tamil", "unit": "word", "mean_len": 9.6, "n": 30, "cer": 0.711, "exact_match": "1/30", "synthetic_cer": 0.259},
    ],
    "leakage": [
        {"script": "Tamil", "with_latin": "17/30", "cer_with": 0.912, "cer_without": 0.461},
        {"script": "Hindi", "with_latin": "2/30", "cer_with": 1.167, "cer_without": 0.455},
    ],
    "script_id": {
        "design": {"n": 90, "accuracy": 0.767, "per_script": "English 28/30 / Tamil 15/30 / Hindi 26/30"},
        "heldout": {"n": 60, "accuracy": 0.783, "per_script": "English 20/20 / Tamil 10/20 / Hindi 17/20"},
        "note": "Current rule (argmax + Indic override). First argmax rule: 65.3% overall (held-out 70.0%). The current rule was revised before the evaluation, so its held-out figure is not a clean unseen-data estimate. Handwritten Tamil is the weak case: the Tamil engine often emits no Tamil at all, so choose Tamil manually when you know it.",
    },
    "vendor_benchmarks": {
        "english": {"value": 0.8525, "model": "en_PP-OCRv5_mobile_rec (this project's English pipeline uses PP-OCRv6, so this is NOT the model in use)"},
        "tamil": {"value": 0.942, "model": "ta_PP-OCRv5_mobile_rec"},
        "hindi": {"value": 0.8496, "model": "devanagari_PP-OCRv5_mobile_rec"},
    },
    "caveats": [
        "The three CERs are NOT comparable across languages: English is scored on ~61-character sentence lines, Tamil/Hindi on 6-10-character isolated words, from different datasets and writers. No claim about which script is harder is supported.",
        "n=30 per language, first rows of each dataset (not random); no confidence intervals. CER counts Unicode code points, which penalizes Tamil/Devanagari more than a grapheme-level metric would.",
        "The synthetic-vs-real comparison mixes an engine change (V1 -> V2) with a data change; it supports a caution about clean synthetic pilots, not a clean estimate of a synthetic-to-real gap.",
        "Vendor benchmark figures were measured by PaddleOCR on its own test sets with specific PP-OCRv5 mobile models; they are context, not a comparison point.",
        "V1 was never run on the real datasets, so V2 is not claimed to be more accurate than V1 on the same data.",
        "A patent claim was investigated and dropped (see docs/patent_ip_assessment.md). Script identification is an established problem; this implementation is a usability feature, not a novelty claim.",
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

    was_warm = _WARM["done"] if script == "auto" else _WARM.get(script, False)
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
