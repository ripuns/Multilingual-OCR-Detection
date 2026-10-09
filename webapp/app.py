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

ALLOWED_SCRIPTS = {"english", "tamil", "hindi"}

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
        "This project produced two linked findings, not one invented mechanism. "
        "V1: a modular OCR pipeline's detector fails differently, and to "
        "different degrees, when extended to a script it wasn't built for -- a "
        "measured severity gradient. V2: replacing that patchwork with a "
        "unified, actively-maintained engine (PaddleOCR) fixes the detection "
        "failures -- but evaluating it against real, independently-sourced "
        "handwriting (not synthetic renders) shows official/synthetic OCR "
        "benchmarks substantially overestimate real-world accuracy, and a real "
        "script-difficulty gradient persists even with a strong, unified backend."
    ),
    "v1_gradient": [
        {
            "script": "English",
            "detector": "EAST (legacy_v1)",
            "detection_succeeds": "Yes",
            "failure_mode": "Was a grouping bug (fixed); now none observed",
            "metric_value": "19/32 lines recovered clipped text after fix (qualitative)",
            "mitigated": "Fixed at the source",
        },
        {
            "script": "Tamil",
            "detector": "EAST (legacy_v1)",
            "detection_succeeds": "Yes, but under-sized",
            "failure_mode": "Systematic trailing-edge clipping",
            "metric_value": "46.3% -> 25.9% CER with 20px box padding (synthetic pilot)",
            "mitigated": "Partially, via box padding",
        },
        {
            "script": "Hindi",
            "detector": "EAST (legacy_v1)",
            "detection_succeeds": "No -- zero boxes at any tested confidence",
            "failure_mode": "N/A for EAST; required a full detector swap",
            "metric_value": "7.1% CER post-swap, via EasyOCR's own detector (synthetic pilot)",
            "mitigated": "Fixed by swapping detectors",
        },
    ],
    "v2_real_data": [
        {"script": "English", "dataset": "Teklia/IAM-line (real handwriting)", "n": 30, "cer": 0.241, "exact_match": "0/30", "synthetic_cer": None},
        {"script": "Hindi", "dataset": "IIIT-INDIC-HW-WORDS-Hindi (real handwriting)", "n": 30, "cer": 0.477, "exact_match": "2/30", "synthetic_cer": 0.071},
        {"script": "Tamil", "dataset": "IIIT-INDIC-HW-WORDS-Tamil (real handwriting)", "n": 30, "cer": 0.711, "exact_match": "1/30", "synthetic_cer": 0.259},
    ],
    "official_benchmarks": {"english": 0.8525, "tamil": 0.942, "hindi": 0.8496},
    "caveats": [
        "All real-data numbers are from n=30 samples per non-English language -- a real signal, not a statistically robust benchmark.",
        "Why PaddleOCR's official Tamil benchmark (94.2%) diverges so sharply from this project's real-handwriting measurement (71.1% CER) was not isolated.",
        "V1's findings remain valid descriptions of that architecture's failure modes; V1 is superseded as the production path, not 'wrong.'",
        "A patent claim was investigated and explicitly dropped as not viable -- see docs/patent_ip_assessment.md. This is a paper-track, engineering/evaluation contribution, not a novel algorithm.",
    ],
}


@app.get("/api/research")
def research():
    return RESEARCH_FINDINGS


@app.post("/api/ocr")
async def ocr(file: UploadFile = File(...), script: str = Form("english")):
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

    was_warm = _WARM.get(script, False)
    start = time.monotonic()
    try:
        results = run_pipeline(input_path, run_dir, config)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")
    elapsed = time.monotonic() - start
    _WARM[script] = True

    for r in results:
        r["crop_url"] = f"/api/crop/{run_id}/{r['index']}.png"

    return JSONResponse({
        "run_id": run_id,
        "script": script,
        "results": results,
        "elapsed_seconds": round(elapsed, 2),
        "was_warm": was_warm,
    })


@app.get("/api/crop/{run_id}/{filename}")
def get_crop(run_id: str, filename: str):
    if not re.fullmatch(r"[a-f0-9]{12}", run_id) or not re.fullmatch(r"\d+\.png", filename):
        raise HTTPException(status_code=400, detail="invalid run_id or filename")

    path = os.path.join(UPLOADS_DIR, run_id, "cropped", filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="crop not found")
    return FileResponse(path)


app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")
