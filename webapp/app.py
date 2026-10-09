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
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

from config import load_config
from main import run_pipeline, warm_up

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

ALLOWED_SCRIPTS = {"english", "tamil", "hindi"}

_WARM = {"english": False, "tamil": False, "hindi": False, "done": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Warming up all 3 recognizers (English, Tamil, Hindi) at startup...")
    config = load_config("config.yaml")
    start = time.monotonic()
    warm_up(config)
    elapsed = time.monotonic() - start
    for script in ("english", "tamil", "hindi"):
        _WARM[script] = True
    _WARM["done"] = True
    print(f"Warm-up complete in {elapsed:.1f}s. All 3 languages ready for fast requests.")
    yield


app = FastAPI(title="Multilingual OCR Demo", lifespan=lifespan)

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
# docs/research_contribution.md's gradient table and padding-fix result --
# duplicated here (not parsed from the doc) so the frontend has a stable,
# simple JSON shape independent of markdown formatting.
RESEARCH_FINDINGS = {
    "claim": (
        "A modular OCR pipeline's detector fails differently, and to different "
        "degrees, when extended to a script it wasn't built for -- a measured "
        "severity gradient, not a uniform 'works' or 'doesn't work.' Where the "
        "failure is geometric under-sizing (not complete detection failure), it "
        "is also partially correctable with a simple, cheap mitigation (box "
        "padding)."
    ),
    "gradient": [
        {
            "script": "English",
            "detector": "EAST",
            "detection_succeeds": "Yes",
            "failure_mode": "Was a grouping bug (fixed); now none observed",
            "metric_label": "Qualitative (before/after text diff)",
            "metric_value": "19/32 lines recovered clipped text after fix",
            "mitigated": "Fixed at the source",
        },
        {
            "script": "Tamil",
            "detector": "EAST",
            "detection_succeeds": "Yes, but under-sized",
            "failure_mode": "Systematic trailing-edge clipping",
            "metric_label": "Character Error Rate (CER)",
            "metric_value": "46.3% -> 25.9% with 20px box padding",
            "mitigated": "Partially, via box padding",
        },
        {
            "script": "Hindi",
            "detector": "EAST",
            "detection_succeeds": "No -- zero boxes at any tested confidence",
            "failure_mode": "N/A for EAST; required a full detector swap",
            "metric_label": "Character Error Rate (CER)",
            "metric_value": "7.1% (post-swap, via EasyOCR's own detector)",
            "mitigated": "Fixed by swapping detectors",
        },
    ],
    "padding_sweep": [
        {"padding_px": 0, "cer": 0.463, "exact_matches": "0/5"},
        {"padding_px": 10, "cer": 0.352, "exact_matches": "1/5"},
        {"padding_px": 20, "cer": 0.259, "exact_matches": "1/5"},
        {"padding_px": 30, "cer": 0.370, "exact_matches": "1/5"},
    ],
    "caveats": [
        "All non-English numbers are from a small (n=5), synthetic pilot set -- not a general accuracy claim.",
        "Padding does not fix everything: one Tamil sample remains a severe outlier (77% CER) after the fix.",
        "Why EAST fails completely on Devanagari (vs. only under-sizing for Tamil) was not isolated -- only one font/rendering method was tested.",
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
