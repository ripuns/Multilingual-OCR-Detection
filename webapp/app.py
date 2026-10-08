import os
import shutil
import tempfile
import uuid

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config
from main import run_pipeline

app = FastAPI(title="Multilingual OCR Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

ALLOWED_SCRIPTS = {"english", "tamil", "hindi"}


@app.get("/api/health")
def health():
    return {"status": "ok"}


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

    try:
        results = run_pipeline(input_path, run_dir, config)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    for r in results:
        r["crop_url"] = f"/api/crop/{run_id}/{r['index']}.png"

    return JSONResponse({"run_id": run_id, "script": script, "results": results})


@app.get("/api/crop/{run_id}/{filename}")
def get_crop(run_id: str, filename: str):
    import re
    if not re.fullmatch(r"[a-f0-9]{12}", run_id) or not re.fullmatch(r"\d+\.png", filename):
        raise HTTPException(status_code=400, detail="invalid run_id or filename")

    path = os.path.join(UPLOADS_DIR, run_id, "cropped", filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="crop not found")
    return FileResponse(path)


app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")
