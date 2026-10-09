"""Mock of the Lekhak API for UI testing: serves the real static UI and returns canned
responses (the 16-region result is the real recorded output for input/images/sample.png in
output/ocr_results.json; the Tamil/Hindi responses are illustrative). No OCR models are loaded.
Run from anywhere:  python webapp/ui_tests/mock_server.py   (listens on 127.0.0.1:8765)"""
import io
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
sys.path.insert(0, REPO)
from webapp.app import RESEARCH_FINDINGS  # noqa: E402

SAMPLE = f"{REPO}/input/images/sample.png"
real = json.load(open(f"{REPO}/output/ocr_results.json", encoding="utf-8"))
for r in real:
    r["script"] = "english"

TAMIL = [{"index": 0, "bbox": [0, 13, 224, 78], "text": "வணக்கம்", "score": 0.93, "route": "PaddleOCR PP-OCRv5 (ta)", "script": "tamil"},
         {"index": 1, "bbox": [240, 13, 400, 78], "text": "தமிழ்நாடு", "score": 0.41, "route": "PaddleOCR PP-OCRv5 (ta)", "script": "tamil"}]
HINDI = [{"index": 0, "bbox": [3, 15, 123, 75], "text": "नमस्ते", "score": 0.97, "route": "PaddleOCR PP-OCRv5 (hi)", "script": "hindi"}]

PNG_CACHE = {}


def crop_png(box):
    key = tuple(box)
    if key not in PNG_CACHE:
        im = Image.open(SAMPLE).convert("RGB")
        w, h = im.size
        x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
        buf = io.BytesIO(); im.crop((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1))).save(buf, "PNG")
        PNG_CACHE[key] = buf.getvalue()
    return PNG_CACHE[key]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/api/health":
            return self.send(200, {"status": "ok", "warm": {"english": True, "tamil": True, "hindi": True, "done": True}})
        if p == "/api/research":
            return self.send(200, RESEARCH_FINDINGS)
        if p == "/api/sample":
            return self.send(200, open(SAMPLE, "rb").read(), "image/png")
        m = re.fullmatch(r"/api/crop/[a-f0-9]{12}/(\d+)\.png", p)
        if m:
            i = int(m.group(1)); box = real[i]["bbox"] if i < len(real) else [0, 0, 40, 20]
            return self.send(200, crop_png(box), "image/png")
        if p == "/":
            p = "/index.html"
        try:
            data = open(f"{REPO}/webapp/static{p}", "rb").read()
        except OSError:
            return self.send(404, {"detail": "not found"})
        self.send(200, data, "text/html; charset=utf-8" if p.endswith(".html") else "application/octet-stream")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = self.rfile.read(n)
        m = re.search(rb'name="script"\r\n\r\n(\w+)', body)
        script = m.group(1).decode() if m else "auto"
        time.sleep(1.8)
        if script == "tamil":
            results, det = TAMIL, {"mode": "manual", "script": "tamil", "fallback": False, "share": None, "masses": None}
        elif script == "hindi":
            results, det = HINDI, {"mode": "manual", "script": "hindi", "fallback": False, "share": None, "masses": None}
        elif script == "english":
            results, det = real, {"mode": "manual", "script": "english", "fallback": False, "share": None, "masses": None}
        else:
            results, det = real, {"mode": "auto", "script": "english", "fallback": False, "share": 1.0,
                                  "masses": {"english": 612.4, "tamil": 0.0, "hindi": 3.2}}
        rs = [dict(r, crop_url=f"/api/crop/aabbccddeeff/{r['index']}.png") for r in results]
        self.send(200, {"run_id": "aabbccddeeff", "requested_script": script, "script": det["script"], "detection": det,
                        "results": rs, "elapsed_seconds": 13.27 if script == "auto" else 3.1, "was_warm": True})


ThreadingHTTPServer(("127.0.0.1", 8765), H).serve_forever()
