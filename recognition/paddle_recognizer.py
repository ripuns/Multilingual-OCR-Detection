from recognition.registry import register

LANG_CODES = {"english": "en", "tamil": "ta", "hindi": "hi"}

# Which model family PaddleOCR selects per language is decided inside
# paddleocr==3.7.0 (see PaddleOCR._get_ocr_model_names): English resolves to the
# unified PP-OCRv6 medium det+rec models, Tamil/Devanagari to PP-OCRv5
# (ta_PP-OCRv5_mobile_rec / devanagari_PP-OCRv5_mobile_rec + PP-OCRv5_server_det).
# Re-verify these labels if the paddleocr pin changes.
ENGINE_LABELS = {
    "english": "PaddleOCR PP-OCRv6 (en)",
    "tamil": "PaddleOCR PP-OCRv5 (ta)",
    "hindi": "PaddleOCR PP-OCRv5 (hi)",
}

for _script, _label in ENGINE_LABELS.items():
    register(_script, _label)


class PaddleOcrRecognizer:
    """Unified detection+recognition backend for all 3 languages, replacing
    the V1 patchwork (EAST+TrOCR, ocr_tamil, EasyOCR -- see legacy_v1/ and
    docs/research_contribution.md for why). PaddleOCR owns both detection and
    recognition internally, so this is always self-detecting: one call per
    image, no separate grouping/clamping/classification stage needed."""

    def __init__(self, script):
        if script not in LANG_CODES:
            raise ValueError(f"Unsupported script '{script}'. Supported: {sorted(LANG_CODES)}")
        self.script = script
        self.lang_code = LANG_CODES[script]
        self._ocr = None

    def _load(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                lang=self.lang_code,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                # Disabling MKL-DNN works around a PaddlePaddle/oneDNN crash
                # (NotImplementedError on ConvertPirAttribute2RuntimeAttribute)
                # seen on this CPU during evaluation -- see docs/research_contribution.md.
                enable_mkldnn=False,
            )
        return self._ocr

    def warm_up(self):
        self._load()

    def detect_and_recognize(self, image_path):
        """Returns a list of {bbox: [x1,y1,x2,y2], text, score, route} for the whole image."""
        ocr = self._load()
        route = ENGINE_LABELS[self.script]

        results = []
        index = 0
        for page in ocr.predict(image_path):
            texts = page.get("rec_texts", [])
            scores = page.get("rec_scores", [])
            boxes = page.get("rec_boxes", [])
            for text, score, box in zip(texts, scores, boxes):
                results.append({
                    "index": index,
                    "bbox": [int(v) for v in box],
                    "text": text,
                    "score": round(float(score), 4),
                    "route": route,
                })
                index += 1

        return sort_reading_order(results)


def sort_reading_order(regions):
    """Orders regions top-to-bottom by line, then left-to-right within a line.
    PaddleOCR's own order can put a word before its left neighbour on the same
    line (e.g. 'सूत्र' before 'क्षार'). Two regions share a line when their
    vertical centres are within half the line's box height. Re-numbers `index`."""
    def centre_y(r):
        return (r["bbox"][1] + r["bbox"][3]) / 2

    lines = []
    for r in sorted(regions, key=centre_y):
        if lines:
            line = lines[-1]
            height = max(line[0]["bbox"][3] - line[0]["bbox"][1], 1)
            if abs(centre_y(r) - centre_y(line[0])) <= height / 2:
                line.append(r)
                continue
        lines.append([r])

    ordered = [r for line in lines for r in sorted(line, key=lambda r: r["bbox"][0])]
    for i, r in enumerate(ordered):
        r["index"] = i
    return ordered
