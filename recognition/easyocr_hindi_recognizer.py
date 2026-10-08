import numpy as np

from recognition.registry import register

register("hindi", "easyocr (CRAFT detection + CRNN recognition, lang='hi')")


class EasyOcrHindiRecognizer:
    """EAST produces zero detections for Devanagari even at near-zero confidence
    (see docs/research_contribution.md), so this recognizer owns its own
    detection via EasyOCR's bundled CRAFT model instead of taking an
    EAST-cropped region. main.py routes the whole image here directly for
    script=='hindi', bypassing EAST/grouping/clamping entirely."""

    def __init__(self):
        self._reader = None

    def _load(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(["hi", "en"], gpu=False, verbose=False)
        return self._reader

    def detect_and_recognize(self, image):
        """Returns a list of {bbox: [x1,y1,x2,y2], text, route} for the whole image."""
        reader = self._load()
        arr = np.array(image)

        detections = reader.readtext(arr, detail=1)
        route = "easyocr (CRAFT detection + CRNN recognition, lang='hi')"

        results = []
        for bbox_points, text, _confidence in detections:
            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
            results.append({"bbox": bbox, "text": text, "route": route})

        return results
