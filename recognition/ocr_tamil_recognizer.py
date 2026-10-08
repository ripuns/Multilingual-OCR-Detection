import os
import tempfile

from ocr_tamil.ocr import OCR

from recognition.registry import register

register("tamil", "ocr_tamil (CRAFT detection + PARSEQ recognition)")


class OcrTamilRecognizer:
    def __init__(self):
        self._ocr = None

    def _load(self):
        if self._ocr is None:
            self._ocr = OCR()
        return self._ocr

    def recognize(self, image, label="tamil"):
        """Returns (text, route). ocr_tamil's API takes an image path, so the
        PIL crop is written to a temp file for the call."""
        ocr = self._load()

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            image.save(path)
            result = ocr.predict(path)
        finally:
            os.remove(path)

        text = result[0] if result else ""
        return text, "ocr_tamil (CRAFT detection + PARSEQ recognition)"
