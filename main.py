import argparse
import logging
import os
import sys

import cv2
from PIL import Image

from detection.east_detector import EASTDetector
from grouping.text_grouping import group_text
from classification.classifier import TextClassifier
from recognition.trocr_recognizer import TrOCRRecognizer
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Multilingual OCR pipeline")
    parser.add_argument("--input", default="input/images/sample.png", help="Path to the input image")
    parser.add_argument("--output-dir", default="output", help="Directory for cropped images and results")
    return parser.parse_args()


def configure_logging():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def run_pipeline(image_path, output_dir):
    detector = EASTDetector()
    classifier = TextClassifier()
    recognizer = TrOCRRecognizer()

    boxes, image = detector.detect_text(image_path)
    sentence_boxes = group_text(boxes)

    cropped_dir = os.path.join(output_dir, "cropped")
    os.makedirs(cropped_dir, exist_ok=True)

    results = []
    h, w = image.shape[:2]

    for i, (x1, y1, x2, y2) in enumerate(sentence_boxes):
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]

        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        label = classifier.classify(pil_img)
        text = recognizer.recognize(pil_img, label)

        results.append(text)

        cv2.imwrite(os.path.join(cropped_dir, f"{i}.png"), crop)

        logger.info("[%d] %s -> %s", i, label, text)

    results_path = os.path.join(output_dir, "ocr_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        for line in results:
            f.write(line + "\n")


if __name__ == "__main__":
    configure_logging()
    args = parse_args()
    run_pipeline(args.input, args.output_dir)
