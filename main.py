import argparse
import json
import logging
import os
import sys

import cv2
from PIL import Image

from boxes import clamp_box
from config import load_config
from detection.east_detector import EASTDetector
from grouping.text_grouping import group_text
from classification.classifier import TextClassifier
from recognition.trocr_recognizer import TrOCRRecognizer
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Multilingual OCR pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to the config YAML file")
    parser.add_argument("--input", default=None, help="Path to the input image (overrides config)")
    parser.add_argument("--output-dir", default=None, help="Directory for cropped images and results (overrides config)")
    return parser.parse_args()


def configure_logging(level_name):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")


def run_pipeline(image_path, output_dir, config):
    detector = EASTDetector(
        min_confidence=config["detection"]["min_confidence"],
        nms_overlap_thresh=config["detection"]["nms_overlap_thresh"],
    )
    classifier = TextClassifier()
    recognizer = TrOCRRecognizer(device=config["device"])

    boxes, image = detector.detect_text(image_path)
    sentence_boxes = group_text(
        boxes,
        v_tol_multiplier=config["grouping"]["v_tol_multiplier"],
        h_gap_multiplier=config["grouping"]["h_gap_multiplier"],
    )

    cropped_dir = os.path.join(output_dir, "cropped")
    os.makedirs(cropped_dir, exist_ok=True)

    results = []
    h, w = image.shape[:2]

    for i, box in enumerate(sentence_boxes):
        clamped = clamp_box(*box, width=w, height=h)
        if clamped is None:
            continue
        x1, y1, x2, y2 = clamped

        crop = image[y1:y2, x1:x2]

        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        label = classifier.classify(pil_img)
        text, route = recognizer.recognize(pil_img, label)

        results.append({
            "index": i,
            "bbox": [x1, y1, x2, y2],
            "label": label,
            "route": route,
            "text": text,
        })

        cv2.imwrite(os.path.join(cropped_dir, f"{i}.png"), crop)

        logger.info("[%d] %s -> %s", i, label, text)

    txt_path = os.path.join(output_dir, "ocr_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(r["text"] + "\n")

    json_path = os.path.join(output_dir, "ocr_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    configure_logging(config["logging"]["level"])

    image_path = args.input or config["paths"]["input"]
    output_dir = args.output_dir or config["paths"]["output_dir"]

    run_pipeline(image_path, output_dir, config)
