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
from recognition.ocr_tamil_recognizer import OcrTamilRecognizer
from recognition.easyocr_hindi_recognizer import EasyOcrHindiRecognizer
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Multilingual OCR pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to the config YAML file")
    parser.add_argument("--input", default=None, help="Path to the input image (overrides config)")
    parser.add_argument("--output-dir", default=None, help="Directory for cropped images and results (overrides config)")
    parser.add_argument(
        "--script",
        default=None,
        choices=["english", "tamil", "hindi"],
        help="Which recognition path to use (overrides config). 'english' and 'tamil' "
        "run EAST detection + grouping, then route the cropped region to that "
        "language's recognizer. 'hindi' bypasses EAST entirely and uses EasyOCR's "
        "own detector, because EAST produces zero detections for Devanagari (see "
        "docs/research_contribution.md). This is an explicit operator choice, not "
        "automatic script detection.",
    )
    return parser.parse_args()


# Recognizers that take an EAST-cropped region: recognize(image, label) -> (text, route)
CROP_BASED_RECOGNIZERS = {
    "tamil": OcrTamilRecognizer,
}

# Recognizers that own their own detection (EAST fails on these scripts):
# detect_and_recognize(image) -> [{bbox, text, route}, ...]
SELF_DETECTING_RECOGNIZERS = {
    "hindi": EasyOcrHindiRecognizer,
}


def configure_logging(level_name):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")


# Process-lifetime caches so a long-running server (the web demo) doesn't
# reload multi-GB models on every request. Keyed by whatever makes two
# configs equivalent for that object.
_DETECTOR_CACHE = {}
_RECOGNIZER_CACHE = {}


def _get_detector(config):
    key = (config["detection"]["min_confidence"], config["detection"]["nms_overlap_thresh"])
    if key not in _DETECTOR_CACHE:
        _DETECTOR_CACHE[key] = EASTDetector(
            min_confidence=config["detection"]["min_confidence"],
            nms_overlap_thresh=config["detection"]["nms_overlap_thresh"],
        )
    return _DETECTOR_CACHE[key]


def _get_recognizer(script, config):
    if script not in _RECOGNIZER_CACHE:
        if script in CROP_BASED_RECOGNIZERS:
            _RECOGNIZER_CACHE[script] = CROP_BASED_RECOGNIZERS[script]()
        elif script in SELF_DETECTING_RECOGNIZERS:
            _RECOGNIZER_CACHE[script] = SELF_DETECTING_RECOGNIZERS[script]()
        else:
            _RECOGNIZER_CACHE[script] = TrOCRRecognizer(device=config["device"])
    return _RECOGNIZER_CACHE[script]


def run_pipeline(image_path, output_dir, config):
    script = config["script"]
    cropped_dir = os.path.join(output_dir, "cropped")
    os.makedirs(cropped_dir, exist_ok=True)

    if script in SELF_DETECTING_RECOGNIZERS:
        results = _run_self_detecting(image_path, cropped_dir, _get_recognizer(script, config))
    else:
        results = _run_east_based(image_path, config, cropped_dir, script)

    _write_outputs(results, output_dir)
    return results


def _run_self_detecting(image_path, cropped_dir, recognizer):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(
            f"Unable to read input image: {image_path}. "
            "Check that the path exists and the file is a supported image."
        )
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    detections = recognizer.detect_and_recognize(pil_img)

    results = []
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        crop = image[max(0, y1):y2, max(0, x1):x2]

        results.append({
            "index": i,
            "bbox": det["bbox"],
            "label": "hindi",
            "route": det["route"],
            "text": det["text"],
        })

        if crop.size > 0:
            cv2.imwrite(os.path.join(cropped_dir, f"{i}.png"), crop)

        logger.info("[%d] %s -> %s", i, "hindi", det["text"])

    return results


def _run_east_based(image_path, config, cropped_dir, script):
    detector = _get_detector(config)
    recognizer = _get_recognizer(script, config)
    classifier = None if script in CROP_BASED_RECOGNIZERS else TextClassifier()

    boxes, image = detector.detect_text(image_path)
    sentence_boxes = group_text(
        boxes,
        v_tol_multiplier=config["grouping"]["v_tol_multiplier"],
        h_gap_multiplier=config["grouping"]["h_gap_multiplier"],
    )

    results = []
    h, w = image.shape[:2]

    for i, box in enumerate(sentence_boxes):
        clamped = clamp_box(*box, width=w, height=h)
        if clamped is None:
            continue
        x1, y1, x2, y2 = clamped

        crop = image[y1:y2, x1:x2]
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        label = script if script in CROP_BASED_RECOGNIZERS else classifier.classify(pil_img)
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

    return results


def _write_outputs(results, output_dir):
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
    if args.script is not None:
        config["script"] = args.script

    run_pipeline(image_path, output_dir, config)
