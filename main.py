import argparse
import json
import logging
import os
import sys

import cv2

from config import load_config
from recognition.paddle_recognizer import PaddleOcrRecognizer

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
        help="Which language model to use (overrides config). This is an explicit "
        "operator choice, not automatic script detection -- see docs/limitations.md.",
    )
    return parser.parse_args()


def configure_logging(level_name):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")


# Process-lifetime cache so a long-running server (the web demo) doesn't
# reload multi-GB models on every request.
_RECOGNIZER_CACHE = {}


def _get_recognizer(script):
    if script not in _RECOGNIZER_CACHE:
        _RECOGNIZER_CACHE[script] = PaddleOcrRecognizer(script)
    return _RECOGNIZER_CACHE[script]


def warm_up(config=None):
    """Forces all 3 language models to load now instead of lazily on first
    request. Used by the web demo at startup so a live audience never hits a
    cold load mid-demo."""
    for script in ("english", "tamil", "hindi"):
        _get_recognizer(script).warm_up()


def run_pipeline(image_path, output_dir, config):
    script = config["script"]
    recognizer = _get_recognizer(script)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(
            f"Unable to read input image: {image_path}. "
            "Check that the path exists and the file is a supported image."
        )

    results = recognizer.detect_and_recognize(image_path)

    cropped_dir = os.path.join(output_dir, "cropped")
    os.makedirs(cropped_dir, exist_ok=True)
    h, w = image.shape[:2]

    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            cv2.imwrite(os.path.join(cropped_dir, f"{r['index']}.png"), image[y1:y2, x1:x2])
        logger.info("[%d] %s -> %s (score=%.3f)", r["index"], script, r["text"], r["score"])

    _write_outputs(results, output_dir)
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
