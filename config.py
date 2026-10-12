import copy
import os

import yaml

DEFAULTS = {
    "script": "auto",
    "selector": "calibrated",
    "suppress_leakage": True,
    "models": {"tamil": "models/ta_PP-OCRv5_mobile_rec_ft", "hindi": "models/devanagari_PP-OCRv5_mobile_rec_ft"},
    "paths": {"input": "input/images/sample.png", "output_dir": "output"},
    "logging": {"level": "INFO"},
}


def _merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path="config.yaml"):
    if not os.path.exists(path):
        return copy.deepcopy(DEFAULTS)

    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    return _merge(DEFAULTS, loaded)
