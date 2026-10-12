"""Calibrated script selection from recognizer outputs.

A small multinomial logistic regression over length-invariant features of the three
engines' outputs (confidences, in-script fractions, Latin fractions, mass shares). It
replaces the hand-set 25% override with weights learned from labelled outputs and
returns class probabilities instead of a saturated "share".

Features deliberately avoid absolute character counts: the evaluation set pairs long
English lines with short Indic words, so counts would let a model "detect" line length.
"""

import json
import math
import os

from recognition.script_detector import SCRIPT_BLOCKS, script_mass

SCRIPTS = ("english", "tamil", "hindi")
FEATURE_NAMES = [f"{s}_{f}" for s in SCRIPTS for f in ("conf", "own_frac", "latin_frac", "empty", "share")]
_HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(_HERE, "script_classifier_weights.json")  # original Tamil model
WEIGHTS_PATH_FT = os.path.join(_HERE, "script_classifier_weights_ft.json")  # fine-tuned Tamil model
WEIGHTS_PATH_FT2 = os.path.join(_HERE, "script_classifier_weights_ft2.json")  # fine-tuned Tamil and Hindi models


def _is_latin(ch):
    return SCRIPT_BLOCKS["english"](ch)


def extract_features(results_by_script):
    masses = {s: script_mass(s, results_by_script.get(s, [])) for s in SCRIPTS}
    total = sum(masses.values())
    feats = []
    for s in SCRIPTS:
        text = "".join(ch for r in results_by_script.get(s, []) for ch in r["text"] if not ch.isspace())
        n = len(text)
        if n == 0:
            conf = own = latin = 0.0
            empty = 1.0
        else:
            weighted = sum(r["score"] * sum(1 for ch in r["text"] if not ch.isspace()) for r in results_by_script[s])
            conf = weighted / n
            own = sum(1 for ch in text if SCRIPT_BLOCKS[s](ch)) / n
            latin = sum(1 for ch in text if _is_latin(ch)) / n
            empty = 0.0
        share = masses[s] / total if total > 0 else 0.0
        feats.extend([conf, own, latin, empty, share])
    return feats


class ScriptClassifier:
    def __init__(self, mean, scale, weights, bias):
        self.mean, self.scale, self.weights, self.bias = mean, scale, weights, bias

    @classmethod
    def load(cls, path=WEIGHTS_PATH):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d["features"] != FEATURE_NAMES or d["classes"] != list(SCRIPTS):
            raise ValueError("script_classifier weights do not match the current feature set")
        return cls(d["mean"], d["scale"], d["weights"], d["bias"])

    def predict_proba(self, feats):
        z = [(x - m) / s for x, m, s in zip(feats, self.mean, self.scale)]
        logits = [sum(w * v for w, v in zip(row, z)) + b for row, b in zip(self.weights, self.bias)]
        top = max(logits)
        exps = [math.exp(v - top) for v in logits]
        tot = sum(exps)
        return [e / tot for e in exps]


_DEFAULT = {}


def load_default(variant="base"):
    """variant 'base' = weights for the stock models, 'ft' = fine-tuned Tamil model, 'ft2' = fine-tuned Tamil and Hindi.
    They differ because the engines' output statistics differ; use the matching one."""
    if variant not in _DEFAULT:
        path = {"ft": WEIGHTS_PATH_FT, "ft2": WEIGHTS_PATH_FT2}.get(variant, WEIGHTS_PATH)
        _DEFAULT[variant] = ScriptClassifier.load(path) if os.path.exists(path) else None
    return _DEFAULT[variant]


def pick_script_calibrated(results_by_script, clf=None, variant="base"):
    """Same contract as script_detector.pick_script plus per-script probabilities.

    Returns (script_or_None, info). None means no engine produced any text at all; unlike the
    rule, the classifier still decides when the text is out-of-script garbage (e.g. a Tamil
    word read as low-confidence Latin letters), because that pattern is itself evidence.
    """
    clf = clf or load_default(variant)
    if clf is None:
        raise RuntimeError("no script classifier weights available")
    masses = {s: script_mass(s, results_by_script.get(s, [])) for s in SCRIPTS}
    total = sum(masses.values())
    if not any(r["text"].strip() for regs in results_by_script.values() for r in regs):
        return None, {"masses": masses, "share": 0.0, "probabilities": None}
    probs = clf.predict_proba(extract_features(results_by_script))
    best = SCRIPTS[max(range(len(SCRIPTS)), key=lambda i: probs[i])]
    return best, {
        "masses": masses, "share": masses[best] / total if total > 0 else 0.0,
        "probabilities": {s: round(p, 4) for s, p in zip(SCRIPTS, probs)},
    }
