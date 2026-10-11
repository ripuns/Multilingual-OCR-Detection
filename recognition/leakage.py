"""Suppression of Latin-letter "script leakage" from Indic engines.

When the Tamil or Hindi engine cannot read a region it often emits low-confidence Latin
letters instead of Indic text. A region from an Indic engine that contains Latin letters,
no characters of its own script, and a recognition score below LEAK_MAX_SCORE is treated
as leakage and withheld from the result. Genuine Latin words read by an Indic engine
usually score higher; on the evaluation images a 0.7 score cut flagged 96% of leaked
regions and 6% of genuine Latin ones.
"""

from recognition.script_detector import INDIC_SCRIPTS, SCRIPT_BLOCKS

LEAK_MAX_SCORE = 0.7


def is_latin_leak(region, script, max_score=LEAK_MAX_SCORE):
    if script not in INDIC_SCRIPTS:
        return False
    text = region["text"]
    latin = sum(1 for ch in text if SCRIPT_BLOCKS["english"](ch))
    own = sum(1 for ch in text if SCRIPT_BLOCKS[script](ch))
    return latin > 0 and own == 0 and region.get("score", 0.0) < max_score


def suppress_leakage(regions, script, max_score=LEAK_MAX_SCORE):
    """Returns (kept_regions, suppressed_regions); kept regions are re-numbered."""
    kept = [r for r in regions if not is_latin_leak(r, script, max_score)]
    suppressed = [r for r in regions if is_latin_leak(r, script, max_score)]
    for i, r in enumerate(kept):
        r["index"] = i
    return kept, suppressed
