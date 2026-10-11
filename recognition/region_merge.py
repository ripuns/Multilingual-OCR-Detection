"""Per-line script identification for pages that mix scripts.

All engines read the whole page. Their detections are grouped into horizontal bands (text
lines) by vertical overlap, the script of every band is chosen separately from the engines'
outputs inside that band, and the band's regions are taken from the winning engine. A line
that itself mixes scripts still receives a single script.
"""

from recognition.paddle_recognizer import sort_reading_order
from recognition.script_detector import SCRIPT_BLOCKS

SCRIPTS = ("english", "tamil", "hindi")
FALLBACK_SCRIPT = "english"
BAND_OVERLAP = 0.3


def _centre_y(r):
    return (r["bbox"][1] + r["bbox"][3]) / 2


def line_bands(by_script, overlap=BAND_OVERLAP):
    """Merges the vertical extents of every region of every engine into bands."""
    spans = sorted((r["bbox"][1], r["bbox"][3]) for regs in by_script.values() for r in regs)
    bands = []
    for y1, y2 in spans:
        if bands:
            b = bands[-1]
            inter = min(b[1], y2) - max(b[0], y1)
            if inter >= overlap * max(min(y2 - y1, b[1] - b[0]), 1):
                b[0], b[1] = min(b[0], y1), max(b[1], y2)
                continue
        bands.append([y1, y2])
    return bands


def merge_by_line(by_script, selector):
    """selector(results_by_script) -> (script_or_None, info). Returns (regions, detection)."""
    merged, band_info, chars = [], [], {s: 0 for s in SCRIPTS}
    any_fallback = False
    for y1, y2 in line_bands(by_script):
        sub = {s: [r for r in regs if y1 <= _centre_y(r) <= y2] for s, regs in by_script.items()}
        picked, info = selector(sub)
        chosen = picked or FALLBACK_SCRIPT
        any_fallback = any_fallback or picked is None
        regs = [dict(r, script=chosen, band=len(band_info)) for r in sub.get(chosen, [])]
        merged.extend(regs)
        chars[chosen] += sum(1 for r in regs for ch in r["text"] if SCRIPT_BLOCKS[chosen](ch))
        band_info.append({
            "y": [y1, y2], "script": chosen, "fallback": picked is None,
            "probabilities": info.get("probabilities"),
        })
    dominant = max(chars, key=chars.get) if any(chars.values()) else FALLBACK_SCRIPT
    return sort_reading_order(merged), {
        "mode": "mixed", "script": dominant, "fallback": any_fallback,
        "share": None, "masses": None, "bands": band_info,
    }
