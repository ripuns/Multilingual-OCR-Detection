"""Automatic script identification from recognizer output.

Each language's recognizer is run on the image; the script whose recognizer
produces the most confident text *inside its own Unicode block* wins. A
recognizer fed the wrong script mostly emits low-confidence garbage or
characters from a different block, which contributes nothing to its own
score.
"""

SCRIPT_BLOCKS = {
    "tamil": lambda ch: "஀" <= ch <= "௿",
    "hindi": lambda ch: "ऀ" <= ch <= "ॿ",
    "english": lambda ch: ("a" <= ch <= "z") or ("A" <= ch <= "Z"),
}

# The Indic engines also read Latin text well, but the English engine turns
# Indic text into Latin garbage that inflates English mass. So on mixed-script
# images (e.g. a Hindi poster with English words) an Indic script wins once it
# holds at least this share of the total mass, even if English mass is higher.
INDIC_SCRIPTS = ("tamil", "hindi")
INDIC_MIN_SHARE = 0.25


def script_mass(script, regions):
    """Confidence-weighted count of characters belonging to `script`'s block."""
    in_block = SCRIPT_BLOCKS[script]
    return sum(
        r.get("score", 0.0) * sum(1 for ch in r["text"] if in_block(ch))
        for r in regions
    )


def pick_script(results_by_script):
    """results_by_script: {script: [region dicts with 'text' and 'score']}.

    Returns (script_or_None, info). None means no script produced any
    in-block text (blank image, symbols only); callers decide the fallback.
    info carries per-script masses and the winner's share of total mass.
    """
    masses = {s: script_mass(s, regions) for s, regions in results_by_script.items()}
    total = sum(masses.values())
    if total <= 0:
        return None, {"masses": masses, "share": 0.0}

    best = max(masses, key=masses.get)
    indic = [s for s in INDIC_SCRIPTS if s in masses]
    if indic:
        best_indic = max(indic, key=masses.get)
        if masses[best_indic] / total >= INDIC_MIN_SHARE:
            best = best_indic
    return best, {"masses": masses, "share": masses[best] / total}
