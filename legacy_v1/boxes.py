def clamp_box(x1, y1, x2, y2, width, height):
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def pad_and_clamp_box(x1, y1, x2, y2, padding, width, height):
    """Expands a box by `padding` on all four edges before clamping. Mitigates
    detectors that systematically under-size boxes for a given script (see
    docs/research_contribution.md) — too much padding reintroduces noise, so
    this is a tunable per-script knob, not a fix applied unconditionally."""
    return clamp_box(x1 - padding, y1 - padding, x2 + padding, y2 + padding, width, height)
