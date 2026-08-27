# grouping/

## What
Merges word-level bounding boxes produced by the EAST detector into
sentence/line-level bounding boxes, in reading order.

## Why
EAST returns one box per detected word. Recognition (TrOCR) works better on
coherent lines/phrases than isolated words, so boxes on the same line need to
be merged into a single crop region before recognition.

## How
`group_text(boxes)` in [`text_grouping.py`](text_grouping.py):
1. Sorts input boxes by `y1` (top edge) ascending.
2. Walks the sorted boxes, accumulating each into the current group if it is
   within a vertical tolerance (`0.5 * average box height`) of the group's
   top edge and within a horizontal gap tolerance (`1.5 * average box
   height`) of the group's right edge.
3. While a box is absorbed into a group, the group's bounding rectangle is
   expanded to `x1=min(...), y1=min(...), x2=max(...), y2=max(...)` across
   every box in that group — not just the first box — so the crop taken from
   the resulting rectangle cannot clip content from a box that started to the
   left of (or above) the group's first box.
4. Final groups are sorted by `(y1, x1)` so downstream consumers (recognition,
   output writer) see them in top-to-bottom, left-to-right reading order.

This is a purely geometric heuristic — no clustering model, no learned
parameters.

## Structure
- `text_grouping.py` — `group_text(boxes)`, the only function in this module.

## Summary
Fixes a previously silent bug where a group's left/top edge stayed pinned to
the first box added to it, causing crops to clip words that drifted left or
appeared above the group's starting box. Regression coverage lives in
[`../tests/test_grouping.py`](../tests/test_grouping.py).
