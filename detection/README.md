# detection/

## What
Detects word-level text bounding boxes in an input image using OpenCV's EAST
(Efficient and Accurate Scene Text) detector.

## Why
Recognition (TrOCR) needs cropped regions to run on, not a whole image. EAST
provides those regions as axis-aligned word boxes with confidence scores.

## How
`EASTDetector` in [`east_detector.py`](east_detector.py):
1. Loads the frozen EAST model (`models/frozen_east_text_detection.pb`, not
   tracked in git — see root README's Model Setup section).
2. `detect_text(image_path)` resizes the image to a multiple of 32, runs the
   EAST forward pass, decodes per-cell rotated-box geometry/scores
   (`decode()`), applies non-max suppression, and rescales boxes back to the
   original image coordinates.
3. `min_confidence` (score threshold) and `nms_overlap_thresh` (NMS overlap
   threshold) are constructor parameters, driven by `config.yaml`'s
   `detection.*` keys — see root README's Configuration section.

## Structure
- `east_detector.py` — `EASTDetector`, the only class in this module.

## Summary
Detection stays a single fixed implementation (no interface/abstraction) since
only one detector is in scope for this project. Confidence/NMS thresholds are
configurable; the model architecture itself is not swappable without code
changes.
