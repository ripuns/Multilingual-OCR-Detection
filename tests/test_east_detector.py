import pytest

from detection.east_detector import EASTDetector


def test_missing_image_raises_filenotfounderror():
    detector = EASTDetector.__new__(EASTDetector)  # skip __init__, avoid loading the EAST model
    with pytest.raises(FileNotFoundError):
        detector.detect_text("does/not/exist.png")
