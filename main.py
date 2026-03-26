import os
import cv2
from PIL import Image

from detection.east_detector import EASTDetector
from grouping.text_grouping import group_text
from classification.classifier import TextClassifier
from recognition.trocr_recognizer import TrOCRRecognizer

def run_pipeline(image_path):
    detector = EASTDetector()
    classifier = TextClassifier()
    recognizer = TrOCRRecognizer()

    boxes, image = detector.detect_text(image_path)
    sentence_boxes = group_text(boxes)

    os.makedirs("output/cropped", exist_ok=True)

    results = []

    for i, (x1, y1, x2, y2) in enumerate(sentence_boxes):
        crop = image[y1:y2, x1:x2]

        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        label = classifier.classify(pil_img)
        text = recognizer.recognize(pil_img, label)

        results.append(text)

        cv2.imwrite(f"output/cropped/{i}.png", crop)

        print(f"[{i}] {label} → {text}")

    with open("output/ocr_results.txt", "w", encoding="utf-8") as f:
        for line in results:
            f.write(line + "\n")


if __name__ == "__main__":
    run_pipeline("input/images/sample.jpg")