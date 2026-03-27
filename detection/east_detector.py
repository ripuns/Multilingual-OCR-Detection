import cv2
import numpy as np
import os
from imutils.object_detection import non_max_suppression

class EASTDetector:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "..", "models", "frozen_east_text_detection.pb")

        self.net = cv2.dnn.readNet(model_path)

    def detect_text(self, image_path):
        image = cv2.imread(image_path)
        orig = image.copy()
        (H, W) = image.shape[:2]

        newW = max(320,(W // 32) * 32)
        newH = max(320,(H // 32) * 32)
        rW, rH = W / float(newW), H / float(newH)

        image = cv2.resize(image, (newW, newH))

        blob = cv2.dnn.blobFromImage(
            image, 1.0, (newW, newH),
            (123.68, 116.78, 103.94), swapRB=True, crop=False
        )

        self.net.setInput(blob)
        (scores, geometry) = self.net.forward([
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"
        ])

        rects, confidences = self.decode(scores, geometry)

        boxes = non_max_suppression(np.array(rects), probs=confidences)

        results = []
        for (startX, startY, endX, endY) in boxes:
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            results.append((startX, startY, endX, endY))

        return results, orig

    def decode(self, scores, geometry, min_confidence=0.3):
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        for y in range(numRows):
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            for x in range(numCols):
                if scoresData[x] < min_confidence:
                    continue

                offsetX, offsetY = x * 4.0, y * 4.0
                angle = anglesData[x]
                cos, sin = np.cos(angle), np.sin(angle)

                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        return rects, confidences