import numpy as np
from PIL import Image

class TextClassifier:
    def classify(self, image):
        gray = image.convert("L")
        arr = np.array(gray)

        if np.std(arr) > 70:
            return "handwritten"
        # elif np.mean(arr) < 150:
        #     return "chinese"
        else:
            return "printed"