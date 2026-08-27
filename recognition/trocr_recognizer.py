import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from recognition.registry import get_route, register

register("printed", "microsoft/trocr-large-printed")
register("handwritten", "microsoft/trocr-large-handwritten")


class TrOCRRecognizer:
    def __init__(self, device="auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self._loaded = {}

    def _load(self, label):
        if label not in self._loaded:
            model_id = get_route(label)
            processor = TrOCRProcessor.from_pretrained(model_id, use_fast=False)
            model = VisionEncoderDecoderModel.from_pretrained(model_id).to(self.device)
            self._loaded[label] = (processor, model)
        return self._loaded[label]

    def recognize(self, image, label):
        """Returns (text, route) where route is the model id the label resolved to."""
        processor, model = self._load(label)
        image = image.resize((384, 384))

        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = model.generate(pixel_values)

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text, get_route(label)
