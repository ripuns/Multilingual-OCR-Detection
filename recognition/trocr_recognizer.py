import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

class TrOCRRecognizer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.models = {
            "printed": self.load_model("microsoft/trocr-large-printed"),
            "handwritten": self.load_model("microsoft/trocr-large-handwritten"),
            # "chinese": self.load_model("ZihCiLin/trocr-traditional-chinese-baseline")
        }

    def load_model(self, name):
        processor = TrOCRProcessor.from_pretrained(name, use_fast=False)
        model = VisionEncoderDecoderModel.from_pretrained(name).to(self.device)
        return processor, model

    def recognize(self, image, label):
        processor, model = self.models[label]
        image = image.resize((384, 384))
        
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = model.generate(pixel_values)

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text