from app.extractors.base import BaseModel
from ultralytics import YOLO

class YoloBase(BaseModel):
    def load(self):
        self.model = YOLO(self.model_path)
        self.loaded = True
        return self

    @staticmethod
    def yolo_preprocess(batch_tensor, device):
        return (
        batch_tensor
        .to(device)
        .float()
        .div_(255.0)
    )


    def predict(self, frame_tensor):
        raise NotImplementedError("Use a concrete YOLO model (Detector, Classifier)")
