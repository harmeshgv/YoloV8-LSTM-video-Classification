from app.extractors.yolo_base import YoloBase
from app.config import get_settings

class YoloDetector(YoloBase):
    def __init__(self):
        settings = get_settings()
        cfg = settings.YOLO_MODELS["detector"]

        super().__init__(
            model_name="yolo_detector",
            model_path=cfg["weights"]
        )

        self.conf_threshold = cfg["conf_threshold"]

    def predict(self, batch_tensor):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() before predict().")

        batch_tensor= self.yolo_preprocess(batch_tensor, self.device)
        results = self.model(
            batch_tensor,
            device=self.device
        )

        batch_outputs = []

        for r in results:
            detections = []

            if r.boxes is None:
                batch_outputs.append(detections)
                continue

            for box in r.boxes:
                conf = float(box.conf)
                if conf < self.conf_threshold:
                    continue

                class_id = int(box.cls)

                detections.append({
                    "bbox": box.xyxy[0].tolist(),
                    "class_id": class_id,
                    "class_name": self.model.names[class_id],
                    "confidence": conf
                })

            batch_outputs.append(detections)

        return batch_outputs
