from app.extractors.yolo_base import YoloBase
from app.config import get_settings
import torch

class YoloPose(YoloBase):
    def __init__(self):

        settings = get_settings()
        cfg = settings.YOLO_MODELS["pose"]

        super().__init__(
            model_name="yolo_pose",
            model_path=cfg["weights"]
        )

        self.conf_threshold = cfg["conf_threshold"]

    def predict(self, batch_tensor):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() before predict().")

        batch_tensor = self.yolo_preprocess(batch_tensor, self.device)

        with torch.no_grad():
            results = self.model(batch_tensor, device=self.device)

        batch_outputs = []

        for r in results:
            frame_outputs = []

            if r.keypoints is None or r.boxes is None:
                batch_outputs.append(frame_outputs)
                continue

            boxes = r.boxes.xyxy
            confs = r.boxes.conf
            keypoints = r.keypoints.xy  # (N, 17, 2)

            for i in range(len(boxes)):
                if float(confs[i]) < self.conf_threshold:
                    continue

                frame_outputs.append({
                    "bbox": boxes[i].tolist(),
                    "confidence": float(confs[i]),
                    "keypoints": keypoints[i].tolist()
                })

            batch_outputs.append(frame_outputs)

        return batch_outputs
