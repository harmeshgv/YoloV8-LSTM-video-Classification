import torch
from app.preprocess.video_reader import BatchedVideoTensorReader
from app.extractors.yolo_detector import YoloDetector
from app.extractors.yolo_pose import YoloPose
from app.config import get_settings
from app.shared.device import get_device
from app.outputs.normalize import normalize_detections
from app.features.kinematics import KinematicsCalculator

class ExtractionPipeline:
    def __init__(self):
        settings = get_settings()

        self.reader = BatchedVideoTensorReader(
            batch_size=settings.BATCH_SIZE,
            device=get_device(),
            resize=settings.RESIZE
        )

        self.detector = YoloDetector()
        self.pose = YoloPose()
        self.detector.load()
        self.pose.load()

        self.kinematics_calculator = KinematicsCalculator(device=get_device())

    def stream(self, video_path):
        with torch.no_grad():
            for frame_ids, fps, batch in self.reader.read(video_path):
                detections_batch = self.detector.predict(batch)
                poses_batch = self.pose.predict(batch)

                for idx, frame_id in enumerate(frame_ids):
                    rows = normalize_detections(
                        frame_id,
                        fps,
                        detections_batch[idx],
                        poses_batch[idx]
                    )
                    print(rows)




                    for row in rows:
                        yield row

    def collect(self, video_path):
        return list(self.stream(video_path))
