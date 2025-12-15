import cv2
import torch

class BatchedVideoTensorReader:
    def __init__(self, batch_size, device, resize=None):
        self.batch_size = batch_size
        self.device = device
        self.resize = resize

    def read(self, video_path):
        cap = cv2.VideoCapture(video_path)

        frames, frame_ids = [], []
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # BGR → RGB (IMPORTANT)
            fps = cap.get(cv2.CAP_PROP_FPS)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.resize:
                frame = cv2.resize(frame, self.resize)

            tensor = torch.from_numpy(frame).permute(2, 0, 1)
            frames.append(tensor)
            frame_ids.append(frame_id)

            if len(frames) == self.batch_size:
                yield frame_ids, fps ,torch.stack(frames).to(self.device)
                frames, frame_ids = [], []

            frame_id += 1

        if frames:
            yield frame_ids, fps, torch.stack(frames).to(self.device)

        cap.release()
