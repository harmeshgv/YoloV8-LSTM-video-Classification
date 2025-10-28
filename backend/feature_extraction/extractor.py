import torch
from config import DETECT_MODEL, POSE_MODEL, CONF_THRESHOLD
from utils.gpu import GPUConfigurator
from preprocessing.preprocessor import FramePreprocessor
from data_extraction.interaction_analyzer import InteractionAnalyzer
from data_extraction.person_tracker import PersonTracker
from utils.visualizer import Visualizer
import numpy as np
from ultralytics import YOLO


class VideoFeatureExtractor:
    def __init__(self):
        self.gpu_config = GPUConfigurator()
        self.device = self.gpu_config.device

        self.detection_model = YOLO(DETECT_MODEL).to(self.device)
        self.pose_model = YOLO(POSE_MODEL).to(self.device)

        self.preprocessor = FramePreprocessor()
        self.interaction_analyzer = InteractionAnalyzer()
        self.person_tracker = PersonTracker()
        self.visualizer = Visualizer()

        self.conf_threshold = CONF_THRESHOLD
        self.prev_poses = None

        self.person_tracker.reset()
        self.prev_poses = None

    def extract_features(self, frame, frame_idx):
        """Extract features from a frame."""
        try:
            processed_frame, scale_info = self.preprocessor.preprocess_frame(frame)
            if processed_frame is None:
                return None, frame

            frame_tensor = (
                torch.from_numpy(processed_frame)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.device)
            )

            if frame_idx % 5 == 0:
                torch.cuda.empty_cache()

            with (
                torch.no_grad(),
                torch.amp.autocast(device_type="cuda", dtype=torch.float16),
            ):
                det_results = self.detection_model(
                    frame_tensor, conf=self.conf_threshold, verbose=False
                )
                pose_results = (
                    self.pose_model(
                        frame_tensor, conf=self.conf_threshold, verbose=False
                    )
                    if len(det_results[0].boxes) > 0
                    else []
                )

            frame_data = {
                "frame_index": frame_idx,
                "timestamp": frame_idx / 30,
                "persons": [],
                "objects": [],
                "interactions": [],
                "resized_width": scale_info.get("resized_size", (0, 0))[1],
                "resized_height": scale_info.get("resized_size", (0, 0))[0],
            }

            # Process detections
            person_boxes = []
            for result in det_results:
                for box in result.boxes:
                    try:
                        cls = result.names[int(box.cls[0])]
                        box_coords = box.xyxy[0].cpu().numpy().tolist()
                        if cls == "person":
                            person_boxes.append(box_coords)
                        else:
                            frame_data["objects"].append(
                                {
                                    "class": cls,
                                    "confidence": float(box.conf[0]),
                                    "box": box_coords,
                                }
                            )
                    except Exception as e:
                        print(f"Detection processing error: {e}")
                        continue

            # Track persons
            tracked_persons = self.person_tracker.assign_person_ids(person_boxes)

            # Process poses
            current_poses = []
            if pose_results:
                for result in pose_results:
                    if result.keypoints:
                        for kpts in result.keypoints:
                            try:
                                pose_data = kpts.data[0].cpu().numpy().tolist()
                                current_poses.append(pose_data)
                            except Exception as e:
                                print(f"Pose processing error: {e}")
                                continue

            # Match persons to poses
            frame_data["persons"] = []
            for i, box in enumerate(person_boxes):
                try:
                    pose = current_poses[i] if i < len(current_poses) else None
                    if pose is None:
                        continue

                    # Find the person ID for this box
                    person_id = None
                    for pid, tracked_box in tracked_persons.items():
                        if np.array_equal(box, tracked_box):
                            person_id = pid
                            break

                    if person_id is None:
                        continue

                    frame_data["persons"].append(
                        {
                            "person_idx": i,
                            "person_id": person_id,
                            "box": box,
                            "center": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                            "keypoints": pose,
                        }
                    )

                except Exception as e:
                    print(f"Skipping person {i} due to error: {e}")
                    continue

            # Calculate motion features
            motion_features = {
                "average_speed": 0,
                "motion_intensity": 0,
                "sudden_movements": 0,
            }

            if self.prev_poses and current_poses:
                try:
                    motion_features = (
                        self.interaction_analyzer.calculate_motion_features(
                            self.prev_poses, current_poses
                        )
                    )
                except Exception as e:
                    print(f"Motion calculation error: {e}")

            frame_data["motion_features"] = motion_features
            self.prev_poses = current_poses

            # Create interactions
            frame_data["interactions"] = (
                self.interaction_analyzer.calculate_interactions(
                    person_boxes, current_poses, tracked_persons
                )
            )

            # Add motion features to frame data

            annotated_frame = self.visualizer.draw_detections(
                frame, det_results, pose_results, scale_info, tracked_persons
            )

            return frame_data, annotated_frame

        except Exception as e:
            print(f"Frame {frame_idx} failed completely: {e}")
            return None, frame

    def reset(self):
        """Reset state for a new video."""
        self.person_tracker.reset()
        self.prev_poses = None
