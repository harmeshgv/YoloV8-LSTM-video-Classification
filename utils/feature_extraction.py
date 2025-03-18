import yaml
import torch
import numpy as np
import cv2
import os
import io
import imageio
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from PIL import Image


class ViolenceFeatureExtractor:
    """
    A class to extract violence-related features from video frames using YOLO models.
    """

    def __init__(self, detection_model_path, segmentation_model_path, pose_model_path):
        """
        Initialize the ViolenceFeatureExtractor with YOLO models.

        Args:
            detection_model_path (str): Path to the detection model.
            segmentation_model_path (str): Path to the segmentation model.
            pose_model_path (str): Path to the pose estimation model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_gpu()

        self.detection_model = YOLO(detection_model_path).to(self.device)
        self.segmentation_model = YOLO(segmentation_model_path).to(self.device)
        self.pose_model = YOLO(pose_model_path).to(self.device)

        self.violence_objects = ["knife", "gun", "baseball bat", "stick", "bottle"]
        self.relevant_classes = ["person"] + self.violence_objects

        self.colors = {
            "violence": (0, 0, 255),  # Red
            "person": (0, 255, 0),  # Green
            "interaction": (255, 0, 0),  # Blue
            "keypoint": (255, 255, 0),  # Yellow
            "connection": (0, 255, 255),  # Cyan
        }

        self.frame_skip = 2
        self.input_size = 640
        self.conf_threshold = 0.5
        self.interaction_threshold = 0.5
        self.current_risk_level = 0.0

    def _setup_gpu(self):
        """Configure GPU settings if available."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("No GPU available. Using CPU.")

    def preprocess_frame(self, frame):
        """Preprocess a frame for model input."""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get original dimensions
            h, w = frame_rgb.shape[:2]

            # Calculate scaling factor
            r = self.input_size / max(h, w)

            # Resize the frame
            new_h, new_w = int(h * r), int(w * r)
            resized = cv2.resize(frame_rgb, (new_w, new_h))

            # Pad the frame to match input_size
            canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            pad_h = (self.input_size - new_h) // 2
            pad_w = (self.input_size - new_w) // 2
            canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

            # Normalize the frame
            normalized = canvas.astype(np.float32) / 255.0

            # Return the normalized frame and scaling/padding info
            return normalized, (r, pad_w, pad_h)

        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None, None

    def analyze_person_interactions(self, person_boxes):
        """Analyze interactions between detected people."""
        interactions = []
        if len(person_boxes) < 2:
            return interactions

        for i in range(len(person_boxes)):
            for j in range(i + 1, len(person_boxes)):
                box1 = person_boxes[i]
                box2 = person_boxes[j]
                center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
                center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
                distance = np.sqrt(
                    (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
                )
                box1_size = (box1[2] - box1[0]) * (box1[3] - box1[1])
                box2_size = (box2[2] - box2[0]) * (box2[3] - box2[1])
                avg_size = (box1_size + box2_size) / 2

                if distance < avg_size * self.interaction_threshold:
                    interactions.append(
                        {
                            "person1_idx": i,
                            "person2_idx": j,
                            "distance": distance,
                            "relative_distance": distance / avg_size,
                            "center1": center1,
                            "center2": center2,
                            "box1": box1,
                            "box2": box2,
                        }
                    )

        return interactions

    def calculate_motion_features(self, prev_poses, current_poses):
        """Calculate motion features between consecutive frames."""
        try:
            if not prev_poses or not current_poses:
                return {
                    "average_speed": 0,
                    "motion_intensity": 0,
                    "sudden_movements": 0,
                }

            prev_poses = np.array(prev_poses)
            current_poses = np.array(current_poses)

            if prev_poses.shape == current_poses.shape:
                displacement = np.linalg.norm(current_poses - prev_poses, axis=2)
                average_speed = np.mean(displacement)
                motion_intensity = np.std(displacement)
                sudden_movements = np.sum(
                    displacement > np.mean(displacement) + 2 * np.std(displacement)
                )

                return {
                    "average_speed": float(average_speed),
                    "motion_intensity": float(motion_intensity),
                    "sudden_movements": int(sudden_movements),
                }

            return {"average_speed": 0, "motion_intensity": 0, "sudden_movements": 0}

        except Exception as e:
            print(f"Error in motion calculation: {e}")
            return {"average_speed": 0, "motion_intensity": 0, "sudden_movements": 0}

    def analyze_poses_for_violence(self, poses):
        """Analyze poses for potential aggressive/violent behavior."""
        try:
            if not poses:
                return False

            for pose in poses:
                pose_array = np.array(pose)
                arm_keypoints = [5, 7, 9, 6, 8, 10]
                arm_positions = pose_array[arm_keypoints]
                arm_confidences = arm_positions[:, 2]

                if np.mean(arm_confidences) > 0.5:
                    return True

            return False

        except Exception as e:
            print(f"Error in pose analysis: {e}")
            return False

    def rescale_coords(self, x, y, scale_info):
        """Rescale coordinates back to original image size."""
        scale, pad_w, pad_h = scale_info
        x_orig = (x - pad_w) / scale
        y_orig = (y - pad_h) / scale
        return int(x_orig), int(y_orig)

    def draw_detections(self, frame, det_results, pose_results, interactions, scale_info):
        """Draw detections, poses, and interactions on the frame."""
        display_frame = frame.copy()

        for result in det_results:
            boxes = result.boxes
            for box in boxes:
                try:
                    x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
                    x1, y1 = self.rescale_coords(x1, y1, scale_info)
                    x2, y2 = self.rescale_coords(x2, y2, scale_info)
                    cls = result.names[int(box.cls[0])]
                    conf = float(box.conf[0])

                    if cls in self.relevant_classes:
                        color = (
                            self.colors["violence"]
                            if cls in self.violence_objects
                            else self.colors["person"]
                        )

                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{cls} {conf:.2f}"
                        (text_w, text_h), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                        )
                        cv2.rectangle(
                            display_frame,
                            (x1, y1 - text_h - 5),
                            (x1 + text_w, y1),
                            color,
                            -1,
                        )
                        cv2.putText(
                            display_frame,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2,
                        )

                except Exception as e:
                    print(f"Error in detection drawing: {e}")
                    continue

        for interaction in interactions:
            try:
                x1, y1 = self.rescale_coords(
                    interaction["center1"][0], interaction["center1"][1], scale_info
                )
                x2, y2 = self.rescale_coords(
                    interaction["center2"][0], interaction["center2"][1], scale_info
                )
                cv2.line(
                    display_frame, (x1, y1), (x2, y2), self.colors["interaction"], 2
                )
                mid_point = ((x1 + x2) // 2, (y1 + y2) // 2)
                distance_label = f"D: {interaction['relative_distance']:.2f}"
                cv2.putText(
                    display_frame,
                    distance_label,
                    mid_point,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.colors["interaction"],
                    2,
                )

            except Exception as e:
                print(f"Error drawing interaction: {e}")
                continue

        if pose_results:
            for result in pose_results:
                if result.keypoints is not None:
                    for kpts in result.keypoints:
                        try:
                            keypoints_data = kpts.data[0].cpu().numpy()
                            for keypoint in keypoints_data:
                                x, y, conf = keypoint
                                if conf > 0.5:
                                    x, y = self.rescale_coords(x, y, scale_info)
                                    cv2.circle(
                                        display_frame,
                                        (x, y),
                                        4,
                                        self.colors["keypoint"],
                                        -1,
                                    )

                            connections = [
                                (5, 7),
                                (7, 9),
                                (6, 8),
                                (8, 10),
                                (5, 6),
                                (11, 13),
                                (13, 15),
                                (12, 14),
                                (14, 16),
                                (11, 12),
                            ]
                            for connection in connections:
                                pt1 = keypoints_data[connection[0]]
                                pt2 = keypoints_data[connection[1]]

                                if pt1[2] > 0.5 and pt2[2] > 0.5:
                                    x1, y1 = self.rescale_coords(
                                        pt1[0], pt1[1], scale_info
                                    )
                                    x2, y2 = self.rescale_coords(
                                        pt2[0], pt2[1], scale_info
                                    )
                                    cv2.line(
                                        display_frame,
                                        (x1, y1),
                                        (x2, y2),
                                        self.colors["connection"],
                                        2,
                                    )
                        except Exception as e:
                            print(f"Error in pose drawing: {e}")
                            continue

        if self.current_risk_level > 0.7:
            cv2.putText(
                display_frame,
                "HIGH RISK",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        cv2.putText(
            display_frame,
            "Press 'q' to quit, 'p' to pause/resume",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return display_frame

    def extract_features(self, frame, prev_frame_data=None):
        """Extract violence-relevant features from a frame."""
        try:
            processed_frame, scale_info = self.preprocess_frame(frame)
            if processed_frame is None:
                return None, frame

            frame_tensor = (
                torch.from_numpy(processed_frame)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.device)
            )

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                det_results = self.detection_model(frame_tensor, verbose=False)
                pose_results = self.pose_model(frame_tensor, verbose=False)

            features = {
                "objects": [],
                "poses": [],
                "interactions": [],
                "motion": {},
                "violence_indicators": {
                    "weapon_present": False,
                    "close_interaction": False,
                    "rapid_motion": False,
                    "aggressive_pose": False,
                },
            }

            person_boxes = []
            for result in det_results:
                for box in result.boxes:
                    try:
                        cls = result.names[int(box.cls[0])]
                        if cls in self.relevant_classes:
                            conf = float(box.conf[0])
                            box_coords = box.xyxy[0].cpu().numpy().tolist()

                            features["objects"].append(
                                {"class": cls, "confidence": conf, "box": box_coords}
                            )

                            if cls == "person":
                                person_boxes.append(box_coords)
                            elif cls in self.violence_objects:
                                features["violence_indicators"]["weapon_present"] = True
                    except Exception as e:
                        print(f"Error processing detection: {e}")
                        continue

            if len(person_boxes) >= 2:
                interactions = self.analyze_person_interactions(person_boxes)
                features["interactions"] = interactions
                features["violence_indicators"]["close_interaction"] = len(interactions) > 0

            if pose_results:
                for result in pose_results:
                    if result.keypoints is not None:
                        for kpts in result.keypoints:
                            try:
                                pose_data = kpts.data[0].cpu().numpy().tolist()
                                features["poses"].append(pose_data)
                            except Exception as e:
                                print(f"Error processing pose: {e}")
                                continue

                features["violence_indicators"]["aggressive_pose"] = (
                    self.analyze_poses_for_violence(features["poses"])
                )

            if prev_frame_data and "poses" in prev_frame_data:
                motion_features = self.calculate_motion_features(
                    prev_frame_data["poses"], features["poses"]
                )
                features["motion"] = motion_features
                features["violence_indicators"]["rapid_motion"] = (
                    motion_features.get("average_speed", 0) > 10
                )

            risk_weights = {
                "weapon_present": 0.4,
                "close_interaction": 0.3,
                "rapid_motion": 0.2,
                "aggressive_pose": 0.1,
            }

            self.current_risk_level = sum(
                risk_weights[indicator] * int(value)
                for indicator, value in features["violence_indicators"].items()
            )

            annotated_frame = self.draw_detections(
                frame, det_results, pose_results, features["interactions"], scale_info
            )

            return features, annotated_frame

        except Exception as e:
            print(f"Error in feature extraction: {e}")
            return None, frame

    def convert_numpy_to_python(self, obj):
        """Recursively convert NumPy objects to native Python types."""
        if isinstance(obj, np.generic):
            return obj.item()  # Convert NumPy scalar to Python scalar
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_to_python(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_numpy_to_python(item) for item in obj)
        else:
            return obj

    def process_video(self, video_path, yaml_path):
        """Process a video file to extract violence-related features and return the video in memory."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return None

        # Extract video metadata
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize data storage
        video_data = {
            "metadata": {
                "path": video_path,
                "fps": fps,
                "frame_count": frame_count,
                "width": frame_width,
                "height": frame_height,
            },
            "frames": [],
        }

        # Load existing data if the YAML file exists
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, "r") as yaml_file:
                    existing_data = yaml.safe_load(yaml_file) or {}
                    if "frames" in existing_data:
                        video_data["frames"].extend(existing_data["frames"])
            except yaml.YAMLError as e:
                print(f"Error loading YAML file: {e}")
                # If the YAML file is corrupted, start with an empty data structure
                video_data["frames"] = []

        # Create an in-memory byte stream to store the video
        video_buffer = io.BytesIO()

        # Use imageio to write the video to the in-memory buffer
        with imageio.get_writer(video_buffer, format="mp4", fps=fps, macro_block_size = 1) as writer:
            frame_idx = 0
            prev_frame_data = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # Exit loop if no more frames

                # Skip frames based on frame_skip
                if frame_idx % self.frame_skip != 0:
                    frame_idx += 1
                    continue

                # Extract features and draw detections
                features, annotated_frame = self.extract_features(frame, prev_frame_data)

                if features is not None:
                    # Append detailed frame data to the frames list
                    frame_data = {
                        "frame_index": frame_idx,
                        "timestamp": frame_idx / fps,  # Calculate timestamp
                        "features": features,
                    }
                    video_data["frames"].append(frame_data)

                    # Convert the annotated frame to RGB (required by imageio)
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                    # Append the frame to the video buffer
                    writer.append_data(annotated_frame_rgb)

                    prev_frame_data = features

                frame_idx += 1

        # Release the video capture object
        cap.release()

        # Convert NumPy objects to native Python types before saving
        video_data_converted = self.convert_numpy_to_python(video_data)

        # Save all data (metadata + frame features) to a YAML file
        with open(yaml_path, "w") as yaml_file:
            yaml.dump(video_data_converted, yaml_file, default_flow_style=False)

        # Return the in-memory video buffer
        video_buffer.seek(0)  # Reset the buffer position to the beginning
        return video_buffer