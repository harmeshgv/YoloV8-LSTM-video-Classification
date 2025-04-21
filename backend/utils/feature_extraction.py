import pandas as pd
import numpy as np
import torch
import cv2
import os
import ast
from ultralytics import YOLO

# In feature_extraction.py
from .preprocessor import FramePreprocessor
class ViolenceFeatureExtractor:
    def __init__(self, detection_model_path, pose_model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_gpu()

        self.detection_model = YOLO(detection_model_path).to(self.device)
        self.pose_model = YOLO(pose_model_path).to(self.device)

        # Initialize preprocessor
        self.preprocessor = FramePreprocessor()

        self.violence_objects = ["knife", "gun", "baseball bat", "stick", "bottle"]
        self.relevant_classes = ["person"] + self.violence_objects

        self.colors = {
            "violence": (0, 0, 255),  # Red
            "person": (0, 255, 0),    # Green
            "interaction": (255, 0, 0),# Blue
            "keypoint": (255, 255, 0), # Yellow
            "connection": (0, 255, 255)# Cyan
        }

        self.frame_skip = self.preprocessor.frame_skip
        self.input_size = self.preprocessor.input_size
        self.conf_threshold = 0.3
        self.interaction_threshold = 0.5
        self.current_risk_level = 0.0
        self.prev_poses = None
        self.person_id_counter = 0
        self.tracked_persons = {}  # Dictionary to store tracked persons
        self.inactive_persons = {}  # Store persons who are temporarily out of frame
        self.inactive_timeout = 30  # Number of frames to keep a person in inactive state

    def _assign_person_ids(self, current_boxes):
        """Assign consistent IDs to persons across frames using IoU matching."""
        new_tracked = {}
        used_ids = set()

        if not self.tracked_persons:
            # First frame - assign new IDs to all
            for box in current_boxes:
                person_id = self.person_id_counter
                new_tracked[person_id] = box
                self.person_id_counter += 1
        else:
            # Calculate IoU between current boxes and previous boxes
            current_boxes_np = np.array([box[:4] for box in current_boxes])  # Ensure we only use x1,y1,x2,y2
            prev_boxes_np = np.array([box[:4] for box in self.tracked_persons.values()])

            if len(current_boxes_np) > 0 and len(prev_boxes_np) > 0:
                # Calculate IoU matrix
                iou_matrix = np.zeros((len(current_boxes_np), len(prev_boxes_np)))
                for i, curr_box in enumerate(current_boxes_np):
                    for j, prev_box in enumerate(prev_boxes_np):
                        iou_matrix[i, j] = self._calculate_iou(curr_box, prev_box)

                # Match current boxes to previous IDs
                matched_pairs = []
                for i in range(len(current_boxes_np)):
                    max_j = np.argmax(iou_matrix[i])
                    if iou_matrix[i, max_j] > 0.3:  # IoU threshold
                        matched_pairs.append((i, max_j))

                # Assign matched IDs
                for i, j in matched_pairs:
                    person_id = list(self.tracked_persons.keys())[j]
                    new_tracked[person_id] = current_boxes_np[i]
                    used_ids.add(person_id)

                # Assign new IDs to unmatched boxes
                for i, box in enumerate(current_boxes_np):
                    if i not in [pair[0] for pair in matched_pairs]:
                        person_id = self.person_id_counter
                        new_tracked[person_id] = box
                        self.person_id_counter += 1

        self.tracked_persons = new_tracked
        return new_tracked

    def _match_inactive_person(self, box):
        """Match a box with inactive persons based on IoU."""
        for person_id, (inactive_box, frames_inactive) in self.inactive_persons.items():
            if frames_inactive < self.inactive_timeout:
                iou = self._calculate_iou(box, inactive_box)
                if iou > 0.3:  # IoU threshold for reappearance
                    return person_id
        return None

    def _update_inactive_persons(self, new_tracked, used_ids):
        """Update the list of inactive persons."""
        # Increment frames inactive for all inactive persons
        for person_id in list(self.inactive_persons.keys()):
            self.inactive_persons[person_id] = (
                self.inactive_persons[person_id][0],
                self.inactive_persons[person_id][1] + 1
            )
            # Remove persons who have been inactive for too long
            if self.inactive_persons[person_id][1] >= self.inactive_timeout:
                del self.inactive_persons[person_id]

        # Add newly inactive persons
        for person_id, box in self.tracked_persons.items():
            if person_id not in used_ids:
                self.inactive_persons[person_id] = (box, 0)

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two boxes."""
        # Convert boxes to x1,y1,x2,y2 format if needed
        box1 = [box1[0], box1[1], box1[2], box1[3]]
        box2 = [box2[0], box2[1], box2[2], box2[3]]
        
        # Calculate intersection area
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    # ... rest of your ViolenceFeatureExtractor methods ...
    # ... rest of your ViolenceFeatureExtractor methods ...
    def _setup_gpu(self):
        """Configure GPU settings if available."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("No GPU available. Using CPU.")

    # ... rest of your ViolenceFeatureExtractor methods ...

# Fourth cell: Create instance and process video
# Replace with your actual model paths
        
    def rescale_coords(self, x, y, scale_info):
       """Convert model coordinates back to original video dimensions"""
       scale, pad_w, pad_h = scale_info
    # Remove padding and rescale
       x_orig = (x - pad_w) / scale
       y_orig = (y - pad_h) / scale
       return int(x_orig), int(y_orig)



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
                    # Get person IDs from tracked_persons
                    person1_id = [k for k, v in self.tracked_persons.items() if np.array_equal(v, box1)][0]
                    person2_id = [k for k, v in self.tracked_persons.items() if np.array_equal(v, box2)][0]
                    
                    interactions.append(
                        {
                            "person1_idx": i,
                            "person2_idx": j,
                            "person1_id": person1_id,
                            "person2_id": person2_id,
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
            if not poses or len(poses) == 0:
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



    def draw_detections(self, frame, det_results, pose_results, interactions, scale_info):
        """Draw detections, poses, and interactions on the frame."""
        try:
            display_frame = frame.copy()

            # Draw person boxes with IDs first
            for person_id, box in self.tracked_persons.items():
                try:
                    if len(box) != 4:
                        continue

                    x1, y1, x2, y2 = map(float, box)  # Ensure floating point values
                    x1, y1 = self.preprocessor.rescale_coords(x1, y1, scale_info)
                    x2, y2 = self.preprocessor.rescale_coords(x2, y2, scale_info)

                    # Ensure coordinates are valid
                    if any(coord < 0 for coord in [x1, y1, x2, y2]):
                        continue

                    # Draw person box
                    cv2.rectangle(display_frame,
                                  (int(x1), int(y1)),
                                  (int(x2), int(y2)),
                                  self.colors["person"],
                                  2)

                    # Draw person ID
                    id_text = f"ID:{person_id}"
                    (text_w, text_h), _ = cv2.getTextSize(
                        id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    cv2.rectangle(
                        display_frame,
                        (int(x2 - text_w - 5), int(y1)),
                        (int(x2), int(y1 + text_h + 5)),
                        self.colors["person"],
                        -1,
                    )
                    cv2.putText(
                        display_frame,
                        id_text,
                        (int(x2 - text_w - 2), int(y1 + text_h + 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )
                except Exception as e:
                    print(f"Error drawing person ID {person_id}: {e}")
                    continue

            # Draw keypoints
            if pose_results:
                for result in pose_results:
                    if result.keypoints:
                        for kpts in result.keypoints:
                            try:
                                keypoints = kpts.data[0].cpu().numpy()
                                for kp in keypoints:
                                    x, y, conf = kp
                                    if conf > 0.5:  # Only draw keypoints with high confidence
                                        x, y = self.preprocessor.rescale_coords(x, y, scale_info)
                                        cv2.circle(display_frame, (int(x), int(y)), 3, self.colors["keypoint"], -1)
                            except Exception as e:
                                print(f"Error drawing keypoints: {e}")
                                continue

            # Draw other objects (weapons)
            if det_results:
                for result in det_results:
                    if not hasattr(result, 'boxes'):
                        continue

                    boxes = result.boxes
                    for box in boxes:
                        try:
                            if not hasattr(box, 'xyxy') or not hasattr(box, 'cls') or not hasattr(box, 'conf'):
                                continue

                            x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
                            x1, y1 = self.preprocessor.rescale_coords(x1, y1, scale_info)
                            x2, y2 = self.preprocessor.rescale_coords(x2, y2, scale_info)

                            cls = result.names[int(box.cls[0])]
                            conf = float(box.conf[0])

                            if cls in self.violence_objects:
                                color = self.colors["violence"]
                                cv2.rectangle(display_frame,
                                              (int(x1), int(y1)),
                                              (int(x2), int(y2)),
                                              color,
                                              2)

                                label = f"{cls} {conf:.2f}"
                                (text_w, text_h), _ = cv2.getTextSize(
                                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                                )
                                cv2.rectangle(
                                    display_frame,
                                    (int(x1), int(y1 - text_h - 5)),
                                    (int(x1 + text_w), int(y1)),
                                    color,
                                    -1,
                                )
                                cv2.putText(
                                    display_frame,
                                    label,
                                    (int(x1), int(y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    2,
                                )
                        except Exception as e:
                            print(f"Error in detection drawing: {e}")
                            continue

            # Draw interactions
            for interaction in interactions:
                try:
                    x1, y1 = self.preprocessor.rescale_coords(
                        interaction["center1"][0], interaction["center1"][1], scale_info
                    )
                    x2, y2 = self.preprocessor.rescale_coords(
                        interaction["center2"][0], interaction["center2"][1], scale_info
                    )

                    cv2.line(
                        display_frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        self.colors["interaction"],
                        2
                    )

                    mid_point = ((x1 + x2) // 2, (y1 + y2) // 2)
                    distance_label = f"D: {interaction['relative_distance']:.2f}"
                    cv2.putText(
                        display_frame,
                        distance_label,
                        (int(mid_point[0]), int(mid_point[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        self.colors["interaction"],
                        2,
                    )
                except Exception as e:
                    print(f"Error drawing interaction: {e}")
                    continue

            # Draw risk level
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

            # Draw controls info
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

        except Exception as e:
            print(f"Error in draw_detections: {e}")
            return frame




    def extract_features(self, frame, frame_idx):
        """Extract violence-relevant features from a frame with robust error handling."""
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

            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                det_results = self.detection_model(frame_tensor, conf=self.conf_threshold, verbose=False)
                pose_results = self.pose_model(frame_tensor, conf=self.conf_threshold, verbose=False) if len(det_results[0].boxes) > 0 else []

            frame_data = {
                "frame_index": frame_idx,
                "timestamp": frame_idx / 30,
                "persons": [],
                "objects": [],
                "interactions": [],
            }

            # Process detections
            person_boxes = []
            for result in det_results:
                for box in result.boxes:
                    try:
                        cls = result.names[int(box.cls[0])]
                        if cls in self.relevant_classes:
                            box_coords = box.xyxy[0].cpu().numpy().tolist()
                            if cls == "person":
                                person_boxes.append(box_coords)
                            else:
                                frame_data["objects"].append({
                                    "class": cls,
                                    "confidence": float(box.conf[0]),
                                    "box": box_coords
                                })
                    except Exception as e:
                        print(f"Detection processing error: {e}")
                        continue

            self._assign_person_ids(person_boxes)

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

                    matched_ids = [k for k, v in self.tracked_persons.items() if np.array_equal(v, box)]
                    person_id = matched_ids[0] if matched_ids else self.person_id_counter

                    frame_data["persons"].append({
                        "person_idx": i,
                        "person_id": person_id,
                        "box": box,
                        "center": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                        "keypoints": pose
                    })

                    if not matched_ids:
                        self.tracked_persons[person_id] = box
                        self.person_id_counter += 1

                except Exception as e:
                    print(f"Skipping person {i} due to error: {e}")
                    continue

            # Calculate motion features with default values
            motion_features = {
                "average_speed": 0,
                "motion_intensity": 0,
                "sudden_movements": 0
            }
            if self.prev_poses and current_poses:
                try:
                    motion_features = self.calculate_motion_features(self.prev_poses, current_poses)
                except Exception as e:
                    print(f"Motion calculation error: {e}")
            self.prev_poses = current_poses

            # Create interactions with all required fields
            interactions = []
            if len(person_boxes) >= 2:
                for i in range(len(person_boxes)):
                    for j in range(i + 1, len(person_boxes)):
                        try:
                            if i >= len(current_poses) or j >= len(current_poses):
                                continue

                            box1, box2 = person_boxes[i], person_boxes[j]
                            pose1, pose2 = current_poses[i], current_poses[j]

                            id1 = next((k for k, v in self.tracked_persons.items() if np.array_equal(v, box1)), None)
                            id2 = next((k for k, v in self.tracked_persons.items() if np.array_equal(v, box2)), None)

                            if id1 is None or id2 is None:
                                continue

                            center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
                            center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
                            distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
                            avg_size = ((box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1])) / 2

                            interaction = {
                                "person1_idx": i,
                                "person2_idx": j,
                                "person1_id": id1,
                                "person2_id": id2,
                                "box1": box1,
                                "box2": box2,
                                "center1": center1,
                                "center2": center2,
                                "distance": distance,
                                "relative_distance": distance / (avg_size ** 0.5),
                                "motion_average_speed": motion_features.get("average_speed", 0),
                                "motion_motion_intensity": motion_features.get("motion_intensity", 0),
                                "motion_sudden_movements": motion_features.get("sudden_movements", 0),
                                "violence_aggressive_pose": self.analyze_poses_for_violence([pose1, pose2]),
                                "violence_close_interaction": distance < (avg_size ** 0.5) * self.interaction_threshold,
                                "violence_rapid_motion": motion_features.get("average_speed", 0) > 10,
                                "violence_weapon_present": any(obj["class"] in self.violence_objects for obj in frame_data["objects"]),
                                "keypoints": {
                                    "person1": pose1,
                                    "person2": pose2,
                                    "relative": (np.array(pose2) - np.array(pose1)).tolist()
                                }
                            }
                            interactions.append(interaction)
                        except Exception as e:
                            print(f"Skipping interaction {i}-{j}: {e}")
                            continue

            frame_data["interactions"] = interactions

            # Calculate risk level
            if frame_data["interactions"]:
                risk_scores = []
                for x in frame_data["interactions"]:
                    score = (0.4 * int(x["violence_weapon_present"]) + 0.3 * int(x["violence_close_interaction"]) + 0.2 * int(x["violence_rapid_motion"]) + 0.1 * int(x["violence_aggressive_pose"]))
                    risk_scores.append(min(score, 1.0))
                self.current_risk_level = sum(risk_scores) / len(risk_scores)
            else:
                self.current_risk_level = 0.0

            annotated_frame = self.draw_detections(frame, det_results, pose_results, frame_data["interactions"], scale_info)

            return frame_data, annotated_frame
        except Exception as e:
            print(f"Frame {frame_idx} failed completely: {e}")
            return None, frame



    def convert_numpy_to_python(self, obj):
        """Recursively convert NumPy objects to native Python types."""
        if isinstance(obj, np.generic):
            return obj.item()  # Convert NumPy scalar to Python scalar
        elif isinstance(obj, dict):
            return {
                key: self.convert_numpy_to_python(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self.convert_numpy_to_python(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_numpy_to_python(item) for item in obj)
        else:
            return obj

    def process_video(self, video_path, output_csv_path, output_folder=None, show_video=False):
        """
        Process a video file to extract pairwise interactions between all persons.
        """
        # Initialize variables before try block
        cap = None
        video_writer = None
        csv_data = []

        # Initialize a set to track seen interactions in the current frame
        seen_interactions = set()

        try:
            # Validate input video
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Error: Could not open video file")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Set appropriate configuration based on video resolution
            batch_size = self.preprocessor.set_resolution_config(frame_width, frame_height)
            self.frame_skip = self.preprocessor.frame_skip  # Update frame_skip based on resolution         

            print(f"Processing video: {frame_width}x{frame_height} at {fps} fps")
            print(f"Using frame_skip: {self.frame_skip}, batch_size: {batch_size}")

            # Initialize video writer if output folder is provided
            if output_folder:
                try:
                    os.makedirs(output_folder, exist_ok=True)
                    output_video_path = os.path.join(
                        output_folder, 
                        f"{os.path.splitext(os.path.basename(video_path))[0]}_detections.mp4"
                    )
                    video_writer = cv2.VideoWriter(
                        output_video_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps / self.frame_skip,
                        (frame_width, frame_height)
                    )
                    if not video_writer.isOpened():
                        print("Warning: Failed to create video writer")
                        video_writer = None
                except Exception as e:
                    print(f"Error creating video writer: {e}")
                    video_writer = None

            # Define expected columns
            expected_columns = [
                "frame_index", "timestamp", "person1_id", "person2_id",
                "box1", "box2", "center1", "center2", "distance",
                "person1_idx", "person2_idx", "relative_distance",
                "motion_average_speed", "motion_motion_intensity",
                "motion_sudden_movements", "violence_aggressive_pose",
                "violence_close_interaction", "violence_rapid_motion",
                "violence_weapon_present", "keypoints"
            ]

            # Initialize or load existing CSV
            try:
                if os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 0:
                    existing_df = pd.read_csv(output_csv_path)
                    # Add missing columns if they don't exist
                    for col in expected_columns:
                        if col not in existing_df.columns:
                            existing_df[col] = None
                else:
                    existing_df = pd.DataFrame(columns=expected_columns)
            except Exception as e:
                print(f"Error loading existing CSV: {e}")
                existing_df = pd.DataFrame(columns=expected_columns)

            # Main processing loop
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Only process frames that match our frame_skip pattern
                if frame_idx % self.frame_skip != 0:
                    frame_idx += 1
                    continue

                # Clear seen interactions for each new frame
                seen_interactions.clear()

                # Progress update
                progress = (frame_idx / total_frames) * 100
                print(f"\rProcessing frame {frame_idx}/{total_frames} ({progress:.1f}%)", end="")

                try:
                    # Extract features
                    frame_data, annotated_frame = self.extract_features(frame, frame_idx)

                    if frame_data is not None:
                        # Process interactions
                        for interaction in frame_data["interactions"]:
                            interaction_id = (interaction["person1_id"], interaction["person2_id"], frame_idx)

                            if interaction_id not in seen_interactions:
                                seen_interactions.add(interaction_id)

                                row = {
                                    "frame_index": frame_data["frame_index"],
                                    "timestamp": frame_data["timestamp"],
                                    "person1_id": interaction["person1_id"],
                                    "person2_id": interaction["person2_id"],
                                    "box1_x_min": interaction["box1"][0],
                                    "box1_y_min": interaction["box1"][1],
                                    "box1_x_max": interaction["box1"][2],
                                    "box1_y_max": interaction["box1"][3],
                                    "box2_x_min": interaction["box2"][0],
                                    "box2_y_min": interaction["box2"][1],
                                    "box2_x_max": interaction["box2"][2],
                                    "box2_y_max": interaction["box2"][3],
                                    "center1_x": interaction["center1"][0],
                                    "center1_y": interaction["center1"][1],
                                    "center2_x": interaction["center2"][0],
                                    "center2_y": interaction["center2"][1],
                                    "distance": interaction["distance"],
                                    "person1_idx": interaction["person1_idx"],
                                    "person2_idx": interaction["person2_idx"],
                                    "relative_distance": interaction["relative_distance"],
                                    "motion_average_speed": interaction["motion_average_speed"],
                                    "motion_motion_intensity": interaction["motion_motion_intensity"],
                                    "motion_sudden_movements": interaction["motion_sudden_movements"],
                                    "violence_aggressive_pose": interaction["violence_aggressive_pose"],
                                    "violence_close_interaction": interaction["violence_close_interaction"],
                                    "violence_rapid_motion": interaction["violence_rapid_motion"],
                                    "violence_weapon_present": interaction["violence_weapon_present"],
                                    "keypoints": str(interaction["keypoints"])
                                }
                                csv_data.append(row)

                        # Write frame to output video
                        if video_writer is not None and annotated_frame is not None:
                            try:
                                video_writer.write(annotated_frame)
                            except Exception as e:
                                print(f"\nError writing video frame: {e}")

                        # Show video if enabled
                        if show_video and annotated_frame is not None:
                            try:
                                cv2.imshow("Violence Detection", annotated_frame)
                                key = cv2.waitKey(1) & 0xFF
                                if key == ord('q'):
                                    break
                                elif key == ord('p'):
                                    while True:
                                        key = cv2.waitKey(1) & 0xFF
                                        if key in [ord('p'), ord('q')]:
                                            break
                                    if key == ord('q'):
                                        break
                            except Exception as e:
                                print(f"\nError displaying frame: {e}")
                                show_video = False

                except Exception as e:
                    print(f"\nError processing frame {frame_idx}: {str(e)}")
                    continue

                if frame_idx % 100 == 0:
                    torch.cuda.empty_cache()

                frame_idx += 1

        finally:
            if cap is not None:
                cap.release()
            if video_writer is not None:
                video_writer.release()
            if show_video:
                cv2.destroyAllWindows()
            torch.cuda.empty_cache()


            try:
                if csv_data:
                    # Flatten keypoints data for each interaction
                    flattened_data = []
                    for interaction in csv_data:
                        flat_interaction = interaction.copy()

                        # Handle keypoints data which might be string or dict
                        keypoints_str = flat_interaction.pop('keypoints', '{}')

                        try:
                            # Convert string representation to dictionary if needed
                            if isinstance(keypoints_str, str):
                                keypoints_data = ast.literal_eval(keypoints_str)
                            else:
                                keypoints_data = keypoints_str

                            # Initialize all keypoint columns to None first
                            for prefix in ['person1_kp', 'person2_kp', 'relative_kp']:
                                for i in range(17):  # COCO format has 17 keypoints
                                    for dim in ['_x', '_y', '_conf']:
                                        flat_interaction[f"{prefix}{i}{dim}"] = None

                            # Fill in actual values if they exist
                            if isinstance(keypoints_data, dict):
                                # Flatten person1 keypoints
                                if 'person1' in keypoints_data and isinstance(keypoints_data['person1'], list):
                                    for i, kp in enumerate(keypoints_data['person1']):
                                        if i >= 17:
                                            continue
                                        if isinstance(kp, (list, tuple)) and len(kp) >= 3:
                                            flat_interaction[f'person1_kp{i}_x'] = float(kp[0])
                                            flat_interaction[f'person1_kp{i}_y'] = float(kp[1])
                                            flat_interaction[f'person1_kp{i}_conf'] = float(kp[2])

                                # Flatten person2 keypoints
                                if 'person2' in keypoints_data and isinstance(keypoints_data['person2'], list):
                                    for i, kp in enumerate(keypoints_data['person2']):
                                        if i >= 17:
                                            continue
                                        if isinstance(kp, (list, tuple)) and len(kp) >= 3:
                                            flat_interaction[f'person2_kp{i}_x'] = float(kp[0])
                                            flat_interaction[f'person2_kp{i}_y'] = float(kp[1])
                                            flat_interaction[f'person2_kp{i}_conf'] = float(kp[2])

                                # Flatten relative keypoints
                                if 'relative' in keypoints_data and isinstance(keypoints_data['relative'], list):
                                    for i, kp in enumerate(keypoints_data['relative']):
                                        if i >= 17:
                                            continue
                                        if isinstance(kp, (list, tuple)) and len(kp) >= 3:
                                            flat_interaction[f'relative_kp{i}_x'] = float(kp[0])
                                            flat_interaction[f'relative_kp{i}_y'] = float(kp[1])
                                            flat_interaction[f'relative_kp{i}_conf'] = float(kp[2])

                        except (ValueError, SyntaxError) as e:
                            print(f"Error processing keypoints data: {e}")
                            # Keep all keypoint columns as None if parsing fails

                        flattened_data.append(flat_interaction)

                    df = pd.DataFrame(flattened_data)

                    # Define all expected columns (including keypoint columns)
                    base_columns = [
                        "frame_index", "timestamp", "person1_id", "person2_id",
                        "box1_x_min", "box1_y_min", "box1_x_max", "box1_y_max",
                        "box2_x_min", "box2_y_min", "box2_x_max", "box2_y_max",
                        "center1_x", "center1_y", "center2_x", "center2_y",
                        "distance", "person1_idx", "person2_idx", "relative_distance",
                        "motion_average_speed", "motion_motion_intensity",
                        "motion_sudden_movements", "violence_aggressive_pose",
                        "violence_close_interaction", "violence_rapid_motion",
                        "violence_weapon_present"
                    ]

                    # Add keypoint columns (assuming 17 keypoints per person as in COCO format)
                    keypoint_columns = []
                    for prefix in ['person1_kp', 'person2_kp', 'relative_kp']:
                        for i in range(17):
                            for dim in ['_x', '_y', '_conf']:
                                keypoint_columns.append(f"{prefix}{i}{dim}")

                    expected_columns = base_columns + keypoint_columns

                    # Ensure all columns exist in the new DataFrame
                    for col in expected_columns:
                        if col not in df.columns:
                            df[col] = None

                    # Reorder columns to match expected order
                    df = df[expected_columns]

                    # Save to CSV
                    df.to_csv(output_csv_path, index=False)
                    print(f"\nSuccessfully saved {len(csv_data)} interactions to {output_csv_path}")
                else:
                    print("\nNo interactions were detected in the video")
            except Exception as e:
                print(f"\nError saving data to CSV: {str(e)}")
                # Try saving just the new data if concatenation failed
                if csv_data:
                    try:
                        pd.DataFrame(csv_data).to_csv(output_csv_path, index=False)
                        print(f"Saved only new data to {output_csv_path}")
                    except Exception as e2:
                        print(f"Failed to save fallback CSV: {e2}")

        return frame_width, frame_height
