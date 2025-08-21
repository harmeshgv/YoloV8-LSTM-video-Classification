# backend/utils/feature_extraction.py
import pandas as pd
import numpy as np
import torch
import cv2
import os
from ultralytics import YOLO
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.gpu import GPUConfigurator
from src.utils.state_manager import get_person_id_counter, set_person_id_counter
from src.preprocessing.preprocessor import FramePreprocessor

from src.utils.visualizer import Visualizer
from src.data_extraction.interaction import Interaction
from src.config import POSE_MODEL, DETECT_MODEL


class ViolenceFeatureExtractor:
    def __init__(self):
        # Initialize GPU configuration first
        self.gpu_config = GPUConfigurator()
        self.interact = Interaction()
        self.device = self.gpu_config.device  # Now properly initialized

        # Initialize models with the device
        self.detection_model = YOLO(DETECT_MODEL).to(self.device)
        self.pose_model = YOLO(POSE_MODEL).to(self.device)

        print(f"Detection model on: {next(self.detection_model.model.parameters()).device}")
        print(f"Pose model on: {next(self.pose_model.model.parameters()).device}")

        # Initialize preprocessor
        self.preprocessor = FramePreprocessor()

        # Rest of your initialization...
        self.violence_objects = ["knife", "gun", "baseball bat", "stick", "bottle"]
        self.relevant_classes = ["person"] + self.violence_objects
        

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
        self.visualize = Visualizer()

        
    def reset(self):
        self.person_id_counter = 0              # Counter for assigning new person IDs
        self.person_tracker = {}                # Dict to hold person-specific tracking info
        self.motion_history = {}                # Dict for storing past positions/speeds
        self.keypoint_buffers = {}              # For temporal keypoint analysis
        self.person_id_mapping = {}             # Optional: map temp person IDs across frames
        self.seen_interactions = set()          # Clear seen interactions

    def _assign_person_ids(self, current_boxes):
        """Assign consistent IDs to persons across frames using IoU matching."""
        new_tracked = {}
        used_ids = set()

        if not self.tracked_persons:
            # First frame - assign new IDs to all
            for box in current_boxes:
                person_id = get_person_id_counter()
                new_tracked[person_id] = box
                set_person_id_counter(get_person_id_counter() + 1)
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
                        person_id = get_person_id_counter()
                        new_tracked[person_id] = box
                        set_person_id_counter(get_person_id_counter() + 1)

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
            if frame_idx % 5 == 0:
               torch.cuda.empty_cache()
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                det_results = self.detection_model(frame_tensor, conf=self.conf_threshold, verbose=False)
                pose_results = self.pose_model(frame_tensor, conf=self.conf_threshold, verbose=False) if len(det_results[0].boxes) > 0 else []

            frame_data = {
                "frame_index": frame_idx,
                "timestamp": frame_idx / 30,
                "persons": [],
                "objects": [],
                "interactions": [],
                  "resized_width": scale_info.get('resized_size', (0, 0))[1],  
            "resized_height": scale_info.get('resized_size', (0, 0))[0]  
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
                    person_id = matched_ids[0] if matched_ids else get_person_id_counter()

                    frame_data["persons"].append({
                        "person_idx": i,
                        "person_id": person_id,
                        "box": box,
                        "center": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                        "keypoints": pose
                    })

                    if not matched_ids:
                        self.tracked_persons[person_id] = box
                        set_person_id_counter(get_person_id_counter() + 1)

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
                    motion_features = self.interact.calculate_motion_features(self.prev_poses, current_poses)
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
                                "violence_aggressive_pose": self.interact.analyze_poses_for_violence([pose1, pose2]),
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

            annotated_frame = self.visualize.draw_detections(frame, det_results, pose_results, frame_data["interactions"], scale_info, self.tracked_persons, self.violence_objects, self.current_risk_level)

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

    def process_video(self, video_path, output_csv_path, output_folder=None, show_video=False, save_video=False):
        """
        Process a video file to extract pairwise interactions between all persons.
        Args:
            video_path: Path to the input video file
            output_csv_path: Path to save the output CSV file
            output_folder: Optional folder to save output video with detections
            show_video: Whether to display the video during processing
            save_video: Whether to save the output video with detections
        Returns:
            Tuple of (frame_width, frame_height) of the video
        """
        # Initialize variables
        cap = None
        video_writer = None
        csv_data = []


        video_name = os.path.splitext(os.path.basename(video_path))[0]

        seen_interactions = set()

        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Error: Could not open video file")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            batch_size, self.frame_skip = self.preprocessor.set_resolution_config(frame_width, frame_height)

            print(f"Processing video: {frame_width}x{frame_height} at {fps} fps")
            print(f"Using frame_skip: {self.frame_skip}, batch_size: {batch_size}")

            if output_folder and save_video:
                try:
                    os.makedirs(output_folder, exist_ok=True)
                    output_video_path = os.path.join(
                        output_folder, 
                        f"{video_name}_detections.mp4"
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

            expected_columns = [
                "video_name",
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

            for prefix in ['person1_kp', 'person2_kp', 'relative_kp']:
                for i in range(17):
                    for dim in ['_x', '_y', '_conf']:
                        expected_columns.append(f"{prefix}{i}{dim}")

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

                                # Create base row data
                                row = {
                                    "video_name": video_name,

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
                                }

                                # Add keypoints data
                                keypoints_data = interaction["keypoints"]

                                # Initialize all keypoint columns to None
                                for prefix in ['person1_kp', 'person2_kp', 'relative_kp']:
                                    for i in range(17):
                                        for dim in ['_x', '_y', '_conf']:
                                            row[f"{prefix}{i}{dim}"] = None

                                # Fill in actual keypoint values if they exist
                                if isinstance(keypoints_data, dict):
                                    # Person1 keypoints
                                    if 'person1' in keypoints_data and isinstance(keypoints_data['person1'], list):
                                        for i, kp in enumerate(keypoints_data['person1']):
                                            if i >= 17:
                                                continue
                                            if isinstance(kp, (list, tuple)) and len(kp) >= 3:
                                                row[f'person1_kp{i}_x'] = float(kp[0])
                                                row[f'person1_kp{i}_y'] = float(kp[1])
                                                row[f'person1_kp{i}_conf'] = float(kp[2])

                                    # Person2 keypoints
                                    if 'person2' in keypoints_data and isinstance(keypoints_data['person2'], list):
                                        for i, kp in enumerate(keypoints_data['person2']):
                                            if i >= 17:
                                                continue
                                            if isinstance(kp, (list, tuple)) and len(kp) >= 3:
                                                row[f'person2_kp{i}_x'] = float(kp[0])
                                                row[f'person2_kp{i}_y'] = float(kp[1])
                                                row[f'person2_kp{i}_conf'] = float(kp[2])

                                    # Relative keypoints
                                    if 'relative' in keypoints_data and isinstance(keypoints_data['relative'], list):
                                        for i, kp in enumerate(keypoints_data['relative']):
                                            if i >= 17:
                                                continue
                                            if isinstance(kp, (list, tuple)) and len(kp) >= 3:
                                                row[f'relative_kp{i}_x'] = float(kp[0])
                                                row[f'relative_kp{i}_y'] = float(kp[1])
                                                row[f'relative_kp{i}_conf'] = float(kp[2])

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
                    # Create DataFrame with all expected columns
                    df = pd.DataFrame(csv_data)

                    # Ensure all expected columns exist in the DataFrame
                    for col in expected_columns:
                        if col not in df.columns:
                            df[col] = None

                    # Reorder columns to match expected order
                    df = df[expected_columns]

                    # Save to CSV
                    df.to_csv(output_csv_path, index=False)
                    print(f"\nSuccessfully saved {len(csv_data)} interactions to {output_csv_path}")
                    return frame_width, frame_height, len(csv_data)
                else:
                    print("\nNo interactions were detected in the video")
                    return frame_width, frame_height, 0
            except Exception as e:
                print(f"\nError saving data to CSV: {str(e)}")
                return frame_width, frame_height, 0
