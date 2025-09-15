import os
import cv2
import torch
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from backend.config import OUTPUT_DIR
from backend.services.feature_extraction.extractor import VideoFeatureExtractor

class VideoDataExtractor:
    def __init__(self):
        self.extractor = VideoFeatureExtractor()
        
    def extract_video_data(self, video_path, output_csv_path, output_folder=None, show_video=False, save_video=False):
        """
        Extract data from a video file.
        
        Args:
            video_path: Path to input video
            output_csv_path: Path to save CSV output
            output_folder: Folder to save output video
            show_video: Whether to display video during processing
            save_video: Whether to save output video
        
        Returns:
            Tuple of (frame_width, frame_height, num_interactions)
        """
        cap = None
        video_writer = None
        csv_data = []
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

            video_name = os.path.splitext(os.path.basename(video_path))[0]

            # Set frame skip based on resolution
            batch_size, frame_skip = self.extractor.preprocessor.set_resolution_config(frame_width, frame_height)
            self.extractor.preprocessor.frame_skip = frame_skip

            print(f"Processing video: {frame_width}x{frame_height} at {fps} fps")
            print(f"Using frame_skip: {frame_skip}")

            # Initialize video writer if needed
            if output_folder and save_video:
                os.makedirs(output_folder, exist_ok=True)
                output_video_path = os.path.join(output_folder, f"{video_name}_detections.mp4")
                video_writer = cv2.VideoWriter(
                    output_video_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps / frame_skip,
                    (frame_width, frame_height)
                )

            # Reset extractor for new video
            self.extractor.reset()

            # Process frames
            for frame_idx in range(0, total_frames, frame_skip):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                # Extract features
                frame_data, annotated_frame = self.extractor.extract_features(frame, frame_idx)

                if frame_data is not None:
                    # Process interactions
                    for interaction in frame_data["interactions"]:
                        interaction_id = (interaction["person1_id"], interaction["person2_id"], frame_idx)

                        if interaction_id not in seen_interactions:
                            seen_interactions.add(interaction_id)
                            row = self._create_interaction_row(video_name, frame_data, interaction, frame_width, frame_height)
                            csv_data.append(row)

                    # Write frame to output video
                    if video_writer is not None and annotated_frame is not None:
                        video_writer.write(annotated_frame)

                    # Show video if enabled
                    if show_video and annotated_frame is not None:
                        cv2.imshow("Video Data Extraction", annotated_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break

                # Clear memory periodically
                if frame_idx % 100 == 0:
                    torch.cuda.empty_cache()

            if csv_data:
                df = pd.DataFrame(csv_data)
                
                if os.path.exists(output_csv_path):
                    # Append to existing CSV
                    df.to_csv(output_csv_path, mode='a', header=False, index=False)
                    print(f"Appended {len(csv_data)} interactions to {output_csv_path}")
                else:
                    # Save new CSV
                    df.to_csv(output_csv_path, index=False)
                    print(f"Saved {len(csv_data)} interactions to {output_csv_path}")
                
            return frame_width, frame_height, len(csv_data)

        finally:
            if cap is not None:
                cap.release()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()
            torch.cuda.empty_cache()

    def _create_interaction_row(self, video_name, frame_data, interaction, frame_width, frame_height):
        """Create a row of interaction data for CSV output."""
        row = {
            "video_name": video_name,
            "frame_index": frame_data["frame_index"],
            "timestamp": frame_data["timestamp"],
            "frame_width": frame_width,
            "frame_height": frame_height,
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
            "motion_average_speed": frame_data["motion_features"]["average_speed"],
            "motion_motion_intensity": frame_data["motion_features"]["motion_intensity"],
            "motion_sudden_movements": frame_data["motion_features"]["sudden_movements"],
        }

        # Add keypoints data
        keypoints_data = interaction["keypoints"]
        for prefix in ['person1_kp', 'person2_kp', 'relative_kp']:
            for i in range(17):
                for dim in ['_x', '_y', '_conf']:
                    row[f"{prefix}{i}{dim}"] = None

        # Fill in actual keypoint values if they exist
        if isinstance(keypoints_data, dict):
            for person_prefix, kp_data in [('person1_kp', keypoints_data.get('person1')),
                                          ('person2_kp', keypoints_data.get('person2')),
                                          ('relative_kp', keypoints_data.get('relative'))]:
                if isinstance(kp_data, list):
                    for i, kp in enumerate(kp_data):
                        if i >= 17:
                            continue
                        if isinstance(kp, (list, tuple)) and len(kp) >= 3:
                            row[f'{person_prefix}{i}_x'] = float(kp[0])
                            row[f'{person_prefix}{i}_y'] = float(kp[1])
                            row[f'{person_prefix}{i}_conf'] = float(kp[2])

        return row

    def extract_single_frame(self, frame_path):
        """Extract features from a single frame."""
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Could not load frame from {frame_path}")
        
        frame_data, annotated_frame = self.extractor.extract_features(frame, 0)
        return frame_data, annotated_frame