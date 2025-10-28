import os
import cv2
import torch
import pandas as pd
from feature_extraction.extractor import VideoFeatureExtractor
from utils.csv_utils import _create_interaction_row


class VideoDataExtractor:
    def __init__(self):
        self.extractor = VideoFeatureExtractor()

    def extract_video_data(
        self,
        video_path,
        output_csv_path=None,
        output_folder=None,
        show_video=False,
        save_video=False,
    ):
        """
        Extract interaction data from a video file and return a DataFrame.

        Args:
            video_path: Path to input video
            output_csv_path: Optional path to save CSV output
            output_folder: Optional folder to save annotated video
            show_video: Whether to display video during processing
            save_video: Whether to save output video

        Returns:
            pandas.DataFrame containing extracted interactions
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

            # Configure resolution-based settings
            batch_size, frame_skip = self.extractor.preprocessor.set_resolution_config(
                frame_width, frame_height
            )
            self.extractor.preprocessor.frame_skip = frame_skip

            # Initialize video writer if required
            if output_folder and save_video:
                os.makedirs(output_folder, exist_ok=True)
                output_video_path = os.path.join(
                    output_folder, f"{video_name}_detections.mp4"
                )
                video_writer = cv2.VideoWriter(
                    output_video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps / frame_skip,
                    (frame_width, frame_height),
                )

            # Reset extractor for a fresh start
            self.extractor.reset()

            # Process frames
            for frame_idx in range(0, total_frames, frame_skip):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                frame_data, annotated_frame = self.extractor.extract_features(
                    frame, frame_idx
                )

                if frame_data is not None:
                    for interaction in frame_data["interactions"]:
                        interaction_id = (
                            interaction["person1_id"],
                            interaction["person2_id"],
                            frame_idx,
                        )

                        if interaction_id not in seen_interactions:
                            seen_interactions.add(interaction_id)
                            row = _create_interaction_row(
                                video_name,
                                frame_data,
                                interaction,
                                frame_width,
                                frame_height,
                            )
                            csv_data.append(row)

                    if video_writer is not None and annotated_frame is not None:
                        video_writer.write(annotated_frame)

                    if show_video and annotated_frame is not None:
                        cv2.imshow("Video Data Extraction", annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                if frame_idx % 100 == 0:
                    torch.cuda.empty_cache()

            # ✅ Return only the DataFrame
            if csv_data:
                df = pd.DataFrame(csv_data)
                if output_csv_path:
                    df.to_csv(
                        output_csv_path,
                        mode="a" if os.path.exists(output_csv_path) else "w",
                        header=not os.path.exists(output_csv_path),
                        index=False,
                    )
                return df
            else:
                return pd.DataFrame()  # empty DataFrame if nothing found

        finally:
            if cap is not None:
                cap.release()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()
            torch.cuda.empty_cache()
