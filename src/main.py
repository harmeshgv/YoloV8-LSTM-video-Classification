import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import OUTPUT_DIR
from src.pipelines.video_preprocessor import VideoDataExtractor

def main():
    parser = argparse.ArgumentParser(description="Video Data Extraction")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", help="Path to output CSV file")
    parser.add_argument("--output_folder", help="Folder to save output video with detections")
    parser.add_argument("--show_video", action="store_true", help="Display video during processing")
    parser.add_argument("--save_video", action="store_true", help="Save output video with detections")
    
    args = parser.parse_args()
    
    # Set default output paths
    if args.output is None:
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.join(OUTPUT_DIR, f"{video_name}_features.csv")
    
    if args.output_folder is None:
        args.output_folder = OUTPUT_DIR
    
    processor = VideoDataExtractor()
    frame_width, frame_height, num_interactions = processor.extract_video_data(
        args.video, args.output, args.output_folder, args.show_video, args.save_video
    )
    
    print(f"Extraction complete. Found {num_interactions} interactions.")
    print(f"Video dimensions: {frame_width}x{frame_height}")
    print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()