import os
import cv2
import torch
import pandas as pd
from tqdm import tqdm

from backend.utils.gpu import GPUConfigurator
from backend.pipelines.video_preprocessor import VideoDataExtractor

class VideoProcessor:
    def __init__(self):
        self.extractor = VideoDataExtractor()
        self.gpu_config = GPUConfigurator()
        self.device = self.gpu_config.device

    def process_video(self, input_path, output_csv_path, output_folder=None, save_video=False):
        frame_width, frame_height, num_interactions = self.extractor.extract_video_data(
            input_path, output_csv_path, output_folder=output_folder, save_video=save_video
        )
        return frame_width, frame_height, num_interactions
