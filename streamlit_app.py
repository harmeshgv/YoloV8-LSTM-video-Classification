import streamlit as st
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.services.video_data_extraction.video_preprocessor import VideoDataExtractor
from backend.services.prediction.predictor import ViolencePredictor
import tempfile

processor = VideoDataExtractor()
predictor = ViolencePredictor()

st.title("Video Violence Analysis")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        input_path = temp_video.name

    # Process video (same as FastAPI)
    output_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    frame_w, frame_h, num_interactions = processor.extract_video_data(
        input_path, output_csv, save_video=True, output_folder=None
    )

    st.write(f"Processed video with {num_interactions} interactions")

    # Show sample results
    df = pd.read_csv(output_csv)
    st.dataframe(df.head())

    # Predict violence
    preds = predictor.predict(df)
    st.write("Predictions:", preds)
