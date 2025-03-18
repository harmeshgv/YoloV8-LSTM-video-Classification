from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from utils.feature_extraction import ViolenceFeatureExtractor
import io

app = FastAPI()

extractor = ViolenceFeatureExtractor(
    detection_model_path="models/detection_model.pt",
    segmentation_model_path="models/segmentation_model.pt",
    pose_model_path="models/pose_model.pt",
)

@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    video_bytes = await file.read()
    video_path = "temp_video.mp4"
    
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    video_buffer = extractor.process_video(video_path, "violence_features.yaml")

    if video_buffer:
        return StreamingResponse(video_buffer, media_type="video/mp4")
    else:
        return {"error": "Video processing failed."}