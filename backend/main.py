import os
import uuid
import tempfile
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import uvicorn

from backend.services.video_processing.processor import VideoProcessor
from backend.services.prediction.predictor import ViolencePredictor

app = FastAPI(title="Video Analysis Backend")

processor = VideoProcessor()
predictor = ViolencePredictor()
jobs: dict[str, dict] = {}


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Service is running"}


@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_video:
            input_video.write(await file.read())
            input_path = input_video.name

        output_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
        output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        frame_w, frame_h, num_interactions = processor.process_video(
            input_path, output_csv,
            output_folder=os.path.dirname(output_video_path),
            save_video=True
        )

        job_id = str(uuid.uuid4())
        jobs[job_id] = {"csv": output_csv, "video": output_video_path}

        return {
            "job_id": job_id,
            "message": f"Processed video with {num_interactions} interactions",
            "frame_width": frame_w,
            "frame_height": frame_h
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)


@app.get("/get-results/{job_id}")
async def get_results(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job ID not found")
    csv_path = jobs[job_id]["csv"]
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="CSV file not found")
    return FileResponse(csv_path, media_type="text/csv",
                        filename="violence_analysis_results.csv")


@app.get("/sample-results/{job_id}")
async def get_sample_results(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job ID not found")
    csv_path = jobs[job_id]["csv"]
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="CSV file not found")
    df = pd.read_csv(csv_path)
    return df.head(5).to_dict(orient="records")


@app.post("/predict/{job_id}")
async def predict_violence(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job ID not found")
    csv_path = jobs[job_id]["csv"]
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="CSV file not found")
    df = pd.read_csv(csv_path)
    preds = predictor.predict(df)
    return {"predictions": preds.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
