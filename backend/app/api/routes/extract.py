from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
import json
import os
import pandas as pd

from app.core.extraction_pipeline import ExtractionPipeline

router = APIRouter(
    prefix="/data",
    tags=["dataset"]
)

TMP_DIR = "/tmp"


# ------------------------
# Helper: save uploaded video
# ------------------------
def save_video(file: UploadFile) -> str:
    os.makedirs(TMP_DIR, exist_ok=True)
    video_path = os.path.join(TMP_DIR, file.filename)

    with open(video_path, "wb") as f:
        f.write(file.file.read())

    return video_path


# ------------------------
# 1️⃣ STREAMING ENDPOINT
# ------------------------
@router.post("/ingest/stream")
async def ingest_stream(file: UploadFile = File(...)):
    video_path = save_video(file)
    pipeline = ExtractionPipeline()

    def event_stream():
        for row in pipeline.stream(video_path):
            yield json.dumps(row) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/json"
    )


# ------------------------
# 2️⃣ PACKED (EXCEL) ENDPOINT
# ------------------------
@router.post("/ingest/packed")
async def ingest_packed(file: UploadFile = File(...)):
    video_path = save_video(file)
    pipeline = ExtractionPipeline()

    rows = pipeline.collect(video_path)
    df = pd.DataFrame(rows)

    output_path = os.path.join(TMP_DIR, "detections.xlsx")
    df.to_excel(output_path, index=False)

    return FileResponse(
        output_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="detections.xlsx"
    )
