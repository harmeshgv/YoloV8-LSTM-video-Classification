import os
import sys
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import tempfile
import uvicorn

import pandas as pd

# Add the parent directory of utils to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.pipeline import ViolenceFeatureExtractor

app = FastAPI()

# Initialize the extractor
extractor = ViolenceFeatureExtractor()

# Store processing results
processing_results = {
    "csv_path": "/extracted_feature_data/output_features.csv",
    "video_path": "/extracted_feature_data/"
}

@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    try:
        # Create temp files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_video:
            input_video.write(await file.read())
            input_path = input_video.name
        
        output_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
        output_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        # Process video
        extractor.process_video(input_path, output_csv, output_folder=os.path.dirname(output_video))
        
        # Store paths
        processing_results["csv_path"] = output_csv
        processing_results["video_path"] = output_video

        # Return the processed video
        return FileResponse(
            output_video,
            media_type="video/mp4",
            filename="processed_video.mp4"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up input file
        if os.path.exists(input_path):
            os.unlink(input_path)

@app.get("/get-results/")
async def get_results():
    if not processing_results["csv_path"]:
        raise HTTPException(status_code=404, detail="No results available")
    
    return FileResponse(
        processing_results["csv_path"],
        media_type="text/csv",
        filename="violence_analysis_results.csv"
    )

@app.get("/sample-results/")
async def get_sample_results():
    if not processing_results["csv_path"]:
        raise HTTPException(status_code=404, detail="No results available")
    
    df = pd.read_csv(processing_results["csv_path"])
    return df.head(5).to_dict(orient="records")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 