import os
import sys
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import tempfile
import uvicorn
import pandas as pd

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipelines.video_processor import VideoProcessor

app = FastAPI()

# Initialize the processor
processor = VideoProcessor()

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
        frame_width, frame_height, num_interactions = processor.process_video(
            input_path, output_csv, output_folder=os.path.dirname(output_video), save_video=True
        )
        
        # Return the processed video
        return {
            "message": f"Processed video with {num_interactions} interactions",
            "video_path": output_video,
            "csv_path": output_csv,
            "frame_width": frame_width,
            "frame_height": frame_height
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up input file
        if os.path.exists(input_path):
            os.unlink(input_path)

@app.get("/get-results/")
async def get_results(csv_path: str):
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="CSV file not found")
    
    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename="violence_analysis_results.csv"
    )

@app.get("/sample-results/")
async def get_sample_results(csv_path: str):
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="CSV file not found")
    
    df = pd.read_csv(csv_path)
    return df.head(5).to_dict(orient="records")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)