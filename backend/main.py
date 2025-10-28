from services.prediction.predictor import ViolencePredictor
from services.video_data_extraction.video_preprocessor import VideoDataExtractor
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import os
import logging
import uuid

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("main")

app = FastAPI(title="Violence Prediction System")

# ✅ Enable CORS
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)
# Initialize shared service objects
try:
    extractor = VideoDataExtractor()
    predictor = ViolencePredictor()
    logger.info("Initialized shared service objects")
except Exception as e:
    logger.error(f"Failed to create service objects: {str(e)}")
    # Create mock objects for testing
    extractor = None
    predictor = None


def to_python(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(i) for i in obj]
    return obj


# Health Check endpoint
@app.get("/")
async def health():
    return {"status": "ok", "message": "Violence Detection API is running"}


# Extract video data
@app.post("/analyze")
async def extract_data(mode: str = Form(...), file: UploadFile = File(...)):
    if not extractor or not predictor:
        raise HTTPException(status_code=500, detail="Service not initialized properly")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Create temp file with proper path
    tmp_file_code = uuid.uuid4()
    temp_path = os.path.join(UPLOAD_DIR, f"{tmp_file_code}_{file.filename}")

    try:
        # Save uploaded file
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Processing file: {file.filename}, mode: {mode}")

        # Extract video data
        data = extractor.extract_video_data(temp_path)

        if mode == "extract":
            result = {"data": data.to_dict(orient="records")}
        else:
            prediction = predictor.predict(data)
            prediction = to_python(prediction)
            result = {"prediction": prediction}

        return result

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process video: {str(e)}"
        )
    finally:
        # Clean up temp file
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            logger.warning(f"Could not remove temp file: {e}")


# Run app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
