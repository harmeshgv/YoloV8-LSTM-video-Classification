from src.data_extraction.pipeline import ViolenceFeatureExtractor

def predict_violence(video_path):
    try:

        extractor = ViolenceFeatureExtractor()
        extractor.