# Use Python base image
FROM python:3.9-slim

# Create working directory
WORKDIR /app

COPY /backend /app/

RUN pip install --upgrade pip

RUN pip install -r requirements.txt 

# Pre-download YOLOv8 and YOLOv8n-pose models
RUN mkdir -p /models && \
    apt-get update && \
    apt-get install -y curl && \
    curl -L https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt -o /models/yolov8n.pt && \
    curl -L https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n-pose.pt -o /models/yolov8n-pose.pt && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

EXPOSE 8000

# Just to verify models exist
CMD ["python", "main.py"]
