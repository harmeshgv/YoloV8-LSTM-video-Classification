FROM python:3.9-slim

# Install system dependencies for OpenCV and other requirements
RUN apt-get update && \
    apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

# Copy the backend directory (including models) to the container
COPY ./backend/ /app/

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install the dependencies listed in requirements.txt
RUN pip install -r /app/requirements.txt

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Expose port 8000 for the FastAPI app
EXPOSE 8000

# Command to start the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
