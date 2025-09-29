# ------------------------------------------------------------
# Base image
# ------------------------------------------------------------
FROM python:3.12-slim-bookworm

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------
#   PYTHONUNBUFFERED  : ensures stdout/stderr aren’t buffered (logs show up instantly)
#   PYTHONDONTWRITEBYTECODE : avoids .pyc files
#   PIP_NO_CACHE_DIR  : prevents pip from caching wheels
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# ------------------------------------------------------------
# Working directory inside container
# ------------------------------------------------------------
WORKDIR /app

# ------------------------------------------------------------
# Copy application files
# ------------------------------------------------------------
# Copy only what’s required first (so Docker can cache layers)
COPY requirements.txt pyproject.toml ./
COPY . ./backend

# ------------------------------------------------------------
# System + Python dependencies
# ------------------------------------------------------------
# --no-install-recommends  : install only what we ask for
# Remove apt cache & docs to reduce size
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 && \
    rm -rf /var/lib/apt/lists/* /usr/share/doc /usr/share/man /usr/share/locale && \
    \
    # Upgrade pip & install Python dependencies (CPU-only Torch)
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.4.1+cpu \
        torchvision==0.19.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    # Editable install so the package can be imported elsewhere
    pip install --no-cache-dir -e .

# ------------------------------------------------------------
# Networking
# ------------------------------------------------------------
EXPOSE 8000

# ------------------------------------------------------------
# Start FastAPI with auto-reload (remove --reload for production)
# ------------------------------------------------------------
CMD ["uvicorn", "backend.fast_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
