FROM python:3.9-slim

WORKDIR /app

COPY backend/ /app/

RUN pip install --upgrade pip

# Install dependencies including PyTorch, torchvision, and torchaudio with the correct versions
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
