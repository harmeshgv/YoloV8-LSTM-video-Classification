from app.config import get_settings
import torch

def get_device() -> str:
  settings = get_settings()
  if settings.DEVICE_TYPE == "cuda" and torch.cuda.is_available():
    return "cuda"
  return "cpu"