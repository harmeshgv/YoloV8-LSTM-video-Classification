from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Dict


class Settings(BaseSettings):
    # --------------------
    # App
    # --------------------
    ENV: str = "dev"

    # --------------------
    # Device
    # --------------------
    DEVICE_TYPE: str = "cuda"  # "cuda" or "cpu"
    GPU_ID: int = 0

    # --------------------
    # YOLO Models
    # --------------------
    YOLO_MODELS: Dict[str, Dict[str, str | float]] = {
        "detector": {
            "weights": "app/models/yolo11m.pt",
            "conf_threshold": 0.4,
        },
        "pose": {
            "weights": "app/models/yolo11m-pose.pt",
            "conf_threshold": 0.3,
        },
        "action": {
            "weights": "app/models/yolo11n-action.pt",
            "conf_threshold": 0.5,
        },
    }
    BATCH_SIZE: int = 8
    RESIZE: tuple = (640,640)
    # --------------------
    # Settings config
    # --------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="APP_",
        case_sensitive=False
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
