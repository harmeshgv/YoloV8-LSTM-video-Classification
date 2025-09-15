import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECT_MODEL = os.path.join(BASE_DIR, "models", "yolov8n.pt")
POSE_MODEL = os.path.join(BASE_DIR, "models", "yolov8n-pose.pt")


# Thresholds and params
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.3))
INACTIVE_TIMEOUT = int(os.getenv("INACTIVE_TIMEOUT", 30))
FRAME_SKIP = int(os.getenv("FRAME_SKIP", 2))
INPUT_SIZE = int(os.getenv("INPUT_SIZE", 640))

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(BASE_DIR, "output"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
    