import os

# Model paths
DETECT_MODEL = "yolov8n.pt"
POSE_MODEL = "yolov8n-pose.pt"

# Extraction parameters
CONF_THRESHOLD = 0.3
INACTIVE_TIMEOUT = 30
FRAME_SKIP = 2
INPUT_SIZE = 640

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)