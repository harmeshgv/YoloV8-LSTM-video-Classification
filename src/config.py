import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

POSE_MODEL = os.path.join(BASE_DIR,"src", "models","yolo8n-pose.pt")
DETECT_MODEL = os.path.join(BASE_DIR,"src", "models","yolo8n.pt")