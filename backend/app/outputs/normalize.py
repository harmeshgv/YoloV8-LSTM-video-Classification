from typing import List, Dict
from app.utils.geometry import iou

def flatten_keypoints(keypoints):
    """
    keypoints: List[List[float]] -> [[x, y], ...]
    returns: dict {kp_0_x, kp_0_y, kp_1_x, kp_1_y, ...}
    """
    flat = {}

    for i, (x, y) in enumerate(keypoints):
        flat[f"kp_{i}_x"] = float(x)
        flat[f"kp_{i}_y"] = float(y)

    return flat

from app.utils.geometry import iou

def normalize_detections(
    frame_id: int,
    fps: int,
    detections: list[dict],
    poses: list[dict],
    iou_thresh: float = 0.5
) -> list[dict]:

    rows = []

    for person_id, det in enumerate(detections):
        best_pose = None
        best_iou = 0.0

        for pose in poses:
            score = iou(det["bbox"], pose["bbox"])
            if score > best_iou:
                best_iou = score
                best_pose = pose

        if best_iou < iou_thresh or best_pose is None:
            continue

        # 🔑 Flatten keypoints
        kp_features = {}
        for i, (x, y) in enumerate(best_pose["keypoints"]):
            kp_features[f"kp_{i}_x"] = float(x)
            kp_features[f"kp_{i}_y"] = float(y)

        rows.append({
            "frame_id": frame_id,
            "fps": fps,
            "person_id": person_id,
            "class_id": det["class_id"],
            "class_name": det["class_name"],
            "x1": float(det["bbox"][0]),
            "y1": float(det["bbox"][1]),
            "x2": float(det["bbox"][2]),
            "y2": float(det["bbox"][3]),
            "confidence": float(det["confidence"]),
            **kp_features,  # 🔥 spread keypoints as columns,
        })

    return rows
