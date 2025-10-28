import numpy as np
from utils.motion_utils import (
    calc_avg_speed,
    calc_motion_intensity,
    calc_sudden_movements,
)
from utils.interaction_utils import (
    get_box_center,
    euclidean_distance,
    relative_distance,
    relative_keypoints,
)


class InteractionAnalyzer:
    """
    Analyze human motion and interactions between people based on poses and bounding boxes.
    """

    def __init__(self):
        # You can later add thresholds or state here if needed
        pass

    def calculate_motion_features(
        self,
        prev_poses: list[list[list[float]]],
        current_poses: list[list[list[float]]],
    ) -> dict:
        """
        Calculate motion features between consecutive frames.

        Args:
            prev_poses: List of keypoints for all people in previous frame
            current_poses: List of keypoints for all people in current frame

        Returns:
            dict: {
                "average_speed": float,
                "motion_intensity": float,
                "sudden_movements": int
            }
        """
        return {
            "average_speed": calc_avg_speed(prev_poses, current_poses),
            "motion_intensity": calc_motion_intensity(prev_poses, current_poses),
            "sudden_movements": calc_sudden_movements(prev_poses, current_poses),
        }

    def calculate_interactions(
        self,
        person_boxes: list[list[float]],
        current_poses: list[list[list[float]]],
        tracked_persons: dict,
    ) -> list[dict]:
        """
        Calculate interactions between people based on bounding boxes and keypoints.

        Args:
            person_boxes: List of bounding boxes [[x1,y1,x2,y2], ...] for each person
            current_poses: List of keypoints for each person
            tracked_persons: Dict mapping person_id -> last tracked box

        Returns:
            List of dictionaries describing interactions between people
        """
        interactions = []

        if len(person_boxes) < 2:
            return interactions

        for i in range(len(person_boxes)):
            for j in range(i + 1, len(person_boxes)):
                try:
                    # Ensure poses exist for both people
                    if i >= len(current_poses) or j >= len(current_poses):
                        continue

                    box1, box2 = person_boxes[i], person_boxes[j]
                    pose1, pose2 = current_poses[i], current_poses[j]

                    # Find person IDs
                    id1, id2 = None, None
                    for pid, tracked_box in tracked_persons.items():
                        if np.array_equal(box1, tracked_box):
                            id1 = pid
                        if np.array_equal(box2, tracked_box):
                            id2 = pid

                    if id1 is None or id2 is None:
                        continue

                    # Build interaction dictionary using utils
                    interaction = {
                        "person1_idx": i,
                        "person2_idx": j,
                        "person1_id": id1,
                        "person2_id": id2,
                        "box1": box1,
                        "box2": box2,
                        "center1": get_box_center(box1),
                        "center2": get_box_center(box2),
                        "distance": euclidean_distance(
                            get_box_center(box1), get_box_center(box2)
                        ),
                        "relative_distance": relative_distance(box1, box2),
                        "keypoints": {
                            "person1": pose1,
                            "person2": pose2,
                            "relative": relative_keypoints(pose1, pose2),
                        },
                    }
                    interactions.append(interaction)

                except Exception as e:
                    print(f"Skipping interaction {i}-{j}: {e}")
                    continue

        return interactions
