import numpy as np


class InteractionAnalyzer:
    def __init__(self, interaction_threshold=0.5):
        self.interaction_threshold = interaction_threshold

    def calculate_motion_features(self, prev_poses, current_poses):
        """Calculate motion features between consecutive frames."""
        try:
            if not prev_poses or not current_poses:
                return {
                    "average_speed": 0,
                    "motion_intensity": 0,
                    "sudden_movements": 0,
                }

            prev_poses = np.array(prev_poses)
            current_poses = np.array(current_poses)

            if prev_poses.shape == current_poses.shape:
                displacement = np.linalg.norm(current_poses - prev_poses, axis=2)
                average_speed = np.mean(displacement)
                motion_intensity = np.std(displacement)
                sudden_movements = np.sum(
                    displacement > np.mean(displacement) + 2 * np.std(displacement)
                )

                return {
                    "average_speed": float(average_speed),
                    "motion_intensity": float(motion_intensity),
                    "sudden_movements": int(sudden_movements),
                }

            return {"average_speed": 0, "motion_intensity": 0, "sudden_movements": 0}

        except Exception as e:
            print(f"Error in motion calculation: {e}")
            return {"average_speed": 0, "motion_intensity": 0, "sudden_movements": 0}

    def calculate_interactions(self, person_boxes, current_poses, tracked_persons):
        """Calculate interactions between people."""
        interactions = []

        if len(person_boxes) < 2:
            return interactions

        for i in range(len(person_boxes)):
            for j in range(i + 1, len(person_boxes)):
                try:
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

                    center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
                    center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
                    distance = np.sqrt(
                        (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
                    )
                    avg_size = (
                        (box1[2] - box1[0]) * (box1[3] - box1[1])
                        + (box2[2] - box2[0]) * (box2[3] - box2[1])
                    ) / 2

                    interaction = {
                        "person1_idx": i,
                        "person2_idx": j,
                        "person1_id": id1,
                        "person2_id": id2,
                        "box1": box1,
                        "box2": box2,
                        "center1": center1,
                        "center2": center2,
                        "distance": distance,
                        "relative_distance": distance / (avg_size**0.5),
                        "keypoints": {
                            "person1": pose1,
                            "person2": pose2,
                            "relative": (np.array(pose2) - np.array(pose1)).tolist(),
                        },
                    }
                    interactions.append(interaction)
                except Exception as e:
                    print(f"Skipping interaction {i}-{j}: {e}")
                    continue

        return interactions
