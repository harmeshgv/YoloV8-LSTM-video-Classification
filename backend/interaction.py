import numpy as np

class Interaction:
    def __init__(self):
        self.interaction_threshold = 0.5
        
    def analyze_person_interactions(self, person_boxes):
        """Analyze interactions between detected people."""
        interactions = []
        if len(person_boxes) < 2:
            return interactions

        for i in range(len(person_boxes)):
            for j in range(i + 1, len(person_boxes)):
                box1 = person_boxes[i]
                box2 = person_boxes[j]
                center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
                center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
                distance = np.sqrt(
                    (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
                )
                box1_size = (box1[2] - box1[0]) * (box1[3] - box1[1])
                box2_size = (box2[2] - box2[0]) * (box2[3] - box2[1])
                avg_size = (box1_size + box2_size) / 2

                if distance < avg_size * self.interaction_threshold:
                    # Get person IDs from tracked_persons
                    person1_id = [k for k, v in self.tracked_persons.items() if np.array_equal(v, box1)][0]
                    person2_id = [k for k, v in self.tracked_persons.items() if np.array_equal(v, box2)][0]
                    
                    interactions.append(
                        {
                            "person1_idx": i,
                            "person2_idx": j,
                            "person1_id": person1_id,
                            "person2_id": person2_id,
                            "distance": distance,
                            "relative_distance": distance / avg_size,
                            "center1": center1,
                            "center2": center2,
                            "box1": box1,
                            "box2": box2,
                        }
                    )

        return interactions

    
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


    def analyze_poses_for_violence(self, poses):
        """Analyze poses for potential aggressive/violent behavior."""
        try:
            if not poses or len(poses) == 0:
                return False

            for pose in poses:
                pose_array = np.array(pose)
                arm_keypoints = [5, 7, 9, 6, 8, 10]
                arm_positions = pose_array[arm_keypoints]
                arm_confidences = arm_positions[:, 2]

                if np.mean(arm_confidences) > 0.5:
                    return True

            return False

        except Exception as e:
            print(f"Error in pose analysis: {e}")
            return False