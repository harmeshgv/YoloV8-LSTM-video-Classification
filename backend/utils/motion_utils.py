import numpy as np


def calc_avg_speed(prev_poses: list, current_poses: list) -> float:
    if not prev_poses or not current_poses:
        return 0.0

    prev_poses = np.array(prev_poses)
    current_poses = np.array(current_poses)

    if prev_poses.shape != current_poses.shape:
        return 0.0

    displacement = np.linalg.norm(current_poses - prev_poses, axis=2)
    return float(np.mean(displacement))


def calc_motion_intensity(prev_poses: list, current_poses: list) -> float:
    if not prev_poses or not current_poses:
        return 0.0

    prev_poses = np.array(prev_poses)
    current_poses = np.array(current_poses)

    if prev_poses.shape != current_poses.shape:
        return 0.0

    displacement = np.linalg.norm(current_poses - prev_poses, axis=2)
    return float(np.std(displacement))


def calc_sudden_movements(prev_poses: list, current_poses: list) -> int:
    if not prev_poses or not current_poses:
        return 0

    prev_poses = np.array(prev_poses)
    current_poses = np.array(current_poses)

    if prev_poses.shape != current_poses.shape:
        return 0

    displacement = np.linalg.norm(current_poses - prev_poses, axis=2)
    threshold = np.mean(displacement) + 2 * np.std(displacement)
    return int(np.sum(displacement > threshold))
