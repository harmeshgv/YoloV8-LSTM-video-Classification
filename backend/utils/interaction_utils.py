import numpy as np


def get_box_center(box):
    """
    Calculate the center of a bounding box.

    box: [x1, y1, x2, y2]
    returns: [center_x, center_y]
    """
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def euclidean_distance(point1, point2):
    """
    Compute Euclidean distance between two points.

    point1, point2: [x, y]
    returns: float
    """
    return float(np.linalg.norm(np.array(point1) - np.array(point2)))


def relative_distance(box1, box2):
    """
    Compute relative distance between two boxes.

    Returns distance normalized by sqrt(average box area)
    """
    center1 = get_box_center(box1)
    center2 = get_box_center(box2)
    distance = euclidean_distance(center1, center2)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    avg_area = (area1 + area2) / 2

    return distance / (avg_area**0.5)


def relative_keypoints(pose1, pose2):
    """
    Compute difference between keypoints of two people.

    Returns a list of [dx, dy] for each keypoint.
    """
    return (np.array(pose2) - np.array(pose1)).tolist()
