import numpy as np
from utils.iou_utils import calculate_iou
from utils.id_utils import get_new_id


class PersonTracker:
    """
    Tracks people across frames by assigning consistent IDs to bounding boxes.
    """

    def __init__(self, inactive_timeout=30):
        self.person_id_counter = 0
        self.tracked_persons = {}  # {id: box}
        self.inactive_persons = {}  # future use
        self.inactive_timeout = inactive_timeout

    def assign_person_ids(self, current_boxes):
        """
        Assign IDs to current frame boxes based on IoU with previous frame.

        Args:
            current_boxes (list of list): [[x1, y1, x2, y2], ...]

        Returns:
            dict: {person_id: box} for current frame
        """
        new_tracked = {}
        used_ids = set()

        if not self.tracked_persons:
            # First frame - assign new IDs to all boxes
            for box in current_boxes:
                person_id, self.person_id_counter = get_new_id(self.person_id_counter)
                new_tracked[person_id] = box
        else:
            # Convert boxes to numpy arrays
            current_boxes_np = np.array(current_boxes)
            prev_boxes_np = np.array(list(self.tracked_persons.values()))

            if len(current_boxes_np) > 0 and len(prev_boxes_np) > 0:
                # Compute IoU matrix
                iou_matrix = np.zeros((len(current_boxes_np), len(prev_boxes_np)))
                for i, curr_box in enumerate(current_boxes_np):
                    for j, prev_box in enumerate(prev_boxes_np):
                        iou_matrix[i, j] = calculate_iou(curr_box, prev_box)

                # Match boxes based on IoU > 0.3
                matched_pairs = []
                for i in range(len(current_boxes_np)):
                    max_j = np.argmax(iou_matrix[i])
                    if iou_matrix[i, max_j] > 0.3:
                        matched_pairs.append((i, max_j))

                # Assign matched IDs
                prev_ids = list(self.tracked_persons.keys())
                for i, j in matched_pairs:
                    person_id = prev_ids[j]
                    new_tracked[person_id] = current_boxes_np[i]
                    used_ids.add(person_id)

                # Assign new IDs to unmatched boxes
                for i, box in enumerate(current_boxes_np):
                    if i not in [pair[0] for pair in matched_pairs]:
                        person_id, self.person_id_counter = get_new_id(
                            self.person_id_counter
                        )
                        new_tracked[person_id] = box

        self.tracked_persons = new_tracked
        return new_tracked

    def reset(self):
        """Reset the tracker for a new video."""
        self.person_id_counter = 0
        self.tracked_persons = {}
        self.inactive_persons = {}
