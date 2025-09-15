import numpy as np

class PersonTracker:
    def __init__(self, inactive_timeout=30):
        self.person_id_counter = 0
        self.tracked_persons = {}
        self.inactive_persons = {}
        self.inactive_timeout = inactive_timeout

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two boxes."""
        # Convert boxes to x1,y1,x2,y2 format
        box1 = [box1[0], box1[1], box1[2], box1[3]]
        box2 = [box2[0], box2[1], box2[2], box2[3]]
        
        # Calculate intersection area
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    def assign_person_ids(self, current_boxes):
        """Assign consistent IDs to persons across frames using IoU matching."""
        new_tracked = {}
        used_ids = set()

        if not self.tracked_persons:
            # First frame - assign new IDs to all
            for box in current_boxes:
                person_id = self.person_id_counter
                new_tracked[person_id] = box
                self.person_id_counter += 1
        else:
            # Calculate IoU between current boxes and previous boxes
            current_boxes_np = np.array([box[:4] for box in current_boxes])
            prev_boxes_np = np.array([box[:4] for box in self.tracked_persons.values()])

            if len(current_boxes_np) > 0 and len(prev_boxes_np) > 0:
                # Calculate IoU matrix
                iou_matrix = np.zeros((len(current_boxes_np), len(prev_boxes_np)))
                for i, curr_box in enumerate(current_boxes_np):
                    for j, prev_box in enumerate(prev_boxes_np):
                        iou_matrix[i, j] = self._calculate_iou(curr_box, prev_box)

                # Match current boxes to previous IDs
                matched_pairs = []
                for i in range(len(current_boxes_np)):
                    max_j = np.argmax(iou_matrix[i])
                    if iou_matrix[i, max_j] > 0.3:  # IoU threshold
                        matched_pairs.append((i, max_j))

                # Assign matched IDs
                for i, j in matched_pairs:
                    person_id = list(self.tracked_persons.keys())[j]
                    new_tracked[person_id] = current_boxes_np[i]
                    used_ids.add(person_id)

                # Assign new IDs to unmatched boxes
                for i, box in enumerate(current_boxes_np):
                    if i not in [pair[0] for pair in matched_pairs]:
                        person_id = self.person_id_counter
                        new_tracked[person_id] = box
                        self.person_id_counter += 1

        self.tracked_persons = new_tracked
        return new_tracked

    def reset(self):
        """Reset the tracker for a new video"""
        self.person_id_counter = 0
        self.tracked_persons = {}
        self.inactive_persons = {}