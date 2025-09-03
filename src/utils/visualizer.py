import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        self.colors = {
            "person": (0, 255, 0),    # Green
            "keypoint": (255, 255, 0), # Yellow
            "connection": (0, 255, 255)# Cyan
        }

    def rescale_coords(self, x, y, scale_info):
        """Convert model coordinates back to original video dimensions"""
        scale = scale_info['scale']
        pad_w = scale_info['pad_w']
        pad_h = scale_info['pad_h']
        original_h, original_w = scale_info['original_size']

        # Remove padding and scale back to original dimensions
        x_orig = int((x - pad_w) / scale)
        y_orig = int((y - pad_h) / scale)

        # Ensure coordinates are within bounds
        x_orig = max(0, min(x_orig, original_w - 1))
        y_orig = max(0, min(y_orig, original_h - 1))

        return (x_orig, y_orig)

    def draw_detections(self, frame, det_results, pose_results, scale_info, tracked_persons):
        """Draw detections and poses on the frame."""
        try:
            display_frame = frame.copy()

            # Draw person boxes with IDs first
            for person_id, box in tracked_persons.items():
                try:
                    if len(box) != 4:
                        continue

                    x1, y1, x2, y2 = map(float, box)
                    x1, y1 = self.rescale_coords(x1, y1, scale_info)
                    x2, y2 = self.rescale_coords(x2, y2, scale_info)

                    # Ensure coordinates are valid
                    if any(coord < 0 for coord in [x1, y1, x2, y2]):
                        continue

                    # Draw person box
                    cv2.rectangle(display_frame,
                                  (int(x1), int(y1)),
                                  (int(x2), int(y2)),
                                  self.colors["person"],
                                  2)

                    # Draw person ID
                    id_text = f"ID:{person_id}"
                    (text_w, text_h), _ = cv2.getTextSize(
                        id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    cv2.rectangle(
                        display_frame,
                        (int(x2 - text_w - 5), int(y1)),
                        (int(x2), int(y1 + text_h + 5)),
                        self.colors["person"],
                        -1,
                    )
                    cv2.putText(
                        display_frame,
                        id_text,
                        (int(x2 - text_w - 2), int(y1 + text_h + 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )
                except Exception as e:
                    print(f"Error drawing person ID {person_id}: {e}")
                    continue

            # Draw keypoints
            if pose_results:
                for result in pose_results:
                    if result.keypoints:
                        for kpts in result.keypoints:
                            try:
                                keypoints = kpts.data[0].cpu().numpy()
                                for kp in keypoints:
                                    x, y, conf = kp
                                    if conf > 0.5:  # Only draw keypoints with high confidence
                                        x, y = self.rescale_coords(x, y, scale_info)
                                        cv2.circle(display_frame, (int(x), int(y)), 3, self.colors["keypoint"], -1)
                            except Exception as e:
                                print(f"Error drawing keypoints: {e}")
                                continue

            return display_frame

        except Exception as e:
            print(f"Error in draw_detections: {e}")
            return frame