import cv2
from .preprocessor import FramePreprocessor


class Visualizer:
    def __init__(self):
        self.colors = {
            "violence": (0, 0, 255),  # Red
            "person": (0, 255, 0),    # Green
            "interaction": (255, 0, 0),# Blue
            "keypoint": (255, 255, 0), # Yellow
            "connection": (0, 255, 255)# Cyan
        }
        self.preprocessor = FramePreprocessor()
    def rescale_coords(self, x, y, scale_info):
       """Convert model coordinates back to original video dimensions"""
       scale, pad_w, pad_h = scale_info
    # Remove padding and rescale
       x_orig = (x - pad_w) / scale
       y_orig = (y - pad_h) / scale
       return int(x_orig), int(y_orig)

    def draw_detections(self, frame, det_results, pose_results, interactions, scale_info, tracked_persons, violence_objects, current_risk_level):
        """Draw detections, poses, and interactions on the frame."""
        try:
            display_frame = frame.copy()

            # Draw person boxes with IDs first
            for person_id, box in tracked_persons.items():
                try:
                    if len(box) != 4:
                        continue

                    x1, y1, x2, y2 = map(float, box)  # Ensure floating point values
                    x1, y1 = self.preprocessor.rescale_coords(x1, y1, scale_info)
                    x2, y2 = self.preprocessor.rescale_coords(x2, y2, scale_info)

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
                                        x, y = self.preprocessor.rescale_coords(x, y, scale_info)
                                        cv2.circle(display_frame, (int(x), int(y)), 3, self.colors["keypoint"], -1)
                            except Exception as e:
                                print(f"Error drawing keypoints: {e}")
                                continue

            # Draw other objects (weapons)
            if det_results:
                for result in det_results:
                    if not hasattr(result, 'boxes'):
                        continue

                    boxes = result.boxes
                    for box in boxes:
                        try:
                            if not hasattr(box, 'xyxy') or not hasattr(box, 'cls') or not hasattr(box, 'conf'):
                                continue

                            x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
                            x1, y1 = self.preprocessor.rescale_coords(x1, y1, scale_info)
                            x2, y2 = self.preprocessor.rescale_coords(x2, y2, scale_info)

                            cls = result.names[int(box.cls[0])]
                            conf = float(box.conf[0])

                            if cls in violence_objects:
                                color = self.colors["violence"]
                                cv2.rectangle(display_frame,
                                              (int(x1), int(y1)),
                                              (int(x2), int(y2)),
                                              color,
                                              2)

                                label = f"{cls} {conf:.2f}"
                                (text_w, text_h), _ = cv2.getTextSize(
                                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                                )
                                cv2.rectangle(
                                    display_frame,
                                    (int(x1), int(y1 - text_h - 5)),
                                    (int(x1 + text_w), int(y1)),
                                    color,
                                    -1,
                                )
                                cv2.putText(
                                    display_frame,
                                    label,
                                    (int(x1), int(y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    2,
                                )
                        except Exception as e:
                            print(f"Error in detection drawing: {e}")
                            continue

            # Draw interactions
            for interaction in interactions:
                try:
                    x1, y1 = self.preprocessor.rescale_coords(
                        interaction["center1"][0], interaction["center1"][1], scale_info
                    )
                    x2, y2 = self.preprocessor.rescale_coords(
                        interaction["center2"][0], interaction["center2"][1], scale_info
                    )

                    cv2.line(
                        display_frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        self.colors["interaction"],
                        2
                    )

                    mid_point = ((x1 + x2) // 2, (y1 + y2) // 2)
                    distance_label = f"D: {interaction['relative_distance']:.2f}"
                    cv2.putText(
                        display_frame,
                        distance_label,
                        (int(mid_point[0]), int(mid_point[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        self.colors["interaction"],
                        2,
                    )
                except Exception as e:
                    print(f"Error drawing interaction: {e}")
                    continue

            # Draw risk level
            if current_risk_level > 0.7:
                cv2.putText(
                    display_frame,
                    "HIGH RISK",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            # Draw controls info
            cv2.putText(
                display_frame,
                "Press 'q' to quit, 'p' to pause/resume",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            return display_frame

        except Exception as e:
            print(f"Error in draw_detections: {e}")
            return frame


