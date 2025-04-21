import numpy as np
import cv2
from ultralytics import YOLO
from collections import defaultdict

# Second cell: FramePreprocessor class
class FramePreprocessor:
    def __init__(self):
        self.max_input_size = 1920
        self.input_size = 640  # YOLO model input size
        self.frame_skip = 2
        
    def set_resolution_config(self, frame_width, frame_height):
        """Set appropriate configuration based on video resolution"""
        max_dim = max(frame_width, frame_height)
        
        # Adjust configuration based on resolution
        if max_dim > 2560:  # 4K
            self.frame_skip = 1
            batch_size = 1
        elif max_dim > 1920:  # 2K
            self.frame_skip = 1
            batch_size = 1
        elif max_dim > 1280:  # Full HD
            self.frame_skip = 1
            batch_size = 1
        else:  # HD or lower
            self.frame_skip = 0
            batch_size = 1

        print(f"Input resolution: {frame_width}x{frame_height}")
        print(f"Frame skip: {self.frame_skip}")
        print(f"Batch size: {batch_size}")
        
        return batch_size

    def preprocess_frame(self, frame):
        """Preprocess frame while maintaining aspect ratio and handling high-res inputs"""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_h, original_w = frame_rgb.shape[:2]

            # Calculate target size maintaining aspect ratio
            scale = self.input_size / max(original_w, original_h)
            target_w = int(original_w * scale)
            target_h = int(original_h * scale)

            # Resize image
            resized = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)

            # Create square canvas
            canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)

            # Calculate padding
            pad_h = (self.input_size - target_h) // 2
            pad_w = (self.input_size - target_w) // 2

            # Place resized image on canvas
            canvas[pad_h:pad_h + target_h, pad_w:pad_w + target_w] = resized

            # Debugging: Log the dimensions and padding
            print(f"Original size: {original_w}x{original_h}, Resized size: {target_w}x{target_h}")
            print(f"Padding: (pad_w: {pad_w}, pad_h: {pad_h})")

            # Normalize
            normalized = canvas.astype(np.float32) / 255.0

            # Store scaling info
            scale_info = {
                'scale': scale,
                'pad_w': pad_w,
                'pad_h': pad_h,
                'original_size': (original_h, original_w),
                'resized_size': (target_h, target_w)
            }

            return normalized, scale_info

        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None, None

    def rescale_coords(self, x, y, scale_info):
        """Convert model coordinates back to original video dimensions"""
        try:
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

        except Exception as e:
            print(f"Rescaling error: {e}")
            return (0, 0)
