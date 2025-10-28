import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    def __init__(self):
        # Initialize scaler (use transform() only during inference)
        self.scaler = MinMaxScaler()

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data by normalizing box coordinates, center coordinates,
        distances, and keypoints.
        """
        df = df.copy()  # prevent modifying original

        # Normalize box coordinates
        frame_height = df["frame_height"]
        frame_width = df["frame_width"]

        for prefix in ["box1", "box2"]:
            for coord in ["x_min", "x_max"]:
                df[f"{prefix}_{coord}"] = df[f"{prefix}_{coord}"] / frame_width
            for coord in ["y_min", "y_max"]:
                df[f"{prefix}_{coord}"] = df[f"{prefix}_{coord}"] / frame_height

        # Normalize center coordinates
        for axis in ["x", "y"]:
            df[f"center1_{axis}"] = df[f"center1_{axis}"] / (
                frame_width if axis == "x" else frame_height
            )
            df[f"center2_{axis}"] = df[f"center2_{axis}"] / (
                frame_width if axis == "x" else frame_height
            )

        # Normalize distances
        max_distance = np.sqrt(frame_width**2 + frame_height**2)
        for col in ["distance", "relative_distance"]:
            if col in df.columns:
                df[col] = df[col] / max_distance

        # Drop confidence columns
        drop_columns = (
            [f"person1_kp{i}_conf" for i in range(17)]
            + [f"person2_kp{i}_conf" for i in range(17)]
            + [f"relative_kp{i}_conf" for i in range(17)]
        )
        df = df.drop(
            columns=[c for c in drop_columns if c in df.columns], errors="ignore"
        )

        # Normalize keypoints
        for i in range(17):
            for prefix in ["person1_kp", "person2_kp", "relative_kp"]:
                if f"{prefix}{i}_x" in df.columns:
                    df[f"{prefix}{i}_x"] = df[f"{prefix}{i}_x"] / frame_width
                if f"{prefix}{i}_y" in df.columns:
                    df[f"{prefix}{i}_y"] = df[f"{prefix}{i}_y"] / frame_height

        # Scale motion/distance columns
        for col in [
            "distance",
            "relative_distance",
            "motion_average_speed",
            "motion_motion_intensity",
        ]:
            if col in df.columns:
                df[col] = self.scaler.fit_transform(
                    df[[col]]
                )  # change to transform() in production

        return df
