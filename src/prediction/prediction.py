import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class ViolencePredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.scaler = MinMaxScaler()
        
    def preprocess_data(self, df, frame_width, frame_height):
        """
        Preprocess the data by normalizing box coordinates, center coordinates, distances, and keypoints.
        """
        # Normalize box coordinates
        df['box1_x_min'] = df['box1_x_min'] / frame_width
        df['box1_y_min'] = df['box1_y_min'] / frame_height
        df['box1_x_max'] = df['box1_x_max'] / frame_width
        df['box1_y_max'] = df['box1_y_max'] / frame_height

        df['box2_x_min'] = df['box2_x_min'] / frame_width
        df['box2_y_min'] = df['box2_y_min'] / frame_height
        df['box2_x_max'] = df['box2_x_max'] / frame_width
        df['box2_y_max'] = df['box2_y_max'] / frame_height

        # Normalize center coordinates
        df['center1_x'] = df['center1_x'] / frame_width
        df['center1_y'] = df['center1_y'] / frame_height

        df['center2_x'] = df['center2_x'] / frame_width
        df['center2_y'] = df['center2_y'] / frame_height

        # Normalize distances
        max_distance = np.sqrt(frame_width**2 + frame_height**2)
        df['distance'] = df['distance'] / max_distance
        df['relative_distance'] = df['relative_distance'] / max_distance
        
        # Drop confidence columns
        drop_columns = [f'person1_kp{i}_conf' for i in range(17)] + \
                   [f'person2_kp{i}_conf' for i in range(17)] + \
                   [f'relative_kp{i}_conf' for i in range(17)]
        
        existing_columns = [col for col in drop_columns if col in df.columns]
        df = df.drop(columns=existing_columns)

        # Normalize keypoints
        for i in range(17):
            for prefix in ['person1_kp', 'person2_kp', 'relative_kp']:
                x_col = f'{prefix}{i}_x'
                y_col = f'{prefix}{i}_y'

                if x_col in df.columns:
                    df[x_col] = df[x_col] / frame_width
                if y_col in df.columns:
                    df[y_col] = df[y_col] / frame_height

        # Scale specific columns
        df["distance"] = self.scaler.fit_transform(df[["distance"]])
        df["relative_distance"] = self.scaler.fit_transform(df[["relative_distance"]])
        df["motion_average_speed"] = self.scaler.fit_transform(df[["motion_average_speed"]])
        df["motion_motion_intensity"] = self.scaler.fit_transform(df[["motion_motion_intensity"]])

        return df

    def predict(self, data, frame_width, frame_height):
        """Predict violence probability for given data"""
        processed_data = self.preprocess_data(data, frame_width, frame_height)
        return self.model.predict_proba(processed_data)