# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df, frame_width, frame_height):
    """
    Preprocess the data by normalizing box coordinates, center coordinates, distances, and keypoints.

    Parameters:
    - df: DataFrame containing the data to preprocess.
    - frame_width: Width of the video frame.
    - frame_height: Height of the video frame.

    Returns:
    - df: Preprocessed DataFrame.
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
    
    drop_columns = [f'person1_kp{i}_conf' for i in range(17)] + \
               [f'person2_kp{i}_conf' for i in range(17)] + \
               [f'relative_kp{i}_conf' for i in range(17)]
    
    existing_columns = [col for col in drop_columns if col in df.columns]
    
    df = df.drop(columns = existing_columns)

    # Normalize keypoints
    for i in range(17):
        for prefix in ['person1_kp', 'person2_kp', 'relative_kp']:
            x_col = f'{prefix}{i}_x'
            y_col = f'{prefix}{i}_y'

            if x_col in df.columns:
                df[x_col] = df[x_col] / frame_width
            if y_col in df.columns:
                df[y_col] = df[y_col] / frame_height


    # Initialize the scaler
    scaler = MinMaxScaler()

    # Scale specific columns
    df["distance"] = scaler.fit_transform(df[["distance"]])
    df["relative_distance"] = scaler.fit_transform(df[["relative_distance"]])
    df["motion_average_speed"] = scaler.fit_transform(df[["motion_average_speed"]])
    df["motion_motion_intensity"] = scaler.fit_transform(df[["motion_motion_intensity"]])
    df_columns = [column_name for column_name in df.columns]

    return df , df_columns