def _create_interaction_row(
    video_name, frame_data, interaction, frame_width, frame_height
):
    """Create a row of interaction data for CSV output."""
    row = {
        "video_name": video_name,
        "frame_index": frame_data["frame_index"],
        "timestamp": frame_data["timestamp"],
        "frame_width": frame_width,
        "frame_height": frame_height,
        "person1_id": interaction["person1_id"],
        "person2_id": interaction["person2_id"],
        "box1_x_min": interaction["box1"][0],
        "box1_y_min": interaction["box1"][1],
        "box1_x_max": interaction["box1"][2],
        "box1_y_max": interaction["box1"][3],
        "box2_x_min": interaction["box2"][0],
        "box2_y_min": interaction["box2"][1],
        "box2_x_max": interaction["box2"][2],
        "box2_y_max": interaction["box2"][3],
        "center1_x": interaction["center1"][0],
        "center1_y": interaction["center1"][1],
        "center2_x": interaction["center2"][0],
        "center2_y": interaction["center2"][1],
        "distance": interaction["distance"],
        "person1_idx": interaction["person1_idx"],
        "person2_idx": interaction["person2_idx"],
        "relative_distance": interaction["relative_distance"],
        "motion_average_speed": frame_data["motion_features"]["average_speed"],
        "motion_motion_intensity": frame_data["motion_features"]["motion_intensity"],
        "motion_sudden_movements": frame_data["motion_features"]["sudden_movements"],
    }

    # Add keypoints data
    keypoints_data = interaction["keypoints"]
    for prefix in ["person1_kp", "person2_kp", "relative_kp"]:
        for i in range(17):
            for dim in ["_x", "_y", "_conf"]:
                row[f"{prefix}{i}{dim}"] = None

    # Fill in actual keypoint values if they exist
    if isinstance(keypoints_data, dict):
        for person_prefix, kp_data in [
            ("person1_kp", keypoints_data.get("person1")),
            ("person2_kp", keypoints_data.get("person2")),
            ("relative_kp", keypoints_data.get("relative")),
        ]:
            if isinstance(kp_data, list):
                for i, kp in enumerate(kp_data):
                    if i >= 17:
                        continue
                    if isinstance(kp, (list, tuple)) and len(kp) >= 3:
                        row[f"{person_prefix}{i}_x"] = float(kp[0])
                        row[f"{person_prefix}{i}_y"] = float(kp[1])
                        row[f"{person_prefix}{i}_conf"] = float(kp[2])

    return row
