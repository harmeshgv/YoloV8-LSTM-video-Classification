
import pandas as pd
import networkx as nx

def to_graph_data(csv_path):
    """
    Convert a CSV file of extracted features into a list of NetworkX graphs.

    Parameters:
    - csv_path: Path to the CSV file containing extracted feature data.

    Returns:
    - A list of NetworkX graphs, one for each frame in the CSV.
    """
    df = pd.read_csv(csv_path)

    grouped = df.groupby("frame_index")

    # List to store all graphs
    graphs = []

    # Iterate over each frame
    for frame_index, group in grouped:
        # Create a new graph for each frame
        G = nx.Graph()

        # Add nodes and edges for each row in the group
        for _, row in group.iterrows():
            # Add person1 node with attributes
            G.add_node(
                row["person1_id"],
                box_x_min=row.get("box1_x_min", 0),
                box_y_min=row.get("box1_y_min", 0),
                box_x_max=row.get("box1_x_max", 0),
                box_y_max=row.get("box1_y_max", 0),
                center_x=row.get("center1_x", 0),
                center_y=row.get("center1_y", 0),
                keypoints=[
                    (row.get(f"person1_kp{i}_x", 0), row.get(f"person1_kp{i}_y", 0))
                    for i in range(17)
                ],
            )

            # Add person2 node with attributes
            G.add_node(
                row["person2_id"],
                box_x_min=row.get("box2_x_min", 0),
                box_y_min=row.get("box2_y_min", 0),
                box_x_max=row.get("box2_x_max", 0),
                box_y_max=row.get("box2_y_max", 0),
                center_x=row.get("center2_x", 0),
                center_y=row.get("center2_y", 0),
                keypoints=[
                    (row.get(f"person2_kp{i}_x", 0), row.get(f"person2_kp{i}_y", 0))
                    for i in range(17)
                ],
            )

            # Add edge with attributes
            G.add_edge(
                row["person1_id"],
                row["person2_id"],
                distance=row.get("distance", 0),
                relative_distance=row.get("relative_distance", 0),
                motion_average_speed=row.get("motion_average_speed", 0),
                motion_intensity=row.get("motion_motion_intensity", 0),
                motion_sudden_movements=row.get("motion_sudden_movements", 0),
                violence_aggressive_pose=row.get("violence_aggressive_pose", 0),
                violence_close_interaction=row.get("violence_close_interaction", 0),
                violence_rapid_motion=row.get("violence_rapid_motion", 0),
                violence_weapon_present=row.get("violence_weapon_present", 0),
                relative_keypoints=[
                    (row.get(f"relative_kp{i}_x", 0), row.get(f"relative_kp{i}_y", 0))
                    for i in range(17)
                ],
            )

        # Store the graph in the list
        graphs.append(G)
        
    return graphs