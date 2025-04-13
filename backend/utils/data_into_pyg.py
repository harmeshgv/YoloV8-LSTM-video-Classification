import torch
from torch_geometric.data import Data

def convert_to_pyg_data(G, label):
    """
    Convert a NetworkX graph to a PyTorch Geometric Data object.

    Parameters:
    - G: A NetworkX graph with node and edge attributes.
    - label: An integer label for the graph.

    Returns:
    - A PyTorch Geometric Data object containing node features, edge indices, edge attributes, and a label.
    """
    # Extract node features
    node_features = []
    node_mapping = {node: i for i, node in enumerate(G.nodes())}  # Map nodes to indices
    for node in G.nodes(data=True):
        # Collect all node attributes
        features = [
            node[1].get(attr, 0) for attr in [
                'box_x_min', 'box_y_min', 'box_x_max', 'box_y_max', 
                'center_x', 'center_y'
            ]
        ]
        # Add keypoints
        keypoints = node[1].get('keypoints', [(0, 0)] * 17)
        features.extend([kp for point in keypoints for kp in point])
        node_features.append(features)
    
    x = torch.tensor(node_features, dtype=torch.float)

    # Extract edge indices using the node mapping
    edge_index = torch.tensor([[node_mapping[edge[0]], node_mapping[edge[1]]] for edge in G.edges], dtype=torch.long).t().contiguous()

    # Extract edge features
    edge_features = []
    for edge in G.edges(data=True):
        # Collect all edge attributes
        features = [
            edge[2].get(attr, 0) for attr in [
                'distance', 'relative_distance', 
                'motion_average_speed', 'motion_intensity', 
                'motion_sudden_movements', 'violence_aggressive_pose', 
                'violence_close_interaction', 'violence_rapid_motion', 
                'violence_weapon_present'
            ]
        ]
        # Add relative keypoints
        relative_keypoints = edge[2].get('relative_keypoints', [(0, 0)] * 17)
        features.extend([kp for point in relative_keypoints for kp in point])
        edge_features.append(features)
    
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    # Create a PyTorch Geometric Data object with a label
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label], dtype=torch.long))
    return data