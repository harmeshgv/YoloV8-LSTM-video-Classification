# gnn_lstm_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNLSTM(nn.Module):
    """
    A Graph Neural Network (GNN) combined with an LSTM for graph classification tasks.

    Parameters:
    - input_dim: Dimension of input node features.
    - hidden_dim: Dimension of hidden layers.
    - num_classes: Number of output classes for classification.
    """
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GNNLSTM, self).__init__()
        self.gnn = GCNConv(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer with 50% dropout rate
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        # GNN layer
        x = self.gnn(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)  # Apply dropout

        # Global pooling to aggregate node features into graph features
        x = global_mean_pool(x, batch)

        # LSTM layer
        x, _ = self.lstm(x.unsqueeze(0))  # Add batch dimension
        x = x.squeeze(0)  # Remove batch dimension

        # Fully connected layer
        x = self.fc(x)
        return x

def train_model(loader, input_dim, hidden_dim, num_classes, num_epochs=10, learning_rate=0.01):
    """
    Train the GNNLSTM model.

    Parameters:
    - loader: DataLoader for the training data.
    - input_dim: Dimension of input node features.
    - hidden_dim: Dimension of hidden layers.
    - num_classes: Number of output classes for classification.
    - num_epochs: Number of training epochs.
    - learning_rate: Learning rate for the optimizer.

    Returns:
    - model: Trained GNNLSTM model.
    """
    # Initialize the model, loss function, and optimizer
    model = GNNLSTM(input_dim, hidden_dim, num_classes)
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0
        for data in loader:
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.batch)
            loss = criterion(output, data.y)  # Use CrossEntropyLoss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

    return model

def evaluate_model(model, loader):
    """
    Evaluate the GNNLSTM model.

    Parameters:
    - model: Trained GNNLSTM model.
    - loader: DataLoader for the evaluation data.

    Returns:
    - accuracy: Accuracy of the model on the evaluation data.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            output = model(data.x, data.edge_index, data.batch)
            _, predicted = torch.max(output, 1)
            total += data.y.size(0)
            correct += (predicted == data.y).sum().item()

    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')
    return accuracy