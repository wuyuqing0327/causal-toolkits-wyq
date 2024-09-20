import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, BatchNorm, JumpingKnowledge
import torch.nn.functional as F
import molecular_project.helper as HP
import data_preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import random
import math
from copy import deepcopy
import pandas as pd
import numpy as np


# Hyperparameters
node_input_dim = 6  # node feature dimension
edge_input_dim = 4  # edge feature dimension
hidden_dim = 128
output_dim = 4  # The number of outputs (mass, charge, sigma, epsilon)
num_epochs = 200
learning_rate = 0.001
# Hyperparameters for early stopping
patience = 30  # Stop if no improvement for 30 consecutive epochs
min_delta = 0.00001  # Minimum improvement required to reset patience
best_val_loss = float('inf')
patience_counter = 0


train_data = HP.load_data_from_file("./molecular_project/data.json")
cleaned_data = data_preprocessing.extract_clean_data(train_data)

def count_unique_attributes(data):
    # Sets to store unique values of node attributes
    atomic_set = set()
    formal_charge_set = set()
    valence_set = set()
    hybridization_set = set()
    radical_electrons_set = set()

    # Sets to store unique values of edge attributes
    type_set = set()
    stereo_set = set()

    # Loop through data structure to collect unique values
    for smiles, features in data.items():
        # Collect node attributes
        for node in features['node_id_feature'].values():
            atomic_set.add(node['atomic'])
            formal_charge_set.add(node['formal_charge'])
            valence_set.add(node['valence'])
            hybridization_set.add(node['hybridization'])
            radical_electrons_set.add(node['radical_electrons'])

        # Collect edge attributes
        for edge in features['edge_features']:
            type_set.add(edge['type'])
            stereo_set.add(edge['stereo'])

    # Create a result dictionary for easy display
    result = {
        "unique_num_atomic": len(atomic_set),
        "unique_num_formal_charge": len(formal_charge_set),
        "unique_num_valence": len(valence_set),
        "unique_num_hybridization": len(hybridization_set),
        "unique_num_radical_electrons": len(radical_electrons_set),
        "unique_num_type": len(type_set),
        "unique_num_stereo": len(stereo_set)
    }

    return result

# Calculate the number of unique values for each property
unique_properties = count_unique_attributes(cleaned_data)
print(unique_properties)


# Create a Dataset class
class MolecularGraphDataset(Dataset):
    def __init__(self, cleaned_data, transform=None, pre_transform=None):
        super(MolecularGraphDataset, self).__init__(transform=transform, pre_transform=pre_transform)
        self.graphs = list(cleaned_data.values())
        self._indices = range(len(self.graphs))
        self.targets = []

    def __len__(self):
        return len(self._indices)

    def get(self, idx):
        graph_info = self.graphs[idx]
        return self.create_pyg_data(graph_info)

    def __getitem__(self, idx):
        data = self.get(self._indices[idx])
        data = data if self.transform is None else self.transform(data)
        return data

    def extract_targets(self):
        for graph_info in self.graphs:
            for node_id, node_info in graph_info["target_variable"].items():
                self.targets.append([
                    node_info["mass"],
                    node_info["charge"],
                    node_info["sigma"],
                    node_info["epsilon"]
                ])
        return self.targets

    def create_pyg_data(self, graph_info):
        # Extract nodes and edges from the graph information
        node_id_feature = graph_info["node_id_feature"]
        edge_features = graph_info["edge_features"]
        target_variable = graph_info["target_variable"]

        # Create the node feature matrix
        node_ids = sorted(node_id_feature.keys())
        node_features = []
        for node_id in node_ids:
            features = [
                node_id_feature[node_id]["atomic"],
                node_id_feature[node_id]["valence"],
                node_id_feature[node_id]["formal_charge"],
                node_id_feature[node_id]["aromatic"],
                node_id_feature[node_id]["hybridization"],
                node_id_feature[node_id]["radical_electrons"]
            ]
            node_features.append(features)
        x = torch.tensor(node_features, dtype=torch.float)

        # Create the edge list
        edge_index = []
        edge_attr = []
        for edge in edge_features:
            edge_index.append([edge["source"], edge["target"]])
            edge_attr.append([
                edge["type"],
                edge["stereo"],
                edge["aromatic"],
                edge["conjugated"]
            ])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Create the target variable tensor
        target_list = [target_variable[node_id] for node_id in node_ids]
        y = torch.tensor([[t["mass"], t["charge"], t["sigma"], t["epsilon"]] for t in target_list], dtype=torch.float)

        #mean = torch.mean(y, dim=0)
        #std = torch.std(y, dim=0)

        # Standardize the target variables
        #y = (y - mean) / std

        # Return the graph as a Data object
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    

def apply_random_mask(mol_graph, p, seed=None):
    if seed is not None:
        random.seed(seed)
    N = mol_graph.x.size(0)
    num_mask_nodes = max(1, math.floor(p * N))
    mask_nodes = random.sample(list(range(N)), num_mask_nodes)

    aug_mol_graph = deepcopy(mol_graph)
    for atom_idx in mask_nodes:
        aug_mol_graph.x[atom_idx, :] = torch.zeros(6)

    return aug_mol_graph

def apply_random_bond_deletion(mol_graph, p, seed=None):
    if seed is not None:
        random.seed(seed)
    M = mol_graph.edge_index.size(1) // 2
    num_mask_edges = max([0, math.floor(p * M)])
    mask_edges_single = random.sample(list(range(M)), num_mask_edges)
    mask_edges = [2*i for i in mask_edges_single] + [2*i+1 for i in mask_edges_single]

    aug_mol_graph = deepcopy(mol_graph)

    num_features_per_edge = mol_graph.edge_attr.size(1)
    aug_mol_graph.edge_index = torch.zeros((2, 2 * (M - num_mask_edges)))
    aug_mol_graph.edge_attr = torch.zeros((2 * (M - num_mask_edges), num_features_per_edge))
    count = 0
    for bond_idx in range(2 * M):
        if bond_idx not in mask_edges:
            aug_mol_graph.edge_index[:, count] = mol_graph.edge_index[:, bond_idx]
            aug_mol_graph.edge_attr[count, :] = mol_graph.edge_attr[bond_idx, :]
            count += 1

    return aug_mol_graph



class NodeEmbedding(nn.Module):
    def __init__(self, num_atomic, num_valence, num_formal_charge, num_hybridization, num_radical_electrons, embedding_dim):
        super(NodeEmbedding, self).__init__()
        self.atomic_embedding = nn.Embedding(num_atomic, embedding_dim)
        self.valence_embedding = nn.Embedding(num_valence, embedding_dim)
        self.formal_charge_embedding = nn.Embedding(num_formal_charge, embedding_dim)
        self.hybridization_embedding = nn.Embedding(num_hybridization, embedding_dim)
        self.radical_electrons_embedding = nn.Embedding(num_radical_electrons, embedding_dim)

    def forward(self, atomic, valence, formal_charge, aromatic, hybridization, radical_electrons):
        atomic_embed = self.atomic_embedding(atomic)
        valence_embed = self.valence_embedding(valence)
        formal_charge_embed = self.formal_charge_embedding(formal_charge)
        hybridization_embed = self.hybridization_embedding(hybridization)
        radical_electrons_embed = self.radical_electrons_embedding(radical_electrons)

        # Concatenate boolean features
        other_features = torch.stack([aromatic], dim=1).float()

        # Concatenate all features together
        return torch.cat([atomic_embed, valence_embed, formal_charge_embed, hybridization_embed, radical_electrons_embed, other_features], dim=1)

class EdgeEmbedding(nn.Module):
    def __init__(self, num_type, num_stereo, embedding_dim):
        super(EdgeEmbedding, self).__init__()
        self.type_embedding = nn.Embedding(num_type, embedding_dim)
        self.stereo_embedding = nn.Embedding(num_stereo, embedding_dim)

    def forward(self, type_, stereo, aromatic, conjugated):
        type_embed = self.type_embedding(type_)
        stereo_embed = self.stereo_embedding(stereo)

        # Concatenate boolean features directly
        other_features = torch.stack([aromatic, conjugated], dim=1).float()

        # Concatenate all features together
        return torch.cat([type_embed, stereo_embed, other_features], dim=1)
    
    


class ImprovedGNNWithEmbeddings(torch.nn.Module):

    def __init__(self, node_embedding_dim, edge_embedding_dim, hidden_dim, output_dim, num_layers = 5, num_atomic = 12, num_valence = 7, num_hybridization = 5, num_type = 4, num_stereo = 3 ,num_formal_charge = 3, num_radical_electrons = 3):
        super(ImprovedGNNWithEmbeddings, self).__init__()
        self.node_embedding = NodeEmbedding(num_atomic, num_valence, num_formal_charge, num_hybridization, num_radical_electrons, node_embedding_dim)
        self.edge_embedding = EdgeEmbedding(num_type, num_stereo, edge_embedding_dim)
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        # Define the first GINEConv layer, with the correct edge_dim specified
        self.convs.append(GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(node_input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_input_dim
        ))
        self.norms.append(BatchNorm(hidden_dim))

        # Additional GINEConv layers, each with the correct edge_dim
        for _ in range(num_layers - 1):
            self.convs.append(GINEConv(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim)
                ),
                edge_dim=edge_input_dim
            ))
            self.norms.append(BatchNorm(hidden_dim))

        # Jumping Knowledge mechanism
        self.jump = JumpingKnowledge(mode="cat")

        # Final fully connected layers
        self.fc1 = torch.nn.Linear(hidden_dim * num_layers, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        features = []

        # Pass through GINEConv layers and apply batch normalization
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = F.relu(norm(x))
            features.append(x)

        # Apply Jumping Knowledge (JK) to concatenate all layers
        x = self.jump(features)

        # Directly pass through the linear layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x




# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0

    with torch.no_grad():  # No gradient calculation during validation
        for data in val_loader:
            data = data.to(device)  # Move data to the correct device
            out = model(data)  # Forward pass

            #out_unscaled = unstandardize(out, mean, std)
            #target_unscaled = unstandardize(data.y, mean, std)

            loss = criterion(out, data.y)  # Compute validation loss
            total_loss += loss.item() * data.num_graphs

    return total_loss / len(val_loader.dataset)

# Training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()  # Set the model to training mode
    total_loss = 0

    for data in train_loader:
        data = data.to(device)  # Move data to GPU or CPU
        optimizer.zero_grad()  # Clear gradients from the last step
        out = model(data)  # Forward pass
        loss = criterion(out, data.y)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model weights
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)




dataset = MolecularGraphDataset(cleaned_data)
# Split indices for train and validation sets
train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
# Create Subset datasets using the split indices
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
# Create DataLoaders for the subsets
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)


# model = ImprovedGNNWithEdgeFeatures(node_input_dim, edge_input_dim, hidden_dim, output_dim)
model = ImprovedGNNWithEmbeddings( node_embedding_dim = 32, edge_embedding_dim = 32, hidden_dim = 128, output_dim = 4)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()
# Move model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# Split indices for train and validation sets
train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

# Create Subset datasets using the split indices
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# Data Augmentation
for i in range(len(train_dataset)):
    apply_random_mask(train_dataset[i], 0.1)
    apply_random_bond_deletion(train_dataset[i], 0.1)

# Create DataLoaders for the subsets
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)



# Training loop with early stopping
for epoch in range(1, num_epochs + 1):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss = validate(model, val_loader, criterion, device)

    # Check for improvement
    if best_val_loss - val_loss > min_delta:
        best_val_loss = val_loss  # New best validation loss
        patience_counter = 0  # Reset patience counter
    else:
        patience_counter += 1  # Increment patience counter

    print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Check if patience counter has been exceeded
    if patience_counter >= patience:
        print("Early stopping triggered. Stopping training...")

        break

torch.save(model.state_dict(), "improved_gnn_model_v3.pth")
print("model save sucessfully")


def predict_with_indices_and_true_values(model, graph_data, device):
    graph_data = graph_data.to(device)
    with torch.no_grad():
        predictions = model(graph_data)

    node_indices = torch.arange(graph_data.num_nodes)
    true_values = graph_data.y

    # Combine predictions with indices and true values
    indexed_predictions = list(zip(node_indices.cpu().numpy(), predictions.cpu().numpy(), true_values.cpu().numpy()))
    return indexed_predictions


mass_val = []
mass_true = []
charge_val = []
charge_true = []
sigma_val = []
sigma_true = []
epsilon_val = []
epsilon_true = []
# Example usage
for graph in val_loader:
    # graph = val_loader  # Get a single graph
    graph_predictions_and_true_values = predict_with_indices_and_true_values(model, graph, device)

    # Display predictions and true values for each node
    print("Predictions and True Values for each node (index, prediction, true value):")
    for index, prediction, true_value in graph_predictions_and_true_values:
        # print(f"Node {index}: Prediction: {prediction}, True Value: {true_value}")
        mass_val.append(prediction[0])
        mass_true.append(true_value[0])
        charge_val.append(prediction[1])
        charge_true.append(true_value[1])
        sigma_val.append(prediction[2])
        sigma_true.append(true_value[2])
        epsilon_val.append(prediction[3])
        epsilon_true.append(true_value[3])

print(f"\nAnalysis for {'mass'}")
sq_diff = (np.array(mass_val) - np.array(mass_true)) ** 2
print(f"Root Mean Squared Difference: {np.sqrt(np.mean(sq_diff))}")
print(f"Max difference: {np.sqrt(np.max(sq_diff))}")
print("\n")

print(f"\nAnalysis for {'charge'}")
sq_diff = (np.array(charge_val) - np.array(charge_true)) ** 2
print(f"Root Mean Squared Difference: {np.sqrt(np.mean(sq_diff))}")
print(f"Max difference: {np.sqrt(np.max(sq_diff))}")
print("\n")

print(f"\nAnalysis for {'sigma'}")
sq_diff = (np.array(sigma_val) - np.array(sigma_true)) ** 2
print(f"Root Mean Squared Difference: {np.sqrt(np.mean(sq_diff))}")
print(f"Max difference: {np.sqrt(np.max(sq_diff))}")
print("\n")

print(f"\nAnalysis for {'epsilon'}")
sq_diff = (np.array(epsilon_val) - np.array(epsilon_true)) ** 2
print(f"Root Mean Squared Difference: {np.sqrt(np.mean(sq_diff))}")
print(f"Max difference: {np.sqrt(np.max(sq_diff))}")
print("\n")




