import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, BatchNorm, JumpingKnowledge
import torch.nn.functional as F


node_input_dim = 6  # node feature dimension
edge_input_dim = 4  # edge feature dimension
hidden_dim = 128
output_dim = 4  # The number of outputs (mass, charge, sigma, epsilon)
num_epochs = 200
learning_rate = 0.001

class NodeEmbedding(nn.Module):
    def __init__(self, num_atomic, num_valence, num_hybridization, embedding_dim):
        super(NodeEmbedding, self).__init__()
        self.atomic_embedding = nn.Embedding(num_atomic, embedding_dim)
        self.valence_embedding = nn.Embedding(num_valence, embedding_dim)
        self.hybridization_embedding = nn.Embedding(num_hybridization, embedding_dim)

    def forward(self, atomic, valence, formal_charge, aromatic, hybridization, radical_electrons):
        atomic_embed = self.atomic_embedding(atomic)
        valence_embed = self.valence_embedding(valence)
        hybridization_embed = self.hybridization_embedding(hybridization)

        # Concatenate numerical and boolean features
        other_features = torch.stack([formal_charge, aromatic, radical_electrons], dim=1).float()

        # Concatenate all features together
        return torch.cat([atomic_embed, valence_embed, hybridization_embed, other_features], dim=1)

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

    def __init__(self, node_embedding_dim, edge_embedding_dim, hidden_dim, output_dim, num_layers = 5, num_atomic = 12, num_valence = 7, num_hybridization = 5, num_type = 4, num_stereo = 3 ):
        super(ImprovedGNNWithEmbeddings, self).__init__()
        self.node_embedding = NodeEmbedding(num_atomic, num_valence, num_hybridization, node_embedding_dim)
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


model = ImprovedGNNWithEmbeddings( node_embedding_dim = 32, edge_embedding_dim = 32, hidden_dim = 128, output_dim = 4)
checkpoint = torch.load('improved_gnn_model_v2.pth')
print("checkpoint:", checkpoint.keys())

model.load_state_dict(checkpoint)



