import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, BatchNorm, JumpingKnowledge
import torch.nn.functional as F
import molecular_project.helper as HP
import data_preprocessing
import json
import molecular_project.final_evaluation as FE


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



model = ImprovedGNNWithEmbeddings( node_embedding_dim = 32, edge_embedding_dim = 32, hidden_dim = 128, output_dim = 4)
checkpoint = torch.load('improved_gnn_model_v3.pth')
model.load_state_dict(checkpoint)



# Create a Dataset class
class MolecularGraphDatasetvali(Dataset):
    def __init__(self, cleaned_data, transform=None, pre_transform=None):
        super(MolecularGraphDatasetvali, self).__init__(transform=transform, pre_transform=pre_transform)
        self.graphs = list(cleaned_data.values())
        self._indices = range(len(self.graphs))

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

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    


ref_dict = HP.load_data_from_file("./molecular_project/data.json")
result_dict = HP.load_data_from_file("./molecular_project/validation_example.json")


cleaned_data = data_preprocessing.extract_clean_data_forvali(result_dict)
val_dataset = MolecularGraphDatasetvali(cleaned_data)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
model.eval()

result=[]
for data in val_loader:
    rng = model(data)
    print("RNG shape:", len(rng), len(rng[0]))
    FE.add_data_from_prediction(result_dict, rng)
    
    FE.compare_property("epsilon", result_dict, ref_dict)
    FE.compare_property("mass", result_dict, ref_dict)
    FE.compare_property("sigma", result_dict, ref_dict)
    FE.compare_property("charge", result_dict, ref_dict)
    print("compare Done and check charge:", FE.compare_property("charge", result_dict, ref_dict))


    # print("Permutation check The real SMILES string we will test with is not in your training data")
    # result_perm_dict = HP.load_data_from_file("./molecular_project/permutation_example_masked.json")
    # with open("./molecular_project/permutation_example.json", "r") as json_handle:
    #     permutation_dict = json.load(json_handle)
    # # This step is again replaced with your model data
    # FE.add_data_from_prediction(result_perm_dict, rng)

        
    # for name in ref_dict:
    #     # ref_graph = ref_dict[name]
    #     ref_graph = ref_dict["O=C(c1ccc2c(c1)OCO2)c1ccc2n1CCC2C(=O)O"]
        
    #     for index, node in enumerate(graph.nodes(data=True)):
    #         node[1]["param"] = get_random_param(rng[index])
                

    #     print("permutation_dict:", permutation_dict)
    #     print("result_perm_dict:", result_perm_dict)
    #     print("ref_graph:", ref_graph)
    
    
    #     print("epsilin:", FE.compare_permutation("epsilon", result_perm_dict, ref_graph, permutation_dict))
    #     print("mass:", FE.compare_permutation("mass", result_perm_dict, ref_graph, permutation_dict))
    #     print("sigma:", FE.compare_permutation("sigma", result_perm_dict, ref_graph, permutation_dict))
    #     print("charge:", FE.compare_permutation("charge", result_perm_dict, ref_graph, permutation_dict))


