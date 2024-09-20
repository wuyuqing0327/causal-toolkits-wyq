import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw
import molecular_project.helper as HP
import data_preprocessing


train_data = HP.load_data_from_file("./molecular_project/data.json")
cleaned_data = data_preprocessing.extract_clean_data(train_data)

smile='C/C=C/C=C/C(=O)O[C@H]1CCCC[C@H](O)[C@H](O)[C@@H](CCC)OC1=O'
node = pd.DataFrame(cleaned_data[smile]['node_id_feature']).transpose().reset_index()
node.rename(columns={'index':'node_id'}, inplace=True)
edge = pd.DataFrame(cleaned_data[smile]['edge_features'])


G = nx.Graph()
# Add nodes and edges to the graph
for index, row in node.iterrows():
    G.add_node(row['node_id'], atomic=row['atomic'], valence=row['valence'],
               formal_charge=row['formal_charge'], aromatic=row['aromatic'],
               hybridization=row['hybridization'], radical_electrons=row['radical_electrons'])
for index, row in edge.iterrows():
    G.add_edge(row['source'], row['target'], type=row['type'],
               stereo=row['stereo'], aromatic=row['aromatic'], conjugated=row['conjugated'])
# Draw the graph
pos = nx.spring_layout(G)  # positions for all nodes
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700)
plt.title("C/C=C/C=C/C(=O)O[C@H]1CCCC[C@H](O)[C@H](O)[C@@H](CCC)OC1=O Graph")
plt.show()



nodes = pd.DataFrame()
edges = pd.DataFrame()
for key in cleaned_data.keys():
    # Example: Convert a SMILES string to a molecular structure and visualize it
    smiles = key  # Example SMILES
    mol = Chem.MolFromSmiles(smiles)
    Draw.MolToImage(mol)

    # Suppose `data` is your dataset in a pandas DataFrame
    node = pd.DataFrame(cleaned_data[smiles]['node_id_feature']).transpose().reset_index()
    node.rename(columns={'index':'node_id'}, inplace=True)
    nodes = pd.concat([nodes, node], axis=0)

    edge = pd.DataFrame(cleaned_data[smiles]['edge_features'])
    edges = pd.concat([edges, edge], axis=0)






# Plotting distributions of node attributes
plt.figure(figsize=(12, 8))
sns.countplot(x='atomic', data=nodes)
plt.title('Distribution of Atomic Types')
plt.show()


plt.figure(figsize=(12, 8))
sns.countplot(x='valence', data=nodes)
plt.title('Distribution of Valence in Molecule')
plt.show()


# Hybridization state distribution
sns.countplot(x='hybridization', data=nodes)
plt.title('Distribution of Hybridization States')
plt.show()

# Hybridization state distribution
sns.countplot(x='aromatic', data=nodes)
plt.title('Distribution of aromatic States')
plt.show()



sns.countplot(x='type', data=edges)
plt.title('Distribution of type States')
plt.show()


sns.countplot(x='stereo', data=edges)
plt.title('Distribution of stereo States')
plt.show()

sns.countplot(x='aromatic', data=edges)
sns.countplot(x='conjugated', data=edges)
plt.title('Distribution of aromatic & conjugated States')
plt.show()

