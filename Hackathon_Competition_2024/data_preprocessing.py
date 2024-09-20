import networkx as nx
from molecular_project.helper import load_data_from_file


def extract_clean_data(graphs_dict):
    """
    Extract cleaned molecular graph data.

    Parameters:
    - graphs_dict: dict of NetworkX graphs loaded from the original JSON file.

    Returns:
    - cleaned_dict: dict with SMILES strings as keys and specific node/edge data as values.
    """
    cleaned_dict = {}

    for smiles, graph in graphs_dict.items():
        # Extract node features
        node_id_features = {}
        for node_id, node_data in graph.nodes(data=True):
            node_features = {
                "e": node_data["atomic"],
                "valence": node_data["valence"],
                "formal_charge": node_data["formal_charge"],
                "aromatic": node_data["aromatic"],
                "hybridization": node_data["hybridization"],
                "radical_electrons": node_data["radical_electrons"]
            }
            node_id_features[node_id] = node_features

        # Extract edge features
        edges = []
        for source, target, edge_data in graph.edges(data=True):
            edge_features = {
                "type": edge_data["type"],
                "stereo": edge_data["stereo"],
                "aromatic": edge_data["aromatic"],
                "conjugated": edge_data["conjugated"]
            }
            edges.append({"source": source, "target": target, **edge_features})

        # Extract target variables
        target_vars = {}
        for node_id, node_data in graph.nodes(data=True):
            target_vars[node_id] = {
                "mass": node_data["param"]["mass"],
                "charge": node_data["param"]["charge"],
                "sigma": node_data["param"]["sigma"],
                "epsilon": node_data["param"]["epsilon"]
            }

        cleaned_dict[smiles] = {
            "node_id_feature": node_id_features,
            "target_variable": target_vars,
            "edge_features": edges
        }

    return cleaned_dict


# train_data = load_data_from_file("./molecular_project/data.json")
# cleaned_data = extract_clean_data(train_data)
#
# print(cleaned_data['O=C(c1cccc(Cl)c1)c1ccc2n1CCC2C(=O)O'])
# print(cleaned_data['O=C(NCCc1c2n(c3ccccc13)CCc1ccccc1-2)C1CCC1'])




def extract_clean_data_forvali(graphs_dict):
    """
    Extract cleaned molecular graph data.

    Parameters:
    - graphs_dict: dict of NetworkX graphs loaded from the original JSON file.

    Returns:
    - cleaned_dict: dict with SMILES strings as keys and specific node/edge data as values.
    """
    cleaned_dict = {}

    for smiles, graph in graphs_dict.items():
        # Extract node features
        node_id_features = {}
        for node_id, node_data in graph.nodes(data=True):
            node_features = {
                "atomic": node_data["atomic"],
                "valence": node_data["valence"],
                "formal_charge": node_data["formal_charge"],
                "aromatic": node_data["aromatic"],
                "hybridization": node_data["hybridization"],
                "radical_electrons": node_data["radical_electrons"]
            }
            node_id_features[node_id] = node_features

        # Extract edge features
        edges = []
        for source, target, edge_data in graph.edges(data=True):
            edge_features = {
                "type": edge_data["type"],
                "stereo": edge_data["stereo"],
                "aromatic": edge_data["aromatic"],
                "conjugated": edge_data["conjugated"]
            }
            edges.append({"source": source, "target": target, **edge_features})


        cleaned_dict[smiles] = {
            "node_id_feature": node_id_features,
            "edge_features": edges
        }

    return cleaned_dict
