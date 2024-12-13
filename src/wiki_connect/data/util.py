import json
import pickle

import matplotlib.pyplot as plt
import networkx as nx


def draw_graph(graph: nx.Graph):
    """
    Visualize the Wikipedia page-page graph.

    Parameters
    ----------
    graph : nx.Graph
        The graph to draw.
    """
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(graph, k=0.5, iterations=50)
    nx.draw(graph, pos, with_labels=True, node_size=50,
            font_size=8, arrowstyle='->', arrowsize=10)
    plt.title("Wikipedia Referenced Pages Graph")
    plt.show(block=False)


def load_graph(file_path: str) -> nx.Graph:
    """
    Load a NetworkX graph from a file.

    Parameters
    ----------
    file_path : str
        The path to the file containing the graph.

    Returns
    -------
    nx.Graph
        The loaded graph.
    """
    if file_path.endswith(".pkl"):
        with open(file_path, "rb") as f:
            graph = pickle.load(f)
        return graph
    elif file_path.endswith(".json"):
        with open(file_path, "r") as f:
            loaded_data = json.load(f)
        return nx.node_link_graph(loaded_data, edges="edges")
    else:
        raise ValueError("Invalid file format. Must be .pkl or .json")


def save_graph(graph: nx.Graph, file_path: str):
    """
    Save a NetworkX graph to a file.

    Parameters
    ----------
    graph : nx.Graph
        The graph to save.
    file_path : str
        The path to save the graph to.
    """
    if file_path.endswith(".pkl"):
        with open(file_path, "wb") as f:
            pickle.dump(graph, f)
    elif file_path.endswith(".json"):
        with open(file_path, "w") as f:
            json.dump(nx.node_link_data(graph, edges="edges"), f, indent=4)
    else:
        raise ValueError("Invalid file format. Must be .pkl or .json")
