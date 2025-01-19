import json
from heapq import nlargest

import torch
from torch_geometric.utils import to_undirected

from wiki_connect.model.encoder import GCNEncoder
from wiki_connect.model.predictor import LinkPredictor


def complement_edge_index(edge_index, num_nodes, directed=True):
    """
    Add the complement of the edge index to include all possible edges in the graph.

    Parameters
    ----------
    edge_index: Tensor
        The edge index tensor.
    num_nodes: int
        The total number of nodes in the graph.
    directed: bool
        Whether the result is directed or not.

    Returns
    -------
    Tensor
        The edge index tensor with the complement of the edges.
    """
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    # Fill the adjacency matrix with original edges
    u, v = edge_index
    adj[u, v] = True
    adj[v, u] = True

    # The complement is all pairs that are not edges, excluding self-loops
    mask = ~adj & ~torch.eye(num_nodes, dtype=torch.bool)

    # For undirected graphs, we only need upper-triangular part to avoid duplicates
    u_comp, v_comp = mask.triu(1).nonzero(as_tuple=True)

    if directed:
        edge_index_comp = torch.stack([u_comp, v_comp], dim=0)
    else:
        # Add the symmetric edges to maintain undirected format
        edge_index_comp = torch.stack(
            [torch.cat([u_comp, v_comp]), torch.cat([v_comp, u_comp])], dim=0)

    return edge_index_comp


class LinkPredictionInference:
    def __init__(self, encoder_path, predictor_path, num_features, hidden_channels, out_channels, device):
        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')

        self._load_model(encoder_path, predictor_path)

    def _load_model(self, encoder_path, predictor_path):
        self.encoder = GCNEncoder(
            self.num_features, self.hidden_channels, self.out_channels).to(self.device)
        self.predictor = LinkPredictor(self.out_channels).to(self.device)

        self.encoder.load_state_dict(torch.load(
            encoder_path, map_location=self.device))
        self.predictor.load_state_dict(torch.load(
            predictor_path, map_location=self.device))

        self.encoder.eval()
        self.predictor.eval()

    @torch.no_grad()
    def predict_all(self, graph_data):
        """
        Predict the probabilities of all possible edges in the graph.

        Parameters
        ----------
        graph_data: PyG Data object containing the graph.

        Returns
        -------
        all_edges: List of all possible edges with their probabilities.
        """
        graph_data.edge_index = to_undirected(graph_data.edge_index)
        graph_data = graph_data.to(self.device)

        x = self.encoder(graph_data.x, graph_data.edge_index)

        # Generate all possible edges between two nodes (excluding self-loops)
        num_nodes = graph_data.num_nodes
        all_edges = torch.combinations(torch.arange(num_nodes), r=2).T
        all_edges = torch.cat(
            [all_edges, all_edges.flip(0)], dim=1).to(self.device)

        edge_probs = self.predictor(
            x[all_edges[0]], x[all_edges[1]]).squeeze().cpu()

        all_edges = all_edges.cpu().T.numpy().tolist()
        edge_probs = edge_probs.numpy().tolist()

        return list(zip(all_edges, edge_probs))

    @torch.no_grad()
    def predict_extreme_k_edges(self, graph_data, k):
        """
        Predict the top-k most likely and least likely edges in the graph.

        Parameters
        ----------
        graph_data: PyG Data object containing the graph.
        k: Number of top edges to predict.

        Returns
        -------
        top_k_edges: List of top-k edges with their probabilities.
        bottom_k_edges: List of bottom-k edges with their probabilities.
        """
        graph_data.edge_index = to_undirected(graph_data.edge_index)
        graph_data = graph_data.to(self.device)

        x = self.encoder(graph_data.x, graph_data.edge_index)

        # Generate all possible additional edges (excluding self-loops)
        num_nodes = graph_data.num_nodes
        all_edges = complement_edge_index(graph_data.edge_index, num_nodes)

        edge_probs = self.predictor(
            x[all_edges[0]], x[all_edges[1]]).squeeze().cpu()

        top_k_indices = nlargest(
            k, range(len(edge_probs)), key=lambda i: edge_probs[i])
        top_k_edges = [(all_edges[0][i].item(), all_edges[1]
                        [i].item(), edge_probs[i].item()) for i in top_k_indices]

        bottom_k_indices = nlargest(
            k, range(len(edge_probs)), key=lambda i: -edge_probs[i])
        bottom_k_edges = [(all_edges[0][i].item(), all_edges[1][i].item(
        ), edge_probs[i].item()) for i in bottom_k_indices]

        return top_k_edges, bottom_k_edges


class PredictionSummary:

    def __init__(self, graph_data, link_prediction_inference):
        self.graph_data = graph_data
        self.graph_data.edge_index = to_undirected(self.graph_data.edge_index)
        self.link_prediction_inference = link_prediction_inference

    def write_summary(self, file_path, k):
        all_predictions = self.link_prediction_inference.predict_all(
            self.graph_data)

        data_structure = {}

        for (source, target), prob in all_predictions:
            data_structure.setdefault(source, {}).setdefault(
                'predictions', {})[target] = prob

        edge_index = self.graph_data.edge_index
        for key in data_structure:
            neighbors = set(edge_index[1][edge_index[0] == key].tolist())
            predictions = data_structure[key].pop(
                'predictions')  # Remove predictions early
            data_structure[key]['neighbors'] = {
                n: predictions[n] for n in neighbors if n in predictions}

            non_neighbors = [(n, p)
                             for n, p in predictions.items() if n not in neighbors]
            data_structure[key]['top_non_neighbors'] = dict(
                nlargest(k, non_neighbors, key=lambda x: x[1]))

        with open(file_path, 'w') as f:
            json.dump(data_structure, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Link Prediction Inference")
    parser.add_argument("--graph_path", type=str,
                        required=True, help="Path to the graph data file")
    parser.add_argument("--encoder_path", type=str, required=True,
                        help="Path to the trained encoder model")
    parser.add_argument("--predictor_path", type=str, required=True,
                        help="Path to the trained predictor model")
    parser.add_argument("--num_features", type=int,
                        default=768, help="Number of node features")
    parser.add_argument("--hidden_channels", type=int,
                        default=128, help="Number of hidden channels")
    parser.add_argument("--out_channels", type=int,
                        default=64, help="Number of output channels")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of top edges to predict")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for inference (cuda or cpu)")

    args = parser.parse_args()

    graph_data = torch.load(args.graph_path)
    inference = LinkPredictionInference(
        encoder_path=args.encoder_path,
        predictor_path=args.predictor_path,
        num_features=args.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        device=args.device
    )

    summary = PredictionSummary(graph_data, inference)
    summary.write_summary(args.graph_path.replace(
        '.pt', '_predictions.json'), args.k)

    top_k_edges, bot_k_edges = inference.predict_extreme_k_edges(
        graph_data, args.k)
    print("Top-k Predicted Edges:")
    for edge in top_k_edges:
        print(f"Edge: ({graph_data.title[edge[0]]}, {
              graph_data.title[edge[1]]}), Probability: {edge[2]:.4f}")

    print("\nBottom-k Predicted Edges:")
    for edge in bot_k_edges:
        print(f"Edge: ({graph_data.title[edge[0]]}, {
              graph_data.title[edge[1]]}), Probability: {edge[2]:.4f}")
