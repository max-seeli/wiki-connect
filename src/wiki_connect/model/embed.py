import torch
from torch_geometric.utils import from_networkx
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from warnings import warn

from wiki_connect.data.util import load_graph


class NodeEmbedding:
    def __init__(self, graph_path: str, model_name: str = "bert-base-uncased"):
        self.graph_path = graph_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.graph = None

    def load_and_preprocess_graph(self):
        """Load the graph and remove nodes without 'info_text'."""
        self.graph = load_graph(self.graph_path)

        # Identify nodes missing 'info_text'
        missing_nodes = [
            node for node in self.graph.nodes if not self.graph.nodes[node].get("info_text", "")]
        if missing_nodes:
            warn(f"Nodes without info text: {missing_nodes}")

        self.graph.remove_nodes_from(missing_nodes)

    def generate_embeddings(self):
        """Generate embeddings for each node using the pre-trained model on the 'info_text'."""
        if self.graph is None:
            raise ValueError(
                "Graph not loaded. Please run 'load_and_preprocess_graph' first.")

        for node in tqdm(self.graph.nodes, desc="Processing nodes"):
            info_text = self.graph.nodes[node]["info_text"]

            # Tokenize and process with model
            inputs = self.tokenizer(
                info_text, return_tensors="pt", truncation=True, padding=True)
            outputs = self.model(**inputs)

            # Compute mean of last hidden state as embedding
            embeddings = outputs.last_hidden_state.mean(1)
            self.graph.nodes[node]["embeddings"] = embeddings.detach(
            ).numpy().squeeze()

    def save_data(self, output_path):
        """Save the PyTorch Geometric data object to a file."""
        data = from_networkx(self.graph, group_node_attrs=["embeddings"])
        torch.save(data, output_path)
        print(f"Data saved to {output_path}")


if __name__ == "__main__":
    from argparse import ArgumentParser
    import os
    parser = ArgumentParser(description="Node Embedding")
    parser.add_argument("--graph_path", type=str,
                        required=True, help="Path to the input graph")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save the output data object (default: change input extension to .pt)")
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = os.path.splitext(args.graph_path)[0] + ".pt"

    pipeline = NodeEmbedding(graph_path=args.graph_path)
    pipeline.load_and_preprocess_graph()
    pipeline.generate_embeddings()
    pipeline.save_data(output_path=args.output_path)
