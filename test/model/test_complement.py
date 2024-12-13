import torch

from wiki_connect.model.inference import complement_edge_index


def test_compement_edge_index():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    num_nodes = 4

    required_edges = [(0, 2), (0, 3), (1, 3)]
    result = complement_edge_index(edge_index, num_nodes)

    assert result.size(1) == len(required_edges)
    assert all((u, v) in required_edges for u, v in result.numpy().T)
