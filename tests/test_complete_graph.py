import torch

from qm9_gap.graph import complete_directed_edge_index, pairwise_edge_distances


def test_complete_directed_graph_has_expected_size_and_no_self_edges() -> None:
    for num_atoms in (1, 2, 5, 9):
        edge_index = complete_directed_edge_index(num_atoms)
        assert edge_index.shape == (2, num_atoms * (num_atoms - 1))
        assert torch.all(edge_index[0] != edge_index[1])


def test_pairwise_distances_match_direct_euclidean_calculation() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0], [-1.0, 0.0, 0.0]])
    edge_index = complete_directed_edge_index(3)
    distances = pairwise_edge_distances(positions, edge_index)
    expected = torch.stack(
        [torch.linalg.vector_norm(positions[source] - positions[target]) for source, target in edge_index.t()]
    )
    assert torch.allclose(distances, expected)
