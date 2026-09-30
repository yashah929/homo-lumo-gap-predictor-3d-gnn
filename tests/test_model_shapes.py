import torch
from torch_geometric.data import Batch

from conftest import make_graph
from qm9_gap.model import QM9GapMPNN


def test_model_returns_one_scalar_per_molecule() -> None:
    model = QM9GapMPNN(hidden_dim=16, num_message_passing_layers=2)
    batch = Batch.from_data_list([make_graph(4, 0), make_graph(6, 1)])
    output = model(batch)
    assert output.shape == (2,)
    assert torch.isfinite(output).all()


def test_batched_and_individual_predictions_are_consistent() -> None:
    torch.manual_seed(7)
    model = QM9GapMPNN(hidden_dim=16, num_message_passing_layers=2).eval()
    graphs = [make_graph(4, 0), make_graph(6, 1)]
    with torch.no_grad():
        individual = torch.cat([model(graph) for graph in graphs])
        batched = model(Batch.from_data_list(graphs))
    assert torch.allclose(individual, batched, atol=2e-6, rtol=2e-6)
