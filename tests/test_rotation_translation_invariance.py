import torch

from conftest import make_graph
from qm9_gap.model import QM9GapMPNN


def test_prediction_is_invariant_to_rigid_rotation_and_translation() -> None:
    torch.manual_seed(11)
    model = QM9GapMPNN(hidden_dim=16, num_message_passing_layers=2).eval()
    graph = make_graph(5)
    angle = torch.tensor(0.731)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotation = torch.tensor([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    transformed = graph.clone()
    transformed.pos = graph.pos @ rotation.T + torch.tensor([4.2, -1.7, 8.1])
    with torch.no_grad():
        original_prediction = model(graph)
        transformed_prediction = model(transformed)
    assert torch.allclose(original_prediction, transformed_prediction, atol=2e-6, rtol=2e-6)
