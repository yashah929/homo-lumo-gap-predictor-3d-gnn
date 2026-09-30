import numpy as np
import torch

from conftest import make_graph
from qm9_gap.data import target_statistics
from qm9_gap.train import TargetStandardizer


def test_statistics_use_only_training_subset() -> None:
    dataset = [make_graph(3, index) for index in range(6)]
    dataset[-1].y = torch.tensor([1000.0])
    training_indices = np.array([0, 1, 2, 3])
    mean, standard_deviation = target_statistics(dataset, training_indices)
    expected = np.array([float(dataset[i].y) for i in training_indices])
    assert np.isclose(mean, expected.mean())
    assert np.isclose(standard_deviation, expected.std(ddof=0))
    assert mean < 10.0
    standardizer = TargetStandardizer(mean, standard_deviation)
    values = torch.tensor(expected, dtype=torch.float32)
    assert torch.allclose(standardizer.inverse(standardizer.transform(values)), values)
