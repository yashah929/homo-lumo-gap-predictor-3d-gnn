import math

import torch

from qm9_gap.rbf import GaussianRBF


def test_rbf_dimensions_finiteness_and_center_response() -> None:
    rbf = GaussianRBF(num_centers=50, minimum=0.0, maximum=10.0)
    distances = torch.tensor([0.0, 1.0, 10.0])
    values = rbf(distances)
    assert values.shape == (3, 50)
    assert torch.isfinite(values).all()
    assert torch.isclose(values[0, 0], torch.tensor(1.0))
    assert torch.isclose(values[-1, -1], torch.tensor(1.0))
    assert math.isclose(float(rbf(rbf.centers[:2])[0, 1]), math.exp(-1.0), rel_tol=1e-5)
