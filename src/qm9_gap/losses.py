"""Optimization objectives."""

from torch import Tensor
from torch.nn import functional as F


def standardized_mse(prediction: Tensor, target: Tensor) -> Tensor:
    """Mean squared error on standardized scalar targets."""
    return F.mse_loss(prediction.reshape(-1), target.reshape(-1))
