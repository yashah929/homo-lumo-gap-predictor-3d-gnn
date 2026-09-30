"""Gaussian radial basis encoding of pairwise distances."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GaussianRBF(nn.Module):
    """Expand distances in evenly spaced Gaussian radial basis functions.

    For adjacent center spacing ``delta``, ``gamma = 1 / delta**2``. Adjacent
    basis functions therefore have value ``exp(-1)`` at one another's center,
    providing deterministic, smooth overlap without a learned width.
    """

    def __init__(self, num_centers: int = 50, minimum: float = 0.0, maximum: float = 10.0):
        super().__init__()
        if num_centers < 2:
            raise ValueError("num_centers must be at least 2")
        if maximum <= minimum:
            raise ValueError("maximum must be greater than minimum")
        centers = torch.linspace(minimum, maximum, num_centers)
        spacing = float(centers[1] - centers[0])
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / spacing**2
        self.minimum = float(minimum)
        self.maximum = float(maximum)

    def forward(self, distances: Tensor) -> Tensor:
        """Return RBF values with shape ``[..., num_centers]``."""
        return torch.exp(-self.gamma * (distances.unsqueeze(-1) - self.centers) ** 2)
