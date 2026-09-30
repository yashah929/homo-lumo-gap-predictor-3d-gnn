"""Complete three-dimensional MPNN for standardized gap regression."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .features import AtomicFeatureEncoder
from .graph import BOND_FEATURE_DIM, pairwise_edge_distances
from .message_passing import ContinuousFilterGRUBlock
from .pooling import MolecularSet2Set
from .rbf import GaussianRBF


class QM9GapMPNN(nn.Module):
    """Predict one standardized HOMO-LUMO gap for each molecular graph."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_message_passing_layers: int = 4,
        num_rbf: int = 50,
        rbf_min: float = 0.0,
        rbf_max: float = 10.0,
        dropout: float = 0.1,
        set2set_steps: int = 3,
        max_atomic_number: int = 100,
    ) -> None:
        super().__init__()
        if num_message_passing_layers < 1:
            raise ValueError("At least one message-passing layer is required")
        self.atom_encoder = AtomicFeatureEncoder(hidden_dim, max_atomic_number)
        self.rbf = GaussianRBF(num_rbf, rbf_min, rbf_max)
        edge_dim = num_rbf + BOND_FEATURE_DIM
        self.message_passing = nn.ModuleList(
            ContinuousFilterGRUBlock(hidden_dim, edge_dim, dropout)
            for _ in range(num_message_passing_layers)
        )
        self.readout = MolecularSet2Set(hidden_dim, set2set_steps)
        self.prediction = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QM9GapMPNN":
        """Build a model from the ``model`` section of a YAML configuration."""
        return cls(**config)

    def forward(self, data: Any) -> Tensor:
        atomic_mass = data.atomic_mass
        if atomic_mass.ndim == 1:
            atomic_mass = atomic_mass.unsqueeze(-1)
        hidden = self.atom_encoder(data.z, data.x_cat, atomic_mass)
        distances = pairwise_edge_distances(data.pos, data.edge_index)
        edge_embedding = torch.cat((self.rbf(distances), data.edge_chem), dim=-1)
        for layer in self.message_passing:
            hidden = layer(hidden, data.edge_index, edge_embedding)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(hidden.shape[0], dtype=torch.long, device=hidden.device)
        molecular = self.readout(hidden, batch)
        return self.prediction(molecular).squeeze(-1)
