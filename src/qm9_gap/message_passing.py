"""Continuous-filter message passing with GRU atom updates."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ContinuousFilterGRUBlock(nn.Module):
    """One independent edge-gated message-passing/GRU update block."""

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.sender_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: Tensor, edge_index: Tensor, edge_embedding: Tensor) -> Tensor:
        source, target = edge_index
        gates = self.edge_mlp(edge_embedding)
        sender_values = self.sender_projection(hidden[source])
        messages = gates * sender_values
        aggregate = torch.zeros_like(hidden)
        aggregate.index_add_(0, target, messages)
        updated = self.gru(aggregate, hidden)
        return self.normalization(updated)
