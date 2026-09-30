"""Permutation-invariant molecular readout."""

from torch import Tensor, nn
from torch_geometric.nn import Set2Set


class MolecularSet2Set(nn.Module):
    """Iterative content-based attention over unordered atom embeddings."""

    def __init__(self, hidden_dim: int, processing_steps: int = 3) -> None:
        super().__init__()
        self.output_dim = 2 * hidden_dim
        self.pool = Set2Set(in_channels=hidden_dim, processing_steps=processing_steps)

    def forward(self, atom_embeddings: Tensor, batch: Tensor) -> Tensor:
        return self.pool(atom_embeddings, batch)
