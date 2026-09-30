from __future__ import annotations

import torch
from torch_geometric.data import Data

from qm9_gap.graph import BOND_FEATURE_DIM, complete_directed_edge_index, pairwise_edge_distances


def make_graph(num_atoms: int, molecule_index: int = 0) -> Data:
    generator = torch.Generator().manual_seed(100 + molecule_index)
    positions = torch.randn((num_atoms, 3), generator=generator)
    edge_index = complete_directed_edge_index(num_atoms)
    return Data(
        z=torch.tensor(([6] + [1] * (num_atoms - 1)), dtype=torch.long),
        x_cat=torch.zeros((num_atoms, 7), dtype=torch.long),
        atomic_mass=torch.tensor([[12.011]] + [[1.008]] * (num_atoms - 1)),
        pos=positions,
        edge_index=edge_index,
        edge_distance=pairwise_edge_distances(positions, edge_index),
        edge_chem=torch.zeros((edge_index.shape[1], BOND_FEATURE_DIM)),
        y=torch.tensor([float(molecule_index) / 10 + 1.0]),
        molecule_index=torch.tensor([molecule_index]),
    )
