"""Complete directed molecular graph construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol


BOND_FEATURE_DIM = 7


def complete_directed_edge_index(num_atoms: int, device: torch.device | None = None) -> Tensor:
    """Construct all ordered atom pairs ``(j, i)`` with ``j != i``.

    Returns an ``edge_index`` tensor of shape ``[2, N(N-1)]`` following the
    PyTorch Geometric source-to-target convention.
    """
    if num_atoms < 1:
        raise ValueError("A molecular graph must contain at least one atom")
    atoms = torch.arange(num_atoms, device=device)
    source = atoms.repeat_interleave(num_atoms)
    target = atoms.repeat(num_atoms)
    mask = source != target
    return torch.stack((source[mask], target[mask]), dim=0)


def pairwise_edge_distances(positions: Tensor, edge_index: Tensor) -> Tensor:
    """Compute Euclidean source-target distances for ``edge_index``."""
    source, target = edge_index
    return torch.linalg.vector_norm(positions[source] - positions[target], dim=-1)


def chemical_edge_features(molecule: "Mol", edge_index: Tensor) -> Tensor:
    """Encode bond attributes for every complete-graph edge.

    Columns are bonded, single, double, triple, aromatic, conjugated, and
    ring-membership. All seven values are zero for nonbonded pairs.
    """
    from rdkit import Chem

    features = torch.zeros((edge_index.shape[1], BOND_FEATURE_DIM), dtype=torch.float32)
    for edge_id, (source, target) in enumerate(edge_index.t().tolist()):
        bond = molecule.GetBondBetweenAtoms(int(source), int(target))
        if bond is None:
            continue
        features[edge_id, 0] = 1.0
        bond_type = bond.GetBondType()
        if bond_type == Chem.BondType.SINGLE:
            features[edge_id, 1] = 1.0
        elif bond_type == Chem.BondType.DOUBLE:
            features[edge_id, 2] = 1.0
        elif bond_type == Chem.BondType.TRIPLE:
            features[edge_id, 3] = 1.0
        elif bond_type == Chem.BondType.AROMATIC:
            features[edge_id, 4] = 1.0
        else:
            raise ValueError(f"Unsupported QM9 bond type: {bond_type}")
        features[edge_id, 5] = float(bond.GetIsConjugated())
        features[edge_id, 6] = float(bond.IsInRing())
    return features
