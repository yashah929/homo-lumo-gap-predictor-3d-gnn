"""Non-quantum-mechanical atomic feature extraction and encoding."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


CATEGORICAL_FEATURE_NAMES = (
    "degree",
    "formal_charge",
    "hybridization",
    "aromaticity",
    "total_valence",
    "ring_membership",
    "chirality",
)
CATEGORICAL_CARDINALITIES = (8, 8, 9, 2, 10, 2, 4)

_HYBRIDIZATION_NAMES = {
    "UNSPECIFIED": 0,
    "S": 1,
    "SP": 2,
    "SP2": 3,
    "SP3": 4,
    "SP3D": 5,
    "SP3D2": 6,
    "OTHER": 7,
}
_CHIRAL_NAMES = {
    "CHI_UNSPECIFIED": 0,
    "CHI_TETRAHEDRAL_CW": 1,
    "CHI_TETRAHEDRAL_CCW": 2,
}


def _bounded_category(value: int, lower: int, upper: int, unknown: int) -> int:
    return value - lower if lower <= value <= upper else unknown


def atom_features_from_rdkit(molecule: object) -> tuple[Tensor, Tensor, Tensor]:
    """Extract atomic numbers, seven categorical features, and mass.

    The molecule must contain explicit hydrogens in the same atom order as the
    associated Cartesian coordinates. Atomic mass is represented in daltons.
    """
    atomic_numbers: list[int] = []
    categorical: list[list[int]] = []
    masses: list[list[float]] = []
    for atom in molecule.GetAtoms():  # type: ignore[attr-defined]
        atomic_numbers.append(int(atom.GetAtomicNum()))
        degree = _bounded_category(int(atom.GetDegree()), 0, 6, 7)
        charge = _bounded_category(int(atom.GetFormalCharge()), -3, 3, 7)
        hybridization = _HYBRIDIZATION_NAMES.get(str(atom.GetHybridization()), 8)
        aromatic = int(atom.GetIsAromatic())
        valence = _bounded_category(int(atom.GetTotalValence()), 0, 8, 9)
        in_ring = int(atom.IsInRing())
        chirality = _CHIRAL_NAMES.get(str(atom.GetChiralTag()), 3)
        categorical.append([degree, charge, hybridization, aromatic, valence, in_ring, chirality])
        masses.append([float(atom.GetMass())])
    return (
        torch.tensor(atomic_numbers, dtype=torch.long),
        torch.tensor(categorical, dtype=torch.long),
        torch.tensor(masses, dtype=torch.float32),
    )


class AtomicFeatureEncoder(nn.Module):
    """Encode atomic identity and chemical attributes into hidden states."""

    def __init__(
        self,
        hidden_dim: int,
        max_atomic_number: int = 100,
        categorical_cardinalities: Sequence[int] = CATEGORICAL_CARDINALITIES,
    ) -> None:
        super().__init__()
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        atomic_embedding_dim = min(32, hidden_dim)
        self.atomic_number_embedding = nn.Embedding(max_atomic_number + 1, atomic_embedding_dim)
        self.categorical_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, min(8, max(2, cardinality)))
            for cardinality in categorical_cardinalities
        )
        input_dim = atomic_embedding_dim + sum(e.embedding_dim for e in self.categorical_embeddings) + 1
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, atomic_numbers: Tensor, categorical: Tensor, atomic_mass: Tensor) -> Tensor:
        if categorical.shape[-1] != len(self.categorical_embeddings):
            raise ValueError("Unexpected number of categorical atom features")
        encoded = [self.atomic_number_embedding(atomic_numbers)]
        encoded.extend(embedding(categorical[:, i]) for i, embedding in enumerate(self.categorical_embeddings))
        # A fixed physical scale avoids fitting statistics on held-out molecules.
        encoded.append(atomic_mass / 100.0)
        return self.projection(torch.cat(encoded, dim=-1))
