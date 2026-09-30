#!/usr/bin/env python3
"""Run a short synthetic end-to-end training smoke test without downloading QM9."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from qm9_gap.data import target_statistics
from qm9_gap.graph import BOND_FEATURE_DIM, complete_directed_edge_index, pairwise_edge_distances
from qm9_gap.model import QM9GapMPNN
from qm9_gap.train import TargetStandardizer, make_loader, train_with_early_stopping
from qm9_gap.utils import seed_everything


def synthetic_graph(index: int) -> Data:
    generator = torch.Generator().manual_seed(index)
    num_atoms = 3 + index % 4
    pos = torch.randn((num_atoms, 3), generator=generator)
    edge_index = complete_directed_edge_index(num_atoms)
    return Data(
        z=torch.tensor([6] + [1] * (num_atoms - 1)),
        x_cat=torch.zeros((num_atoms, 7), dtype=torch.long),
        atomic_mass=torch.tensor([[12.011]] + [[1.008]] * (num_atoms - 1)),
        pos=pos,
        edge_index=edge_index,
        edge_distance=pairwise_edge_distances(pos, edge_index),
        edge_chem=torch.zeros((edge_index.shape[1], BOND_FEATURE_DIM)),
        y=torch.tensor([1.0 + 0.05 * index]),
        molecule_index=torch.tensor([index]),
    )


def main() -> None:
    seed_everything(42)
    dataset = [synthetic_graph(index) for index in range(16)]
    training_indices = np.arange(12)
    validation_indices = np.arange(12, 16)
    mean, standard_deviation = target_statistics(dataset, training_indices)
    standardizer = TargetStandardizer(mean, standard_deviation)
    train_loader = make_loader(dataset, training_indices, 4, True, 42, 0, False)
    validation_loader = make_loader(dataset, validation_indices, 4, False, 42, 0, False)
    model = QM9GapMPNN(hidden_dim=16, num_message_passing_layers=2)
    best, curves = train_with_early_stopping(
        model,
        train_loader,
        validation_loader,
        standardizer,
        torch.device("cpu"),
        learning_rate=1e-3,
        max_epochs=3,
        patience=3,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "smoke_curve.csv"
        curves.to_csv(path, index=False)
        assert path.exists()
    print(
        f"Synthetic smoke test completed: {len(curves)} epochs; "
        f"best validation MAE={best['metrics']['mae_ev']:.6f} eV."
    )


if __name__ == "__main__":
    main()
