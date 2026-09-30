"""Shared training loops for cross-validation and final retraining."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from .losses import standardized_mse
from .metrics import regression_metrics
from .utils import worker_seed

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetStandardizer:
    mean: float
    standard_deviation: float

    def transform(self, target: torch.Tensor) -> torch.Tensor:
        return (target - self.mean) / self.standard_deviation

    def inverse(self, standardized: torch.Tensor) -> torch.Tensor:
        return standardized * self.standard_deviation + self.mean

    def as_dict(self) -> dict[str, float]:
        return {"mean_ev": self.mean, "standard_deviation_ev": self.standard_deviation}


def make_loader(
    dataset: Any,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        worker_init_fn=worker_seed,
        persistent_workers=num_workers > 0,
    )


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    standardizer: TargetStandardizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        target_standardized = standardizer.transform(batch.y.reshape(-1))
        loss = standardized_mse(prediction, target_standardized)
        loss.backward()
        optimizer.step()
        count = int(batch.num_graphs)
        total_loss += float(loss.detach()) * count
        total_examples += count
    return total_loss / total_examples


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    standardizer: TargetStandardizer,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    references: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    molecule_indices: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        prediction_ev = standardizer.inverse(model(batch))
        references.append(batch.y.reshape(-1).detach().cpu().numpy())
        predictions.append(prediction_ev.detach().cpu().numpy())
        molecule_indices.append(batch.molecule_index.reshape(-1).detach().cpu().numpy())
    return (
        np.concatenate(references),
        np.concatenate(predictions),
        np.concatenate(molecule_indices).astype(np.int64),
    )


def train_with_early_stopping(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    standardizer: TargetStandardizer,
    device: torch.device,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    weight_decay: float = 0.0,
    eta_min: float = 0.0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=eta_min)
    best: dict[str, Any] | None = None
    rows: list[dict[str, float | int]] = []
    epochs_without_improvement = 0
    for epoch in range(1, max_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, standardizer, device)
        reference, prediction, _ = predict(model, validation_loader, standardizer, device)
        metrics = regression_metrics(reference, prediction)
        rows.append({"epoch": epoch, "train_standardized_mse": loss, "learning_rate": scheduler.get_last_lr()[0], **metrics})
        if best is None or metrics["mae_ev"] < best["metrics"]["mae_ev"]:
            best = {
                "epoch": epoch,
                "metrics": metrics,
                "model_state_dict": copy.deepcopy(model.state_dict()),
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        scheduler.step()
        if epochs_without_improvement >= patience:
            LOGGER.info("Early stopping at epoch %d; best epoch was %d", epoch, best["epoch"])
            break
    assert best is not None
    return best, pd.DataFrame(rows)


def train_fixed_epochs(
    model: torch.nn.Module,
    train_loader: DataLoader,
    standardizer: TargetStandardizer,
    device: torch.device,
    learning_rate: float,
    num_epochs: int,
    weight_decay: float = 0.0,
    eta_min: float = 0.0,
) -> pd.DataFrame:
    if num_epochs < 1:
        raise ValueError("num_epochs must be positive")
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)
    rows: list[dict[str, float | int]] = []
    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, standardizer, device)
        rows.append({"epoch": epoch, "train_standardized_mse": loss, "learning_rate": scheduler.get_last_lr()[0]})
        scheduler.step()
    return pd.DataFrame(rows)


def save_checkpoint(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
