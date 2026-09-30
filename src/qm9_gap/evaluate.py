"""Checkpoint loading and physical-unit final evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .metrics import regression_metrics
from .model import QM9GapMPNN
from .train import TargetStandardizer, predict


def load_final_checkpoint(path: str | Path, device: torch.device) -> tuple[QM9GapMPNN, TargetStandardizer, dict[str, Any]]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("training_stage") != "final_development_retraining":
        raise ValueError("Refusing final-test evaluation: checkpoint is not a final retrained model")
    model = QM9GapMPNN.from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    normalization = checkpoint["target_standardization"]
    standardizer = TargetStandardizer(normalization["mean_ev"], normalization["standard_deviation_ev"])
    return model, standardizer, checkpoint


def evaluate_loader(
    model: torch.nn.Module,
    loader: Any,
    standardizer: TargetStandardizer,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    reference, prediction, molecule_index = predict(model, loader, standardizer, device)
    metrics = regression_metrics(reference, prediction)
    predictions = pd.DataFrame(
        {
            "molecule_index": molecule_index,
            "reference_gap_ev": reference,
            "predicted_gap_ev": prediction,
            "residual_ev": prediction - reference,
        }
    ).sort_values("molecule_index")
    return metrics, predictions
