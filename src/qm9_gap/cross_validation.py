"""Development-only grid construction, CV execution, and model selection."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .data import QM9GapDataset, load_cv_fold, target_statistics
from .model import QM9GapMPNN
from .train import TargetStandardizer, make_loader, save_checkpoint, train_with_early_stopping
from .utils import environment_metadata, save_json, save_yaml, seed_everything, select_device


def grid_configurations(grid: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the deterministic 36-member depth/dimension/rate Cartesian grid."""
    depths = list(grid["num_message_passing_layers"])
    dimensions = list(grid["hidden_dim"])
    rates = list(grid["learning_rate"])
    if depths != [3, 4, 5, 6] or dimensions != [64, 128, 256] or rates != [0.0001, 0.0003, 0.001]:
        raise ValueError("The initial search grid must exactly match the preregistered 4x3x3 values")
    configurations = []
    for index, (depth, hidden_dim, learning_rate) in enumerate(itertools.product(depths, dimensions, rates)):
        configurations.append(
            {
                "configuration_id": f"cfg_{index:03d}",
                "num_message_passing_layers": int(depth),
                "hidden_dim": int(hidden_dim),
                "learning_rate": float(learning_rate),
            }
        )
    assert len(configurations) == 36
    return configurations


def array_index_to_experiment(array_index: int, grid: dict[str, Any]) -> tuple[dict[str, Any], int]:
    num_folds = int(grid.get("num_folds", 4))
    configurations = grid_configurations(grid)
    if num_folds != 4:
        raise ValueError("Exactly four development folds are required")
    if not 0 <= array_index < len(configurations) * num_folds:
        raise ValueError("SLURM array index must be in [0, 143]")
    return configurations[array_index // num_folds], array_index % num_folds


def run_cv_experiment(
    base_config: dict[str, Any],
    experiment: dict[str, Any],
    fold: int,
    repository: str | Path = ".",
    dataset: Any | None = None,
) -> dict[str, Any]:
    """Train one configuration/fold pair without loading final-test indices."""
    seed = int(base_config["seed"]) + 1000 * int(experiment["configuration_id"].split("_")[-1]) + fold
    seed_everything(seed)
    paths = base_config["paths"]
    model_config = dict(base_config["model"])
    model_config["hidden_dim"] = int(experiment["hidden_dim"])
    model_config["num_message_passing_layers"] = int(experiment["num_message_passing_layers"])
    if dataset is None:
        dataset = QM9GapDataset(
            paths["data_root"],
            rbf_max=model_config["rbf_max"],
            domain_policy=base_config["dataset"]["rbf_domain_policy"],
            domain_tolerance=base_config["dataset"]["rbf_domain_tolerance"],
        )
    training_indices, validation_indices = load_cv_fold(paths["splits_dir"], fold)
    mean, standard_deviation = target_statistics(dataset, training_indices)
    standardizer = TargetStandardizer(mean, standard_deviation)
    training = base_config["training"]
    device = select_device(training["device"])
    train_loader = make_loader(
        dataset, training_indices, training["batch_size"], True, seed, training["num_workers"], training["pin_memory"]
    )
    validation_loader = make_loader(
        dataset, validation_indices, training["batch_size"], False, seed, training["num_workers"], training["pin_memory"]
    )
    model = QM9GapMPNN.from_config(model_config).to(device)
    best, curves = train_with_early_stopping(
        model,
        train_loader,
        validation_loader,
        standardizer,
        device,
        learning_rate=float(experiment["learning_rate"]),
        max_epochs=int(training["max_epochs"]),
        patience=int(training["patience"]),
        weight_decay=float(training["weight_decay"]),
        eta_min=float(training["scheduler"]["eta_min"]),
    )
    run_name = f"{experiment['configuration_id']}_fold_{fold}"
    results_root = Path(paths["results_dir"]) / "cv"
    checkpoint_path = results_root / "checkpoints" / f"{run_name}.pt"
    curve_path = results_root / "curves" / f"{run_name}.csv"
    run_path = results_root / "runs" / f"{run_name}.json"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(curve_path, index=False)
    metadata = environment_metadata(repository)
    checkpoint = {
        "model_state_dict": best["model_state_dict"],
        "model_config": model_config,
        "experiment": experiment,
        "fold": fold,
        "seed": seed,
        "best_epoch": best["epoch"],
        "target_standardization": standardizer.as_dict(),
        "metadata": metadata,
    }
    save_checkpoint(checkpoint, checkpoint_path)
    result = {
        "configuration_id": experiment["configuration_id"],
        "fold": fold,
        "num_message_passing_layers": experiment["num_message_passing_layers"],
        "hidden_dim": experiment["hidden_dim"],
        "learning_rate": experiment["learning_rate"],
        "seed": seed,
        "best_epoch": best["epoch"],
        "best_validation_mae_ev": best["metrics"]["mae_ev"],
        "validation_rmse_ev": best["metrics"]["rmse_ev"],
        "validation_r2": best["metrics"]["r2"],
        "checkpoint_path": str(checkpoint_path),
        "curve_path": str(curve_path),
        "target_standardization": standardizer.as_dict(),
        "split_identity": "seed42_80-20_development_test_fourfold_v1",
        "metadata": metadata,
    }
    save_json(result, run_path)
    save_yaml({"base": base_config, "experiment": experiment, "fold": fold, "seed": seed}, run_path.with_suffix(".yaml"))
    return result


def summarize_cv(results_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Aggregate 144 atomic run records and select the CV winner."""
    root = Path(results_dir)
    run_files = sorted((root / "runs").glob("cfg_*_fold_*.json"))
    if len(run_files) != 144:
        raise RuntimeError(f"Expected 144 completed CV run files, found {len(run_files)}")
    import json

    rows = []
    for path in run_files:
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        rows.append({key: value for key, value in record.items() if key not in {"metadata", "target_standardization"}})
    detailed = pd.DataFrame(rows).sort_values(["configuration_id", "fold"]).reset_index(drop=True)
    grouped = detailed.groupby(
        ["configuration_id", "num_message_passing_layers", "hidden_dim", "learning_rate"], as_index=False
    ).agg(
        mean_validation_mae_ev=("best_validation_mae_ev", "mean"),
        std_validation_mae_ev=("best_validation_mae_ev", "std"),
        mean_validation_rmse_ev=("validation_rmse_ev", "mean"),
        std_validation_rmse_ev=("validation_rmse_ev", "std"),
        mean_validation_r2=("validation_r2", "mean"),
        std_validation_r2=("validation_r2", "std"),
        mean_best_epoch=("best_epoch", "mean"),
    )
    counts = detailed.groupby("configuration_id")["fold"].nunique()
    if not bool((counts == 4).all()):
        raise RuntimeError("Every configuration must contain four distinct folds")
    # Tie-break order is preregistered and does not involve the locked test set.
    summary = grouped.sort_values(
        ["mean_validation_mae_ev", "std_validation_mae_ev", "hidden_dim", "num_message_passing_layers"],
        kind="stable",
    ).reset_index(drop=True)
    winner = summary.iloc[0]
    winner_epochs = detailed.loc[detailed["configuration_id"] == winner["configuration_id"], "best_epoch"].to_numpy()
    final_epochs = int(np.floor(np.median(winner_epochs) + 0.5))
    selected = {
        "configuration_id": str(winner["configuration_id"]),
        "num_message_passing_layers": int(winner["num_message_passing_layers"]),
        "hidden_dim": int(winner["hidden_dim"]),
        "learning_rate": float(winner["learning_rate"]),
        "fold_best_epochs": [int(value) for value in winner_epochs],
        "final_epochs": final_epochs,
        "selection_metric": "mean_validation_mae_ev",
        "mean_validation_mae_ev": float(winner["mean_validation_mae_ev"]),
        "std_validation_mae_ev": float(winner["std_validation_mae_ev"]),
    }
    return detailed, summary, selected
