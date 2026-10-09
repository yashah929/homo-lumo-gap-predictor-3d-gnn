#!/usr/bin/env python3
"""Perform the single explicit evaluation on the locked final test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qm9_gap.data import QM9GapDataset, load_split_indices
from qm9_gap.evaluate import evaluate_loader, load_final_checkpoint
from qm9_gap.plotting import plot_test_diagnostics
from qm9_gap.train import make_loader
from qm9_gap.utils import environment_metadata, load_yaml, save_json, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Required acknowledgement that this consumes the locked final test evaluation.",
    )
    args = parser.parse_args()
    if not args.confirm_final_test:
        parser.error("Refusing to access final-test indices without --confirm-final-test")
    base = load_yaml(args.config)
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_root = Path(base["paths"].get("checkpoints_dir", "results/final/checkpoints"))
        checkpoint_path = str(checkpoint_root / "final" / "final_model.pt")
    training = base["training"]
    device = select_device(training["device"])
    model, standardizer, checkpoint = load_final_checkpoint(checkpoint_path, device)
    dataset = QM9GapDataset(
        base["paths"]["data_root"],
        rbf_max=checkpoint["model_config"]["rbf_max"],
        domain_policy=base["dataset"]["rbf_domain_policy"],
        domain_tolerance=base["dataset"]["rbf_domain_tolerance"],
    )
    test_indices = load_split_indices(base["paths"]["splits_dir"], "final_test_indices")
    loader = make_loader(
        dataset,
        test_indices,
        int(training["batch_size"]),
        False,
        int(checkpoint["seed"]),
        int(training["num_workers"]),
        bool(training["pin_memory"]),
    )
    metrics, predictions = evaluate_loader(model, loader, standardizer, device)
    prediction_ids = predictions["molecule_index"].to_numpy(dtype=np.int64)
    if len(predictions) != len(test_indices):
        raise RuntimeError("Final-test prediction count does not match the locked split")
    if not np.array_equal(np.sort(prediction_ids), np.sort(test_indices)):
        raise RuntimeError("Final-test prediction identifiers do not match the locked split")
    if not np.isfinite(
        predictions[["reference_gap_ev", "predicted_gap_ev", "residual_ev"]].to_numpy()
    ).all():
        raise RuntimeError("Final-test predictions contain non-finite values")
    final_root = Path(base["paths"]["results_dir"]) / "final"
    predictions.to_csv(final_root / "test_predictions.csv", index=False)
    metrics_record = {
        "evaluation_stage": "locked_final_test",
        "split_identity": "seed42_80-20_development_test_fourfold_v1",
        "mae_ev": metrics["mae_ev"],
        "rmse_ev": metrics["rmse_ev"],
        "r2": metrics["r2"],
        "num_test_molecules": len(predictions),
        "checkpoint": checkpoint_path,
        "metadata": environment_metadata(),
    }
    save_json(metrics_record, final_root / "test_metrics.json")
    plot_test_diagnostics(predictions, Path(base["paths"]["results_dir"]) / "figures")
    print(json.dumps(metrics_record, indent=2))


if __name__ == "__main__":
    main()
