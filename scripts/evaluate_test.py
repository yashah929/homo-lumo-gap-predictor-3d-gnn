#!/usr/bin/env python3
"""Perform the single explicit evaluation on the locked final test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qm9_gap.data import QM9GapDataset, load_split_indices
from qm9_gap.evaluate import evaluate_loader, load_final_checkpoint
from qm9_gap.plotting import plot_test_diagnostics
from qm9_gap.train import make_loader
from qm9_gap.utils import environment_metadata, load_yaml, save_json, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", default="results/final/checkpoints/final_model.pt")
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Required acknowledgement that this consumes the locked final test evaluation.",
    )
    args = parser.parse_args()
    if not args.confirm_final_test:
        parser.error("Refusing to access final-test indices without --confirm-final-test")
    base = load_yaml(args.config)
    training = base["training"]
    device = select_device(training["device"])
    model, standardizer, checkpoint = load_final_checkpoint(args.checkpoint, device)
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
    final_root = Path(base["paths"]["results_dir"]) / "final"
    predictions.to_csv(final_root / "test_predictions.csv", index=False)
    metrics_record = {
        "evaluation_stage": "locked_final_test",
        "split_identity": "seed42_80-20_development_test_fourfold_v1",
        "mae_ev": metrics["mae_ev"],
        "rmse_ev": metrics["rmse_ev"],
        "r2": metrics["r2"],
        "num_test_molecules": len(predictions),
        "checkpoint": str(args.checkpoint),
        "metadata": environment_metadata(),
    }
    save_json(metrics_record, final_root / "test_metrics.json")
    plot_test_diagnostics(predictions, Path(base["paths"]["results_dir"]) / "figures")
    print(json.dumps(metrics_record, indent=2))


if __name__ == "__main__":
    main()
