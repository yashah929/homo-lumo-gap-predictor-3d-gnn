#!/usr/bin/env python3
"""Retrain a fresh selected model on the complete 80% development set."""

from __future__ import annotations

import argparse
from pathlib import Path

from qm9_gap.data import QM9GapDataset, load_split_indices, target_statistics
from qm9_gap.model import QM9GapMPNN
from qm9_gap.train import TargetStandardizer, make_loader, save_checkpoint, train_fixed_epochs
from qm9_gap.utils import (
    environment_metadata,
    load_yaml,
    save_json,
    save_yaml,
    seed_everything,
    select_device,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--selected", default="results/cv/selected_configuration.yaml")
    args = parser.parse_args()
    base = load_yaml(args.config)
    selected = load_yaml(args.selected)
    seed = int(base["final_seed"])
    seed_everything(seed)
    model_config = dict(base["model"])
    model_config.update(
        hidden_dim=int(selected["hidden_dim"]),
        num_message_passing_layers=int(selected["num_message_passing_layers"]),
    )
    dataset = QM9GapDataset(
        base["paths"]["data_root"],
        rbf_max=model_config["rbf_max"],
        domain_policy=base["dataset"]["rbf_domain_policy"],
        domain_tolerance=base["dataset"]["rbf_domain_tolerance"],
    )
    # Only development indices are read. No validation or test DataLoader exists here.
    development = load_split_indices(base["paths"]["splits_dir"], "development_indices")
    mean, standard_deviation = target_statistics(dataset, development)
    standardizer = TargetStandardizer(mean, standard_deviation)
    training = base["training"]
    loader = make_loader(
        dataset,
        development,
        int(training["batch_size"]),
        True,
        seed,
        int(training["num_workers"]),
        bool(training["pin_memory"]),
    )
    device = select_device(training["device"])
    # This is a new initialization after setting final_seed; no CV weights are loaded.
    model = QM9GapMPNN.from_config(model_config).to(device)
    final_epochs = int(selected["final_epochs"])
    curves = train_fixed_epochs(
        model,
        loader,
        standardizer,
        device,
        learning_rate=float(selected["learning_rate"]),
        num_epochs=final_epochs,
        weight_decay=float(training["weight_decay"]),
        eta_min=float(training["scheduler"]["eta_min"]),
    )
    final_root = Path(base["paths"]["results_dir"]) / "final"
    curves_path = final_root / "final_training_curve.csv"
    curves_path.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(curves_path, index=False)
    metadata = environment_metadata()
    checkpoint = {
        "training_stage": "final_development_retraining",
        "split_identity": "seed42_80-20_development_test_fourfold_v1",
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "selected_configuration": selected,
        "final_epochs": final_epochs,
        "seed": seed,
        "target_standardization": standardizer.as_dict(),
        "metadata": metadata,
    }
    checkpoint_root = Path(base["paths"].get("checkpoints_dir", final_root / "checkpoints"))
    checkpoint_path = checkpoint_root / "final" / "final_model.pt"
    save_checkpoint(checkpoint, checkpoint_path)
    run_record = {
        "training_stage": checkpoint["training_stage"],
        "split_identity": checkpoint["split_identity"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "selected_configuration": selected,
        "final_epochs": final_epochs,
        "seed": seed,
        "target_standardization": standardizer.as_dict(),
        "metadata": metadata,
    }
    save_json(run_record, final_root / "final_training_metadata.json")
    save_yaml({"base": base, "selected": selected, "final_seed": seed}, final_root / "final_training_config.yaml")
    print(f"Saved final checkpoint after exactly {final_epochs} epochs: {checkpoint_path}")


if __name__ == "__main__":
    main()
