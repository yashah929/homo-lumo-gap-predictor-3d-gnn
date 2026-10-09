#!/usr/bin/env python3
"""Run a development-only, real-QM9 GPU execution smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.datasets import QM9

from qm9_gap.data import QM9GapDataset, enriched_qm9_graph, load_cv_fold, target_statistics
from qm9_gap.model import QM9GapMPNN
from qm9_gap.train import (
    TargetStandardizer,
    make_loader,
    save_checkpoint,
    train_with_early_stopping,
)
from qm9_gap.utils import environment_metadata, load_yaml, save_json, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--fold", type=int, default=0, choices=range(4))
    parser.add_argument("--num-train", type=int, default=192)
    parser.add_argument("--num-validation", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--prepared-cache", action="store_true")
    parser.add_argument("--output", default="results/smoke/qm9_gpu_smoke.json")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/checkpoints/smoke/qm9_gpu_smoke.pt",
    )
    args = parser.parse_args()
    if args.num_train < 2 or args.num_validation < 1 or args.epochs < 1:
        parser.error("Smoke-test subset sizes and epoch count must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("The real-QM9 smoke test requires a CUDA allocation")

    base = load_yaml(args.config)
    seed = int(base["seed"])
    seed_everything(seed)
    expected = int(base["dataset"]["expected_num_molecules"])
    training_pool, validation_pool = load_cv_fold(base["paths"]["splits_dir"], args.fold)
    training_source_indices = training_pool[: args.num_train]
    validation_source_indices = validation_pool[: args.num_validation]
    if args.prepared_cache:
        dataset = QM9GapDataset(
            base["paths"]["data_root"],
            rbf_max=float(base["model"]["rbf_max"]),
            domain_policy=base["dataset"]["rbf_domain_policy"],
            domain_tolerance=float(base["dataset"]["rbf_domain_tolerance"]),
        )
        training_indices = training_source_indices
        validation_indices = validation_source_indices
        data_source = "prepared_enriched_cache"
    else:
        source_root = Path(base["paths"]["data_root"]) / "pyg_source"
        source = QM9(root=str(source_root))
        selected = np.concatenate((training_source_indices, validation_source_indices))
        sdf_path = Path(source.raw_dir) / "gdb9.sdf"
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
        dataset = [enriched_qm9_graph(supplier, source[int(index)], int(index)) for index in selected]
        training_indices = np.arange(args.num_train, dtype=np.int64)
        validation_indices = np.arange(args.num_train, len(dataset), dtype=np.int64)
        data_source = "on_demand_enrichment"
    if len(dataset) != expected and args.prepared_cache:
        raise RuntimeError(f"Expected {expected} usable QM9 molecules, found {len(dataset)}")

    mean, standard_deviation = target_statistics(dataset, training_indices)
    standardizer = TargetStandardizer(mean, standard_deviation)
    training = base["training"]
    batch_size = min(32, args.num_train)
    train_loader = make_loader(dataset, training_indices, batch_size, True, seed, 0, True)
    validation_loader = make_loader(dataset, validation_indices, batch_size, False, seed, 0, True)
    model_config = dict(base["model"])
    model_config.update(hidden_dim=64, num_message_passing_layers=3)
    device = torch.device("cuda")
    model = QM9GapMPNN.from_config(model_config).to(device)
    best, curves = train_with_early_stopping(
        model,
        train_loader,
        validation_loader,
        standardizer,
        device,
        learning_rate=3.0e-4,
        max_epochs=args.epochs,
        patience=args.epochs,
        weight_decay=float(training["weight_decay"]),
        eta_min=float(training["scheduler"]["eta_min"]),
    )

    checkpoint_path = Path(args.checkpoint)
    save_checkpoint(
        {
            "training_stage": "development_only_real_qm9_gpu_smoke",
            "model_state_dict": best["model_state_dict"],
            "model_config": model_config,
            "fold": args.fold,
            "seed": seed,
            "best_epoch": best["epoch"],
            "target_standardization": standardizer.as_dict(),
            "metadata": environment_metadata(),
        },
        checkpoint_path,
    )
    output_path = Path(args.output)
    curve_path = output_path.with_name("qm9_gpu_smoke_curve.csv")
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(curve_path, index=False)
    result = {
        "execution_only": True,
        "scientific_result": False,
        "data_scope": "development_only",
        "data_source": data_source,
        "fold": args.fold,
        "num_source_molecules": expected,
        "num_training_molecules": args.num_train,
        "num_validation_molecules": args.num_validation,
        "epochs_completed": len(curves),
        "best_epoch": best["epoch"],
        "smoke_validation_metrics": best["metrics"],
        "checkpoint_path": str(checkpoint_path),
        "curve_path": str(curve_path),
        "target_standardization": standardizer.as_dict(),
        "metadata": environment_metadata(),
    }
    save_json(result, output_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
