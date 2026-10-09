#!/usr/bin/env python3
"""Write a portable machine-readable manifest for the cluster run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from qm9_gap.utils import environment_metadata, load_yaml, save_json, sha256_file


def optional_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def optional_yaml(path: Path) -> dict[str, object] | None:
    return load_yaml(path) if path.exists() else None


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", default="results/run_manifest.json")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--gpu-type", required=True)
    parser.add_argument("--smoke-job-id")
    parser.add_argument("--preprocessing-job-id")
    parser.add_argument("--cv-job-id")
    parser.add_argument("--cv-array-range", default="0-143")
    parser.add_argument("--retry-job-id")
    parser.add_argument("--retry-array-range")
    parser.add_argument("--final-training-job-id")
    parser.add_argument("--final-training-runtime")
    parser.add_argument("--final-test-job-id")
    parser.add_argument("--pre-repair-predictions-sha256")
    args = parser.parse_args()

    base = load_yaml(args.config)
    results_root = Path(base["paths"]["results_dir"])
    splits_root = Path(base["paths"]["splits_dir"])
    split_manifest = optional_json(splits_root / "split_manifest.json")
    final_metadata = optional_json(results_root / "final" / "final_training_metadata.json")
    test_metrics = optional_json(results_root / "final" / "test_metrics.json")
    selected = optional_yaml(results_root / "cv" / "selected_configuration.yaml")
    selected_id = None if selected is None else str(selected["configuration_id"])
    selected_folds = sorted(
        (
            row
            for row in csv_rows(results_root / "cv" / "cv_results.csv")
            if row["configuration_id"] == selected_id
        ),
        key=lambda row: int(row["fold"]),
    )
    selected_summary = next(
        (
            row
            for row in csv_rows(results_root / "cv" / "cv_summary.csv")
            if row["configuration_id"] == selected_id
        ),
        None,
    )
    status_lines = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], text=True
    ).splitlines()
    working_tree_files: dict[str, str] = {}
    for line in status_lines:
        path = line[3:].split(" -> ")[-1]
        candidate = Path(path)
        if candidate == Path(args.output):
            continue
        if candidate.is_file() and not candidate.is_symlink():
            working_tree_files[path] = sha256_file(candidate)
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"])
    packages = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--all"], text=True
    ).splitlines()
    base_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    checkpoint = (
        None
        if final_metadata is None
        else {
            "path": final_metadata.get("checkpoint_path"),
            "size_bytes": final_metadata.get("checkpoint_size_bytes"),
            "sha256": final_metadata.get("checkpoint_sha256"),
        }
    )
    prediction_path = results_root / "final" / "test_predictions.csv"
    result_files = [
        "results/cv/cv_results.csv",
        "results/cv/cv_summary.csv",
        "results/cv/selected_configuration.yaml",
        "results/final/final_training_config.yaml",
        "results/final/final_training_curve.csv",
        "results/final/final_training_metadata.json",
        "results/final/test_metrics.json",
        "results/final/test_predictions.csv",
        *[
            f"results/figures/{name}.{extension}"
            for name in (
                "predicted_vs_reference",
                "residual_distribution",
                "residual_vs_reference",
                "cv_validation_mae",
                "selected_cv_training_curves",
            )
            for extension in ("png", "pdf")
        ],
    ]
    manifest = {
        "manifest_version": 2,
        "stage": args.stage,
        "git_base_commit": base_commit,
        "metadata": environment_metadata(),
        "repository_state": {
            "status_porcelain": status_lines,
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "changed_file_sha256": working_tree_files,
        },
        "gpu_type": args.gpu_type,
        "slurm": {
            "account": args.account,
            "partition": args.partition,
            "smoke_job_id": args.smoke_job_id,
            "preprocessing_job_id": args.preprocessing_job_id,
            "cv_array_job_id": args.cv_job_id,
            "cv_array_range": args.cv_array_range if args.cv_job_id else None,
            "retry_array_job_id": args.retry_job_id,
            "retry_array_range": args.retry_array_range if args.retry_job_id else None,
            "final_training_job_id": args.final_training_job_id,
            "final_test_job_id": args.final_test_job_id,
        },
        "random_seeds": {
            "base_seed": int(base["seed"]),
            "final_seed": int(base["final_seed"]),
            "cv_seed_rule": "base_seed + 1000 * configuration_index + fold",
        },
        "dataset_split_manifest": split_manifest,
        "selected_hyperparameters": selected,
        "cv_selection": None
        if selected_summary is None
        else {
            "configuration_id": selected_id,
            "fold_validation_mae_ev": [
                float(row["best_validation_mae_ev"]) for row in selected_folds
            ],
            "fold_validation_rmse_ev": [
                float(row["validation_rmse_ev"]) for row in selected_folds
            ],
            "fold_validation_r2": [float(row["validation_r2"]) for row in selected_folds],
            "mean_validation_mae_ev": float(selected_summary["mean_validation_mae_ev"]),
            "std_validation_mae_ev": float(selected_summary["std_validation_mae_ev"]),
            "mean_validation_rmse_ev": float(selected_summary["mean_validation_rmse_ev"]),
            "mean_validation_r2": float(selected_summary["mean_validation_r2"]),
            "final_epochs": None if selected is None else int(selected["final_epochs"]),
        },
        "package_freeze": packages,
        "final_checkpoint": checkpoint,
        "final_training": None
        if final_metadata is None
        else {
            "job_id": args.final_training_job_id,
            "runtime": args.final_training_runtime,
            "epochs": final_metadata.get("final_epochs"),
            "seed": final_metadata.get("seed"),
            "num_development_molecules": None
            if split_manifest is None
            else split_manifest.get("num_development"),
            "target_standardization": final_metadata.get("target_standardization"),
            "checkpoint": checkpoint,
            "environment": final_metadata.get("metadata"),
        },
        "final_test": None
        if test_metrics is None
        else {
            "job_id": args.final_test_job_id,
            "mae_ev": test_metrics.get("mae_ev"),
            "rmse_ev": test_metrics.get("rmse_ev"),
            "r2": test_metrics.get("r2"),
            "num_test_molecules": test_metrics.get("num_test_molecules"),
            "split_identity": test_metrics.get("split_identity"),
            "environment": test_metrics.get("metadata"),
            "metrics_file": "results/final/test_metrics.json",
            "predictions_file": "results/final/test_predictions.csv",
            "predictions_sha256": sha256_file(prediction_path),
            "prediction_index_provenance": {
                "identifier": "zero-based retained PyG QM9 dataset index",
                "legacy_file_sha256": args.pre_repair_predictions_sha256,
                "correction": "Removed PyG batching offsets from identifiers by deterministic alignment to the completed non-shuffled locked-test loader; references, predictions, residuals, and metrics were unchanged.",
            },
        },
        "portable_result_files": result_files,
    }
    save_json(manifest, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
