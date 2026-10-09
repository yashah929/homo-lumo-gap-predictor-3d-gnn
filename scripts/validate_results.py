#!/usr/bin/env python3
"""Validate completed split, prediction, metric, figure, and provenance artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def require_close(name: str, observed: float, expected: float, tolerance: float) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{name} mismatch: {observed} != {expected}")


def main() -> None:
    splits = ROOT / "splits"
    split_manifest = load_json(splits / "split_manifest.json")
    arrays: dict[str, np.ndarray] = {}
    for filename, expected_hash in split_manifest["files"].items():
        path = splits / filename
        observed_hash = sha256(path)
        if observed_hash != expected_hash:
            raise ValueError(f"Split hash mismatch for {filename}: {observed_hash}")
        values = np.load(path, allow_pickle=False)
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"Invalid split array: {filename}")
        if len(np.unique(values)) != len(values):
            raise ValueError(f"Duplicate indices in {filename}")
        arrays[filename] = values.astype(np.int64, copy=False)

    development = arrays["development_indices.npy"]
    test = arrays["final_test_indices.npy"]
    if len(development) != int(split_manifest["num_development"]):
        raise ValueError("Development count does not match the split manifest")
    if len(test) != int(split_manifest["num_final_test"]):
        raise ValueError("Test count does not match the split manifest")
    if np.intersect1d(development, test).size:
        raise ValueError("Development and final-test splits overlap")
    complete = np.sort(np.concatenate([development, test]))
    if not np.array_equal(complete, np.arange(int(split_manifest["num_molecules"]))):
        raise ValueError("Development and test indices do not partition the retained dataset")

    folds = [arrays[f"cv_fold_{fold}_validation_indices.npy"] for fold in range(4)]
    if not np.array_equal(np.sort(np.concatenate(folds)), np.sort(development)):
        raise ValueError("CV folds do not partition the development set")

    final_root = ROOT / "results" / "final"
    predictions = pd.read_csv(final_root / "test_predictions.csv")
    expected_columns = [
        "molecule_index",
        "reference_gap_ev",
        "predicted_gap_ev",
        "residual_ev",
    ]
    if predictions.columns.tolist() != expected_columns:
        raise ValueError(f"Unexpected prediction columns: {predictions.columns.tolist()}")
    if len(predictions) != len(test):
        raise ValueError(f"Prediction count mismatch: {len(predictions)} != {len(test)}")
    prediction_ids = predictions["molecule_index"].to_numpy(dtype=np.int64)
    if len(np.unique(prediction_ids)) != len(prediction_ids):
        raise ValueError("Prediction molecule indices are not unique")
    if not np.array_equal(np.sort(prediction_ids), np.sort(test)):
        raise ValueError("Prediction molecule indices do not equal the locked test split")
    if np.intersect1d(prediction_ids, development).size:
        raise ValueError("Development indices appear in the test predictions")

    numeric = predictions[expected_columns[1:]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Non-finite reference, prediction, or residual value")
    reference = predictions["reference_gap_ev"].to_numpy(dtype=np.float64)
    predicted = predictions["predicted_gap_ev"].to_numpy(dtype=np.float64)
    residual = predictions["residual_ev"].to_numpy(dtype=np.float64)
    if not np.allclose(residual, predicted - reference, rtol=0.0, atol=2.0e-6):
        raise ValueError("Residual values are inconsistent with prediction minus reference")

    mae = float(np.mean(np.abs(predicted - reference)))
    rmse = float(np.sqrt(np.mean((predicted - reference) ** 2)))
    r2 = float(1.0 - np.sum((predicted - reference) ** 2) / np.sum((reference - reference.mean()) ** 2))
    metrics = load_json(final_root / "test_metrics.json")
    if int(metrics["num_test_molecules"]) != len(predictions):
        raise ValueError("Metric and prediction molecule counts differ")
    require_close("MAE", mae, float(metrics["mae_ev"]), 2.0e-8)
    require_close("RMSE", rmse, float(metrics["rmse_ev"]), 2.0e-8)
    require_close("R2", r2, float(metrics["r2"]), 2.0e-8)

    figure_names = (
        "predicted_vs_reference",
        "residual_distribution",
        "residual_vs_reference",
        "cv_validation_mae",
        "selected_cv_training_curves",
        "model_architecture",
    )
    for name in figure_names:
        for extension in ("png", "pdf"):
            path = ROOT / "results" / "figures" / f"{name}.{extension}"
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing or empty figure: {path}")

    training = load_json(final_root / "final_training_metadata.json")
    checkpoint = ROOT / str(training["checkpoint_path"])
    if checkpoint.stat().st_size != int(training["checkpoint_size_bytes"]):
        raise ValueError("Checkpoint size does not match training metadata")
    if sha256(checkpoint) != training["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 does not match training metadata")

    manifest = load_json(ROOT / "results" / "run_manifest.json")
    if manifest["dataset_split_manifest"] != split_manifest:
        raise ValueError("Run manifest does not embed the immutable split manifest")
    manifest_test = manifest["final_test"]
    require_close("manifest MAE", float(manifest_test["mae_ev"]), mae, 2.0e-8)
    require_close("manifest RMSE", float(manifest_test["rmse_ev"]), rmse, 2.0e-8)
    require_close("manifest R2", float(manifest_test["r2"]), r2, 2.0e-8)
    if manifest_test["predictions_sha256"] != sha256(final_root / "test_predictions.csv"):
        raise ValueError("Run manifest prediction hash is stale")

    print(
        json.dumps(
            {
                "split_integrity": "passed",
                "prediction_integrity": "passed",
                "num_test_molecules": len(predictions),
                "mae_ev": mae,
                "rmse_ev": rmse,
                "r2": r2,
                "figures": 2 * len(figure_names),
                "checkpoint_sha256": training["checkpoint_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
