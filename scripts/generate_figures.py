#!/usr/bin/env python3
"""Regenerate every project figure from committed result artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
PLOTTING_PATH = ROOT / "src" / "qm9_gap" / "plotting.py"
SPEC = importlib.util.spec_from_file_location("qm9_gap_result_plotting", PLOTTING_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load plotting utilities from {PLOTTING_PATH}")
plotting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plotting)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results = Path(args.results_dir)
    figures = results / "figures"
    predictions = pd.read_csv(results / "final" / "test_predictions.csv")
    with (results / "final" / "test_metrics.json").open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    summary = pd.read_csv(results / "cv" / "cv_summary.csv")
    selected_runs = pd.read_csv(results / "cv" / "cv_results.csv").query(
        "configuration_id == 'cfg_034'"
    )
    curves = pd.read_csv(results / "cv" / "selected_cv_training_curves.csv")
    with (results / "cv" / "selected_configuration.yaml").open(encoding="utf-8") as handle:
        selected = yaml.safe_load(handle)

    plotting.plot_test_diagnostics(predictions, metrics, figures)
    plotting.plot_cv_summary(summary, figures / "cv_validation_mae")
    plotting.plot_training_curves(
        curves,
        selected_runs,
        int(selected["final_epochs"]),
        figures / "selected_cv_training_curves",
    )
    plotting.plot_architecture(figures / "model_architecture")
    print(f"Regenerated six PNG/PDF figure pairs in {figures}")


if __name__ == "__main__":
    main()
