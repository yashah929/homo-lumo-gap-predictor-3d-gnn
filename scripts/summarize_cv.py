#!/usr/bin/env python3
"""Aggregate all CV runs, select the winner, and generate CV figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from qm9_gap.cross_validation import summarize_cv
from qm9_gap.plotting import plot_cv_summary, plot_training_curves
from qm9_gap.utils import load_yaml, save_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    results = Path(config["paths"]["results_dir"])
    cv_root = results / "cv"
    detailed, summary, selected = summarize_cv(cv_root)
    detailed.to_csv(cv_root / "cv_results.csv", index=False)
    summary.to_csv(cv_root / "cv_summary.csv", index=False)
    save_yaml(selected, cv_root / "selected_configuration.yaml")
    plot_cv_summary(summary, results / "figures" / "cv_validation_mae")
    curves, labels = [], []
    for fold in range(4):
        path = cv_root / "curves" / f"{selected['configuration_id']}_fold_{fold}.csv"
        curves.append(pd.read_csv(path))
        labels.append(f"Fold {fold + 1}")
    plot_training_curves(curves, labels, results / "figures" / "selected_cv_training_curves")
    print(f"Selected {selected['configuration_id']} for {selected['final_epochs']} final epochs")


if __name__ == "__main__":
    main()
