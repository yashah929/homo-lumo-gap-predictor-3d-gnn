"""Publication-oriented diagnostic plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless, deterministic file rendering on compute nodes.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


def _save_figure(figure: plt.Figure, output_stem: str | Path) -> None:
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_test_diagnostics(predictions: pd.DataFrame, output_dir: str | Path) -> None:
    output = Path(output_dir)
    reference = predictions["reference_gap_ev"].to_numpy()
    predicted = predictions["predicted_gap_ev"].to_numpy()
    residual = predictions["residual_ev"].to_numpy()

    figure, axis = plt.subplots(figsize=(4.6, 4.3))
    axis.scatter(reference, predicted, s=7, alpha=0.35, linewidths=0)
    limits = [min(reference.min(), predicted.min()), max(reference.max(), predicted.max())]
    axis.plot(limits, limits, color="black", linewidth=1)
    axis.set(xlabel="Reference HOMO-LUMO gap (eV)", ylabel="Predicted HOMO-LUMO gap (eV)", xlim=limits, ylim=limits)
    _save_figure(figure, output / "predicted_vs_reference")

    figure, axis = plt.subplots(figsize=(4.8, 3.6))
    axis.hist(residual, bins=60, color="0.35", edgecolor="white", linewidth=0.3)
    axis.set(xlabel="Prediction residual (eV)", ylabel="Molecule count")
    _save_figure(figure, output / "residual_distribution")

    figure, axis = plt.subplots(figsize=(4.8, 3.6))
    axis.scatter(reference, residual, s=7, alpha=0.35, linewidths=0)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set(xlabel="Reference HOMO-LUMO gap (eV)", ylabel="Prediction residual (eV)")
    _save_figure(figure, output / "residual_vs_reference")


def plot_cv_summary(summary: pd.DataFrame, output_stem: str | Path) -> None:
    ordered = summary.sort_values("mean_validation_mae_ev").reset_index(drop=True)
    x = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(9.0, 4.0))
    axis.errorbar(
        x,
        ordered["mean_validation_mae_ev"],
        yerr=ordered["std_validation_mae_ev"],
        fmt="o",
        markersize=3.5,
        capsize=2,
        color="black",
        ecolor="0.55",
    )
    axis.set(xlabel="Hyperparameter configuration (ordered by mean MAE)", ylabel="Four-fold validation MAE (eV)")
    axis.set_xticks(x[:: max(1, len(x) // 12)], ordered["configuration_id"].iloc[:: max(1, len(x) // 12)], rotation=45, ha="right")
    _save_figure(figure, output_stem)


def plot_training_curves(curves: list[pd.DataFrame], labels: list[str], output_stem: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(5.4, 3.8))
    for curve, label in zip(curves, labels, strict=True):
        axis.plot(curve["epoch"], curve["mae_ev"], linewidth=1.2, label=label)
    axis.set(xlabel="Epoch", ylabel="Validation MAE (eV)")
    axis.legend(frameon=False, ncol=2)
    _save_figure(figure, output_stem)
