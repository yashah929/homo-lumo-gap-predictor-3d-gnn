"""Consistent Seaborn/Matplotlib plotting utilities for committed results."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless, deterministic file rendering on compute nodes.
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


FIGURE_BACKGROUND = "#FBFCFE"
AXES_BACKGROUND = "#FFFFFF"
TEXT_COLOR = "#233044"
GRID_COLOR = "#DCE3EC"
SELECTED_COLOR = "#C23B55"


def apply_plot_style() -> None:
    """Apply the shared visual language used by every project figure."""

    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        font_scale=1.1,
        rc={
            "figure.facecolor": FIGURE_BACKGROUND,
            "axes.facecolor": AXES_BACKGROUND,
            "axes.edgecolor": "#8793A3",
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.58,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "legend.frameon": False,
            "savefig.facecolor": FIGURE_BACKGROUND,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


def _save_figure(figure: plt.Figure, output_stem: str | Path) -> None:
    """Save a vector PDF and a 300 dpi PNG, then release the figure."""

    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    figure.savefig(
        stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(figure)


def _despine(axis: plt.Axes) -> None:
    sns.despine(ax=axis, top=True, right=True)


def plot_test_diagnostics(
    predictions: pd.DataFrame,
    metrics: Mapping[str, float],
    output_dir: str | Path,
) -> None:
    """Plot held-out predictions and residuals without changing or filtering data."""

    apply_plot_style()
    output = Path(output_dir)
    reference = predictions["reference_gap_ev"].to_numpy(dtype=float)
    predicted = predictions["predicted_gap_ev"].to_numpy(dtype=float)
    residual = predictions["residual_ev"].to_numpy(dtype=float)
    density_cmap = sns.color_palette("mako", as_cmap=True)

    figure, axis = plt.subplots(figsize=(5.35, 4.8), constrained_layout=True)
    bounds = [min(reference.min(), predicted.min()), max(reference.max(), predicted.max())]
    padding = 0.025 * (bounds[1] - bounds[0])
    limits = [bounds[0] - padding, bounds[1] + padding]
    density = axis.hexbin(
        reference,
        predicted,
        gridsize=82,
        mincnt=1,
        bins="log",
        cmap=density_cmap,
        linewidths=0,
        rasterized=True,
    )
    axis.plot(limits, limits, color="#17202D", linewidth=1.7, zorder=3, label="Identity")
    axis.set(
        xlabel="Reference HOMO–LUMO gap (eV)",
        ylabel="Predicted HOMO–LUMO gap (eV)",
        xlim=limits,
        ylim=limits,
        aspect="equal",
    )
    metric_text = (
        f"MAE  {float(metrics['mae_ev']):.6f} eV\n"
        f"RMSE  {float(metrics['rmse_ev']):.6f} eV\n"
        f"$R^2$  {float(metrics['r2']):.6f}"
    )
    axis.text(
        0.045,
        0.955,
        metric_text,
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=9.2,
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": GRID_COLOR,
            "alpha": 0.94,
        },
    )
    colorbar = figure.colorbar(density, ax=axis, pad=0.025, fraction=0.047)
    colorbar.set_label("Molecules per hexagon (log scale)")
    axis.grid(True, alpha=0.35)
    _despine(axis)
    _save_figure(figure, output / "predicted_vs_reference")

    figure, axis = plt.subplots(figsize=(5.6, 3.9), constrained_layout=True)
    sns.histplot(
        x=residual,
        bins=90,
        kde=True,
        color=sns.color_palette("deep")[0],
        alpha=0.72,
        edgecolor="white",
        linewidth=0.35,
        line_kws={"linewidth": 1.7},
        ax=axis,
    )
    mean_residual = float(np.mean(residual))
    median_residual = float(np.median(residual))
    axis.axvline(0.0, color="#17202D", linewidth=1.25, linestyle="--", label="Zero")
    axis.axvline(
        mean_residual,
        color=sns.color_palette("colorblind")[1],
        linewidth=1.7,
        label=f"Mean: {mean_residual:+.4f} eV",
    )
    axis.axvline(
        median_residual,
        color=sns.color_palette("colorblind")[2],
        linewidth=1.7,
        linestyle=":",
        label=f"Median: {median_residual:+.4f} eV",
    )
    axis.set(xlabel="Prediction residual (eV)", ylabel="Molecule count")
    axis.legend(loc="upper left", ncols=1)
    axis.grid(axis="x", visible=False)
    _despine(axis)
    _save_figure(figure, output / "residual_distribution")

    figure, axis = plt.subplots(figsize=(6.15, 4.2), constrained_layout=True)
    density = axis.hexbin(
        reference,
        residual,
        gridsize=78,
        mincnt=1,
        bins="log",
        cmap=sns.color_palette("crest", as_cmap=True),
        linewidths=0,
        rasterized=True,
    )
    trend_frame = pd.DataFrame({"reference": reference, "residual": residual})
    trend_frame["bin"] = pd.qcut(trend_frame["reference"], q=24, duplicates="drop")
    trend = (
        trend_frame.groupby("bin", observed=True)
        .agg(reference=("reference", "median"), residual=("residual", "median"))
        .reset_index(drop=True)
    )
    sns.lineplot(
        data=trend,
        x="reference",
        y="residual",
        color=SELECTED_COLOR,
        marker="o",
        markersize=4.2,
        linewidth=2.0,
        label="Binned median residual",
        ax=axis,
        zorder=4,
    )
    axis.axhline(0.0, color="#17202D", linewidth=1.3, linestyle="--", label="Zero residual")
    axis.set(
        xlabel="Reference HOMO–LUMO gap (eV)",
        ylabel="Residual: predicted − reference (eV)",
    )
    axis.legend(loc="lower left")
    colorbar = figure.colorbar(density, ax=axis, pad=0.025, fraction=0.047)
    colorbar.set_label("Molecules per hexagon (log scale)")
    axis.grid(True, alpha=0.35)
    _despine(axis)
    _save_figure(figure, output / "residual_vs_reference")


def _learning_rate_label(value: float) -> str:
    labels = {
        1.0e-4: r"$1\times10^{-4}$",
        3.0e-4: r"$3\times10^{-4}$",
        1.0e-3: r"$1\times10^{-3}$",
    }
    return labels.get(float(value), f"{value:.1e}")


def plot_cv_summary(summary: pd.DataFrame, output_stem: str | Path) -> None:
    """Show the full depth × width × learning-rate search and fold variability."""

    apply_plot_style()
    hidden_dims = sorted(summary["hidden_dim"].unique())
    learning_rates = sorted(summary["learning_rate"].unique())
    colors = sns.color_palette("colorblind", n_colors=len(hidden_dims))
    palette = {str(value): color for value, color in zip(hidden_dims, colors, strict=True)}
    plot_data = summary.copy()
    plot_data["hidden_dimension"] = plot_data["hidden_dim"].astype(str)

    figure, axes = plt.subplots(
        1,
        len(learning_rates),
        figsize=(10.6, 4.05),
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(left=0.075, right=0.992, top=0.84, bottom=0.24, wspace=0.04)
    for axis, learning_rate in zip(axes, learning_rates, strict=True):
        subset = plot_data[np.isclose(plot_data["learning_rate"], learning_rate)].copy()
        sns.lineplot(
            data=subset,
            x="num_message_passing_layers",
            y="mean_validation_mae_ev",
            hue="hidden_dimension",
            hue_order=[str(value) for value in hidden_dims],
            palette=palette,
            marker="o",
            markersize=5.0,
            linewidth=1.45,
            errorbar=None,
            legend=False,
            ax=axis,
        )
        for row in subset.itertuples(index=False):
            axis.errorbar(
                row.num_message_passing_layers,
                row.mean_validation_mae_ev,
                yerr=row.std_validation_mae_ev,
                fmt="none",
                ecolor=palette[str(row.hidden_dim)],
                elinewidth=1.0,
                capsize=2.2,
                alpha=0.8,
                zorder=2,
            )
        selected = subset[subset["configuration_id"] == "cfg_034"]
        if not selected.empty:
            row = selected.iloc[0]
            axis.scatter(
                row["num_message_passing_layers"],
                row["mean_validation_mae_ev"],
                s=135,
                marker="*",
                facecolor="#FFD166",
                edgecolor=SELECTED_COLOR,
                linewidth=1.2,
                zorder=6,
            )
            axis.annotate(
                "cfg_034",
                (row["num_message_passing_layers"], row["mean_validation_mae_ev"]),
                xytext=(-7, 13),
                textcoords="offset points",
                ha="right",
                va="bottom",
                color=SELECTED_COLOR,
                fontsize=8.7,
                fontweight="bold",
            )
        axis.set_title(f"Learning rate = {_learning_rate_label(learning_rate)}", pad=8)
        axis.set_xlabel("Message-passing depth")
        axis.set_xticks([3, 4, 5, 6])
        axis.grid(axis="x", visible=False)
        _despine(axis)

    axes[0].set_ylabel("Mean four-fold validation MAE (eV)")
    for axis in axes[1:]:
        axis.set_ylabel("")
    handles = [
        Line2D(
            [0],
            [0],
            color=palette[str(value)],
            marker="o",
            linewidth=1.5,
            label=f"{value}",
        )
        for value in hidden_dims
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="none",
            marker="*",
            markerfacecolor="#FFD166",
            markeredgecolor=SELECTED_COLOR,
            markersize=10,
            label="Selected cfg_034",
        )
    )
    figure.legend(
        handles=handles,
        title="Hidden dimension",
        loc="lower center",
        ncols=4,
        bbox_to_anchor=(0.5, 0.015),
    )
    figure.text(
        0.075,
        0.965,
        "Four-fold mean validation MAE; error bars show fold-to-fold SD.",
        ha="left",
        va="top",
        fontsize=8.6,
        color="#59677A",
    )
    _save_figure(figure, output_stem)


def plot_training_curves(
    curves: pd.DataFrame,
    selected_runs: pd.DataFrame,
    final_epoch: int,
    output_stem: str | Path,
) -> None:
    """Plot unsmoothed selected-CV validation paths and their exact best epochs."""

    apply_plot_style()
    colors = sns.color_palette("colorblind", n_colors=4)
    figure, axis = plt.subplots(figsize=(7.2, 4.15), constrained_layout=True)
    for fold, color in enumerate(colors):
        curve = curves[curves["fold"] == fold].sort_values("epoch")
        sns.lineplot(
            data=curve,
            x="epoch",
            y="mae_ev",
            color=color,
            linewidth=1.35,
            label=f"Fold {fold + 1}",
            ax=axis,
        )
        selected = selected_runs[selected_runs["fold"] == fold].iloc[0]
        axis.scatter(
            selected["best_epoch"],
            selected["best_validation_mae_ev"],
            s=37,
            facecolor=color,
            edgecolor="white",
            linewidth=0.85,
            zorder=5,
        )
        label_offsets = [(-9, 15), (-12, 29), (8, 15), (11, 29)]
        axis.annotate(
            str(int(selected["best_epoch"])),
            (selected["best_epoch"], selected["best_validation_mae_ev"]),
            xytext=label_offsets[fold],
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=color,
            fontsize=7.6,
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.65},
        )

    axis.axvline(final_epoch, color=SELECTED_COLOR, linewidth=1.55, linestyle="--", zorder=1)
    axis.text(
        final_epoch - 2,
        0.18,
        f"Final epoch = {final_epoch}",
        color=SELECTED_COLOR,
        rotation=90,
        ha="right",
        va="top",
        fontsize=8.4,
    )
    handles, labels = axis.get_legend_handles_labels()
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#718096",
                markeredgecolor="white",
                markersize=6,
                label="Fold-best epoch",
            ),
            Line2D(
                [0],
                [0],
                color=SELECTED_COLOR,
                linestyle="--",
                label=f"Final epoch ({final_epoch})",
            ),
        ]
    )
    labels.extend(["Fold-best epoch", f"Final epoch ({final_epoch})"])
    axis.legend(
        handles,
        labels,
        loc="upper right",
        ncols=2,
        frameon=True,
        facecolor="white",
        edgecolor=GRID_COLOR,
        framealpha=0.94,
    )
    axis.set(
        xlabel="Epoch",
        ylabel="Validation MAE (eV)",
        xlim=(0, 300),
        ylim=(0.047, 0.267),
    )
    axis.text(
        0.015,
        0.965,
        "Selected configuration: cfg_034",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    axis.grid(axis="x", visible=False)
    _despine(axis)
    _save_figure(figure, output_stem)


def _add_box(
    axis: plt.Axes,
    center: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: tuple[float, float, float],
    edgecolor: tuple[float, float, float],
    fontsize: float = 9.0,
) -> FancyBboxPatch:
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.35,
        zorder=2,
    )
    axis.add_patch(patch)
    axis.text(x, y, text, ha="center", va="center", fontsize=fontsize, linespacing=1.25, zorder=3)
    return patch


def _add_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    connectionstyle: str = "arc3",
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.25,
            color="#526173",
            connectionstyle=connectionstyle,
            shrinkA=1,
            shrinkB=1,
            zorder=1,
        )
    )


def plot_architecture(output_stem: str | Path) -> None:
    """Draw the model architecture and the distance-only geometry pathway."""

    apply_plot_style()
    palette = sns.color_palette("Set2", n_colors=8)
    fills = [sns.light_palette(color, n_colors=5)[1] for color in palette]
    figure, axis = plt.subplots(figsize=(13.2, 4.0), constrained_layout=True)
    axis.set_xlim(0, 18.9)
    axis.set_ylim(0.2, 5.5)
    axis.axis("off")

    _add_box(
        axis,
        (1.0, 2.9),
        1.55,
        1.25,
        "QM9 molecule\n(elements, bonds, xyz)",
        fills[0],
        palette[0],
    )
    _add_box(
        axis,
        (3.25, 2.9),
        1.8,
        1.25,
        "Complete directed\natomic graph\n$N(N-1)$ edges",
        fills[1],
        palette[1],
    )
    _add_box(
        axis,
        (5.75, 4.35),
        1.9,
        1.0,
        "Atom chemical\nfeatures + bond\nindicators",
        fills[2],
        palette[2],
    )
    _add_box(
        axis,
        (5.75, 1.45),
        1.9,
        1.0,
        "Pairwise distances\n$d_{ij}=\\|x_i-x_j\\|_2$",
        fills[3],
        palette[3],
    )
    _add_box(
        axis,
        (8.15, 1.45),
        1.85,
        1.0,
        "50 Gaussian radial\nbasis functions",
        fills[4],
        palette[4],
    )
    _add_box(axis, (10.1, 3.05), 1.8, 1.25, "6 message-passing\nblocks", fills[5], palette[5])
    _add_box(
        axis,
        (12.25, 3.05),
        1.75,
        1.25,
        "GRU + LayerNorm\natom updates",
        fills[6],
        palette[6],
    )
    _add_box(axis, (14.3, 3.05), 1.55, 1.15, "Set2Set\npooling", fills[7], palette[7])
    _add_box(axis, (16.15, 3.05), 1.45, 1.15, "Prediction\nMLP", fills[0], palette[0])
    _add_box(axis, (18.0, 3.05), 1.45, 1.15, "HOMO–LUMO\ngap (eV)", fills[1], palette[1])

    _add_arrow(axis, (1.8, 2.9), (2.3, 2.9))
    _add_arrow(axis, (4.17, 3.15), (4.78, 4.08), "arc3,rad=-0.08")
    _add_arrow(axis, (4.17, 2.64), (4.78, 1.72), "arc3,rad=0.08")
    _add_arrow(axis, (6.72, 1.45), (7.2, 1.45))
    _add_arrow(axis, (6.72, 4.25), (9.2, 3.35), "arc3,rad=0.08")
    _add_arrow(axis, (9.08, 1.62), (9.45, 2.4), "arc3,rad=-0.08")
    _add_arrow(axis, (11.02, 3.05), (11.35, 3.05))
    _add_arrow(axis, (13.15, 3.05), (13.5, 3.05))
    _add_arrow(axis, (15.1, 3.05), (15.4, 3.05))
    _add_arrow(axis, (16.9, 3.05), (17.25, 3.05))

    axis.text(
        7.0,
        0.53,
        "Geometry enters only through invariant pairwise distances; raw xyz coordinates are not passed to the network.",
        ha="center",
        va="center",
        fontsize=9.1,
        color="#48566A",
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": GRID_COLOR},
    )
    axis.text(5.75, 5.07, "Chemical pathway", ha="center", fontsize=8.5, color="#637084")
    axis.text(6.95, 2.13, "Geometric pathway", ha="center", fontsize=8.5, color="#637084")
    _save_figure(figure, output_stem)


def plot_model_overview(output_stem: str | Path) -> None:
    """Draw an intuitive molecule-to-scalar overview of the prediction task."""

    apply_plot_style()
    palette = sns.color_palette("Set2", n_colors=8)
    figure, axis = plt.subplots(figsize=(13.2, 4.25), constrained_layout=True)
    axis.set_xlim(0, 20)
    axis.set_ylim(0, 6.2)
    axis.axis("off")

    stages = [
        (0.25, 4.35, palette[0], "1  Molecular input"),
        (5.05, 4.15, palette[1], "2  Geometric GNN"),
        (9.65, 4.0, palette[2], "3  Prediction MLP"),
        (14.1, 5.65, palette[3], "4  Scalar gap"),
    ]
    for x, width, color, title in stages:
        panel = FancyBboxPatch(
            (x, 0.35),
            width,
            5.35,
            boxstyle="round,pad=0.04,rounding_size=0.15",
            facecolor=sns.light_palette(color, n_colors=8)[1],
            edgecolor=color,
            linewidth=1.2,
            zorder=0,
        )
        axis.add_patch(panel)
        axis.text(
            x + 0.25,
            5.38,
            title,
            ha="left",
            va="center",
            fontsize=10.0,
            fontweight="bold",
            color=TEXT_COLOR,
        )

    # A recognizable nine-heavy-atom molecule is shown without naming it so
    # the overview retains the intended "can you guess it?" prompt.
    molecule_coordinates = np.array(
        [
            [1.8243, -1.9450],
            [0.7981, -0.8510],
            [-0.6625, -1.1926],
            [-1.6886, -0.0985],
            [-3.1492, -0.4401],
            [-1.2542, 1.3372],
            [0.2064, 1.6788],
            [1.2325, 0.5848],
            [2.6931, 0.9264],
        ]
    )
    molecule_elements = ["C", "C", "C", "C", "O", "O", "C", "C", "O"]
    molecule_bonds = [
        (0, 1, 1.0),
        (1, 2, 1.5),
        (2, 3, 1.5),
        (3, 4, 2.0),
        (3, 5, 1.5),
        (5, 6, 1.5),
        (6, 7, 1.5),
        (7, 8, 1.0),
        (7, 1, 1.5),
    ]
    coordinate_center = (molecule_coordinates.min(axis=0) + molecule_coordinates.max(axis=0)) / 2
    mapped_atoms = np.column_stack(
        [
            2.4 + 0.56 * (molecule_coordinates[:, 0] - coordinate_center[0]),
            3.12 + 0.58 * (molecule_coordinates[:, 1] - coordinate_center[1]),
        ]
    )
    for first, second, order in molecule_bonds:
        start = mapped_atoms[first]
        end = mapped_atoms[second]
        offsets = [0.0]
        if order == 2.0:
            offsets = [-0.035, 0.035]
        delta = end - start
        normal = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
        for offset in offsets:
            shift = offset * normal
            axis.plot(
                [start[0] + shift[0], end[0] + shift[0]],
                [start[1] + shift[1], end[1] + shift[1]],
                color="#788699",
                linewidth=2.2,
                solid_capstyle="round",
                zorder=1,
            )
    atom_colors = {"C": "#39414D", "O": "#E05A5A"}
    atom_edges = {"C": "#202733", "O": "#B63D48"}
    atom_labels = {0: "CH₃", 4: "O", 5: "O", 8: "OH"}
    for index, (position, element) in enumerate(zip(mapped_atoms, molecule_elements, strict=True)):
        size = 86 if element == "C" else 205
        axis.scatter(
            position[0],
            position[1],
            s=size,
            facecolor=atom_colors[element],
            edgecolor=atom_edges[element],
            linewidth=1.25,
            zorder=3,
        )
        if index in atom_labels:
            axis.text(
                position[0],
                position[1],
                atom_labels[index],
                ha="center",
                va="center",
                color="white",
                fontsize=6.8,
                fontweight="bold",
                zorder=4,
            )
    axis.text(
        2.4,
        0.92,
        "Can you guess the molecule?\nC₆H₆O₃  •  9 heavy atoms",
        ha="center",
        fontsize=8.8,
    )

    # Complete directed graph with an emphasized incoming message aggregation.
    graph_center = np.array([7.0, 3.2])
    graph_offsets = np.column_stack(
        [
            0.45 * (molecule_coordinates[:, 0] - coordinate_center[0]),
            0.60 * (molecule_coordinates[:, 1] - coordinate_center[1]),
        ]
    )
    graph_nodes = graph_center + graph_offsets
    for index, start in enumerate(graph_nodes):
        for end in graph_nodes[index + 1 :]:
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="#AAB5C3",
                linewidth=0.65,
                alpha=0.55,
                zorder=1,
            )
    target_index = 5
    target = graph_nodes[target_index]
    for index in (0, 3, 7):
        _add_arrow(axis, tuple(graph_nodes[index]), tuple(target), "arc3,rad=0.10")
    node_colors = [atom_colors[element] for element in molecule_elements]
    for index, (position, color) in enumerate(zip(graph_nodes, node_colors, strict=True)):
        axis.scatter(
            position[0],
            position[1],
            s=175 if index == target_index else 125,
            facecolor=color,
            edgecolor=SELECTED_COLOR if index == target_index else "#526173",
            linewidth=2.0 if index == target_index else 1.0,
            zorder=3,
        )
    axis.text(
        7.0,
        1.05,
        "Complete directed graph\n6 message-passing blocks + Set2Set",
        ha="center",
        fontsize=8.8,
    )

    # Dense multilayer perceptron.
    layer_x = [10.25, 11.35, 12.45, 13.35]
    layer_counts = [5, 4, 3, 1]
    layer_colors = sns.color_palette("crest", n_colors=len(layer_x))
    layer_positions: list[list[tuple[float, float]]] = []
    for x, count in zip(layer_x, layer_counts, strict=True):
        ys = np.linspace(1.75, 4.35, count)
        layer_positions.append([(x, float(y)) for y in ys])
    for left, right in zip(layer_positions[:-1], layer_positions[1:], strict=True):
        for start in left:
            for end in right:
                axis.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    color="#AAB5C3",
                    linewidth=0.62,
                    alpha=0.62,
                    zorder=1,
                )
    for color, positions in zip(layer_colors, layer_positions, strict=True):
        for x, y in positions:
            axis.scatter(
                x,
                y,
                s=105,
                facecolor=color,
                edgecolor="white",
                linewidth=1.0,
                zorder=3,
            )
    axis.text(11.8, 1.05, "2d → d → d/2 → 1", ha="center", fontsize=9.0, fontweight="bold")

    # Scalar output and conventional HOMO/LUMO energy-level representation.
    output_color = sns.color_palette("deep")[4]
    axis.text(
        15.35,
        4.75,
        "One predicted scalar",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#59677A",
    )
    axis.text(
        15.35,
        4.2,
        "ΔEₕₗ  (eV)",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=output_color,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": output_color},
    )
    homo_y, lumo_y = 1.65, 3.25
    axis.plot([16.7, 18.8], [homo_y, homo_y], color="#273244", linewidth=2.2)
    axis.plot([16.7, 18.8], [lumo_y, lumo_y], color="#273244", linewidth=2.2)
    axis.text(18.9, homo_y, "HOMO", ha="left", va="center", fontsize=9.0)
    axis.text(18.9, lumo_y, "LUMO", ha="left", va="center", fontsize=9.0)
    axis.text(17.05, homo_y + 0.10, "↑↓", ha="center", va="bottom", fontsize=11)
    axis.annotate(
        "",
        xy=(18.15, lumo_y - 0.05),
        xytext=(18.15, homo_y + 0.05),
        arrowprops={"arrowstyle": "<->", "color": output_color, "linewidth": 1.8},
    )
    axis.text(
        18.02,
        (homo_y + lumo_y) / 2,
        "ΔEₕₗ",
        ha="right",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color=output_color,
    )
    axis.annotate(
        "Energy",
        xy=(16.35, 3.7),
        xytext=(16.35, 1.2),
        ha="center",
        va="bottom",
        rotation=90,
        fontsize=8.2,
        color="#637084",
        arrowprops={"arrowstyle": "-|>", "color": "#637084", "linewidth": 1.0},
    )

    _add_arrow(axis, (4.62, 3.0), (5.0, 3.0))
    _add_arrow(axis, (9.22, 3.0), (9.6, 3.0))
    _add_arrow(axis, (13.7, 3.0), (14.08, 3.0))
    _save_figure(figure, output_stem)
