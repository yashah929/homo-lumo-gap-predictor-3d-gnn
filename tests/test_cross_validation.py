import json
from pathlib import Path

from qm9_gap.cross_validation import array_index_to_experiment, grid_configurations, summarize_cv


GRID = {
    "num_message_passing_layers": [3, 4, 5, 6],
    "hidden_dim": [64, 128, 256],
    "learning_rate": [0.0001, 0.0003, 0.001],
    "num_folds": 4,
}


def test_grid_has_exactly_36_configurations_and_144_array_tasks() -> None:
    configurations = grid_configurations(GRID)
    assert len(configurations) == 36
    assert array_index_to_experiment(0, GRID) == (configurations[0], 0)
    assert array_index_to_experiment(143, GRID) == (configurations[-1], 3)


def test_cv_aggregation_selects_by_mean_mae_and_rounds_median_epoch(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    configurations = grid_configurations(GRID)
    winning_epochs = [2, 3, 8, 9]
    for config_index, configuration in enumerate(configurations):
        for fold in range(4):
            mae = 0.1 + 0.001 * fold if config_index == 0 else 1.0 + config_index
            record = {
                **configuration,
                "fold": fold,
                "seed": 42 + fold,
                "best_epoch": winning_epochs[fold] if config_index == 0 else 10,
                "best_validation_mae_ev": mae,
                "validation_rmse_ev": mae * 1.2,
                "validation_r2": 0.8,
                "checkpoint_path": "ignored.pt",
                "curve_path": "ignored.csv",
            }
            path = runs / f"{configuration['configuration_id']}_fold_{fold}.json"
            path.write_text(json.dumps(record), encoding="utf-8")
    detailed, summary, selected = summarize_cv(tmp_path)
    assert len(detailed) == 144
    assert len(summary) == 36
    assert selected["configuration_id"] == "cfg_000"
    assert selected["fold_best_epochs"] == winning_epochs
    assert selected["final_epochs"] == 6
