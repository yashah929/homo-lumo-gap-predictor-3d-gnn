from pathlib import Path

import numpy as np

from qm9_gap.data import load_cv_fold
from qm9_gap.splits import create_fixed_splits, validate_fixed_splits


def test_split_isolation_coverage_and_determinism(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    create_fixed_splits(103, first, seed=42)
    create_fixed_splits(103, second, seed=42)
    validate_fixed_splits(103, first)
    for name in ["development_indices", "final_test_indices", *[f"cv_fold_{i}_validation_indices" for i in range(4)]]:
        assert np.array_equal(np.load(first / f"{name}.npy"), np.load(second / f"{name}.npy"))
    test = np.load(first / "final_test_indices.npy")
    for fold in range(4):
        training, validation = load_cv_fold(first, fold)
        assert np.intersect1d(training, validation).size == 0
        assert np.intersect1d(training, test).size == 0
        assert np.intersect1d(validation, test).size == 0
