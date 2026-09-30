"""Deterministic, versionable development/test and CV split generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import save_json, sha256_file


def create_fixed_splits(num_molecules: int, output_dir: str | Path, seed: int = 42) -> dict[str, object]:
    """Write an 80/20 global split and four exhaustive development folds."""
    if num_molecules < 5:
        raise ValueError("At least five molecules are required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(num_molecules).astype(np.int64)
    development_size = int(np.floor(0.8 * num_molecules))
    development = np.sort(permutation[:development_size])
    final_test = np.sort(permutation[development_size:])
    files: list[Path] = []
    development_path = output / "development_indices.npy"
    test_path = output / "final_test_indices.npy"
    np.save(development_path, development, allow_pickle=False)
    np.save(test_path, final_test, allow_pickle=False)
    files.extend((development_path, test_path))
    fold_order = generator.permutation(development)
    folds = np.array_split(fold_order, 4)
    for fold, indices in enumerate(folds):
        path = output / f"cv_fold_{fold}_validation_indices.npy"
        np.save(path, np.sort(indices), allow_pickle=False)
        files.append(path)
    manifest: dict[str, object] = {
        "identity": "seed42_80-20_development_test_fourfold_v1",
        "seed": seed,
        "algorithm": "numpy.default_rng permutation; floor(0.8*N) development; numpy.array_split four folds",
        "num_molecules": num_molecules,
        "num_development": int(development.size),
        "num_final_test": int(final_test.size),
        "fold_sizes": [int(fold.size) for fold in folds],
        "files": {path.name: sha256_file(path) for path in files},
    }
    save_json(manifest, output / "split_manifest.json")
    return manifest


def validate_fixed_splits(num_molecules: int, splits_dir: str | Path) -> None:
    """Raise if coverage, disjointness, or four-fold isolation is violated."""
    root = Path(splits_dir)
    development = np.load(root / "development_indices.npy", allow_pickle=False)
    test = np.load(root / "final_test_indices.npy", allow_pickle=False)
    if np.intersect1d(development, test).size:
        raise ValueError("Development and final-test sets overlap")
    if not np.array_equal(np.sort(np.concatenate((development, test))), np.arange(num_molecules)):
        raise ValueError("Development/test union does not equal the retained dataset")
    folds = [np.load(root / f"cv_fold_{fold}_validation_indices.npy", allow_pickle=False) for fold in range(4)]
    for left in range(4):
        for right in range(left + 1, 4):
            if np.intersect1d(folds[left], folds[right]).size:
                raise ValueError(f"CV validation folds {left} and {right} overlap")
    if not np.array_equal(np.sort(np.concatenate(folds)), np.sort(development)):
        raise ValueError("Four CV validation folds do not partition the development set")
    if any(np.intersect1d(fold, test).size for fold in folds):
        raise ValueError("A CV validation fold contains a final-test molecule")
