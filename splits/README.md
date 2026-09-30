# Fixed dataset partitions

The `.npy` arrays in this directory are generated deterministically with seed 42 by `scripts/create_splits.py`. They refer to indices in the processed PyG QM9 dataset and are intended for version control.

- `development_indices.npy`: 80% model-development set.
- `final_test_indices.npy`: locked 20% final test set.
- `cv_fold_K_validation_indices.npy`: validation members for development fold `K`.
- `split_manifest.json`: counts, algorithm, and SHA-256 hashes.

Do not regenerate these files during a study. If the upstream dataset identity changes, create a separately versioned split definition rather than silently replacing the arrays.
