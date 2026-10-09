# Reproducibility protocol

## Deterministic state

The default experimental seed is 42. Python, NumPy, PyTorch CPU, and every visible CUDA generator are seeded. PyTorch deterministic algorithms are requested with `warn_only=True`; cuDNN benchmarking is disabled and deterministic cuDNN behavior is enabled. The final development-set model uses the separately documented initialization seed 4242.

Some GPU scatter/reduction kernels and version-dependent vendor libraries may remain nondeterministic. Warnings emitted by PyTorch should be retained with run records. Exact bitwise agreement is therefore most defensible within the same hardware and software environment; scientific comparisons should use recorded eV metrics and tolerances.

## Data identity

`scripts/prepare_data.py` downloads QM9 through PyG, reads the original SDF with explicit hydrogen retention, applies all RDKit sanitization operations except strict property/valence validation, verifies atom-order identity, creates complete directed graphs, and writes an ignored processed cache. This matches PyG's need to admit retained QM9 records with unusual formal valences while still assigning the chemical features used here. Its metadata records molecule count, target identity, sanitization policy, maximum pairwise distance, and RBF-domain policy. Processing fails if the PyG usable count differs from 130,831.

The maximum observed pairwise distance is 12.040427 Å, from the terminal explicit hydrogens of n-nonane. Because the scientific design fixes 50 RBF centers on 0–10 Å and specifies evaluation without clipping, the base configuration warns rather than aborts for this known exceedance.

Versioned files in `splits/` define the development set, locked final test set, and four validation folds. `split_manifest.json` records the generation algorithm, counts, and SHA-256 checksum of every array. Existing splits are not replaced unless `--force` is supplied.

## Run provenance

Every CV and final run records:

- all random seeds and hyperparameters;
- target mean and population standard deviation;
- split identity and fold;
- Python, PyTorch, PyTorch Geometric, RDKit, and CUDA versions;
- GPU model, hostname, platform, UTC timestamp, and Git commit;
- checkpoint and training-curve paths.

Atomic CV JSON files are written independently to avoid concurrent writes from the SLURM array. Aggregation requires all 144 files.

## Isolation safeguards

CV loads `development_indices.npy` and one validation-fold file. It does not read `final_test_indices.npy` and constructs no test loader. Final training reads only development indices, uses no validation loader, and loads no CV weights. The final evaluation command is the only training workflow component that reads test indices; it requires `--confirm-final-test` and rejects a checkpoint not marked as final development retraining.

## Archiving a completed study

Commit source, configuration, split arrays, compact CSV/JSON/YAML records, test predictions, and PDF/PNG figures. Do not commit downloaded data, processed graph caches, checkpoints, scheduler output, or verbose logs. The completed run manifest records the package freeze, training Git base commit, Slurm job identities, split hashes, selected configuration, final checkpoint identity, and held-out metrics.

Before publication, run:

```bash
python scripts/validate_results.py
pytest
python -m pip check
```

The result validator independently recomputes metrics from the prediction CSV and checks that its unique molecule identifiers equal the immutable final-test split with no development-set overlap. During final validation, an identifier-only output issue was corrected: PyG had interpreted the legacy `molecule_index` graph attribute as a node-index tensor and added batching offsets. The completed non-shuffled loader order and reference values provided a deterministic alignment back to the immutable test indices. Reference values, predictions, residuals, and aggregate metrics were not changed. Graph metadata now use `molecule_id` and `raw_qm9_id`, which PyG does not increment during batching; the original and corrected prediction-file hashes are retained in `results/run_manifest.json`.
