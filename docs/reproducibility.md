# Reproducibility protocol

## Deterministic state

The default experimental seed is 42. Python, NumPy, PyTorch CPU, and every visible CUDA generator are seeded. PyTorch deterministic algorithms are requested with `warn_only=True`; cuDNN benchmarking is disabled and deterministic cuDNN behavior is enabled. The final development-set model uses the separately documented initialization seed 4242.

Some GPU scatter/reduction kernels and version-dependent vendor libraries may remain nondeterministic. Warnings emitted by PyTorch should be retained with run records. Exact bitwise agreement is therefore most defensible within the same hardware and software environment; scientific comparisons should use recorded eV metrics and tolerances.

## Data identity

`scripts/prepare_data.py` downloads QM9 through PyG, reads the original SDF with explicit hydrogen retention, verifies atom-order identity, creates complete directed graphs, and writes an ignored processed cache. Its metadata records molecule count, target identity, maximum pairwise distance, and RBF-domain policy. Processing fails if the PyG usable count differs from 130,831.

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

Commit source, configuration, split arrays, compact CSV/JSON/YAML records, test predictions, and PDF/PNG figures. Do not commit downloaded data, processed graph caches, checkpoints, scheduler output, or verbose logs. Before publication, record the environment with `python -m pip freeze`, retain the Git commit used for training, run `pytest`, and replace only the clearly marked README result placeholders after the final test command.
