# SLURM execution

The scripts avoid site-specific accounts, partitions, modules, and filesystem paths. Resource values in the headers are conservative defaults; override them with `sbatch` options according to the cluster policy.

Define:

```bash
export PROJECT_DIR=/absolute/path/to/homo-lumo-gap-predictor-3d-gnn
export ENV_ACTIVATE=/absolute/path/to/venv/bin/activate
export CONFIG_PATH=configs/base.yaml  # data and result roots are specified here
```

On Cannon, use `ENV_ACTIVATE=$PROJECT_DIR/slurm/cannon_env.sh`. The ignored `data`, `artifacts`, and `logs` paths should point to spacious project storage; see `docs/cannon.md`.

Run the development-only real-QM9 GPU smoke test before full enriched preprocessing:

```bash
sbatch --account=<account> --partition=<gpu-partition> slurm/qm9_gpu_smoke.sbatch
```

Prepare data and validate the immutable splits once on a CPU compute node:

```bash
cd "$PROJECT_DIR"
source "$ENV_ACTIVATE"
python scripts/prepare_data.py
python scripts/create_splits.py
```

The equivalent batch entry point is `slurm/prepare_data.sbatch`.

Submit all 144 CV experiments as one array. Array index `a` maps to configuration `floor(a/4)` and fold `a mod 4`:

```bash
sbatch --account=<account> --partition=<gpu-partition> --gres=gpu:1 slurm/cv_array.sbatch
```

After every array task succeeds, aggregate and select on a CPU/login node:

```bash
python scripts/summarize_cv.py
```

Then submit fresh final retraining and, only after it succeeds, the explicit locked evaluation:

```bash
sbatch --account=<account> --partition=<gpu-partition> --gres=gpu:1 slurm/train_final.sbatch
sbatch --account=<account> --partition=<gpu-partition> --gres=gpu:1 slurm/evaluate_test.sbatch
```

For a shared filesystem, all jobs must use the same `PROJECT_DIR`. Set `NUM_WORKERS` only by editing a copied YAML configuration; configuration snapshots are saved with results.
