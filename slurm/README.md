# SLURM execution

The scripts avoid site-specific accounts, partitions, modules, and filesystem paths. Resource values in the headers are conservative defaults; override them with `sbatch` options according to the cluster policy.

Define:

```bash
export PROJECT_DIR=/absolute/path/to/homo-lumo-gap-predictor-3d-gnn
export ENV_ACTIVATE=/absolute/path/to/venv/bin/activate
export CONFIG_PATH=configs/base.yaml  # data and result roots are specified here
```

Prepare data and create the immutable splits once, preferably on a data-transfer or CPU node:

```bash
cd "$PROJECT_DIR"
source "$ENV_ACTIVATE"
python scripts/prepare_data.py
python scripts/create_splits.py
```

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
