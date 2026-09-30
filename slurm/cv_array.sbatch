#!/bin/bash
#SBATCH --job-name=qm9-gap-cv
#SBATCH --array=0-143
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail
: "${PROJECT_DIR:?Set PROJECT_DIR to the repository path}"
: "${ENV_ACTIVATE:?Set ENV_ACTIVATE to the environment activation script}"
cd "$PROJECT_DIR"
source "$ENV_ACTIVATE"
python scripts/run_cv.py --config "${CONFIG_PATH:-configs/base.yaml}" --array-index "$SLURM_ARRAY_TASK_ID"
