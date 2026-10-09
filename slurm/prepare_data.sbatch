#!/bin/bash
#SBATCH --job-name=qm9-gap-prepare
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-prepare-%j.out

set -euo pipefail
: "${PROJECT_DIR:?Set PROJECT_DIR to the repository path}"
: "${ENV_ACTIVATE:?Set ENV_ACTIVATE to the environment activation script}"
cd "$PROJECT_DIR"
source "$ENV_ACTIVATE"
python scripts/prepare_data.py --config "${CONFIG_PATH:-configs/base.yaml}"
python scripts/create_splits.py --config "${CONFIG_PATH:-configs/base.yaml}"
