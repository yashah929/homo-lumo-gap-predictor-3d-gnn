#!/bin/bash
#SBATCH --job-name=qm9-gap-test
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-test-%j.out

set -euo pipefail
: "${PROJECT_DIR:?Set PROJECT_DIR to the repository path}"
: "${ENV_ACTIVATE:?Set ENV_ACTIVATE to the environment activation script}"
cd "$PROJECT_DIR"
source "$ENV_ACTIVATE"
python scripts/evaluate_test.py --config "${CONFIG_PATH:-configs/base.yaml}" --confirm-final-test
