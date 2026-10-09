#!/bin/bash
#SBATCH --job-name=qm9-gap-smoke
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-smoke-%j.out

set -euo pipefail
: "${PROJECT_DIR:?Set PROJECT_DIR to the repository path}"
: "${ENV_ACTIVATE:?Set ENV_ACTIVATE to the environment activation script}"
cd "$PROJECT_DIR"
source "$ENV_ACTIVATE"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
if [[ "${USE_PREPARED_CACHE:-0}" == "1" ]]; then
  python scripts/smoke_test_qm9.py --config "${CONFIG_PATH:-configs/base.yaml}" --prepared-cache
else
  python scripts/smoke_test_qm9.py --config "${CONFIG_PATH:-configs/base.yaml}"
fi
