#!/bin/bash
module load Mambaforge/23.11.0-fasrc01
conda activate /n/holylabs/drliu_lab/Lab/yshah/conda/envs/qm9-gap-gnn

QM9_MPL_CACHE="${TMPDIR:-/tmp}/qm9-gap-matplotlib-${SLURM_JOB_ID:-interactive}"
mkdir -p "$QM9_MPL_CACHE"
export MPLCONFIGDIR="$QM9_MPL_CACHE"
