# Cannon execution environment

This project uses the `drliu_lab` Slurm account with the normal QOS. At setup time, the dedicated `gpu` partition's A100 nodes were unavailable, while `gpu_h200` was available. Production CV therefore targets one NVIDIA H200 per array element on `gpu_h200`.

The reproducible environment is located at:

```text
/n/holylabs/drliu_lab/Lab/yshah/conda/envs/qm9-gap-gnn
```

It was created with:

```bash
module load Mambaforge/23.11.0-fasrc01
conda create -y -p /n/holylabs/drliu_lab/Lab/yshah/conda/envs/qm9-gap-gnn python=3.12 pip
/n/holylabs/drliu_lab/Lab/yshah/conda/envs/qm9-gap-gnn/bin/python -m pip install \
  torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
/n/holylabs/drliu_lab/Lab/yshah/conda/envs/qm9-gap-gnn/bin/python -m pip install \
  torch-geometric==2.8.0.post1 rdkit==2026.3.6 -e '.[test]'
```

`slurm/cannon_env.sh` activates this environment and gives Matplotlib a job-local writable cache. The repository's ignored `data`, `artifacts`, and `logs` paths are symbolic links to `/n/holylabs/drliu_lab/Lab/yshah/qm9-gap-gnn/`, keeping raw data and large checkpoints off home storage.

Portable results remain under `results/`. Retrieve them with:

```bash
rsync -avh --progress \
  yshah@login.rc.fas.harvard.edu:/n/home13/yshah/experiment_repos/gnn_training/results/ \
  /LOCAL/PATH/qm9-gap-gnn/results/
```

The final checkpoint is intentionally separate. After final retraining, copy the exact path recorded in `results/run_manifest.json` only if a local archive is wanted.

## Completed production jobs

The production cross-validation array was job `49527094`; operational retries were submitted as job `50344654`. All 144 configuration-fold runs completed. Final development-set training was job `51427700` and completed in 01:18:34 on an NVIDIA H200. The guarded locked-test evaluation was job `51444636`; it completed with exit code `0:0` in 21 seconds on `holygpu8a12402`, also using an NVIDIA H200.

The final evaluation produced 26,167 predictions with MAE 0.044720 eV, RMSE 0.071658 eV, and \(R^2=0.996884\). Scheduler output remains excluded from version control; Slurm accounting and environment metadata are summarized in `results/run_manifest.json`.
