# Generated results

The experiment pipeline writes commit-ready summaries here:

- `cv/cv_results.csv`: one row per configuration and fold;
- `cv/cv_summary.csv`: four-fold aggregate metrics;
- `cv/selected_cv_training_curves.csv`: unsmoothed displayed path vertices for the selected folds;
- `cv/selected_configuration.yaml`: winning hyperparameters and final epoch count;
- `final/test_metrics.json`: locked-test MAE, RMSE, and \(R^2\);
- `final/test_predictions.csv`: molecule-level references, predictions, and residuals;
- `figures/`: matching PDF and 300 dpi PNG diagnostics plus the architecture schematic.

The original per-epoch CV curve directories were intentionally excluded from the compact result bundle. `selected_cv_training_curves.csv` preserves the path vertices from the originally committed vector figure for `cfg_034`; exact fold-best epochs and validation MAEs come from `cv_results.csv`. It supports deterministic presentation-only regeneration without rerunning cross-validation.

Checkpoints are written below the separately configured `artifacts/checkpoints/` root. Atomic run-record, training-curve, and scheduler-output directories are ignored by Git but remain part of the compact transfer bundle. No numerical result file should be represented as a full-QM9 result unless the documented CV, fresh final retraining, and explicit locked evaluation have completed.

## Completed experiment

The 144-run cross-validation study selected `cfg_034` (six message-passing layers, hidden dimension 256, Adam learning rate \(3\times10^{-4}\)). Its mean validation MAE was \(0.051359 \pm 0.001002\) eV, with mean RMSE 0.086402 eV and mean \(R^2=0.995452\). The fresh final model was trained for 279 epochs on all 104,664 development molecules.

The single evaluation on the 26,167-molecule locked test set produced MAE 0.044720 eV, RMSE 0.071658 eV, and \(R^2=0.996884\). The molecule-level prediction file uses zero-based indices into the retained 130,831-molecule PyG QM9 dataset. `scripts/validate_results.py` verifies split hashes, disjointness, prediction identifiers and finiteness, metric recomputation, figure presence, checkpoint identity, and manifest consistency.
