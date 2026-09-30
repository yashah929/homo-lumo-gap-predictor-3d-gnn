# Generated results

The experiment pipeline writes commit-ready summaries here:

- `cv/cv_results.csv`: one row per configuration and fold;
- `cv/cv_summary.csv`: four-fold aggregate metrics;
- `cv/selected_configuration.yaml`: winning hyperparameters and final epoch count;
- `final/test_metrics.json`: locked-test MAE, RMSE, and \(R^2\);
- `final/test_predictions.csv`: molecule-level references, predictions, and residuals;
- `figures/`: PDF and PNG diagnostics.

Checkpoint, atomic run-record, training-curve, and scheduler-output directories are ignored by Git. No numerical result file should be represented as a full-QM9 result unless the documented CV, fresh final retraining, and explicit locked evaluation have completed.
