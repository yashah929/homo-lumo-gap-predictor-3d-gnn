"""Physical-unit regression metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(reference_ev: np.ndarray, prediction_ev: np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE, and R2 after flattening arrays."""
    reference = np.asarray(reference_ev, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction_ev, dtype=np.float64).reshape(-1)
    return {
        "mae_ev": float(mean_absolute_error(reference, prediction)),
        "rmse_ev": float(np.sqrt(mean_squared_error(reference, prediction))),
        "r2": float(r2_score(reference, prediction)),
    }
