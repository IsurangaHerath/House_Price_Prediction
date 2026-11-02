"""Regression metrics for the housing project.

The target is stored in units of $100,000, so every metric is also
reported in dollars to make the error magnitudes interpretable.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET_UNIT_USD = 100_000


def regression_metrics(y_true, y_pred) -> dict:
    """Compute MAE, RMSE and R^2 for a set of predictions.

    Parameters
    ----------
    y_true:
        Ground-truth target values (units of $100,000).
    y_pred:
        Model predictions in the same units.

    Returns
    -------
    dict
        Keys: mae, rmse, r2 (dataset units), mae_usd, rmse_usd (dollars),
        n_samples.

    Notes
    -----
    MAE is the average absolute error. RMSE is the square root of the
    mean squared error, so large errors count more heavily. R^2 compares
    the model against always predicting the training mean (1.0 is
    perfect, 0.0 equals the mean baseline, negative is worse than it).
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
        "mae_usd": float(mae) * TARGET_UNIT_USD,
        "rmse_usd": rmse * TARGET_UNIT_USD,
        "n_samples": int(len(np.asarray(y_true))),
    }
