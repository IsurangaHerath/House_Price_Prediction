"""Score regression predictions."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import TARGET_UNIT_USD


def regression_metrics(y_true, y_pred):
    """Return MAE, RMSE, R2, and the dollar versions of MAE and RMSE."""
    # Average absolute error.
    mae = mean_absolute_error(y_true, y_pred)

    # Square root of the average squared error.
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # How much better the model is than always predicting the mean.
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        # Convert errors back to dollars.
        "mae_usd": float(mae) * TARGET_UNIT_USD,
        "rmse_usd": float(rmse) * TARGET_UNIT_USD,
        "n_samples": int(len(y_true)),
    }


def residual_summary(y_true, y_pred):
    """Describe the size and direction of prediction errors."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Positive means the true value is higher than the prediction.
    residuals = y_true - y_pred

    return {
        "mean_residual": float(residuals.mean()),
        "median_residual": float(np.median(residuals)),
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "underprediction_rate": float((residuals > 0).mean()),
        "worst_overprediction": float(residuals.min()),
        "worst_underprediction": float(residuals.max()),
    }


def residual_summary_by_cap(y_true, y_pred, cap=5.0):
    """Compare errors for rows at the target cap and below it."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred
    capped = y_true >= cap

    return {
        "capped_rows": int(capped.sum()),
        "capped_mae": (
            float(np.mean(np.abs(residuals[capped])))
            if capped.any()
            else None
        ),
        "uncapped_mae": (
            float(np.mean(np.abs(residuals[~capped])))
            if (~capped).any()
            else None
        ),
    }
