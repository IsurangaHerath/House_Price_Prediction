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
