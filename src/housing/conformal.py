"""Calibrate prediction ranges with split conformal prediction."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import load_dataset, validate_frame
from .models import RANDOM_STATE, build_models
from .train import split_train_test
from .uncertainty import as_frame, interval_coverage, prediction_interval

COVERAGE = 0.80
CALIBRATION_SHARE = 0.25


def split_calibration(X, y, calibration_share=CALIBRATION_SHARE):
    """Split training rows into a fitting set and a calibration set."""
    return train_test_split(
        X,
        y,
        test_size=calibration_share,
        random_state=RANDOM_STATE,
    )


def conformal_quantile(residuals, coverage=COVERAGE):
    """Return the finite-sample conformal quantile of absolute residuals."""
    residuals = np.abs(np.asarray(residuals, dtype=float))
    n = len(residuals)

    if n == 0:
        raise ValueError("Need at least one calibration residual")

    # The small-sample correction keeps coverage at or above the target.
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(residuals, level))


def conformal_interval(model, features, half_width):
    """Return a prediction plus/minus the calibrated half-width."""
    features = as_frame(features)
    prediction = model.predict(features)

    return pd.DataFrame(
        {
            "prediction": prediction,
            "low": prediction - half_width,
            "high": prediction + half_width,
        }
    )


def summarize(intervals, y_true):
    """Return coverage and width statistics for one interval method."""
    low = intervals["low"].to_numpy()
    high = intervals["high"].to_numpy()
    y_true = np.asarray(y_true)
    width = high - low

    # Use positional masks so every array lines up.
    capped = y_true >= 5.0

    return {
        "coverage": interval_coverage(y_true, low, high),
        "mean_width": float(width.mean()),
        "median_width": float(np.median(width)),
        "cap_coverage": interval_coverage(
            y_true[capped],
            low[capped],
            high[capped],
        ),
        "cap_mean_width": float(width[capped].mean()),
    }


def main():
    """Compare conformal and tree-based ranges on the holdout set."""
    df = load_dataset()
    validate_frame(df)

    X_train, X_test, y_train, y_test = split_train_test(df)
    X_fit, X_cal, y_fit, y_cal = split_calibration(X_train, y_train)

    model = build_models()["random_forest"]
    model.fit(X_fit, y_fit)

    # Calibrate on rows the model did not see during fitting.
    residuals = np.abs(y_cal - model.predict(X_cal))
    half_width = conformal_quantile(residuals, COVERAGE)

    conformal = conformal_interval(model, X_test, half_width)
    trees = prediction_interval(model, X_test)

    rows = []
    for name, frame in (
        ("Conformal 80%", conformal),
        ("Tree 10th-90th", trees),
    ):
        rows.append({"method": name, **summarize(frame, y_test)})

    table = pd.DataFrame(rows)

    print(f"Fit rows: {len(X_fit):,} | calibration rows: {len(X_cal):,}")
    print(f"Conformal half-width: {half_width:.4f}")
    print()
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
