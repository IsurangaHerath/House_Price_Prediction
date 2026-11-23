"""Inspect the model's largest holdout errors."""

from pathlib import Path

import numpy as np
import pandas as pd

from .data import TARGET_COLUMN, load_dataset, validate_frame
from .models import build_models
from .train import split_train_test

REPORT_DIR = Path("reports")
WORST_CSV = REPORT_DIR / "worst_predictions.csv"
N_WORST = 20


def worst_predictions(X, y_true, y_pred, n=N_WORST):
    """Return the n rows with the largest absolute error."""
    frame = X.copy()
    frame[TARGET_COLUMN] = np.asarray(y_true)
    frame["prediction"] = np.asarray(y_pred)

    # Positive error means the model under-predicted the true value.
    frame["error"] = frame[TARGET_COLUMN] - frame["prediction"]
    frame["abs_error"] = frame["error"].abs()

    return (
        frame.sort_values("abs_error", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def main():
    """Save and summarize the worst holdout predictions."""
    df = load_dataset()
    validate_frame(df)

    X_train, X_test, y_train, y_test = split_train_test(df)

    model = build_models()["random_forest"]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    worst = worst_predictions(X_test, y_test, predictions)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    worst.to_csv(WORST_CSV, index=False)

    capped = int((worst[TARGET_COLUMN] >= 5.0).sum())
    under = int((worst["error"] > 0).sum())
    small = int((worst["Population"] < 200).sum())
    imputed = int(worst["AveBedrms"].isna().sum())

    print(f"Worst {len(worst)} holdout predictions:")
    print(f"  mean absolute error: {worst['abs_error'].mean():.4f}")
    print(f"  rows at the target cap: {capped}")
    print(f"  under-predictions: {under}")
    print(f"  rows with population under 200: {small}")
    print(f"  rows with imputed bedrooms: {imputed}")
    print(f"  saved {WORST_CSV}")
    print()
    print(
        worst[
            [
                "MedInc",
                "AveOccup",
                "Latitude",
                "Longitude",
                TARGET_COLUMN,
                "prediction",
                "error",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
