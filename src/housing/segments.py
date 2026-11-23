"""Measure prediction errors across groups of block groups."""

import numpy as np
import pandas as pd

from .data import load_dataset, validate_frame
from .evaluate import regression_metrics
from .models import build_models
from .train import split_train_test

# Each segment is a column, its edges, and readable labels.
SEGMENTS = {
    "Income": {
        "column": "MedInc",
        "bins": [0, 2, 3.5, 5, np.inf],
        "labels": ["under $20k", "$20k-$35k", "$35k-$50k", "$50k+"],
    },
    "House age": {
        "column": "HouseAge",
        "bins": [0, 10, 20, 30, 40, np.inf],
        "labels": ["under 10", "10-19", "20-29", "30-39", "40+"],
    },
    "Occupancy": {
        "column": "AveOccup",
        "bins": [0, 1, 2, 3, np.inf],
        "labels": ["under 1", "1-1.9", "2-2.9", "3+"],
    },
    "Latitude": {
        "column": "Latitude",
        "bins": [32, 34, 36, 38, 43],
        "labels": ["under 34", "34-35.9", "36-37.9", "38+"],
    },
}


def add_segment(X, column, bins, labels):
    """Label each row with the segment it falls into."""
    return pd.cut(X[column], bins=bins, labels=labels, right=False)


def segment_errors(y_true, y_pred, segments):
    """Score each segment and return one row per group."""
    frame = pd.DataFrame(
        {
            "segment": segments,
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
        }
    )

    rows = []
    for name, group in frame.groupby("segment", observed=True):
        metrics = regression_metrics(group["y_true"], group["y_pred"])

        # Positive bias means the model is under-predicting this group.
        metrics["bias"] = float(
            (group["y_true"] - group["y_pred"]).mean()
        )
        rows.append({"segment": str(name), **metrics})

    return pd.DataFrame(rows)


def main():
    """Score the tuned forest on holdout subgroups."""
    df = load_dataset()
    validate_frame(df)

    X_train, X_test, y_train, y_test = split_train_test(df)

    model = build_models()["random_forest"]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    overall = regression_metrics(y_test, predictions)
    print("Overall holdout errors:")
    print(
        f"  MAE: {overall['mae']:.4f} | "
        f"RMSE: {overall['rmse']:.4f} | "
        f"R2: {overall['r2']:.4f}"
    )

    for title, spec in SEGMENTS.items():
        segments = add_segment(
            X_test,
            spec["column"],
            spec["bins"],
            spec["labels"],
        )
        table = segment_errors(y_test, predictions, segments)

        print()
        print(f"{title}:")
        print(
            table[["segment", "n_samples", "mae", "rmse", "r2", "bias"]]
            .to_string(index=False, float_format=lambda value: f"{value:.4f}")
        )


if __name__ == "__main__":
    main()
