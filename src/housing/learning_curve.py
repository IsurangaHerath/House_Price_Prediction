"""Measure how much training data the model needs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, learning_curve

from .data import load_dataset, validate_frame
from .models import RANDOM_STATE, build_models
from .train import split_train_test

FIGURE_DIR = Path("reports/figures")
TRAIN_SIZES = np.linspace(0.2, 1.0, 5)

# Keep the same 80/20 split convention as the rest of the project.
CV = ShuffleSplit(n_splits=3, test_size=0.2, random_state=RANDOM_STATE)


def learning_curve_table(model, X, y, train_sizes=TRAIN_SIZES, cv=CV):
    """Return training and validation MAE at each training size."""
    sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        train_sizes=train_sizes,
        cv=cv,
        scoring="neg_mean_absolute_error",
        shuffle=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # sklearn returns negative errors, so flip the sign.
    return pd.DataFrame(
        {
            "train_rows": sizes.astype(int),
            "train_mae": -train_scores.mean(axis=1),
            "train_std": train_scores.std(axis=1),
            "val_mae": -val_scores.mean(axis=1),
            "val_std": val_scores.std(axis=1),
        }
    )


def save_learning_curve_plot(table, output_dir=FIGURE_DIR):
    """Save training and validation MAE against the number of rows."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.plot(
        table["train_rows"],
        table["train_mae"],
        marker="o",
        label="Training",
    )
    plt.plot(
        table["train_rows"],
        table["val_mae"],
        marker="o",
        label="Validation",
    )

    # Show the spread across folds.
    plt.fill_between(
        table["train_rows"],
        table["train_mae"] - table["train_std"],
        table["train_mae"] + table["train_std"],
        alpha=0.15,
    )
    plt.fill_between(
        table["train_rows"],
        table["val_mae"] - table["val_std"],
        table["val_mae"] + table["val_std"],
        alpha=0.15,
    )

    plt.xlabel("Training rows")
    plt.ylabel("MAE")
    plt.title("Learning curve")
    plt.legend()
    plt.tight_layout()

    path = output_dir / "learning_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def main():
    """Build the learning curve on the training split."""
    df = load_dataset()
    validate_frame(df)

    X_train, _, y_train, _ = split_train_test(df)

    model = build_models()["random_forest"]
    table = learning_curve_table(model, X_train, y_train)

    print("Learning curve on the training split:")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    path = save_learning_curve_plot(table)
    print()
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
