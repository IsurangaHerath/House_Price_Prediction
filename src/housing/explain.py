"""Explain which features drive the model."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .data import load_dataset, validate_frame
from .models import RANDOM_STATE, build_models
from .train import split_train_test

REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"
IMPORTANCE_CSV = REPORT_DIR / "feature_importance.csv"

PDP_SAMPLE = 400


def permutation_importance_table(model, X, y, n_repeats=5):
    """Return mean and std permutation importance for each feature."""
    result = permutation_importance(
        model,
        X,
        y,
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    table = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )

    # Largest effect first.
    return table.sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(drop=True)


def partial_dependence_curve(
    model,
    X,
    feature,
    grid_size=20,
    sample_size=PDP_SAMPLE,
):
    """Average prediction as one feature changes, holding the rest fixed."""
    if feature not in X.columns:
        raise ValueError(f"Unknown feature: {feature}")

    sample = X.sample(
        n=min(sample_size, len(X)),
        random_state=RANDOM_STATE,
    )

    # Use the middle 90% of the feature to avoid extreme outliers.
    low, high = X[feature].quantile([0.05, 0.95])
    grid = np.linspace(low, high, grid_size)

    predictions = []
    for value in grid:
        changed = sample.copy()
        changed[feature] = value
        predictions.append(float(model.predict(changed).mean()))

    return pd.DataFrame({feature: grid, "prediction": predictions})


def save_importance_plot(table, output_dir=FIGURE_DIR):
    """Save a horizontal bar chart of permutation importance."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ordered = table.sort_values("importance_mean")

    plt.figure(figsize=(7, 4))
    plt.barh(
        ordered["feature"],
        ordered["importance_mean"],
        xerr=ordered["importance_std"],
    )
    plt.xlabel("Increase in MAE when shuffled")
    plt.title("Permutation importance")
    plt.tight_layout()

    path = output_dir / "permutation_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_partial_dependence_plot(curves, output_dir=FIGURE_DIR):
    """Save one partial-dependence panel per feature."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(curves), figsize=(4 * len(curves), 3.5))

    for axis, (feature, curve) in zip(axes, curves.items()):
        axis.plot(curve[feature], curve["prediction"])
        axis.set_xlabel(feature)
        axis.set_ylabel("Predicted value ($100,000)")
        axis.set_title(feature)

    plt.tight_layout()
    path = output_dir / "partial_dependence.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def build_explanation(model, X, y, n_repeats=5, top_features=3):
    """Return the importance table and curves for the top features."""
    importance = permutation_importance_table(
        model,
        X,
        y,
        n_repeats=n_repeats,
    )

    curves = {}
    for feature in importance["feature"].head(top_features):
        curves[feature] = partial_dependence_curve(model, X, feature)

    return {"importance": importance, "partial_dependence": curves}


def main():
    """Fit the tuned forest on the training split and explain it."""
    df = load_dataset()
    validate_frame(df)

    X_train, X_test, y_train, y_test = split_train_test(df)

    model = build_models()["random_forest"]
    model.fit(X_train, y_train)

    explanation = build_explanation(model, X_test, y_test)
    importance = explanation["importance"]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    importance.to_csv(IMPORTANCE_CSV, index=False)

    print("Permutation importance on the holdout set:")
    print(importance.to_string(index=False))
    print()

    for feature, curve in explanation["partial_dependence"].items():
        print(f"Partial dependence for {feature}:")
        print(curve.to_string(index=False))
        print()

    importance_path = save_importance_plot(importance)
    curves_path = save_partial_dependence_plot(
        explanation["partial_dependence"]
    )

    print(f"Saved {IMPORTANCE_CSV}")
    print(f"Saved {importance_path}")
    print(f"Saved {curves_path}")


if __name__ == "__main__":
    main()
