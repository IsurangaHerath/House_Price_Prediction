"""Look at the residuals from the best model."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .evaluate import residual_summary, residual_summary_by_cap
from .train import train_and_evaluate

FIGURE_DIR = Path("reports/figures")


def save_residual_plots(y_true, y_pred, output_dir=FIGURE_DIR):
    """Save a residual histogram and a predicted-vs-true plot."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Histogram of residuals.
    plt.figure(figsize=(7, 4))
    plt.hist(residuals, bins=40)
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("Residual (true - predicted)")
    plt.ylabel("Count")
    plt.title("Residual distribution")
    plt.tight_layout()
    hist_path = output_dir / "residuals_hist.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()

    # Predicted values against true values.
    plt.figure(figsize=(6, 6))
    plt.scatter(y_pred, y_true, s=8, alpha=0.4)

    low = min(y_true.min(), y_pred.min())
    high = max(y_true.max(), y_pred.max())
    plt.plot([low, high], [low, high], "k--")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Predicted vs true")
    plt.tight_layout()
    scatter_path = output_dir / "predicted_vs_true.png"
    plt.savefig(scatter_path, dpi=150)
    plt.close()

    return [hist_path, scatter_path]


def main():
    """Run the workflow and summarize the best model's residuals."""
    results = train_and_evaluate()
    best_model = results["best_model"]
    y_true = results["y_test"]
    y_pred = results["test_predictions"][best_model]

    summary = residual_summary(y_true, y_pred)
    cap_summary = residual_summary_by_cap(y_true, y_pred)

    print(f"Best model: {best_model}")
    print()

    print("Residual summary:")
    for name, value in summary.items():
        print(f"  {name}: {value}")

    print()
    print("Errors at the target cap:")
    for name, value in cap_summary.items():
        print(f"  {name}: {value}")

    print()
    paths = save_residual_plots(y_true, y_pred)
    print("Saved plots:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
