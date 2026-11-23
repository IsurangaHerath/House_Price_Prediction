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


def save_error_map(X, y_true, y_pred, output_dir=FIGURE_DIR):
    """Save a map of holdout blocks shaded by absolute error."""
    abs_error = np.abs(np.asarray(y_true) - np.asarray(y_pred))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Saturate at the 95th percentile so a few huge errors do not flatten the
    # colours for the rest of the map.
    vmax = np.percentile(abs_error, 95)

    plt.figure(figsize=(8, 6))
    points = plt.scatter(
        X["Longitude"],
        X["Latitude"],
        c=abs_error,
        cmap="viridis",
        s=8,
        alpha=0.6,
        vmin=0,
        vmax=vmax,
    )
    plt.colorbar(points, label="Absolute error")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Holdout errors by location (colors capped at 95th percentile)")
    plt.tight_layout()

    path = output_dir / "error_map.png"
    plt.savefig(path, dpi=150)
    plt.close()

    return path


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
    paths.append(save_error_map(results["X_test"], y_true, y_pred))

    # Show where the largest errors sit without needing the figure.
    locations = results["X_test"][["Latitude", "Longitude"]].copy()
    locations["abs_error"] = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    top = locations.nlargest(int(len(locations) * 0.1), "abs_error")

    print()
    print("Locations of the largest 10% of errors:")
    print(f"  latitude: {top['Latitude'].min():.2f} to {top['Latitude'].max():.2f}")
    print(f"  longitude: {top['Longitude'].min():.2f} to {top['Longitude'].max():.2f}")

    print()
    print("Saved plots:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
