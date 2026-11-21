"""Train and save the final model used by the dashboard."""

from datetime import datetime, timezone
from pathlib import Path

import joblib

from .data import EXPECTED_FEATURES, TARGET_COLUMN, load_dataset, validate_frame
from .models import build_models
from .train import train_and_evaluate

MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "models"
    / "best_model.joblib"
)


def train_final_model(model_name="random_forest", df=None):
    """Fit one model on every row and return it."""
    if df is None:
        df = load_dataset()

    validate_frame(df)

    model = build_models()[model_name]
    model.fit(df[EXPECTED_FEATURES], df[TARGET_COLUMN])
    return model


def save_model(
    model,
    model_name,
    n_rows,
    model_path=MODEL_PATH,
    results=None,
):
    """Save the model, metadata, and comparison results with joblib."""
    payload = {
        "model": model,
        "metadata": {
            "model": model_name,
            "rows": int(n_rows),
            "features": EXPECTED_FEATURES,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "results": results,
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Random forests are large, so compress the artifact on disk.
    joblib.dump(payload, model_path, compress=3)
    return model_path


def load_model(model_path=MODEL_PATH):
    """Load a saved model payload."""
    return joblib.load(model_path)


def train_and_save(model_path=MODEL_PATH, df=None):
    """Pick the best model, fit it on all rows, and save it."""
    if df is None:
        df = load_dataset()

    validate_frame(df)

    # Use cross-validation to choose the winner, then refit on all rows.
    results = train_and_evaluate(df, validate=False)
    best_model = results["best_model"]
    model = train_final_model(best_model, df)

    # Keep the tables the dashboard needs, without the large predictions.
    saved_results = {
        "cv_results": results["cv_results"],
        "test_results": results["test_results"],
        "best_model": best_model,
        "n_train": results["n_train"],
        "n_test": results["n_test"],
    }

    return save_model(
        model,
        best_model,
        len(df),
        model_path=model_path,
        results=saved_results,
    )


def main():
    """Create the saved model artifact."""
    path = train_and_save()
    payload = load_model(path)
    metadata = payload["metadata"]

    print(f"Saved model: {metadata['model']}")
    print(f"Rows: {metadata['rows']}")
    print(f"Path: {path}")


if __name__ == "__main__":
    main()
