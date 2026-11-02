"""Model definitions and the shared cross-validation workflow.

Design rules encoded here:
- One fixed random seed for everything so results are reproducible.
- Every learned preprocessing step lives inside a Pipeline, so it is fit
  only on training folds during cross-validation (no leakage).
- Model selection uses cross-validation on the TRAINING set only; the
  test set is never consulted here.
"""
from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
CV_FOLDS = 5

# sklearn returns "neg_" scores for error metrics; we flip their sign later.
SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}
NEGATED_METRICS = {"mae", "rmse"}


def build_models() -> dict[str, object]:
    """Return the candidate models for this milestone.

    - dummy_mean: predicts the training mean; the honest floor any real
      model must beat.
    - linear_regression: StandardScaler + LinearRegression inside a
      Pipeline, so scaling is fit on training data only.
    """
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "linear_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        ),
    }


def make_cv() -> KFold:
    """Shared 5-fold splitter, shuffled with the project seed."""
    return KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def cross_validate_models(models: dict[str, object], X_train, y_train, cv=None) -> pd.DataFrame:
    """Cross-validate every model on the training data with identical folds.

    Returns one row per model, sorted by mean CV MAE (lower is better).
    """
    cv = make_cv() if cv is None else cv
    rows: list[dict] = []

    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)

        row: dict[str, float | str] = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            if metric in NEGATED_METRICS:
                values = -values
            row[f"cv_{metric}_mean"] = float(values.mean())
            row[f"cv_{metric}_std"] = float(values.std())
        row["mean_fit_seconds"] = float(scores["fit_time"].mean())
        rows.append(row)

    results = pd.DataFrame(rows)
    return results.sort_values("cv_mae_mean").reset_index(drop=True)


def fit_model(model, X_train, y_train):
    """Fit a model on the training set and return it (for chaining)."""
    model.fit(X_train, y_train)
    return model
