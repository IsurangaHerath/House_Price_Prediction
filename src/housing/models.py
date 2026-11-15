"""Models and a simple cross-validation helper."""

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
CV_FOLDS = 5

# sklearn returns negative scores for error metrics, so we flip them later.
SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}


def build_models():
    """Return the models we want to compare."""
    return {
        # Baseline: always predicts the average house value.
        "dummy_mean": DummyRegressor(strategy="mean"),
        # Linear regression with scaling inside a pipeline.
        "linear_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        ),
    }


def make_cv():
    """Return a shuffled 5-fold splitter."""
    return KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def cross_validate_models(models, X_train, y_train, cv=None):
    """Cross-validate each model and return one row per model."""
    if cv is None:
        cv = make_cv()

    rows = []

    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=-1,
        )

        row = {"model": name}

        for metric in SCORING:
            values = scores[f"test_{metric}"]

            # Flip MAE and RMSE back to positive numbers.
            if metric in ("mae", "rmse"):
                values = -values

            row[f"cv_{metric}_mean"] = float(values.mean())
            row[f"cv_{metric}_std"] = float(values.std())

        row["mean_fit_seconds"] = float(scores["fit_time"].mean())
        rows.append(row)

    results = pd.DataFrame(rows)

    # Lowest MAE first.
    return results.sort_values("cv_mae_mean").reset_index(drop=True)


def fit_model(model, X_train, y_train):
    """Train a model and return it."""
    model.fit(X_train, y_train)
    return model
