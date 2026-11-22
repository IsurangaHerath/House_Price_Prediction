"""Models and a simple cross-validation helper."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
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


class BedroomImputer(BaseEstimator, TransformerMixin):
    """Fill missing AveBedrms with AveRooms times the median bedroom share.

    The share is learned in ``fit``, so it only uses training rows.
    """

    def fit(self, X, y=None):
        ratio = X["AveBedrms"] / X["AveRooms"]
        self.bedroom_share_ = float(ratio.median())
        return self

    def transform(self, X):
        X = X.copy()
        X["AveBedrms"] = X["AveBedrms"].fillna(
            X["AveRooms"] * self.bedroom_share_
        )
        return X


def with_imputer(*steps):
    """Build a pipeline that imputes bedrooms before the given steps."""
    return Pipeline([("impute", BedroomImputer()), *steps])


def build_random_forest(**params):
    """Return the random forest, using the tuned settings by default."""
    settings = {
        "n_estimators": 100,
        # These settings came from the tuning workflow in tuning.py.
        "max_depth": 30,
        "max_features": 0.5,
        "min_samples_leaf": 1,
        "random_state": RANDOM_STATE,
        **params,
    }
    return RandomForestRegressor(**settings)


def build_models():
    """Return the models we want to compare."""
    return {
        # Baseline: always predicts the average house value.
        "dummy_mean": with_imputer(
            ("regressor", DummyRegressor(strategy="mean")),
        ),
        # Linear regression with scaling inside a pipeline.
        "linear_regression": with_imputer(
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ),
        # Ridge adds a small penalty to keep coefficients stable.
        "ridge": with_imputer(
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ),
        # Random forest can learn non-linear relationships.
        "random_forest": with_imputer(
            ("regressor", build_random_forest()),
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
