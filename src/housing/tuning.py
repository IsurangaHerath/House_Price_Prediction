"""Tune the random forest and compare it with the default settings."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from .data import load_dataset, validate_frame
from .evaluate import regression_metrics
from .models import RANDOM_STATE, with_imputer
from .train import split_train_test

# A small search space keeps the tuning run quick and reproducible.
PARAM_DISTRIBUTIONS = {
    "max_depth": [None, 20, 30],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [1.0, 0.5, "sqrt"],
}


def tune_random_forest(
    X_train,
    y_train,
    param_distributions=None,
    n_iter=6,
    cv=3,
):
    """Search for better random forest settings using cross-validation."""
    if param_distributions is None:
        param_distributions = PARAM_DISTRIBUTIONS

    # Search over sklearn's default forest, wrapped in the same imputer
    # pipeline as the real models. Parameter names get the step prefix.
    estimator = with_imputer(
        (
            "regressor",
            RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        ),
    )
    search = RandomizedSearchCV(
        estimator,
        param_distributions={
            f"regressor__{name}": values
            for name, values in param_distributions.items()
        },
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)
    return search


def main():
    """Compare the default and tuned random forests on the holdout set."""
    df = load_dataset()
    validate_frame(df)

    X_train, X_test, y_train, y_test = split_train_test(df)

    # sklearn's default forest, not the tuned one from build_models().
    default_model = with_imputer(
        (
            "regressor",
            RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        ),
    )
    default_model.fit(X_train, y_train)
    default_metrics = regression_metrics(
        y_test,
        default_model.predict(X_test),
    )

    search = tune_random_forest(X_train, y_train)
    tuned_metrics = regression_metrics(
        y_test,
        search.predict(X_test),
    )

    print("Default random forest:")
    print(f"  test MAE: {default_metrics['mae']:.6f}")
    print()
    print("Tuned random forest:")
    print(f"  best CV MAE: {-search.best_score_:.6f}")
    print(f"  test MAE: {tuned_metrics['mae']:.6f}")
    print(f"  best params: {search.best_params_}")


if __name__ == "__main__":
    main()
