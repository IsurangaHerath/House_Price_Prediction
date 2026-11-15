"""Train and evaluate the baseline models."""

import pandas as pd
from sklearn.model_selection import train_test_split

from .data import load_dataset, validate_frame
from .evaluate import regression_metrics
from .features import split_features_target
from .models import (
    RANDOM_STATE,
    build_models,
    cross_validate_models,
    fit_model,
)

TEST_SIZE = 0.2


def split_train_test(df):
    """Split the data into training and test sets."""
    X, y = split_features_target(df)

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


def train_and_evaluate(df=None, validate=True):
    """Train all models and return CV and test results."""
    if df is None:
        df = load_dataset()

    if validate:
        validate_frame(df)

    X_train, X_test, y_train, y_test = split_train_test(df)

    models = build_models()
    cv_results = cross_validate_models(models, X_train, y_train)

    # Fit every model on the training set and score it on the test set.
    test_rows = []

    for name, model in models.items():
        fitted = fit_model(model, X_train, y_train)
        metrics = regression_metrics(y_test, fitted.predict(X_test))
        test_rows.append({"model": name, **metrics})

    test_results = pd.DataFrame(test_rows)
    test_results = test_results.sort_values("mae").reset_index(drop=True)

    # Pick the winner using training-only cross-validation.
    best_model = cv_results.iloc[0]["model"]

    return {
        "cv_results": cv_results,
        "test_results": test_results,
        "best_model": best_model,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def main():
    """Run the workflow and print the results."""
    results = train_and_evaluate()

    print(f"Training rows: {results['n_train']}")
    print(f"Test rows: {results['n_test']}")
    print()

    print("Cross-validation on the training set:")
    print(results["cv_results"].to_string(index=False))
    print()

    print("Test results:")
    print(results["test_results"].to_string(index=False))
    print()

    print(f"Best model by CV MAE: {results['best_model']}")


if __name__ == "__main__":
    main()
