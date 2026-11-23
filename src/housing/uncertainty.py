"""Estimate a prediction range from the forest's individual trees."""

import numpy as np
import pandas as pd

from .data import EXPECTED_FEATURES, load_dataset, validate_frame
from .models import build_models
from .train import split_train_test


def as_frame(features):
    """Accept a dict or a DataFrame and return a DataFrame."""
    if isinstance(features, dict):
        features = pd.DataFrame([features])
    else:
        features = pd.DataFrame(features)

    # Keep the same column order the model was trained on.
    if all(col in features.columns for col in EXPECTED_FEATURES):
        features = features[EXPECTED_FEATURES]
    return features


def _transform_features(model, features):
    """Apply the pipeline's preprocessing steps, if any."""
    steps = getattr(model, "steps", None)
    if steps is None:
        return features

    preprocess = model[:-1]
    if len(preprocess) == 0:
        return features
    return preprocess.transform(features)


def prediction_interval(model, features, lower=0.10, upper=0.90):
    """Return the forest's lower and upper prediction quantiles."""
    features = as_frame(features)

    regressor = getattr(model, "named_steps", {}).get("regressor", model)
    trees = getattr(regressor, "estimators_", None)

    # Models without trees can only give a single value, so use a flat range.
    if trees is None:
        prediction = model.predict(features)
        return pd.DataFrame(
            {"prediction": prediction, "low": prediction, "high": prediction}
        )

    transformed = np.asarray(_transform_features(model, features))

    # One prediction per tree gives a small distribution for each row.
    tree_predictions = np.array(
        [tree.predict(transformed) for tree in trees]
    )

    low, high = np.quantile(tree_predictions, [lower, upper], axis=0)
    prediction = model.predict(features)

    return pd.DataFrame(
        {
            "prediction": prediction,
            "low": low,
            "high": high,
        }
    )


def interval_coverage(y_true, low, high):
    """Return the share of true values inside the interval."""
    y_true = np.asarray(y_true)
    low = np.asarray(low)
    high = np.asarray(high)

    inside = (y_true >= low) & (y_true <= high)
    return float(inside.mean())


def main():
    """Measure interval coverage and width on the holdout set."""
    df = load_dataset()
    validate_frame(df)

    X_train, X_test, y_train, y_test = split_train_test(df)

    model = build_models()["random_forest"]
    model.fit(X_train, y_train)

    intervals = prediction_interval(model, X_test)
    coverage = interval_coverage(
        y_test,
        intervals["low"],
        intervals["high"],
    )
    width = intervals["high"] - intervals["low"]

    # Use positional masks so the indices line up with `intervals`.
    capped = (y_test >= 5.0).to_numpy()
    capped_coverage = interval_coverage(
        y_test.to_numpy()[capped],
        intervals["low"].to_numpy()[capped],
        intervals["high"].to_numpy()[capped],
    )

    print("Forest range on the holdout set (10th-90th percentile):")
    print(f"  coverage: {coverage:.4f}")
    print(f"  mean width: {width.mean():.4f}")
    print(f"  median width: {width.median():.4f}")
    print(f"  coverage at the target cap: {capped_coverage:.4f}")


if __name__ == "__main__":
    main()
