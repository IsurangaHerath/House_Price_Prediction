"""Estimate a prediction range from the forest's individual trees."""

from pathlib import Path

import numpy as np
import pandas as pd

from .data import EXPECTED_FEATURES, load_dataset, validate_frame
from .models import build_models
from .segments import SEGMENTS, add_segment
from .train import split_train_test

REPORT_DIR = Path("reports")
SEGMENT_CSV = REPORT_DIR / "uncertainty_segments.csv"


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


def coverage_by_segment(y_true, intervals, segments):
    """Return coverage and mean width for each subgroup."""
    lows = intervals["low"].to_numpy()
    highs = intervals["high"].to_numpy()
    truth = np.asarray(y_true)

    frame = pd.DataFrame(
        {
            "segment": np.asarray(segments),
            "inside": (truth >= lows) & (truth <= highs),
            "width": highs - lows,
        }
    )

    rows = []
    for name, group in frame.groupby("segment", observed=True):
        rows.append(
            {
                "segment": str(name),
                "n_samples": len(group),
                "coverage": float(group["inside"].mean()),
                "mean_width": float(group["width"].mean()),
            }
        )

    return pd.DataFrame(rows)


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

    tables = []
    for title, spec in SEGMENTS.items():
        labels = add_segment(
            X_test,
            spec["column"],
            spec["bins"],
            spec["labels"],
        )
        table = coverage_by_segment(y_test, intervals, labels)
        table.insert(0, "segment_type", title)
        tables.append(table)

    combined = pd.concat(tables, ignore_index=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(SEGMENT_CSV, index=False)

    income = combined[combined["segment_type"] == "Income"]
    print()
    print("Coverage by income band:")
    print(
        income[["segment", "n_samples", "coverage", "mean_width"]]
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print()
    print(f"Saved {SEGMENT_CSV}")


if __name__ == "__main__":
    main()
