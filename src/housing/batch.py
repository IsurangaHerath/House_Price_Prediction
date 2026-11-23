"""Score a CSV of block groups and write predictions with a range."""

import argparse
from pathlib import Path

import pandas as pd

from .bounds import outside_mask, training_bounds
from .data import EXPECTED_FEATURES, TARGET_UNIT_USD, load_dataset
from .finalize import MODEL_PATH, load_model, train_final_model
from .uncertainty import prediction_interval


def validate_features(df):
    """Check the input columns and the gaps the pipeline cannot fill."""
    missing = [col for col in EXPECTED_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    # Only bedrooms are imputed inside the pipeline.
    gaps = [
        col
        for col in EXPECTED_FEATURES
        if col != "AveBedrms" and df[col].isna().any()
    ]
    if gaps:
        raise ValueError(f"Missing values in: {gaps}")


def score_frame(model, df, lower=0.10, upper=0.90, bounds=None):
    """Add predicted values and the tree range, in dollars."""
    validate_features(df)

    intervals = prediction_interval(
        model,
        df[EXPECTED_FEATURES],
        lower=lower,
        upper=upper,
    )

    scored = df.copy()
    scored["prediction_usd"] = intervals["prediction"].to_numpy() * TARGET_UNIT_USD
    scored["low_usd"] = intervals["low"].to_numpy() * TARGET_UNIT_USD
    scored["high_usd"] = intervals["high"].to_numpy() * TARGET_UNIT_USD

    if bounds is not None:
        mask = outside_mask(df[EXPECTED_FEATURES], bounds)
        scored["outside_range"] = mask.any(axis=1).to_numpy()
        scored["outside_features"] = mask.apply(
            lambda row: ";".join(row.index[row]),
            axis=1,
        ).to_numpy()

    return scored


def get_model(model_path=MODEL_PATH):
    """Load the saved model, or train one when the artifact is missing."""
    if Path(model_path).exists():
        return load_model(model_path)["model"]
    return train_final_model()


def main(argv=None):
    """Read a CSV, score it, and save the result."""
    parser = argparse.ArgumentParser(
        description="Score block groups with the California Housing model."
    )
    parser.add_argument("--input", required=True, help="CSV with the 8 features")
    parser.add_argument(
        "--output",
        help="Where to write predictions (default: <input>_predictions.csv)",
    )
    args = parser.parse_args(argv)

    df = pd.read_csv(args.input)
    bounds = training_bounds(load_dataset())
    scored = score_frame(get_model(), df, bounds=bounds)

    if args.output:
        output = Path(args.output)
    else:
        input_path = Path(args.input)
        output = input_path.with_name(f"{input_path.stem}_predictions.csv")

    output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output, index=False)

    flagged = int(scored["outside_range"].sum())
    print(f"Scored {len(scored):,} rows ({flagged:,} outside the training range)")
    print(f"Saved {output}")
    print(
        scored[["prediction_usd", "low_usd", "high_usd"]]
        .head()
        .round(0)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
