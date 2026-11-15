"""Load and validate the California Housing dataset."""

from pathlib import Path

import pandas as pd

EXPECTED_FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
EXPECTED_N_FEATURES = len(EXPECTED_FEATURES)
EXPECTED_ROWS = 20640

TARGET_COLUMN = "MedHouseVal"
TARGET_UNIT_USD = 100_000

DATA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "california_housing.csv"
)


def load_dataset():
    """Load the CSV and convert it to scikit-learn's units."""
    df = pd.read_csv(DATA_PATH)

    # The raw file stores total rooms, total bedrooms, and households.
    # Convert them to scikit-learn's per-household averages.
    households = df["AveOccup"]
    df["AveBedrms"] = df["AveBedrms"].fillna(df["AveRooms"] * 0.5)
    df["AveRooms"] = df["AveRooms"] / households
    df["AveBedrms"] = df["AveBedrms"] / households
    df["AveOccup"] = df["Population"] / households

    # The raw target is in dollars. Store it in $100,000 units.
    df[TARGET_COLUMN] = df[TARGET_COLUMN] / TARGET_UNIT_USD

    # Keep the same column order as scikit-learn.
    expected_cols = EXPECTED_FEATURES + [TARGET_COLUMN]
    if list(df.columns) != expected_cols:
        raise ValueError(
            f"Columns don't match. Expected {expected_cols}, got {list(df.columns)}"
        )

    return df


def validate_frame(df):
    """Check the dataset and return a short summary."""
    errors = []

    if len(df) != EXPECTED_ROWS:
        errors.append(
            f"Row count mismatch: expected {EXPECTED_ROWS}, got {len(df)}"
        )

    # Every feature and the target must be present.
    feature_cols = [col for col in EXPECTED_FEATURES if col in df.columns]
    if len(feature_cols) != EXPECTED_N_FEATURES:
        errors.append(
            f"Feature count mismatch: expected {EXPECTED_N_FEATURES}, "
            f"found {len(feature_cols)} ({feature_cols})"
        )

    missing_features = [
        col for col in EXPECTED_FEATURES if col not in df.columns
    ]
    if missing_features:
        errors.append(f"Missing feature columns: {missing_features}")

    if TARGET_COLUMN not in df.columns:
        errors.append(f"Target column '{TARGET_COLUMN}' missing.")

    # Every required column must be numeric.
    for col in EXPECTED_FEATURES + [TARGET_COLUMN]:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(
                f"Column '{col}' is non-numeric (dtype={df[col].dtype})"
            )

    # No missing values are allowed.
    null_total = int(df.isnull().sum().sum())
    if null_total > 0:
        errors.append(f"Found {null_total} missing values across all columns.")

    # Check the target range.
    min_target = None
    max_target = None
    cap_count = None

    if TARGET_COLUMN in df.columns:
        min_target = float(df[TARGET_COLUMN].min())
        max_target = float(df[TARGET_COLUMN].max())

        if min_target < 0.10:
            errors.append(f"Target minimum too low: {min_target}")

        # The cap is 5.0, but float storage can show a tiny amount above it.
        if max_target > 5.05:
            errors.append(
                f"Target maximum exceeds expected cap: {max_target}"
            )

        cap_count = int((df[TARGET_COLUMN] >= 5.0).sum())

    if errors:
        raise ValueError(
            "Dataset validation failed:\n" + "\n".join(errors)
        )

    return {
        "rows": int(len(df)),
        "features": EXPECTED_N_FEATURES,
        "columns": list(df.columns),
        "null_total": null_total,
        "cap_count_at_5_0": cap_count,
        "target_min": round(min_target, 4) if min_target is not None else None,
        "target_max": round(max_target, 4) if max_target is not None else None,
    }
