"""Split the dataset into features (X) and the target (y)."""

import pandas as pd

from .data import EXPECTED_FEATURES

TARGET_COLUMN = "MedHouseVal"


def split_features_target(df: pd.DataFrame):
    """Return X (features) and y (target)."""
    # Check that every feature column exists.
    missing = [col for col in EXPECTED_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    # Check that the target column exists.
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: '{TARGET_COLUMN}'")

    # Copy the columns so changing X or y does not change the original data.
    X = df[EXPECTED_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y
