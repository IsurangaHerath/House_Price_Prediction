"""Feature/target separation for the California Housing dataset.

Keeping this in one place means every model and notebook separates the
target consistently, which is the first defence against target leakage.
"""
from __future__ import annotations

import pandas as pd

from .data import EXPECTED_FEATURES

TARGET_COLUMN = "MedHouseVal"


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a validated housing DataFrame into (features, target).

    Parameters
    ----------
    df:
        DataFrame containing the 8 expected features and the
        'MedHouseVal' target column.

    Returns
    -------
    (X, y):
        X is a DataFrame with the 8 features in the documented order;
        y is the target Series.

    Raises
    ------
    ValueError
        If required columns are missing or the lengths differ.
    """
    missing_features = [name for name in EXPECTED_FEATURES if name not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: '{TARGET_COLUMN}'")

    X = df.loc[:, EXPECTED_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    if len(X) != len(y):
        raise ValueError("Feature and target lengths differ; the DataFrame is malformed.")

    return X, y
