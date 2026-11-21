"""Helpers for making and formatting dashboard predictions."""

import pandas as pd

from .data import EXPECTED_FEATURES, TARGET_UNIT_USD


def default_features(df):
    """Return the median feature values as a one-row DataFrame."""
    return df[EXPECTED_FEATURES].median().to_frame().T


def predict_house_value(model, features):
    """Predict one house value in dollars.

    ``features`` can be a DataFrame or a plain dictionary of feature values.
    """
    if isinstance(features, dict):
        features = pd.DataFrame([features])

    # Keep the columns in the same order the model was trained on.
    row = pd.DataFrame(features)[EXPECTED_FEATURES]
    prediction = model.predict(row)[0]

    # The model works in $100,000 units, so convert back to dollars.
    return float(prediction) * TARGET_UNIT_USD


def format_usd(amount):
    """Format a dollar amount with thousands separators."""
    return f"${amount:,.0f}"
