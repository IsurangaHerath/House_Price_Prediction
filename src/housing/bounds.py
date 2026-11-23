"""Flag inputs that fall outside the training distribution."""

import pandas as pd

from .data import EXPECTED_FEATURES
from .uncertainty import as_frame


def training_bounds(df, lower=0.01, upper=0.99):
    """Return per-feature percentile bounds from the training data."""
    bounds = df[EXPECTED_FEATURES].quantile([lower, upper]).T
    bounds.columns = ["low", "high"]
    return bounds


def outside_mask(features, bounds):
    """Mark each feature that falls outside the training bounds."""
    frame = as_frame(features)

    # NaN comparisons are False, so imputed bedrooms are not flagged.
    return (frame < bounds["low"]) | (frame > bounds["high"])
