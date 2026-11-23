"""Tests for the out-of-range input guard."""

import numpy as np
import pandas as pd

from src.housing import bounds as hb
from src.housing import data as hd


def make_bounds():
    """Build bounds from a small slice of the real dataset."""
    df = hd.load_dataset().head(200)
    return hb.training_bounds(df)


def test_training_bounds_cover_every_feature():
    bounds = make_bounds()

    assert list(bounds.index) == hd.EXPECTED_FEATURES
    assert list(bounds.columns) == ["low", "high"]
    assert (bounds["low"] <= bounds["high"]).all()


def test_outside_mask_flags_extreme_rows():
    bounds = make_bounds()
    typical = {col: float(bounds.loc[col, "low"]) for col in hd.EXPECTED_FEATURES}
    extreme = dict(typical, MedInc=1000.0)

    mask = hb.outside_mask([typical, extreme], bounds)

    assert not mask.iloc[0].any()
    assert mask.iloc[1]["MedInc"]
    assert not mask.iloc[1]["HouseAge"]


def test_outside_mask_accepts_dict_and_ignores_missing_bedrooms():
    bounds = make_bounds()
    row = {col: float(bounds.loc[col, "low"]) for col in hd.EXPECTED_FEATURES}
    row["AveBedrms"] = np.nan

    mask = hb.outside_mask(row, bounds)

    assert len(mask) == 1
    assert not mask.iloc[0].any()
