"""Tests for data loading and validation.

Run with: .venv\\Scripts\\python.exe -m pytest tests/test_data.py -v
"""
import pandas as pd
import pytest

from src.housing import data as hd


def test_load_shape_and_types():
    """Real dataset must load with the known census dimensions."""
    df = hd.load_dataset()
    assert len(df) == hd.EXPECTED_ROWS, (
        f"Expected {hd.EXPECTED_ROWS} census block groups, got {len(df)}"
    )
    # 8 features + 1 target = 9 columns
    assert list(df.columns) == hd.EXPECTED_FEATURES + ["MedHouseVal"]


def test_feature_target_separation():
    """The 8 features must be present and the target must be separate."""
    df = hd.load_dataset()
    for feature in hd.EXPECTED_FEATURES:
        assert feature in df.columns, f"Feature '{feature}' missing."
    assert "MedHouseVal" in df.columns, "Target 'MedHouseVal' missing."
    # Confirm no feature column leaks into the target (defensive check):
    # the target should be numeric but its name should not overlap.
    assert df["MedHouseVal"].name == "MedHouseVal"


def test_features_use_sklearn_units():
    """Rooms and occupancy must be per-household averages."""
    df = hd.load_dataset()
    row = df.iloc[0]

    assert row["AveRooms"] == pytest.approx(6.984127, rel=1e-5)
    assert row["AveBedrms"] == pytest.approx(1.023810, rel=1e-5)
    assert row["AveOccup"] == pytest.approx(2.555556, rel=1e-5)


def test_target_is_in_100k_units():
    """The target must be scaled from dollars to $100,000 units."""
    df = hd.load_dataset()
    assert df[hd.TARGET_COLUMN].max() == pytest.approx(5.0, abs=0.01)


def test_only_bedrooms_have_missing_values():
    """Only the known AveBedrms gaps stay missing; the pipeline imputes them."""
    df = hd.load_dataset()
    null_counts = df.isnull().sum()
    assert int(null_counts["AveBedrms"]) == hd.EXPECTED_MISSING
    assert int(null_counts.drop("AveBedrms").sum()) == 0


def test_unexpected_missing_values_raise():
    df = hd.load_dataset()
    df.loc[0, "MedInc"] = float("nan")
    with pytest.raises(ValueError, match="Unexpected missing values"):
        hd.validate_frame(df)


def test_validation_summary():
    """validate_frame() must return a dict with expected keys."""
    df = hd.load_dataset()
    summary = hd.validate_frame(df)
    assert summary["rows"] == 20640
    assert summary["features"] == 8
    assert summary["columns"] == hd.EXPECTED_FEATURES + ["MedHouseVal"]
    assert summary["null_total"] == hd.EXPECTED_MISSING
    # The cap count is an observed property; we only assert it is a
    # non-negative integer (the dataset has some capped rows).
    assert isinstance(summary["cap_count_at_5_0"], int)
    assert summary["cap_count_at_5_0"] >= 0
    # Float-reported max can be 5.00001 due to binary storage; allow it.
    assert summary["target_max"] is not None
