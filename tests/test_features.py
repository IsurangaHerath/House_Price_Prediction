"""Tests for feature/target separation."""
import pandas as pd
import pytest

from src.housing import data as hd
from src.housing import features as hf


def test_split_returns_expected_shapes():
    df = hd.load_dataset()
    X, y = hf.split_features_target(df)

    assert list(X.columns) == hd.EXPECTED_FEATURES
    assert len(X) == len(y) == 20640
    assert y.name == hf.TARGET_COLUMN


def test_target_is_not_inside_features():
    df = hd.load_dataset()
    X, _ = hf.split_features_target(df)
    assert hf.TARGET_COLUMN not in X.columns


def test_missing_feature_raises():
    df = hd.load_dataset().drop(columns=["MedInc"])
    with pytest.raises(ValueError, match="Missing feature columns"):
        hf.split_features_target(df)


def test_missing_target_raises():
    df = hd.load_dataset().drop(columns=[hf.TARGET_COLUMN])
    with pytest.raises(ValueError, match="Missing target column"):
        hf.split_features_target(df)
