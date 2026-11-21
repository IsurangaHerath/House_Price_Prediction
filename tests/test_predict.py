"""Tests for the prediction helpers."""

import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from src.housing.data import EXPECTED_FEATURES
from src.housing.predict import (
    default_features,
    format_usd,
    predict_house_value,
)


def make_frame():
    """Build a tiny frame with every feature column."""
    return pd.DataFrame({col: [1.0, 3.0] for col in EXPECTED_FEATURES})


def test_default_features_returns_one_row():
    defaults = default_features(make_frame())

    assert len(defaults) == 1
    assert list(defaults.columns) == EXPECTED_FEATURES
    assert defaults.iloc[0]["MedInc"] == pytest.approx(2.0)


def test_predict_house_value_returns_dollars():
    df = make_frame()
    model = DummyRegressor(strategy="constant", constant=2.0)
    model.fit(df[EXPECTED_FEATURES], [1.0, 2.0])

    value = predict_house_value(model, default_features(df))

    assert value == pytest.approx(200_000)


def test_predict_house_value_accepts_dict():
    df = make_frame()
    model = DummyRegressor(strategy="constant", constant=1.5)
    model.fit(df[EXPECTED_FEATURES], [1.0, 2.0])

    value = predict_house_value(model, {col: 1.0 for col in EXPECTED_FEATURES})

    assert value == pytest.approx(150_000)


def test_format_usd():
    assert format_usd(1234.6) == "$1,235"
