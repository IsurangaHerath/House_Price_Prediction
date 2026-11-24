"""Tests for the forest-based prediction range."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.housing import data as hd
from src.housing import features as hf
from src.housing import models as hm
from src.housing import uncertainty as hu


def test_prediction_interval_uses_pipeline():
    df = hd.load_dataset().sample(n=200, random_state=42)
    X, y = hf.split_features_target(df)

    model = hm.build_models()["random_forest"]
    # Keep the test fast.
    model.named_steps["regressor"].set_params(n_estimators=5)
    model.fit(X, y)

    interval = hu.prediction_interval(model, X.head(10))

    assert len(interval) == 10
    assert (interval["low"] <= interval["prediction"]).all()
    assert (interval["prediction"] <= interval["high"]).all()


def test_prediction_interval_accepts_dict():
    df = hd.load_dataset().sample(n=200, random_state=42)
    X, y = hf.split_features_target(df)

    model = RandomForestRegressor(n_estimators=5, random_state=42)
    model.fit(X, y)

    row = X.iloc[0].to_dict()
    interval = hu.prediction_interval(model, row)

    assert len(interval) == 1


def test_interval_coverage():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    low = np.array([0.5, 2.5, 2.5, 3.5])
    high = np.array([1.5, 3.5, 3.5, 4.5])

    # The second value is outside its interval.
    assert hu.interval_coverage(y_true, low, high) == pytest.approx(0.75)


def test_coverage_by_segment():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    intervals = pd.DataFrame(
        {
            "low": [0.5, 2.5, 2.5, 3.5],
            "high": [1.5, 3.5, 3.5, 4.5],
        }
    )
    groups = pd.Series(["a", "a", "b", "b"])

    table = hu.coverage_by_segment(y_true, intervals, groups)

    assert list(table["segment"]) == ["a", "b"]
    assert table["coverage"].tolist() == pytest.approx([0.5, 1.0])
    assert table["mean_width"].tolist() == pytest.approx([1.0, 1.0])
