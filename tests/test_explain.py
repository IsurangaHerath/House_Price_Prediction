"""Tests for the model explanation helpers."""

import pandas as pd
import pytest

from src.housing import data as hd
from src.housing import explain as he
from src.housing import features as hf
from src.housing import models as hm


@pytest.fixture(scope="module")
def small_data():
    df = hd.load_dataset().sample(n=300, random_state=42)
    X, y = hf.split_features_target(df)
    model = hm.build_models()["dummy_mean"].fit(X, y)
    return model, X, y


def test_permutation_importance_table(small_data):
    model, X, y = small_data
    table = he.permutation_importance_table(model, X, y, n_repeats=1)

    assert len(table) == len(hd.EXPECTED_FEATURES)
    assert list(table.columns) == [
        "feature",
        "importance_mean",
        "importance_std",
    ]
    # A mean-only model ignores its features.
    assert table["importance_mean"].abs().max() == pytest.approx(0.0)


def test_partial_dependence_curve(small_data):
    model, X, _ = small_data
    curve = he.partial_dependence_curve(
        model,
        X,
        "MedInc",
        grid_size=5,
        sample_size=50,
    )

    assert len(curve) == 5
    assert list(curve.columns) == ["MedInc", "prediction"]
    # A mean-only model predicts the same value everywhere.
    assert curve["prediction"].nunique() == 1
