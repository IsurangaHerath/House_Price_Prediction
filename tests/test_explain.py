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


def test_what_if_curve_changes_only_one_feature(small_data):
    model, X, _ = small_data
    features = X.iloc[0].to_dict()

    curve = he.what_if_curve(model, features, "MedInc", [1.0, 2.0, 3.0])

    assert len(curve) == 3
    assert curve["MedInc"].tolist() == [1.0, 2.0, 3.0]
    # The mean-only model ignores every feature, so the line is flat.
    assert curve["prediction"].nunique() == 1


def test_what_if_curve_rejects_unknown_feature(small_data):
    model, X, _ = small_data
    features = X.iloc[0].to_dict()

    with pytest.raises(ValueError, match="Unknown feature"):
        he.what_if_curve(model, features, "NotAFeature", [1.0])


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
