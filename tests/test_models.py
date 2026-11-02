"""Integration tests for the model workflow.

These use a small slice of the real dataset so the suite stays fast while
still exercising the actual sklearn estimators and the shared CV helper.
"""
import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from src.housing import data as hd
from src.housing import evaluate as ev
from src.housing import features as hf
from src.housing import models as hm


@pytest.fixture(scope="module")
def small_split():
    df = hd.load_dataset().sample(n=800, random_state=hm.RANDOM_STATE)
    X, y = hf.split_features_target(df)
    return train_test_split(X, y, test_size=0.25, random_state=hm.RANDOM_STATE)


def test_build_models_contains_expected_candidates():
    models = hm.build_models()
    assert set(models) == {"dummy_mean", "linear_regression"}


def test_each_model_fits_and_predicts(small_split):
    X_train, X_test, y_train, y_test = small_split

    for name, model in hm.build_models().items():
        fitted = hm.fit_model(model, X_train, y_train)
        predictions = fitted.predict(X_test)

        assert len(predictions) == len(y_test), f"{name} produced wrong length"
        metrics = ev.regression_metrics(y_test, predictions)
        assert set(metrics) >= {"mae", "rmse", "r2"}
        assert metrics["mae"] >= 0


def test_dummy_predictions_are_constant(small_split):
    X_train, X_test, y_train, _ = small_split
    dummy = hm.build_models()["dummy_mean"]
    hm.fit_model(dummy, X_train, y_train)
    predictions = dummy.predict(X_test)
    assert np.allclose(predictions, predictions[0])


def test_cross_validation_returns_all_models(small_split):
    X_train, _, y_train, _ = small_split
    results = hm.cross_validate_models(hm.build_models(), X_train, y_train)

    assert len(results) == 2
    assert {"model", "cv_mae_mean", "cv_rmse_mean", "cv_r2_mean"} <= set(results.columns)
    # Lower MAE must be the first row (sorted ascending).
    assert results["cv_mae_mean"].is_monotonic_increasing
