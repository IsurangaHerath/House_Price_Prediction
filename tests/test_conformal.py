"""Tests for conformal prediction ranges."""

import numpy as np
import pytest

from src.housing import conformal as hc
from src.housing import data as hd
from src.housing import features as hf
from src.housing import models as hm


def test_conformal_quantile_uses_finite_sample_level():
    residuals = np.arange(1, 11)

    # ceil(11 * 0.8) / 10 = 0.9, and the 90th percentile is 9.1.
    assert hc.conformal_quantile(residuals, 0.8) == pytest.approx(9.1)


def test_conformal_quantile_caps_at_largest_residual():
    residuals = np.array([0.5, 1.0])

    assert hc.conformal_quantile(residuals, 0.99) == pytest.approx(1.0)


def test_split_calibration_sizes():
    df = hd.load_dataset().sample(n=200, random_state=42)
    X, y = hf.split_features_target(df)

    X_fit, X_cal, y_fit, y_cal = hc.split_calibration(X, y, 0.25)

    assert len(X_fit) == 150
    assert len(X_cal) == 50
    assert len(y_fit) == 150
    assert len(y_cal) == 50


def test_conformal_interval_is_centered_on_the_prediction():
    df = hd.load_dataset().sample(n=200, random_state=42)
    X, y = hf.split_features_target(df)

    model = hm.build_models()["dummy_mean"].fit(X, y)
    interval = hc.conformal_interval(model, X.head(5), half_width=0.25)

    assert len(interval) == 5
    assert np.allclose(interval["low"], interval["prediction"] - 0.25)
    assert np.allclose(interval["high"], interval["prediction"] + 0.25)
