"""Tests for regression metrics.

The expected values are computed by hand from a tiny four-row example so
the tests check correctness, not just that the function runs.
"""
import math

import pytest

from src.housing import evaluate as ev

Y_TRUE = [3.0, -0.5, 2.0, 7.0]
Y_PRED = [2.5, 0.0, 2.0, 8.0]


def test_mae_matches_hand_computation():
    # abs errors: 0.5, 0.5, 0.0, 1.0 -> mean 0.5
    metrics = ev.regression_metrics(Y_TRUE, Y_PRED)
    assert metrics["mae"] == pytest.approx(0.5)


def test_rmse_matches_hand_computation():
    # squared errors: 0.25, 0.25, 0.0, 1.0 -> mean 0.375 -> sqrt
    metrics = ev.regression_metrics(Y_TRUE, Y_PRED)
    assert metrics["rmse"] == pytest.approx(math.sqrt(0.375))


def test_r2_matches_hand_computation():
    metrics = ev.regression_metrics(Y_TRUE, Y_PRED)
    # mean(y_true) = 2.875, SS_tot = 29.1875, SS_res = 1.5
    assert metrics["r2"] == pytest.approx(1 - (1.5 / 29.1875))


def test_usd_conversion():
    metrics = ev.regression_metrics(Y_TRUE, Y_PRED)
    assert metrics["mae_usd"] == pytest.approx(0.5 * 100_000)
    assert metrics["rmse_usd"] == pytest.approx(math.sqrt(0.375) * 100_000)


def test_perfect_prediction_scores():
    metrics = ev.regression_metrics(Y_TRUE, Y_TRUE)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)
