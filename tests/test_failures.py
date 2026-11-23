"""Tests for the failure-case analysis."""

import numpy as np
import pandas as pd
import pytest

from src.housing import data as hd
from src.housing import failures as hf


def make_inputs():
    """Build a tiny feature frame with known errors."""
    X = pd.DataFrame({col: [1.0, 2.0, 3.0, 4.0] for col in hd.EXPECTED_FEATURES})
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.5, 2.0, 4.2])
    return X, y_true, y_pred


def test_worst_predictions_sorts_by_absolute_error():
    X, y_true, y_pred = make_inputs()

    worst = hf.worst_predictions(X, y_true, y_pred, n=2)

    assert len(worst) == 2
    assert worst["abs_error"].tolist() == pytest.approx([1.0, 0.5])
    assert worst["error"].tolist() == pytest.approx([1.0, -0.5])


def test_worst_predictions_keeps_features_and_target():
    X, y_true, y_pred = make_inputs()

    worst = hf.worst_predictions(X, y_true, y_pred, n=1)

    assert hd.TARGET_COLUMN in worst.columns
    assert "prediction" in worst.columns
    assert worst[hd.TARGET_COLUMN].iloc[0] == pytest.approx(3.0)
    assert worst["MedInc"].iloc[0] == pytest.approx(3.0)


def test_worst_predictions_caps_at_available_rows():
    X, y_true, y_pred = make_inputs()

    worst = hf.worst_predictions(X, y_true, y_pred, n=100)

    assert len(worst) == 4
