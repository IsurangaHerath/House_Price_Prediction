"""Tests for subgroup error analysis."""

import numpy as np
import pandas as pd
import pytest

from src.housing import segments as hseg


def test_add_segment_labels_each_row():
    X = pd.DataFrame(
        {"MedInc": [0.5, 1.99, 2.0, 3.49, 3.5, 4.99, 5.0, 10.0]}
    )
    spec = hseg.SEGMENTS["Income"]

    labels = hseg.add_segment(X, spec["column"], spec["bins"], spec["labels"])

    assert list(labels.astype(str)) == [
        "under $20k",
        "under $20k",
        "$20k-$35k",
        "$20k-$35k",
        "$35k-$50k",
        "$35k-$50k",
        "$50k+",
        "$50k+",
    ]


def test_segment_errors_reports_each_group():
    y_true = np.array([1.0, 1.0, 3.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0, 2.0])
    groups = pd.Series(["a", "a", "b", "b"])

    table = hseg.segment_errors(y_true, y_pred, groups)

    assert len(table) == 2
    assert list(table["segment"]) == ["a", "b"]
    assert table["n_samples"].tolist() == [2, 2]
    assert table["mae"].tolist() == pytest.approx([0.5, 0.5])
    assert table["bias"].tolist() == pytest.approx([-0.5, 0.5])


def test_segment_errors_skips_empty_groups():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.0, 2.0])
    groups = pd.Categorical(["a", "a"], categories=["a", "b"])

    table = hseg.segment_errors(y_true, y_pred, groups)

    assert list(table["segment"]) == ["a"]
