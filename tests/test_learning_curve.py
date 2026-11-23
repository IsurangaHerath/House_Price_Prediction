"""Tests for the learning curve workflow."""

import pandas as pd
import pytest

from src.housing import data as hd
from src.housing import features as hf
from src.housing import learning_curve as hl
from src.housing import models as hm


def test_learning_curve_table_reports_each_size():
    df = hd.load_dataset().head(200)
    X, y = hf.split_features_target(df)
    model = hm.build_models()["dummy_mean"]

    table = hl.learning_curve_table(
        model,
        X,
        y,
        train_sizes=[0.5, 1.0],
        cv=2,
    )

    assert len(table) == 2
    # With cv=2 and shuffle=True, half the rows are reserved for validation,
    # so the sizes are fractions of the remaining 100 rows.
    assert list(table["train_rows"]) == [50, 100]
    assert list(table.columns) == [
        "train_rows",
        "train_mae",
        "train_std",
        "val_mae",
        "val_std",
    ]
    assert (table["train_mae"] >= 0).all()
    assert (table["val_mae"] >= 0).all()


def test_save_learning_curve_plot(tmp_path):
    table = pd.DataFrame(
        {
            "train_rows": [100, 200],
            "train_mae": [0.4, 0.3],
            "train_std": [0.01, 0.01],
            "val_mae": [0.5, 0.4],
            "val_std": [0.02, 0.02],
        }
    )

    path = hl.save_learning_curve_plot(table, tmp_path)

    assert path.exists()
    assert path.name == "learning_curve.png"
    assert path.stat().st_size > 0
