"""Tests for batch scoring."""

import pandas as pd
import pytest

from src.housing import batch as hb
from src.housing import data as hd
from src.housing import features as hf
from src.housing import models as hm


def make_frame(n=20):
    """Return a small feature frame from the real dataset."""
    df = hd.load_dataset().head(n)
    return df[hd.EXPECTED_FEATURES].copy()


def test_score_frame_adds_dollar_columns():
    df = make_frame()
    model = hm.build_models()["dummy_mean"]
    model.fit(df, [2.0] * len(df))

    scored = hb.score_frame(model, df)

    assert len(scored) == len(df)
    assert {"prediction_usd", "low_usd", "high_usd"} <= set(scored.columns)
    # A mean-only model gives a flat range at $200,000.
    assert scored["prediction_usd"].iloc[0] == pytest.approx(200_000)
    assert scored["low_usd"].iloc[0] == pytest.approx(200_000)
    assert scored["high_usd"].iloc[0] == pytest.approx(200_000)


def test_score_frame_uses_the_tree_range():
    df = make_frame()
    X, y = hf.split_features_target(hd.load_dataset().head(200))

    model = hm.build_models()["random_forest"]
    model.named_steps["regressor"].set_params(n_estimators=5)
    model.fit(X, y)

    scored = hb.score_frame(model, df)

    assert (scored["low_usd"] <= scored["prediction_usd"]).all()
    assert (scored["prediction_usd"] <= scored["high_usd"]).all()
    assert (scored["low_usd"] < scored["high_usd"]).all()


def test_validate_features_rejects_missing_columns():
    df = make_frame().drop(columns=["MedInc"])

    with pytest.raises(ValueError, match="Missing feature columns"):
        hb.validate_features(df)


def test_main_writes_predictions(tmp_path, monkeypatch):
    df = hd.load_dataset().head(5)
    input_path = tmp_path / "blocks.csv"
    output_path = tmp_path / "scored.csv"
    df.to_csv(input_path, index=False)

    model = hm.build_models()["dummy_mean"]
    model.fit(df[hd.EXPECTED_FEATURES], df[hd.TARGET_COLUMN])
    monkeypatch.setattr(hb, "get_model", lambda *args, **kwargs: model)

    hb.main(["--input", str(input_path), "--output", str(output_path)])

    scored = pd.read_csv(output_path)
    assert len(scored) == 5
    assert {"prediction_usd", "low_usd", "high_usd"} <= set(scored.columns)
