"""Tests for saving and loading the final model."""

from src.housing import data as hd
from src.housing import finalize as hf


def test_save_and_load_round_trip(tmp_path):
    df = hd.load_dataset()
    model = hf.train_final_model("dummy_mean", df)

    path = hf.save_model(
        model,
        "dummy_mean",
        len(df),
        model_path=tmp_path / "best_model.joblib",
        results={"best_model": "dummy_mean"},
    )
    payload = hf.load_model(path)

    assert payload["metadata"]["model"] == "dummy_mean"
    assert payload["metadata"]["rows"] == hd.EXPECTED_ROWS
    assert payload["metadata"]["features"] == hd.EXPECTED_FEATURES
    assert payload["results"]["best_model"] == "dummy_mean"
    assert len(payload["model"].predict(df[hd.EXPECTED_FEATURES])) == len(df)
