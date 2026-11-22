"""Tests for the split-seed robustness check."""

from src.housing import data as hd
from src.housing import robustness as hr


def test_holdout_by_seed_returns_one_row_per_seed():
    df = hd.load_dataset().sample(n=400, random_state=0)

    results = hr.holdout_by_seed(df, model_name="ridge", seeds=[0, 1])

    assert list(results["seed"]) == [0, 1]
    assert (results["mae"] > 0).all()
    total_missing = int(df["AveBedrms"].isna().sum())
    counts = results["imputed_in_train"] + results["imputed_in_test"]
    assert (counts == total_missing).all()
