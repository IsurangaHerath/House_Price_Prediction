"""Tests for importance stability across split seeds."""

import pandas as pd
import pytest

from src.housing import data as hd
from src.housing import stability as hs


def make_table():
    """A small table with two features and two seeds."""
    return pd.DataFrame(
        {
            "seed": [1, 1, 2, 2],
            "feature": ["MedInc", "Latitude", "MedInc", "Latitude"],
            "importance": [0.4, 0.2, 0.5, 0.1],
            "rank": [1, 2, 1, 2],
        }
    )


def test_importance_by_seed_covers_every_feature():
    df = hd.load_dataset().sample(n=200, random_state=42)

    table = hs.importance_by_seed(df, seeds=(1, 2), n_repeats=1)

    assert len(table) == 2 * len(hd.EXPECTED_FEATURES)
    assert set(table["seed"]) == {1, 2}
    assert set(table["feature"]) == set(hd.EXPECTED_FEATURES)
    assert table.groupby("seed")["rank"].max().tolist() == [
        len(hd.EXPECTED_FEATURES)
    ] * 2


def test_summarize_reports_mean_and_rank_range():
    summary = hs.summarize(make_table())

    assert list(summary["feature"]) == ["MedInc", "Latitude"]
    assert summary["mean_importance"].tolist() == pytest.approx([0.45, 0.15])
    assert summary["best_rank"].tolist() == [1, 2]
    assert summary["worst_rank"].tolist() == [1, 2]


def test_rank_correlations_covers_each_seed_pair():
    correlations = hs.rank_correlations(make_table())

    assert list(correlations["seeds"]) == ["1 vs 2"]
    assert correlations["spearman"].iloc[0] == pytest.approx(1.0)
