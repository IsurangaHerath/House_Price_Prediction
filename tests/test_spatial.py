"""Tests for the spatial validation workflow."""

from src.housing import data as hd
from src.housing import spatial as hs


def test_assign_regions_returns_one_label_per_row():
    df = hd.load_dataset().sample(n=300, random_state=42)
    labels = hs.assign_regions(df, n_regions=3)

    assert len(labels) == len(df)
    assert set(labels) == {0, 1, 2}


def test_spatial_cv_holds_out_every_region_once():
    df = hd.load_dataset().sample(n=300, random_state=42)
    folds, pooled = hs.spatial_cv(df, model_name="dummy_mean", n_regions=3)

    assert len(folds) == 3
    assert folds["n_test"].sum() == 300
    assert pooled["mae"] >= 0
