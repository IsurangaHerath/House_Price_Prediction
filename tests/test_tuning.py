"""Tests for the random forest tuning workflow."""

from src.housing import data as hd
from src.housing import features as hf
from src.housing import tuning as ht


def test_tune_random_forest_returns_a_search():
    df = hd.load_dataset().sample(n=300, random_state=42)
    X, y = hf.split_features_target(df)

    search = ht.tune_random_forest(
        X,
        y,
        param_distributions={"n_estimators": [5], "max_depth": [3]},
        n_iter=1,
        cv=2,
    )

    assert search.best_params_ == {"n_estimators": 5, "max_depth": 3}
    assert len(search.predict(X)) == len(X)
