"""Tests for the baseline training workflow."""

from src.housing import data as hd
from src.housing import train as ht


def test_split_train_test_sizes():
    """The default split should be 80% train and 20% test."""
    df = hd.load_dataset().sample(n=500, random_state=42)
    X_train, X_test, y_train, y_test = ht.split_train_test(df)

    assert len(X_train) == 400
    assert len(X_test) == 100
    assert len(y_train) == 400
    assert len(y_test) == 100


def test_train_and_evaluate_returns_comparison():
    """The workflow should return CV and test results for every model."""
    df = hd.load_dataset().sample(n=500, random_state=42)
    results = ht.train_and_evaluate(df, validate=False)

    assert results["n_train"] == 400
    assert results["n_test"] == 100
    assert len(results["cv_results"]) == 4
    assert len(results["test_results"]) == 4

    assert results["best_model"] == results["cv_results"].iloc[0]["model"]
    assert results["cv_results"]["cv_mae_mean"].is_monotonic_increasing
    assert results["test_results"]["mae"].is_monotonic_increasing
