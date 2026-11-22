"""Check how much the holdout score depends on the split seed."""

import pandas as pd

from .data import load_dataset, validate_frame
from .evaluate import regression_metrics
from .models import build_models
from .train import split_train_test

SEEDS = [0, 1, 2, 3, 42]


def holdout_by_seed(df, model_name="random_forest", seeds=None):
    """Fit one model on several random splits and score each holdout set."""
    if seeds is None:
        seeds = SEEDS

    rows = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = split_train_test(df, seed)
        model = build_models()[model_name].fit(X_train, y_train)
        metrics = regression_metrics(y_test, model.predict(X_test))
        rows.append(
            {
                "seed": seed,
                "imputed_in_train": int(X_train["AveBedrms"].isna().sum()),
                "imputed_in_test": int(X_test["AveBedrms"].isna().sum()),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def main():
    """Print the per-seed holdout scores and their spread."""
    df = load_dataset()
    validate_frame(df)

    results = holdout_by_seed(df)
    print(results.to_string(index=False))
    print()
    print(
        f"Holdout MAE: {results['mae'].mean():.4f} "
        f"+/- {results['mae'].std():.4f} over {len(results)} seeds"
    )


if __name__ == "__main__":
    main()
