"""Check whether feature importance is stable across split seeds."""

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from .data import load_dataset, validate_frame
from .explain import permutation_importance_table
from .models import build_models
from .train import split_train_test

REPORT_DIR = Path("reports")
STABILITY_CSV = REPORT_DIR / "importance_stability.csv"
SEEDS = (42, 7, 123, 2024, 99)
N_REPEATS = 3


def importance_by_seed(df, seeds=SEEDS, n_repeats=N_REPEATS):
    """Return importance and rank for each feature under each split seed."""
    rows = []

    for seed in seeds:
        X_train, X_test, y_train, y_test = split_train_test(
            df, random_state=seed
        )

        model = build_models()["random_forest"]
        model.fit(X_train, y_train)

        table = permutation_importance_table(
            model,
            X_test,
            y_test,
            n_repeats=n_repeats,
        )

        for rank, row in enumerate(table.itertuples(index=False), start=1):
            rows.append(
                {
                    "seed": seed,
                    "feature": row.feature,
                    "importance": row.importance_mean,
                    "rank": rank,
                }
            )

    return pd.DataFrame(rows)


def summarize(table):
    """Give each feature's mean importance and best/worst rank."""
    summary = (
        table.groupby("feature")
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            best_rank=("rank", "min"),
            worst_rank=("rank", "max"),
        )
        .sort_values("mean_importance", ascending=False)
        .reset_index()
    )
    return summary


def rank_correlations(table):
    """Compare the importance rankings for every pair of seeds."""
    wide = table.pivot(index="feature", columns="seed", values="rank")
    seeds = list(wide.columns)

    rows = []
    for index, first in enumerate(seeds):
        for second in seeds[index + 1 :]:
            correlation, _ = spearmanr(wide[first], wide[second])
            rows.append(
                {
                    "seeds": f"{first} vs {second}",
                    "spearman": float(correlation),
                }
            )

    return pd.DataFrame(rows)


def main():
    """Save and summarize importance across five split seeds."""
    df = load_dataset()
    validate_frame(df)

    table = importance_by_seed(df)
    summary = summarize(table)
    correlations = rank_correlations(table)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(STABILITY_CSV, index=False)

    print("Importance across split seeds:")
    print(summary.to_string(index=False))
    print()
    print("Rank agreement between seeds:")
    print(correlations.to_string(index=False))
    print()
    print(f"Saved {STABILITY_CSV}")


if __name__ == "__main__":
    main()
