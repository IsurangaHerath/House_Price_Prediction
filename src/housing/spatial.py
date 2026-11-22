"""Evaluate the model on held-out geographic regions."""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold

from .data import load_dataset, validate_frame
from .evaluate import regression_metrics
from .features import split_features_target
from .models import RANDOM_STATE, build_models

N_REGIONS = 10


def assign_regions(df, n_regions=N_REGIONS):
    """Group nearby census blocks into regions with KMeans on lat/lon."""
    coords = df[["Latitude", "Longitude"]]
    kmeans = KMeans(
        n_clusters=n_regions,
        random_state=RANDOM_STATE,
        n_init=10,
    )
    return kmeans.fit_predict(coords)


def spatial_cv(df, model_name="random_forest", n_regions=N_REGIONS):
    """Train on all but one region and score the held-out region.

    Each region is used as the test set exactly once, so the model never sees
    a nearby census block from the same region during training.
    """
    df = df.copy()
    df["region"] = assign_regions(df, n_regions)

    X, y = split_features_target(df)
    groups = df["region"]

    cv = GroupKFold(n_splits=n_regions)
    rows = []
    all_true = []
    all_pred = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        model = build_models()[model_name]
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        predictions = model.predict(X.iloc[test_idx])
        metrics = regression_metrics(y.iloc[test_idx], predictions)

        rows.append(
            {
                "fold": fold,
                "region": int(groups.iloc[test_idx].iloc[0]),
                "n_test": int(len(test_idx)),
                **metrics,
            }
        )

        all_true.extend(y.iloc[test_idx])
        all_pred.extend(predictions)

    # Pool every held-out prediction to get one overall score.
    pooled = regression_metrics(all_true, all_pred)
    return pd.DataFrame(rows), pooled


def main():
    """Print per-region and pooled spatial scores."""
    df = load_dataset()
    validate_frame(df)

    folds, pooled = spatial_cv(df)

    print("Held-out region scores:")
    print(folds.to_string(index=False))
    print()
    print(f"Rows per fold: {folds['n_test'].min()} to {folds['n_test'].max()}")
    print(f"Pooled MAE:  {pooled['mae']:.6f}")
    print(f"Pooled RMSE: {pooled['rmse']:.6f}")
    print(f"Pooled R2:   {pooled['r2']:.6f}")


if __name__ == "__main__":
    main()
