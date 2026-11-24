# Explainability results

Generated with:

```powershell
.venv\Scripts\python.exe -m src.housing.explain
```

## Method

| Item | Value |
|---|---|
| Model | Tuned random forest pipeline |
| Evaluation rows | Holdout test set (4,128 rows) |
| Permutation importance | 5 shuffles per feature, scored by MAE |
| Partial dependence | 5th to 95th percentile, 400-row sample |
| Random state | 42 |

Permutation importance measures how much the MAE gets worse when one feature
is shuffled. A larger value means the model relies on that feature more.

## Permutation importance

| Feature | Mean increase in MAE | Std |
|---|---:|---:|
| MedInc | 0.416866 | 0.005492 |
| Latitude | 0.322054 | 0.005158 |
| Longitude | 0.280269 | 0.004730 |
| AveOccup | 0.153487 | 0.002972 |
| HouseAge | 0.050899 | 0.002020 |
| AveRooms | 0.044739 | 0.002528 |
| AveBedrms | 0.008400 | 0.000962 |
| Population | 0.005478 | 0.000712 |

The same numbers are stored in
[`feature_importance.csv`](./feature_importance.csv).

### What stands out

- **Median income is the strongest driver** by a clear margin.
- **Latitude and longitude are next**, confirming that location carries a lot
  of information beyond the other features.
- Occupancy matters moderately; house age and room counts matter less.
- Bedroom count and population are almost irrelevant once the other features
  are present.

## Partial dependence

Partial dependence shows the average prediction while one feature changes and
the rest stay fixed.

### Median income

Predicted value rises steadily with income:

| MedInc | Prediction ($100,000) |
|---:|---:|
| 1.60 | 1.46 |
| 3.39 | 1.92 |
| 5.17 | 2.57 |
| 7.25 | 3.49 |

Going from the low end to the high end of the plotted range roughly doubles
the predicted value.

### Latitude

Predicted values are higher in the south and lower toward the north:

| Latitude | Prediction ($100,000) |
|---:|---:|
| 32.80 | 2.60 |
| 36.04 | 1.95 |
| 38.95 | 1.54 |

This reflects the higher-value coastal and southern California markets in the
1990 data, not a causal effect of latitude.

### Longitude

Predicted value falls from the coast toward the inland east:

| Longitude | Prediction ($100,000) |
|---:|---:|
| -122.46 | 2.79 |
| -118.49 | 2.25 |
| -117.07 | 1.62 |

Again, this is a geographic association, not a causal relationship.

## Local what-if view

The dashboard's Predict tab can vary median income from the 1st to the 99th
percentile while holding the other seven inputs fixed. This is the same idea as
the partial dependence curve above, but conditioned on one entered block rather
than averaged over hundreds of holdout rows. The curve uses the selected model
and is generated on the fly.

## Plots

The script writes:

- `reports/figures/permutation_importance.png`
- `reports/figures/partial_dependence.png`

These are ignored by Git and can be regenerated.

## Caveats

- Permutation importance splits credit between correlated features. Latitude
  and longitude overlap, so their individual rankings should be read together.
- The holdout split still has geographic leakage. The spatial validation shows
  that location-based performance drops on unseen regions.
- Partial dependence is descriptive. It does not show causal effects.
- The target is capped at `$500,000`, which limits what the model can learn
  about high-value areas.
