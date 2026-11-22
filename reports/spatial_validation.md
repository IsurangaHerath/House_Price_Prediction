# Spatial validation results

Generated with:

```powershell
.venv\Scripts\python.exe -m src.housing.spatial
```

## Why this check matters

The main workflow uses a random 80/20 split. California census blocks are
geographically clustered, so neighbouring blocks appear in both training and
test. That inflates the score compared with predicting a **new region**.

This check holds out entire geographic regions. Each region is used as the test
set exactly once, and the model never trains on a nearby block from that region.

## Method

| Item | Value |
|---|---|
| Regions | 10 clusters from KMeans on latitude/longitude |
| Splitter | `GroupKFold`, one fold per region |
| Model | Tuned random forest (same pipeline and settings) |
| Target | `MedHouseVal` in `$100,000` units |
| Random state | 42 |

## Per-region scores

| Fold | Region | Rows | MAE | RMSE | R² |
|---:|---:|---:|---:|---:|---:|
| 0 | 3 | 7,269 | 0.540110 | 0.771610 | 0.521045 |
| 1 | 1 | 3,153 | 0.617861 | 0.884351 | 0.410889 |
| 2 | 5 | 1,809 | 0.553331 | 0.700072 | 0.481726 |
| 3 | 8 | 1,769 | 0.602438 | 0.786195 | 0.446092 |
| 4 | 0 | 1,756 | 0.652381 | 0.766076 | 0.348513 |
| 5 | 4 | 1,471 | 0.322317 | 0.424439 | 0.469186 |
| 6 | 2 | 1,013 | 0.410731 | 0.588308 | -1.758584 |
| 7 | 9 | 989 | 0.446887 | 0.542024 | -0.227257 |
| 8 | 7 | 939 | 0.571050 | 0.815514 | 0.496583 |
| 9 | 6 | 472 | 0.306483 | 0.365346 | -0.402647 |

Region labels are arbitrary cluster numbers, not official names. Region sizes
range from 472 to 7,269 rows.

## Pooled result

| Metric | Random holdout | Spatial holdout | Change |
|---|---:|---:|---:|
| MAE | 0.325238 | 0.537766 | +65% |
| MAE ($) | 32,524 | 53,777 | +21,253 |
| RMSE | 0.500555 | 0.743119 | +48% |
| R² | 0.808796 | 0.585276 | -0.224 |

The random-split score overstates performance on an unseen region. The model
still beats the mean baseline, but its error grows by about $21,000 per block
when the region is new.

## Interpretation

- The random forest uses latitude and longitude, but it cannot extrapolate
  reliably to a region it has never seen.
- Some small regions have negative R² because their internal target variance
  is small while the prediction error remains large.
- The large region 3 (7,269 rows) drives much of the pooled score. Even there,
  MAE is 0.540, well above the random-split result.
- This is the more honest estimate for deploying the model outside the
  training geography.

## Caveats

- KMeans regions are not official counties or metros; different boundaries
  would give different numbers.
- The split is spatial but not temporal. The data still describes 1990.
- The 207 rows with missing `AveBedrms` are handled by the pipeline imputer in
  every fold.
- The target is still censored at `$500,000`.
