# Model comparison and residual results

Generated with:

```powershell
.venv\Scripts\python.exe -m src.housing.train
.venv\Scripts\python.exe -m src.housing.analyze
```

## Setup

| Item | Value |
|---|---|
| Dataset | California Housing CSV (20,640 rows) |
| Features | 8 numeric columns |
| Target | `MedHouseVal` in `$100,000` units |
| Train/test split | 80/20, `random_state=42` |
| Training rows | 16,512 |
| Test rows | 4,128 |
| Cross-validation | 5 shuffled folds, `random_state=42` |
| Models | Dummy mean, linear regression, ridge, random forest |

## Cross-validation on the training set

| Model | MAE | RMSE | R² | Mean fit time (s) |
|---|---:|---:|---:|---:|
| Random forest | 0.335053 | 0.511071 | 0.804542 | 20.24 |
| Ridge | 0.529056 | 0.720509 | 0.611457 | 0.05 |
| Linear regression | 0.529061 | 0.720510 | 0.611457 | 0.10 |
| Dummy mean | 0.913915 | 1.156172 | -0.000215 | 0.01 |

## Holdout test set

| Model | MAE | RMSE | R² | MAE ($) | RMSE ($) |
|---|---:|---:|---:|---:|---:|
| Random forest | 0.330565 | 0.507663 | 0.803328 | 33,057 | 50,766 |
| Ridge | 0.573246 | 0.803144 | 0.507757 | 57,325 | 80,314 |
| Linear regression | 0.573295 | 0.803245 | 0.507633 | 57,329 | 80,325 |
| Dummy mean | 0.906069 | 1.144856 | -0.000219 | 90,607 | 114,486 |

## Best model

The random forest wins on both cross-validation and the holdout set:

- holdout MAE: **0.330565** (`$33,057`)
- holdout RMSE: **0.507663** (`$50,766`)
- holdout R²: **0.803328**

Ridge and linear regression are effectively tied. The random forest captures
non-linear relationships that the linear models miss.

## Residual analysis: random forest

| Statistic | Value |
|---|---:|
| Mean residual | -0.018860 |
| Median residual | -0.053675 |
| MAE | 0.330565 |
| RMSE | 0.507663 |
| Underprediction rate | 41.69% |
| Worst overprediction | -3.045841 |
| Worst underprediction | 3.043500 |

Positive residuals mean the true value was higher than the prediction. The
mean residual is slightly below zero, so the model over-predicts a little on
average. The underprediction rate is 41.7%.

### Errors at the target cap

| Group | Rows | MAE |
|---|---:|---:|
| At cap (`MedHouseVal >= 5.0`) | 184 | 0.731050 |
| Below cap | 3,944 | 0.311881 |

The model makes much larger errors on capped rows. That is expected: several
different true values above the cap all appear as `5.0`, so the model cannot
distinguish them.

## Plots

The analysis script writes:

- `reports/figures/residuals_hist.png`
- `reports/figures/predicted_vs_true.png`

These files are ignored by Git and can be regenerated at any time.

## Caveats

- The 207 missing `AveBedrms` values are imputed as `AveRooms * 0.5`. With
  `random_state=42`, all 207 rows fall in the holdout set, so test metrics
  depend on that approximation.
- The split is random, so nearby census blocks appear in both training and test.
  This is geographic leakage and means the scores do not measure performance in
  an unseen region.
- Values at the `$500,000` cap are censored at 5.0.
- The data describes 1990, not the current housing market.
