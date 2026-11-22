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
| Models | Dummy mean, linear regression, ridge, tuned random forest |

## Cross-validation on the training set

| Model | MAE | RMSE | R² | Mean fit time (s) |
|---|---:|---:|---:|---:|
| Random forest (tuned) | 0.330936 | 0.501851 | 0.811527 | 8.54 |
| Ridge | 0.529056 | 0.720509 | 0.611457 | 0.05 |
| Linear regression | 0.529061 | 0.720510 | 0.611457 | 0.10 |
| Dummy mean | 0.913915 | 1.156172 | -0.000215 | 0.01 |

## Holdout test set

| Model | MAE | RMSE | R² | MAE ($) | RMSE ($) |
|---|---:|---:|---:|---:|---:|
| Random forest (tuned) | 0.325238 | 0.500555 | 0.808796 | 32,524 | 50,056 |
| Ridge | 0.534510 | 0.747575 | 0.573516 | 53,451 | 74,758 |
| Linear regression | 0.534518 | 0.747601 | 0.573486 | 53,452 | 74,760 |
| Dummy mean | 0.906069 | 1.144856 | -0.000219 | 90,607 | 114,486 |

## Best model

The random forest wins on both cross-validation and the holdout set:

- holdout MAE: **0.325238** (`$32,524`)
- holdout RMSE: **0.500555** (`$50,056`)
- holdout R²: **0.808796**

Ridge and linear regression are effectively tied. The random forest captures
non-linear relationships that the linear models miss. Its settings came from
the tuning workflow; see [`tuning_results.md`](./tuning_results.md).

## Residual analysis: random forest

| Statistic | Value |
|---|---:|
| Mean residual | -0.010719 |
| Median residual | -0.054341 |
| MAE | 0.325238 |
| RMSE | 0.500555 |
| Underprediction rate | 41.64% |
| Worst overprediction | -2.919841 |
| Worst underprediction | 3.302680 |

Positive residuals mean the true value was higher than the prediction. The
mean residual is slightly below zero, so the model over-predicts a little on
average. The underprediction rate is 41.6%.

### Errors at the target cap

| Group | Rows | MAE |
|---|---:|---:|
| At cap (`MedHouseVal >= 5.0`) | 184 | 0.761061 |
| Below cap | 3,944 | 0.304905 |

The model makes much larger errors on capped rows. That is expected: several
different true values above the cap all appear as `5.0`, so the model cannot
distinguish them.

## Plots

The analysis script writes:

- `reports/figures/residuals_hist.png`
- `reports/figures/predicted_vs_true.png`

These files are ignored by Git and can be regenerated at any time.

## Split-seed robustness

Generated with `python -m src.housing.robustness`. The tuned random forest is
refit on five different 80/20 splits:

| Seed | Imputed rows in train | Imputed rows in test | Holdout MAE |
|---:|---:|---:|---:|
| 0 | 158 | 49 | 0.325460 |
| 1 | 163 | 44 | 0.323287 |
| 2 | 176 | 31 | 0.324249 |
| 3 | 167 | 40 | 0.312985 |
| 42 | 0 | 207 | 0.325238 |

Holdout MAE is **0.3222 ± 0.0052** across seeds. The project seed (42) sits
near the middle, so the headline number is not a lucky split. With seed 42,
every imputed row lands in the test set, which a random split would almost
never do by chance. The missing values in the CSV likely line up with that
split.

## Caveats

- The 207 missing `AveBedrms` values are imputed inside each model pipeline
  as `AveRooms` times the median bedroom share learned from the training rows
  (about 0.20). With `random_state=42`, all 207 rows fall in the holdout set
  (0 in training). See "Split-seed robustness" below.
- The split is random, so nearby census blocks appear in both training and test.
  This is geographic leakage and means the scores do not measure performance in
  an unseen region.
- Values at the `$500,000` cap are censored at 5.0.
- The data describes 1990, not the current housing market.
