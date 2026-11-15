# Baseline model results

Generated with:

```powershell
.venv\Scripts\python.exe -m src.housing.train
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
| Models | Dummy mean, StandardScaler + LinearRegression |

## Cross-validation on the training set

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear regression | 0.529061 | 0.720510 | 0.611457 |
| Dummy mean | 0.913915 | 1.156172 | -0.000215 |

## Holdout test set

| Model | MAE | RMSE | R² | MAE ($) | RMSE ($) |
|---|---:|---:|---:|---:|---:|
| Linear regression | 0.573295 | 0.803245 | 0.507633 | 57,329 | 80,325 |
| Dummy mean | 0.906069 | 1.144856 | -0.000219 | 90,607 | 114,486 |

## Interpretation

- Linear regression beats the dummy mean baseline on both cross-validation and
  the holdout set.
- The holdout MAE is about **$57,300**, so predictions are typically off by
  roughly that amount in 1990 dollars.
- R² is about **0.51** on the holdout set, meaning the model explains roughly
  half of the variance in the target.

## Caveats

- The 207 missing `AveBedrms` values are imputed as `AveRooms * 0.5`. With
  `random_state=42`, all 207 rows fall in the holdout set, so the test metrics
  depend on that approximation.
- The split is random, so nearby census blocks appear in both training and test.
  This is geographic leakage and means the scores do not measure performance in
  an unseen region.
- Values at the `$500,000` cap are censored at 5.0, which makes high-value
  districts harder to predict.
- The data describes 1990, not the current housing market.
