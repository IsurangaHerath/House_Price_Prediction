# Learning curve results

Generated with:

```powershell
.venv\Scripts\python.exe -m src.housing.learning_curve
```

## Why this check matters

The project reports one holdout score. A learning curve asks a different
question: **would more training data make the model better?** It also shows
whether the gap between training and validation error comes from fitting too
little data or from variance the model cannot remove.

## Method

| Item | Value |
|---|---|
| Model | Tuned random forest pipeline |
| Data | Training split only (16,512 rows) |
| Sizes | 20%, 40%, 60%, 80%, 100% of the fitting pool |
| Splitter | 3 shuffle splits, 20% validation |
| Metric | MAE in `$100,000` units |
| Random state | 42 |

The largest curve point uses **13,209 rows**, because each fold reserves 20%
for validation. The plot is saved to `reports/figures/learning_curve.png`.

## Results

| Training rows | Train MAE | Train std | Validation MAE | Validation std |
|---:|---:|---:|---:|---:|
| 2,641 | 0.1407 | 0.0011 | 0.3885 | 0.0059 |
| 5,283 | 0.1321 | 0.0010 | 0.3597 | 0.0079 |
| 7,925 | 0.1270 | 0.0005 | 0.3459 | 0.0069 |
| 10,567 | 0.1247 | 0.0011 | 0.3369 | 0.0092 |
| 13,209 | 0.1229 | 0.0011 | 0.3295 | 0.0081 |

## Interpretation

- **Validation error keeps falling.** MAE drops from **0.389** to **0.330**,
  about **15%**, as the fitting pool grows fivefold.
- **The curve is not flat yet.** The last 2,642 rows still buy **0.0074** MAE
  (about **$740** per block). More data would help modestly, with diminishing
  returns.
- **The train/validation gap is large.** At the largest size, training MAE is
  **0.123** while validation MAE is **0.330**, a gap of about **0.21**. The
  forest fits its training rows much better than it generalizes.
- **Fold spread is small.** Validation standard deviation stays under **0.01**,
  so the trend is not a lucky split.

The practical reading: collecting more block groups is a reasonable lever, but
the gap suggests variance reduction and better features matter at least as
much. The spatial validation already shows that generalizing to a new region is
a harder problem than adding more rows from the same geography.

## Caveats

- **Random splits.** Neighbouring blocks appear in both fitting and validation,
  so the curve measures within-geography learning, not new-region learning.
- **Training split only.** The holdout set is untouched; the largest point is
  13,209 rows, not the full 16,512.
- **One model family.** The curve describes the tuned random forest, not other
  models.
- **Censored target.** Values at `$500,000` are stored as `5.0`, which puts a
  floor on achievable error regardless of data size.
