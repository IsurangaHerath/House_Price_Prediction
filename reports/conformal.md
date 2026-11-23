# Conformal calibration results

Generated with:

```powershell
.venv\Scripts\python.exe -m src.housing.conformal
```

## Why this check matters

The tree-based range covers about 85% of holdout rows even though it asks for
80%. That is close, but it is not a formal guarantee. Split conformal
prediction provides one: if the calibration and test rows are exchangeable,
the interval covers the truth at the requested rate on average.

This check asks a simple question: **is the dashboard's tree range
well calibrated, and would a conformal interval be better?**

## Method

| Item | Value |
|---|---|
| Model | Tuned random forest pipeline |
| Training rows | 16,512 (from the usual 80/20 split) |
| Fitting rows | 12,384 (75% of training) |
| Calibration rows | 4,128 (25% of training) |
| Holdout rows | 4,128 |
| Target coverage | 80% |
| Score | Absolute residual on calibration rows |
| Random state | 42 |

The model is fit on the fitting rows only. Absolute residuals on the
calibration rows give the conformal quantile, with the finite-sample
correction `ceil((n + 1) * coverage) / n`. The interval is the prediction plus
or minus that quantile. The tree range is computed from the same fitted model,
so the comparison is fair.

## Results

| Method | Coverage | Mean width | Median width | Cap coverage | Cap width |
|---|---:|---:|---:|---:|---:|
| Conformal 80% | 0.8021 | 1.0244 | 1.0244 | 0.4728 | 1.0244 |
| Tree 10th–90th | 0.8508 | 1.1815 | 1.0120 | 0.8098 | 1.4289 |

Widths are in `$100,000` units. The conformal half-width is **0.5122**, so its
width is **1.0244** (`$102,440`) for every row.

## Interpretation

- **Marginal coverage is right.** Conformal hits **80.2%** against an 80%
  target, and it is narrower on average than the tree range.
- **The tree range over-covers.** It reaches **85.1%** because averaging 100
  trees widens the 10th and 90th percentiles.
- **Conformal fails at the cap.** It covers only **47.3%** of the 184 capped
  rows. Their true value is `5.0`, while predictions regress toward the mean,
  so a constant-width symmetric interval often stops short of `5.0`.
- **The tree range adapts.** At the cap its mean width grows to **1.4289**
  (`$142,890`), and coverage there stays at **81.0%**.

The cap failure is a bias problem, not a spread problem. Scaling the conformal
interval by the tree width was tested during development and did not fix it,
because the errors at the cap are systematically one-sided.

## Decision

The dashboard keeps the **tree range**. It is slightly conservative overall but
much safer where the model is known to struggle. A global conformal interval
would look better on average while quietly failing on the hardest rows.

If a calibrated interval is needed later, the right follow-up is
group-conditional conformal (one calibration per value band) or a method that
accounts for the censored target.

## Caveats

- **Exchangeability.** Conformal coverage assumes calibration and test rows
  are exchangeable. The random split satisfies this loosely; the spatial
  validation shows it does not hold across regions.
- **Censored target.** Values at `$500,000` are stored as `5.0`, which breaks
  the symmetry the interval relies on.
- **One coverage level.** Only the 80% level is checked. Other levels would
  need their own quantile.
- **Single split.** The numbers come from one fitting/calibration split with
  seed 42.
