# Worst holdout predictions

Generated with:

```powershell
.venv\Scripts\python.exe -m src.housing.failures
```

The full table is committed as
[`worst_predictions.csv`](./worst_predictions.csv).

## Why this check matters

Aggregate metrics say how large errors are on average. They do not show what
the biggest mistakes look like. This workflow saves the 20 holdout rows with
the largest absolute error so the failure modes can be inspected directly.

## Method

| Item | Value |
|---|---|
| Model | Tuned random forest pipeline |
| Evaluation rows | Holdout test set (4,128 rows) |
| Selection | Top 20 by absolute error |
| Error | `true - predicted`; positive means under-prediction |
| Random state | 42 |

## Summary

| Metric | Value |
|---|---:|
| Mean absolute error of the worst 20 | 2.6408 ($264,080) |
| Rows at the target cap | 14 |
| Under-predictions | 16 |
| Rows with population under 200 | 6 |
| Rows with imputed bedrooms | 1 |

For comparison, the overall holdout MAE is **0.3252** ($32,524), so these rows
are about **8 times** worse than the average.

## What stands out

### 1. The cap dominates

14 of the 20 worst rows have a true value of `5.0`. Every one is
under-predicted, with predictions between **1.70** and **2.69** and errors from
**2.31** to **3.30** (`$231,000` to `$330,000`). The model cannot separate
districts whose true values are all stored as the same capped number.

### 2. The over-predictions are anomalously low-value blocks

The four over-predictions have true values at or below **1.25**, near the
bottom of the target range, while the model predicts **2.40** to **3.88**:

| True | Predicted | Error | MedInc | AveOccup | Population | Latitude | Longitude |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.175 | 3.095 | -2.920 | 2.367 | 1.877 | 259 | 34.15 | -118.33 |
| 1.250 | 3.880 | -2.630 | 7.300 | 2.172 | 63 | 34.17 | -119.08 |
| 0.600 | 2.937 | -2.337 | 2.562 | 1.649 | 94 | 37.82 | -122.29 |
| 0.150 | 2.398 | -2.248 | 4.193 | 2.791 | 628 | 34.24 | -117.86 |

Their income, occupancy, and location look ordinary, so the aggregate features
do not explain why the true median is so low. These may be unusual populations
or data quirks; the dataset cannot confirm the cause.

### 3. Tiny block groups are overrepresented

6 of the 20 worst rows have fewer than **200** residents. Across the whole
dataset only **2.4%** of blocks are that small, and the median population is
**1,166**. Small block groups are therefore about **12 times** more common
among the worst errors than in the data overall. A median computed from a few
dozen people is noisy, and the model has little signal to learn from.

### 4. Other details

- One of the worst rows has a missing `AveBedrms` value filled by the pipeline
  imputer.
- Most worst rows sit in Southern California (latitude about 32.8–34.4,
  longitude about -116 to -119), with a few Bay Area and Sacramento blocks.

## Implications

- Treat predictions for **capped** and **very small** block groups as rough
  estimates, not precise values.
- The out-of-range guard flags the smallest populations (below the 1st
  percentile, about 88 residents), but blocks with 100–200 residents can still
  be noisy while staying inside the bounds.
- A future improvement could flag small population explicitly, or model the
  cap separately.

## Caveats

- **Random split.** Neighbouring blocks appear in both training and test, so
  these are within-geography failures.
- **Censored target.** Values at `$500,000` are stored as `5.0`, which creates
  many of the largest errors by construction.
- **Small sample.** Twenty rows illustrate patterns; they are not a statistical
  estimate of failure rates.
- **Aggregate data.** Block-group features cannot reveal why a specific median
  is unusually low or high.
