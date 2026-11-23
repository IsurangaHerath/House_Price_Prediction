# Prediction uncertainty results

Generated with:

```powershell
.venv\Scripts\python.exe -m src.housing.uncertainty
```

## Why this check matters

A single predicted number hides how sure the model is. This workflow uses the
random forest's 100 trees to show a range: each tree predicts a value, and the
10th to 90th percentile of those predictions becomes the low and high bound.

## Method

| Item | Value |
|---|---|
| Model | Tuned random forest pipeline |
| Evaluation rows | Holdout test set (4,128 rows) |
| Range | 10th to 90th percentile of the 100 tree predictions |
| Coverage | Share of true values inside the range |
| Random state | 42 |

The point prediction is the forest average, which always sits inside the tree
range.

## Holdout results

| Metric | Value |
|---|---:|
| Coverage | 0.8520 |
| Mean width | 1.1612 ($116,120) |
| Median width | 0.9877 ($98,770) |
| Coverage at the target cap | 0.8370 |

The requested range was 80%, and the measured coverage is **85.2%**. The
interval is slightly conservative: averaging 100 trees smooths the extremes,
so the 10th and 90th tree quantiles sit a little wider than a calibrated 80%
interval would.

Typical width is about **$100,000 to $116,000**. That is large in absolute
terms, but the target itself spans roughly `$15,000` to `$500,000`.

Coverage on the 184 capped rows is **83.7%**, so the range is not badly
miscalibrated at the `$500,000` ceiling.

## What it means for the dashboard

The Predict tab now shows:

```text
Predicted median house value: $220,487
Likely range: $176,460 to $264,260
```

The range communicates model disagreement, not a guarantee. It is a
convenience signal for the demo, not a statistical confidence interval for a
single house.

## Caveats

- **Tree spread is not data noise.** It measures how much the trees disagree.
  It does not include uncertainty from the training sample or from a changing
  market.
- **Coverage is measured on the random holdout.** Because neighbouring blocks
  appear in both training and test, the spatial-validation result suggests
  coverage could be worse on an unseen region.
- **Capped target.** Many high values are stored as `5.0`, so the upper bound
  often sits near the cap and cannot express how far above it the true value
  might be.
- **One global setting.** The same 10th to 90th percentile is used for every
  row. A per-row calibration check would be a larger follow-up.

A conformal alternative is evaluated in
[`conformal.md`](./conformal.md). It meets 80% average coverage but covers less
than half of capped rows, so the tree range stays in the dashboard.
