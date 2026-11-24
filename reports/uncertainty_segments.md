# Uncertainty across subgroups

Generated with:

```powershell
.venv\Scripts\python.exe -m src.housing.uncertainty
```

The full table is committed as
[`uncertainty_segments.csv`](./uncertainty_segments.csv). The headline interval
results are in [`uncertainty.md`](./uncertainty.md).

## Why this check matters

The dashboard shows a 10th–90th tree range for every prediction. An overall
coverage of 85.2% does not guarantee that the range behaves the same for every
kind of block. This workflow breaks coverage and interval width down by the same
feature bands used in [`error_analysis.md`](./error_analysis.md).

## Method

| Item | Value |
|---|---|
| Model | Tuned random forest pipeline |
| Evaluation rows | Holdout test set (4,128 rows) |
| Interval | 10th–90th percentile of the 100 trees |
| Coverage | Share of true values inside the interval |
| Width | Upper minus lower bound, in target units |

## By income band

| Income | Rows | Coverage | Mean width |
|---|---:|---:|---:|
| Under $20k | 495 | **0.820** | 0.875 |
| $20k–$35k | 1,566 | 0.850 | 1.013 |
| $35k–$50k | 1,215 | 0.856 | 1.255 |
| $50k+ | 852 | **0.869** | 1.467 |

Coverage stays between **82%** and **87%** across income bands, always above the
nominal 80% target. Intervals get wider as income rises, which matches the
larger errors measured in
[`error_analysis.md`](./error_analysis.md). The lowest band is the only one
noticeably below the overall 85.2%, by about 3 percentage points.

## By occupancy band

| Occupancy | Rows | Coverage | Mean width |
|---|---:|---:|---:|
| 1–1.9 | 311 | 0.855 | **2.013** |
| 2–2.9 | 2,212 | 0.851 | 1.273 |
| 3+ | 1,605 | 0.852 | 0.842 |

This is the clearest adaptation: low-occupancy blocks, which have the highest
MAE (**0.582**), get intervals about **2.4 times** wider than high-occupancy
blocks, while coverage stays the same. The forest's tree spread does reflect
that these rows are harder to predict.

## Other bands

- **House age:** coverage ranges from **0.831** (20–29 years) to **0.869**
  (30–39 years). Widths vary from 1.04 to 1.32, with no simple trend.
- **Latitude:** coverage ranges from **0.838** (34–35.9) to **0.860** (under 34).
  The northern band has the narrowest intervals (0.883), consistent with its
  lower error.

## Implications

- The interval is reasonably reliable for all four subgroup definitions. No
  band needs a separate correction.
- Width tracks difficulty where it matters most: income and occupancy. A wide
  range is a useful warning sign in the dashboard.
- The low-income band is slightly under-covered while higher bands are
  over-covered. This is a small calibration difference, not a failure.

## Caveats

- **Nominal vs empirical.** A 10th–90th range would cover 80% if tree spread
  were perfectly calibrated. Actual coverage is 82–87%, so the intervals are
  conservative overall.
- **Censored target.** The 5.0 cap limits how wide an interval can usefully be;
  see [`conformal.md`](./conformal.md) for a related comparison.
- **Correlated bands.** The same holdout rows appear in every table, so the
  findings are not independent tests.
