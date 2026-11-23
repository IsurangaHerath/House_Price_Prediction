# Error geography

Generated with:

```powershell
.venv\Scripts\python.exe -m src.housing.analyze
```

The plot is saved as
[`figures/error_map.png`](./figures/error_map.png). Figures are regenerated on
demand and are not committed.

## Method

| Item | Value |
|---|---|
| Model | Tuned random forest pipeline (best by training CV) |
| Evaluation rows | Holdout test set (4,128 rows) |
| Colour | Absolute error, `abs(true - predicted)` |
| Colour scale | Saturated at the 95th percentile (**1.065**) so a few huge errors do not flatten the rest |
| Random state | 42 |

## What the map shows

Each point is one holdout block group at its latitude and longitude. Dark
points are close to the truth; bright points are large errors. The brightest
spots are scattered rather than forming a single hot region, which fits the
spatial validation result: errors are similar across held-out regions, and the
model has no single geographic blind spot.

The geographic spread of the **largest 10%** of errors is wide:

| Coordinate | Range |
|---|---|
| Latitude | 32.68 to 39.19 |
| Longitude | -122.83 to -115.55 |

That covers most of the populated part of California in the dataset. The very
largest errors come from a handful of unusual blocks (see
[`worst_predictions.md`](./worst_predictions.md)), not from one town.

## Error by latitude band

| Latitude | Rows | Mean absolute error |
|---|---:|---:|
| 34 or less | 1,288 | 0.3244 |
| 34–35 | 1,011 | **0.3613** |
| 35–36 | 99 | 0.3060 |
| 36–37 | 254 | 0.2972 |
| Above 37 | 1,476 | 0.3074 |

The 34–35 band is the hardest, about **11%** above the overall holdout MAE of
**0.3252**. This is the Southern California corridor where the worst capped
blocks also cluster. The northern bands are slightly easier.

## Error by longitude band

| Longitude | Rows | Mean absolute error |
|---|---:|---:|
| -120 or less | 1,586 | 0.3151 |
| -120 to -118 | 1,509 | **0.3436** |
| -118 to -116 | 1,000 | 0.3178 |
| Above -116 | 33 | 0.1989 |

The -120 to -118 band is the hardest. The easternmost group has too few rows
to draw a conclusion.

## Implications

- Geography alone does not explain most errors. A location-based correction
  would not help much beyond what the forest already learns from latitude and
  longitude.
- The modest 34–35 and -120 to -118 increases are consistent with the cap and
  small-population effects found in the failure cases, rather than a distinct
  geographic problem.
- The map is a diagnostic, not a feature. It confirms the spatial validation
  and failure-case results visually.

## Caveats

- **Random split.** Nearby blocks appear in both training and test, so this map
  shows within-geography errors. It is not a held-out-region evaluation; see
  [`spatial_validation.md`](./spatial_validation.md) for that.
- **Saturated colours.** Errors above **1.065** all share the brightest colour,
  so the few extreme blocks cannot be ranked from the map alone. The committed
  table covers those rows.
- **Point overlap.** Dense urban areas overplot, which can hide local patterns.
