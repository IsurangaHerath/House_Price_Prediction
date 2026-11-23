# Model card: California Housing median value predictor

## Model details

| Item | Value |
|---|---|
| Model type | Scikit-learn pipeline: `BedroomImputer` + tuned `RandomForestRegressor` |
| Task | Regression |
| Target | 1990 census block-group median house value, in `$100,000` units |
| Inputs | 8 numeric block-group features |
| Forest settings | 100 trees, `max_depth=30`, `max_features=0.5`, `min_samples_leaf=1` |
| Random state | 42 |
| Artifact | `models/best_model.joblib` (generated, not committed) |

## Intended use

This model is a **learning and portfolio project**. It is intended for:

- exploring the California Housing dataset
- demonstrating a reproducible regression workflow
- estimating a 1990 block-group median from aggregate features

It is **not** intended for:

- individual house appraisals or sale-price estimates
- current market pricing; the data describes 1990
- lending, insurance, tenant screening, or any decision about a person
- comparing neighbourhoods as a value judgment

Every prediction is an aggregate estimate for a census block group, not a
specific property.

## Training data

| Item | Value |
|---|---|
| Source | California Housing (StatLib/UCI, prepared by Scikit-learn) |
| Rows | 20,640 census block groups |
| Features | MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude |
| Target | `MedHouseVal`, 1990 census median |
| Target range | about 0.15 to 5.0 (`$15,000` to `$500,000`) |
| Missing values | 207 missing `AveBedrms` values, imputed inside the pipeline |
| Geography | California |
| Time period | 1990 |

The loader converts raw room/bedroom/household totals to per-household
averages and converts the target from dollars to `$100,000` units. Details are
in [`data/README.md`](../data/README.md).

## Evaluation

### Random 80/20 holdout

| Metric | Value |
|---|---:|
| MAE | 0.325238 |
| MAE ($) | 32,524 |
| RMSE | 0.500555 |
| R² | 0.808796 |

### Cross-validation

5 shuffled folds on the training set: MAE **0.330936**, RMSE **0.501851**,
R² **0.811527**.

### Held-out regions

10 KMeans regions, one held out at a time with `GroupKFold`:

| Metric | Value |
|---|---:|
| MAE | 0.537766 |
| MAE ($) | 53,777 |
| RMSE | 0.743119 |
| R² | 0.585276 |

Performance drops sharply on an unseen region, so the random-split score
should not be read as a cross-region promise.

### Split-seed robustness

Across five random 80/20 splits, holdout MAE is **0.3222 ± 0.0052**. Seed 42
is not a lucky split.

### Error at the target cap

| Group | Rows | MAE |
|---|---:|---:|
| At cap (`MedHouseVal >= 5.0`) | 184 | 0.761061 |
| Below cap | 3,944 | 0.304905 |

Capped rows are much harder because many different high values all appear as
`5.0`.

### Prediction range

The 10th to 90th percentile of the forest's 100 tree predictions covers
**85.2%** of holdout rows. Mean width is **1.161** (`$116,120`) and median
width is **0.988** (`$98,770`). At the `$500,000` cap, coverage is **83.7%**.
The range reflects disagreement between trees, not market or sampling
uncertainty.

A split conformal 80% interval was also checked. It reaches **80.2%** coverage
with a narrower mean width of **1.024**, but it covers only **47.3%** of capped
rows because predictions regress toward the mean while the truth stays at
`5.0`. The dashboard keeps the tree range for that reason.

### Subgroup errors

Holdout MAE by income band:

| Segment | Rows | MAE |
|---|---:|---:|
| under $20k | 495 | 0.258 |
| $20k–$35k | 1,566 | 0.275 |
| $35k–$50k | 1,215 | 0.349 |
| $50k+ | 852 | 0.423 |

Absolute error rises **64%** from the lowest to the highest income band.
Occupancy 1–1.9 is the hardest small group (MAE **0.582** on 311 rows), and
occupancy 3+ is the easiest (**0.240**). The northern latitude band is easiest
(**0.239**), while 34–35.9 is hardest (**0.354**). Bias is within about `±0.02`
in most groups, so the differences are mostly variance rather than a systematic
shift.

### Data sufficiency

A learning curve on the training split drops validation MAE from **0.389** at
2,641 rows to **0.330** at 13,209 rows. The curve is still falling, so more
data would help modestly. Training MAE stays near **0.123**, leaving a gap of
about **0.21** against validation error.

Full reports: [`baseline_results.md`](./baseline_results.md),
[`tuning_results.md`](./tuning_results.md),
[`spatial_validation.md`](./spatial_validation.md),
[`explainability.md`](./explainability.md),
[`uncertainty.md`](./uncertainty.md),
[`conformal.md`](./conformal.md),
[`error_analysis.md`](./error_analysis.md), and
[`learning_curve.md`](./learning_curve.md).

## Explainability

Permutation importance on the holdout set:

| Feature | Mean increase in MAE |
|---|---:|
| MedInc | 0.416866 |
| Latitude | 0.322054 |
| Longitude | 0.280269 |
| AveOccup | 0.153487 |
| HouseAge | 0.050899 |
| AveRooms | 0.044739 |
| AveBedrms | 0.008400 |
| Population | 0.005478 |

Income and location dominate. Partial dependence shows predicted value rising
with median income and falling from the coast toward inland areas. These are
associations, not causal effects.

## Limitations and ethical considerations

1. **Ecological fallacy.** The target is a block-group median. Applying it to
   an individual house or person is invalid.
2. **Historical bias.** The data comes from 1990 and reflects the income,
   housing, and geographic patterns of that time, including segregation and
   disinvestment. The model can reproduce those patterns.
3. **Geographic generalization.** On held-out regions, MAE rises from 0.325 to
   0.538 (`$53,777`). Do not assume it works in a new area.
4. **Censored target.** Values at `$500,000` are capped, so the model cannot
   distinguish high-value districts above the cap.
5. **Imputed values.** 207 `AveBedrms` entries are missing and filled inside
   the pipeline from training-row patterns.
6. **Uneven subgroup accuracy.** Holdout MAE is 0.258 for blocks under `$20k`
   income and 0.423 for blocks at `$50k+`, so a single overall score hides
   where estimates are less reliable.
7. **No current market signal.** The model knows nothing about post-1990
   prices, interest rates, renovations, or individual property condition.

## Input guards

The dashboard and batch scorer compare each input with the 1st and 99th
percentile bounds of the training data. Features outside those bounds are
flagged because the model extrapolates poorly, especially across geography.
Flagged rows are still scored; the warning is a caution, not a rejection.

## Reproducibility

```powershell
.venv\Scripts\python.exe -m src.housing.train
.venv\Scripts\python.exe -m src.housing.tuning
.venv\Scripts\python.exe -m src.housing.robustness
.venv\Scripts\python.exe -m src.housing.spatial
.venv\Scripts\python.exe -m src.housing.explain
.venv\Scripts\python.exe -m src.housing.uncertainty
.venv\Scripts\python.exe -m src.housing.conformal
.venv\Scripts\python.exe -m src.housing.segments
.venv\Scripts\python.exe -m src.housing.learning_curve
.venv\Scripts\python.exe -m src.housing.analyze
.venv\Scripts\python.exe -m src.housing.finalize
```

## License

MIT
