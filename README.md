# House Price Prediction Using Machine Learning

[![CI](https://github.com/IsurangaHerath/House_Price_Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/IsurangaHerath/House_Price_Prediction/actions/workflows/ci.yml)

A small, reproducible regression project that predicts **historical California
census block-group median house values** from the California Housing dataset.
Built as a learning project and portfolio entry for Data Science internship
applications.

> **Important limitation.** The target is the **1990 US Census district median**,
> *not* current individual-house sale prices and *not* the market value of any
> particular property. Every prediction should be read with that caveat. See
> [`data/README.md`](./data/README.md) for the full dataset description.

## Current status

| Capability | Status |
|---|---|
| Dataset loading and validation | Complete, tested |
| Exploratory data analysis notebook | Complete |
| Baseline and regression models | Complete, tested |
| Model comparison and residual analysis | Complete, tested |
| Hyperparameter tuning | Complete, tested |
| Pipeline-based imputation | Complete, tested |
| Split-seed robustness | Complete, tested |
| Spatial validation | Complete, tested |
| Model explainability | Complete, tested |
| Prediction uncertainty | Complete, tested |
| Conformal calibration | Complete, tested |
| Subgroup error analysis | Complete, tested |
| Learning curve | Complete, tested |
| Model card | Complete |
| Streamlit dashboard | Complete |
| Saved model artifact | Complete, tested |
| Batch scoring | Complete, tested |
| Out-of-range warnings | Complete, tested |
| In-app reports | Complete, tested |
| Failure case analysis | Complete, tested |
| Geographic error map | Complete, tested |
| Prediction error context | Complete, tested |
| Continuous integration | Configured |

The training, tuning, robustness, spatial-validation, explainability,
uncertainty, calibration, and analysis workflows are reproducible with
`python -m src.housing.train`, `python -m src.housing.tuning`,
`python -m src.housing.robustness`, `python -m src.housing.spatial`,
`python -m src.housing.explain`, `python -m src.housing.uncertainty`,
`python -m src.housing.conformal`, and `python -m src.housing.analyze`. The
dashboard runs with `streamlit run app.py`. Measured results are stored in the
[`reports/`](./reports) folder, and intended use and ethical limits are
documented in the [`model card`](./reports/model_card.md).

## Repository layout

```text
house-price-prediction-ml/
├── .github/
│   └── workflows/
│       └── ci.yml             # tests and compile checks on push and PR
├── .gitignore                 # excludes venvs, caches, and generated artifacts
├── README.md                  # this file
├── requirements.txt           # pinned dependencies
├── app.py                     # Streamlit dashboard
├── data/
│   ├── README.md              # dataset provenance, units, and caveats
│   └── california_housing.csv # committed dataset (20,640 rows, 9 columns)
├── reports/
│   ├── baseline_results.md    # measured baseline metrics
│   ├── tuning_results.md      # random forest tuning comparison
│   ├── spatial_validation.md  # scores on held-out regions
│   ├── error_map.md           # where holdout errors cluster
│   ├── explainability.md      # importance and partial dependence
│   ├── uncertainty.md         # forest range and coverage
│   ├── conformal.md           # calibrated range comparison
│   ├── error_analysis.md      # errors by income, age, occupancy, latitude
│   ├── segment_errors.csv     # saved band-by-band holdout errors
│   ├── learning_curve.md      # error as the training set grows
│   ├── worst_predictions.md   # inspection of the largest errors
│   ├── worst_predictions.csv  # the 20 worst holdout rows
│   ├── model_card.md          # intended use, metrics, and limits
│   └── feature_importance.csv # permutation importance values
├── notebooks/
│   └── 01_eda.ipynb           # exploratory data analysis
├── models/
│   └── best_model.joblib      # generated artifact, ignored by Git
├── src/
│   └── housing/
│       ├── __init__.py
│       ├── data.py            # dataset loader and validator
│       ├── features.py        # feature/target split
│       ├── models.py          # model definitions and cross-validation
│       ├── evaluate.py        # regression metrics and residual summaries
│       ├── train.py           # training and evaluation workflow
│       ├── analyze.py         # residual analysis and error maps
│       ├── predict.py         # dashboard prediction helpers
│       ├── finalize.py        # train, save, and load the final model
│       ├── tuning.py          # random forest hyperparameter search
│       ├── robustness.py      # holdout score across split seeds
│       ├── spatial.py         # scores on held-out geographic regions
│       ├── explain.py         # permutation importance and dependence
│       ├── uncertainty.py     # prediction range from the forest's trees
│       ├── conformal.py       # calibrated range comparison
│       ├── batch.py           # score a CSV with the saved model
│       ├── segments.py        # errors across feature groups, saved as CSV
│       ├── learning_curve.py  # error as the training set grows
│       ├── bounds.py          # flag inputs outside the training range
│       ├── reports.py         # load committed reports for the dashboard
│       └── failures.py        # save the worst holdout predictions
└── tests/
    ├── test_data.py
    ├── test_features.py
    ├── test_evaluate.py
    ├── test_models.py
    ├── test_train.py
    ├── test_analyze.py
    ├── test_predict.py
    ├── test_finalize.py
    ├── test_tuning.py
    ├── test_robustness.py
    ├── test_spatial.py
    ├── test_explain.py
    ├── test_uncertainty.py
    ├── test_conformal.py
    ├── test_batch.py
    ├── test_segments.py
    ├── test_learning_curve.py
    ├── test_bounds.py
    ├── test_reports.py
    └── test_failures.py
```

## Dataset

`data/california_housing.csv` is committed to the repository, so no download
step is required.

| Property | Value |
|---|---|
| Rows | 20,640 census block groups |
| Features | MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude |
| Target | `MedHouseVal`, median house value in units of **$100,000** (the CSV stores dollars; the loader divides by 100,000) |
| Feature units | `AveRooms`, `AveBedrms`, and `AveOccup` are converted from raw totals/households to per-household averages |
| Target range | approximately 0.15 to 5.0 (about $15k to $500k) |

The file is derived from the California Housing data distributed by the UCI
Machine Learning Repository and prepared by the Scikit-learn project. The
original 1990 figures are preserved; column names follow Scikit-learn's schema,
the loader converts the raw room/occupancy totals to per-household averages,
and the target is converted from dollars to $100,000 units. Provenance,
units, and scientific caveats are documented in [`data/README.md`](./data/README.md).

## Setup

1. Create a virtual environment and install the pinned dependencies:

   ```powershell
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install --upgrade pip
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. Activate the environment when developing:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. Register the virtual environment as a Jupyter kernel (needed for the EDA
   notebook):

   ```powershell
   .venv\Scripts\python.exe -m ipykernel install --user --name housing --display-name "Python (housing)"
   ```

## Usage

```python
from src.housing.data import load_dataset, validate_frame

df = load_dataset()          # loads and prepares the CSV
summary = validate_frame(df) # raises ValueError if constraints are violated
print(summary)
```

`validate_frame` checks the row count, feature names, numeric dtypes, missing
values, and the historical target range. It returns a small summary dictionary
instead of silently accepting a corrupted dataset.

## Exploratory data analysis

Open the notebook with:

```powershell
.venv\Scripts\jupyter.exe notebook notebooks\01_eda.ipynb
```

Select the **Python (housing)** kernel. The notebook covers:

- dataset shape, dtypes, and missing values
- target distribution and the `$500,000` cap
- feature distributions
- the correlation matrix
- house values on a California map

## Running the workflows

Run the model comparison workflow:

```powershell
.venv\Scripts\python.exe -m src.housing.train
```

This loads and validates the data, splits it 80/20, runs 5-fold
cross-validation on the training set, then reports holdout metrics for the
dummy, linear regression, ridge, and random forest models.

Run the random forest tuning search:

```powershell
.venv\Scripts\python.exe -m src.housing.tuning
```

This samples a small set of hyperparameters on the training set and compares
the tuned model with the default on the holdout set.

Run the spatial validation:

```powershell
.venv\Scripts\python.exe -m src.housing.spatial
```

This groups nearby blocks into 10 regions and holds out one region at a time,
so neighbouring blocks cannot appear in both training and test.

Run the explainability analysis:

```powershell
.venv\Scripts\python.exe -m src.housing.explain
```

This computes permutation importance and partial dependence on the holdout
set, then saves the importance table and plots.

Run the uncertainty check:

```powershell
.venv\Scripts\python.exe -m src.housing.uncertainty
```

This uses the forest's 100 trees to build a 10th to 90th percentile range and
reports how often the holdout truth falls inside it.

Run the conformal calibration check:

```powershell
.venv\Scripts\python.exe -m src.housing.conformal
```

This holds out part of the training data to calibrate an 80% interval, then
compares it with the tree range on the same model.

Run the subgroup error analysis:

```powershell
.venv\Scripts\python.exe -m src.housing.segments
```

This scores the holdout set by income, house-age, occupancy, and latitude
bands, including the average direction of error in each group.

Run the learning curve:

```powershell
.venv\Scripts\python.exe -m src.housing.learning_curve
```

This trains the forest at five growing sizes on the training split and saves
the curve plot, showing whether more data would help.

Run the failure-case analysis:

```powershell
.venv\Scripts\python.exe -m src.housing.failures
```

This saves the 20 holdout rows with the largest absolute error and summarizes
what they have in common.

Run the residual analysis:

```powershell
.venv\Scripts\python.exe -m src.housing.analyze
```

This trains the same models, summarizes the best model's residuals, and saves
residual plots and a geographic error map to `reports/figures/`.

## Dashboard

Start the Streamlit app:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard has four tabs:

- **Predict** — enter block-group features and get an estimated median house
  value in dollars, a likely range from the forest's trees, and the typical
  holdout error for that income band. Inputs outside the 1st–99th percentile
  training range trigger a warning.
- **Model comparison** — cross-validation and holdout scores for all four
  models.
- **Data overview** — dataset summary and the target distribution.
- **Reports** — read the committed model card and validation reports without
  leaving the app.

The app loads `models/best_model.joblib` when it exists, including the saved
comparison results, so startup takes about 10 seconds instead of retraining for
over a minute. Without the artifact it trains the models and caches them. The
prediction form uses the model that won cross-validation.

## Saved model

Create or refresh the final model artifact:

```powershell
.venv\Scripts\python.exe -m src.housing.finalize
```

This runs cross-validation to pick the winner, refits it on all 20,640 rows,
and saves the model, its metadata, and the comparison tables with joblib. The
compressed file is about 40 MB and is ignored by Git, so generate it after a
fresh clone. Loading is handled by `src/housing/finalize.py`.

## Batch scoring

Score a CSV of block groups:

```powershell
.venv\Scripts\python.exe -m src.housing.batch --input data/california_housing.csv
```

The input needs the eight feature columns. Missing `AveBedrms` values are
imputed by the pipeline; other gaps are rejected. The command loads
`models/best_model.joblib` when it exists and trains a model otherwise.

By default it writes `<input>_predictions.csv` next to the input; pass
`--output` to choose another path. The output keeps the input columns and adds:

| Column | Meaning |
|---|---|
| `prediction_usd` | Predicted median value in dollars |
| `low_usd` | 10th percentile of the forest's tree predictions |
| `high_usd` | 90th percentile of the forest's tree predictions |
| `outside_range` | True when any feature is outside the training bounds |
| `outside_features` | Semicolon-separated names of the out-of-range features |

The bounds are the 1st and 99th percentiles of each feature in the committed
dataset. Flagged rows are still scored, but the model extrapolates poorly
outside the training distribution, so treat those estimates with caution.

Scoring the committed training CSV is an in-sample demo. Use new block groups
for real estimates, and remember that the values describe 1990. The upper bound
can sit above `$500,000` because it comes from tree disagreement; the target
itself is capped there.

## Hyperparameter tuning

A small randomized search improved the random forest settings to:

```text
max_depth=30, max_features=0.5, min_samples_leaf=1
```

On the same five folds, mean MAE fell from **0.3351** to **0.3309**, and every
fold improved. Holdout MAE fell from **0.3289** to **0.3252**. Full details are
in [`reports/tuning_results.md`](./reports/tuning_results.md).

## Data quality checks

Missing bedrooms are kept as `NaN` by the loader. Every model is a pipeline
whose first step, `BedroomImputer`, learns the median bedroom-to-room share
from the training rows and fills the gaps there. This avoids using test data
and replaces the earlier fixed fill.

`python -m src.housing.robustness` refits the tuned forest on five different
80/20 splits. Holdout MAE is **0.3222 ± 0.0052**, so the project's seed is not
a lucky outlier.

## Spatial validation

A random split puts neighbouring blocks in both training and test.
`python -m src.housing.spatial` groups blocks into 10 regions with KMeans and
holds out one region at a time with `GroupKFold`.

| Metric | Random holdout | Spatial holdout |
|---|---:|---:|
| MAE | 0.325 | 0.538 |
| MAE ($) | 32,524 | 53,777 |
| RMSE | 0.501 | 0.743 |
| R² | 0.809 | 0.585 |

On a genuinely unseen region, error rises by about **$21,000** per block. The
model still beats the mean baseline, but it does not generalise as well across
geography as the random-split number suggests. Per-region details are in
[`reports/spatial_validation.md`](./reports/spatial_validation.md).

## Explainability

Permutation importance on the holdout set, in MAE increase when a feature is
shuffled:

| Feature | Increase in MAE |
|---|---:|
| MedInc | 0.417 |
| Latitude | 0.322 |
| Longitude | 0.280 |
| AveOccup | 0.153 |
| HouseAge | 0.051 |
| AveRooms | 0.045 |
| AveBedrms | 0.008 |
| Population | 0.005 |

Median income is the strongest driver, followed by location. Partial
dependence shows predicted value rising with income and falling from the coast
toward inland areas. These are associations, not causal effects. Full details
are in [`reports/explainability.md`](./reports/explainability.md).

## Prediction uncertainty

Each tree in the forest predicts a slightly different value. The dashboard
shows the 10th to 90th percentile of those 100 predictions as a likely range.
`python -m src.housing.uncertainty` measures that range on the holdout set:

| Metric | Value |
|---|---:|
| Coverage | 85.2% |
| Mean width | 1.161 ($116,120) |
| Median width | 0.988 ($98,770) |
| Coverage at the target cap | 83.7% |

The requested range was 80%, so the measured coverage is slightly
conservative. The range describes how much the trees disagree; it is not a
guarantee and does not include market or sampling uncertainty. Details are in
[`reports/uncertainty.md`](./reports/uncertainty.md).

`python -m src.housing.conformal` checks whether a formally calibrated interval
would be better. On the same fitted model:

| Method | Coverage | Mean width | Cap coverage |
|---|---:|---:|---:|
| Conformal 80% | 80.2% | 1.024 | 47.3% |
| Tree 10th–90th | 85.1% | 1.182 | 81.0% |

Conformal hits its average target and is narrower, but it covers less than half
of the capped rows because their truth sits at `$500,000` while predictions
regress toward the mean. The tree range adapts at the cap, so the dashboard
keeps it. Details are in [`reports/conformal.md`](./reports/conformal.md).

## Subgroup errors

`python -m src.housing.segments` scores the holdout set by feature bands. The
clearest gap is income:

| Segment | Rows | MAE |
|---|---:|---:|
| Income under $20k | 495 | 0.258 |
| Income $20k–$35k | 1,566 | 0.275 |
| Income $35k–$50k | 1,215 | 0.349 |
| Income $50k+ | 852 | 0.423 |

MAE rises **64%** from the lowest to the highest income band. Occupancy 1–1.9
is the hardest small group (MAE **0.582** on 311 rows), while occupancy 3+ is
the easiest (**0.240**). The northern latitude band is easiest (**0.239**), and
34–35.9 is hardest (**0.354**). Bias is within about `±0.02` in most groups, so
these differences are mostly variance, not a systematic shift. Full tables are
in [`reports/error_analysis.md`](./reports/error_analysis.md), and the same
numbers are saved as [`reports/segment_errors.csv`](./reports/segment_errors.csv).
The dashboard uses that table to show the typical error for the income band of
each prediction.

## Learning curve

`python -m src.housing.learning_curve` trains the forest at five growing sizes
on the training split:

| Training rows | Train MAE | Validation MAE |
|---:|---:|---:|
| 2,641 | 0.141 | 0.389 |
| 5,283 | 0.132 | 0.360 |
| 7,925 | 0.127 | 0.346 |
| 10,567 | 0.125 | 0.337 |
| 13,209 | 0.123 | 0.330 |

Validation MAE falls about **15%** across the curve and is still dropping at the
largest size, so more data would help modestly. The train/validation gap of
about **0.21** shows the forest fits its training rows far better than it
generalizes. Details are in
[`reports/learning_curve.md`](./reports/learning_curve.md).

## Worst predictions

`python -m src.housing.failures` saves the 20 largest holdout errors. Their mean
absolute error is **2.641** (`$264,080`), about **8 times** the overall MAE.

- **14 of 20** are capped rows, all under-predicted by `$231,000`–`$330,000`.
- The four over-predictions are blocks with true values at or below **1.25**
  that look ordinary in the features.
- **6 of 20** have fewer than 200 residents, versus **2.4%** of all blocks, so
  tiny block groups are about **12 times** overrepresented among the worst
  errors.

The full table is committed as
[`reports/worst_predictions.csv`](./reports/worst_predictions.csv), and the
interpretation is in
[`reports/worst_predictions.md`](./reports/worst_predictions.md).

## Error geography

`python -m src.housing.analyze` also saves a map of holdout blocks shaded by
absolute error:

| Latitude band | Rows | MAE |
|---|---:|---:|
| 34 or less | 1,288 | 0.324 |
| 34–35 | 1,011 | **0.361** |
| 35–36 | 99 | 0.306 |
| 36–37 | 254 | 0.297 |
| Above 37 | 1,476 | 0.307 |

The 34–35 band is about **11%** harder than the overall holdout MAE, and the
-120 to -118 longitude band is the hardest east–west slice (MAE **0.344**).
The largest 10% of errors spread across almost the whole populated range of the
dataset, so there is no single geographic blind spot. Details are in
[`reports/error_map.md`](./reports/error_map.md). The figure itself is
regenerated into `reports/figures/` rather than committed.

## Model card

[`reports/model_card.md`](./reports/model_card.md) records the model's intended
use, training data, evaluation results, and ethical limitations. The headline
point: this is a 1990 block-group estimator for learning and demonstration, not
a tool for appraising individual homes or making decisions about people.

## Model comparison and residual analysis

### Cross-validation on the training set

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random forest (tuned) | 0.331 | 0.502 | 0.812 |
| Ridge | 0.529 | 0.721 | 0.611 |
| Linear regression | 0.529 | 0.721 | 0.611 |
| Dummy mean | 0.914 | 1.156 | 0.000 |

### Holdout test set

| Model | MAE | RMSE | R² | MAE ($) |
|---|---:|---:|---:|---:|
| Random forest (tuned) | 0.325 | 0.501 | 0.809 | 32,524 |
| Ridge | 0.535 | 0.748 | 0.574 | 53,451 |
| Linear regression | 0.535 | 0.748 | 0.573 | 53,452 |
| Dummy mean | 0.906 | 1.145 | 0.000 | 90,607 |

The random forest is clearly the best model. Its holdout MAE is about
**$32,500**, and it explains about **81%** of the variance.

Residual checks for the random forest:

- mean residual: **-0.011** (slightly over-predicting on average)
- underprediction rate: **41.6%**
- MAE at the target cap: **0.761** vs **0.305** below the cap

The much larger error at the cap is expected: many different high values all
appear as `5.0`, so the model cannot separate them. Full details and caveats
are in [`reports/baseline_results.md`](./reports/baseline_results.md).

## Tests

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

All 69 tests currently pass locally, and the same suite runs on every push and
pull request through GitHub Actions.

## Continuous integration

The workflow in [`.github/workflows/ci.yml`](./.github/workflows/ci.yml) runs on
Ubuntu with Python 3.12. It:

1. installs the pinned dependencies from `requirements.txt`;
2. checks that `app.py` and the `src` package compile;
3. runs the full pytest suite.

The badge at the top of this README shows the latest result.

## Reproducibility

- Dataset: committed CSV, so results do not depend on a network download.
- Environment: pinned versions in `requirements.txt` (Python 3.12).
- Split: 80/20 train/test, `random_state=42`.
- Cross-validation: 5 shuffled folds, `random_state=42`.
- Models: dummy mean baseline, linear regression, ridge, and a tuned random
  forest (100 trees, `max_depth=30`, `max_features=0.5`,
  `min_samples_leaf=1`).
- Tuning: 6 sampled candidates, 3-fold CV on the training set only.
- Imputation: missing bedrooms are filled inside each model pipeline using
  the training rows' median bedroom share.
- Robustness: holdout MAE is checked across five random split seeds.
- Spatial validation: 10 KMeans regions are held out one at a time with
  `GroupKFold`.
- Explainability: permutation importance and partial dependence on the holdout
  set.
- Uncertainty: the 10th to 90th percentile of the forest's tree predictions,
  checked against holdout coverage.
- Calibration: split conformal with a 75/25 fitting/calibration split inside
  the training data.
- Subgroup errors: fixed income, house-age, occupancy, and latitude bands on
  the holdout set, saved as `reports/segment_errors.csv` for the dashboard.
- Learning curve: five training sizes with three 80/20 shuffle splits on the
  training data.
- Input guards: dashboard and batch scoring flag features outside the 1st–99th
  percentile training bounds.
- Failure cases: the 20 largest holdout errors are saved for inspection.
- Error map: holdout blocks are shaded by absolute error, with colours
  saturated at the 95th percentile.
- Metrics: MAE, RMSE, and R² in target units, plus MAE/RMSE in dollars.
- Residuals: mean/median error, underprediction rate, and cap-vs-uncapped MAE.
- Dashboard: reuses the training workflow, caches results, and fits the
  winning model on all rows for predictions.
- Final model: `python -m src.housing.finalize` saves a compressed artifact
  with its metadata and comparison results.
- Batch scoring: `python -m src.housing.batch` adds predictions and the tree
  range to a CSV.

## Known limitations

1. **Target ceiling.** Values at 5.0 are censored at the $500k census cap, so
   high-value districts are harder to predict. The random forest's MAE is
   about 0.761 on capped rows versus 0.305 on the rest.
2. **Geographic leakage.** Rows are geographically clustered, so a random
   train/test split places neighbouring blocks in both folds. The random-split
   MAE is 0.325, but holding out whole regions raises it to 0.538
   (`python -m src.housing.spatial`). Treat the random-split score as an
   in-geography estimate, not a promise for a new region.
3. **Historical data.** The values describe 1990, not the current housing market.
4. **Imputed bedrooms.** The 207 missing `AveBedrms` values are imputed inside
   the model pipeline from the training rows' median bedroom share, so those
   rows are approximate. With seed 42 all 207 fall in the holdout set. Across
   five split seeds the holdout MAE is 0.322 ± 0.005
   (`python -m src.housing.robustness`).
5. **Ranges are not guarantees.** The dashboard's likely range is the spread of
   the forest's trees, not a calibrated confidence interval. It covers 85% of
   holdout rows, but coverage on a new region is unmeasured. A global conformal
   interval meets 80% average coverage yet covers only 47% of capped rows.

## License

MIT
