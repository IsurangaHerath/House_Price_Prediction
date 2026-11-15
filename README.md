# House Price Prediction Using Machine Learning

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
| Exploratory data analysis notebook | Not started |
| Baseline and regression models | Complete, tested |
| Model comparison and residual analysis | Comparison done; residual analysis pending |
| Streamlit dashboard | Not started |

The baseline workflow is reproducible with `python -m src.housing.train`.
Measured results are stored in
[`reports/baseline_results.md`](./reports/baseline_results.md).

## Repository layout

```text
house-price-prediction-ml/
├── .gitignore                 # excludes venvs, caches, and generated artifacts
├── README.md                  # this file
├── requirements.txt           # pinned dependencies
├── data/
│   ├── README.md              # dataset provenance, units, and caveats
│   └── california_housing.csv # committed dataset (20,640 rows, 9 columns)
├── reports/
│   └── baseline_results.md    # measured baseline metrics
├── src/
│   └── housing/
│       ├── __init__.py
│       ├── data.py            # dataset loader and validator
│       ├── features.py        # feature/target split
│       ├── models.py          # model definitions and cross-validation
│       ├── evaluate.py        # regression metrics
│       └── train.py           # baseline training workflow
└── tests/
    ├── test_data.py
    ├── test_features.py
    ├── test_evaluate.py
    ├── test_models.py
    └── test_train.py
```

Planned additions: a notebook for exploration and residual analysis, and
`app.py` for the dashboard.

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

Run the baseline workflow:

```powershell
.venv\Scripts\python.exe -m src.housing.train
```

This loads and validates the data, splits it 80/20, runs 5-fold
cross-validation on the training set, then reports holdout metrics for the
dummy baseline and linear regression models.

## Baseline results

| Stage | Model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| 5-fold CV | Linear regression | 0.529 | 0.721 | 0.611 |
| 5-fold CV | Dummy mean | 0.914 | 1.156 | 0.000 |
| Holdout | Linear regression | 0.573 | 0.803 | 0.508 |
| Holdout | Dummy mean | 0.906 | 1.145 | 0.000 |

Errors are in units of `$100,000`, so the linear model's holdout MAE is about
**$57,300**. Linear regression clearly beats the mean baseline. Full details
and caveats are in [`reports/baseline_results.md`](./reports/baseline_results.md).

## Tests

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

All 21 tests currently pass.

## Reproducibility

- Dataset: committed CSV, so results do not depend on a network download.
- Environment: pinned versions in `requirements.txt` (Python 3.12).
- Split: 80/20 train/test, `random_state=42`.
- Cross-validation: 5 shuffled folds, `random_state=42`.
- Models: dummy mean baseline and `StandardScaler` + linear regression.
- Metrics: MAE, RMSE, and R² in target units, plus MAE/RMSE in dollars.

## Known limitations

1. **Target ceiling.** Values at 5.0 are censored at the $500k census cap, so
   high-value districts are harder to predict and metrics computed without
   noting the cap count understate that difficulty.
2. **Geographic leakage.** Rows are geographically clustered, so a random
   train/test split places neighbouring blocks in both folds. Reported scores
   describe prediction *within* California, not in an unseen region.
3. **Historical data.** The values describe 1990, not the current housing market.
4. **Imputed bedrooms.** The 207 missing `AveBedrms` values are filled with
   `AveRooms * 0.5`, so those rows are approximate. With the current seed, all
   207 rows fall in the holdout set.

## License

MIT
