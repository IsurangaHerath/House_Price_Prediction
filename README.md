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
| Baseline and regression models | Not started |
| Model comparison and residual analysis | Not started |
| Streamlit dashboard | Not started |

No model results are reported yet. This README will be updated with measured
metrics only after the experiments have actually been run.

## Repository layout

```text
house-price-prediction-ml/
├── .gitignore                 # excludes venvs, caches, and generated artifacts
├── README.md                  # this file
├── requirements.txt           # pinned dependencies
├── data/
│   ├── README.md              # dataset provenance, units, and caveats
│   └── california_housing.csv # committed dataset (20,640 rows, 9 columns)
├── src/
│   └── housing/
│       └── data.py            # dataset loader and validator
└── tests/
    └── test_data.py           # pytest suite for loading and validation
```

Planned additions: `notebooks/` (exploration), `src/housing/models.py`
(training and evaluation), and `app.py` (dashboard).

## Dataset

`data/california_housing.csv` is committed to the repository, so no download
step is required.

| Property | Value |
|---|---|
| Rows | 20,640 census block groups |
| Features | MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude |
| Target | `MedHouseVal`, median house value in units of **$100,000** (the CSV stores dollars; the loader divides by 100,000) |
| Target range | approximately 0.15 to 5.0 (about $15k to $500k) |

The file is derived from the California Housing data distributed by the UCI
Machine Learning Repository and prepared by the Scikit-learn project. The
original 1990 figures are preserved; column names follow Scikit-learn's schema
and the loader converts the target from dollars to $100,000 units. Provenance,
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

## Tests

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

All four data tests currently pass.

## Reproducibility

- Dataset: committed CSV, so results do not depend on a network download.
- Environment: pinned versions in `requirements.txt` (Python 3.12).
- Random seeds, train/test split configuration, cross-validation settings, and
  model parameters will be documented once the modeling code exists.

## Known limitations

1. **Target ceiling.** Values at 5.0 are censored at the $500k census cap, so
   high-value districts are harder to predict and metrics computed without
   noting the cap count understate that difficulty.
2. **Geographic leakage.** Rows are geographically clustered, so a random
   train/test split places neighbouring blocks in both folds. Reported scores
   describe prediction *within* California, not in an unseen region.
3. **Historical data.** The values describe 1990, not the current housing market.

## License

MIT
