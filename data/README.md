# Data — California Housing

## Source

- **Dataset name:** California Housing
- **Original provider:** StatLib / UCI Machine Learning Repository, prepared for
  Scikit-learn by Flavian Vasile (figshare file `5976036`, `cal_housing.tgz`).
- **Local file:** `data/california_housing.csv`, committed to this repository so
  that a fresh clone runs without a network download.
- **Relationship to Scikit-learn:** the same underlying 1990 census records as
  `sklearn.datasets.fetch_california_housing()`, but stored as plain CSV with
  Scikit-learn's column names.

## Content

| Field | Description |
|---|---|
| Records | 20,640 census block groups |
| Features | MedInc (median income), HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude |
| Target | `MedHouseVal`, median house value in units of **$100,000** |
| Range | approximately 0.15 – 5.0 (i.e. roughly $15k – $500k) |

## Processing applied

The committed CSV was derived from the UCI distribution. The following changes
were made so it matches Scikit-learn's schema:

1. Column names were mapped to Scikit-learn's names (`median_income` →
   `MedInc`, `median_house_value` → `MedHouseVal`, and so on).
2. The CSV stores the target in **dollars**; `load_dataset()` divides it by
   100,000 so the in-memory target uses Scikit-learn's units. The file itself is
   left unchanged so the raw values stay auditable.
3. `ocean_proximity` was dropped because Scikit-learn's dataset does not include
   it and it is not used by this project.
4. Missing `AveBedrms` values (207 rows) are imputed at load time in
   `src/housing/data.py` as `AveRooms * 0.5`, an explicit approximation rather
   than a silent default.

All other values are unchanged from the 1990 census source.

## What this dataset is **not**

- **Not** current individual-property sale prices. It is the **1990 US Census**
  district median. Recent property-market pricing may differ substantially,
  especially given California's housing prices over the last 30+ years.
- **Not** the value of any single house on the dataset. Each row aggregates all
  houses in one census block group.

## Scientific caveats to state whenever results are discussed

1. **Hard cap at $500k.** Values equal to 5.0 are censored — the topmost
   districts hit the census cap. This biases training difficulty for those
   regions, and any metric computed on the whole dataset without noting the cap
   count will understate the cost of predicting high-value blocks accurately.
2. **Geographic leakage on random splits.** Rows are geographically clustered; a
   train/test split lets the model "see" neighbouring census blocks in both
   folds. Reported test performance therefore reflects prediction within
   California geography and **does not promise performance in an unseen region**.
3. **Imputed values.** The 207 missing `AveBedrms` entries are filled with an
   approximation, so those rows carry slightly lower data quality than the rest.
4. **Historical, aggregated data.** A prediction describes a 1990 block-group
   median, not a house you could buy today.
