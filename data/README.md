# Data — California Housing

## Source
- **Dataset name:** California Housing
- **Acquisition method:** Publish-and-reuse binary, shipped with Scikit-learn as
  `sklearn.datasets.fetch_california_housing()`. The underlying tarball lives on
  figshare (file `5976036`, titled `cal_housing.tgz`), originally prepared by
  Flavian Vasile under the Scikit-learn project.

## Content
| Field | Description |
|---|---|
| Records | 20,640 census block groups |
| Features | MedInc (median income), HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude |
| Target | Median house value, in units of **$100,000** |
| Range | 0.15 – 5.0 (i.e. roughly $15k – $500k) |

## What this dataset is **not**
- **Not** current individual-property sale prices. It is the **1990 US Census**
  district median. Recent property-market pricing may differ substantially,
  especially given California's housing prices over the last 30+ years.
- **Not** the value of any single house on the dataset. Each row aggregates all
  houses in one census block group.

## Scientific caveats to state whenever results are discussed
1. **Hard cap at $500k.** Values equal to 5.0 are censored — the topmost districts
   hit the census cap. This biases training difficulty for those regions and any
   metric computed on the whole dataset without noting the cap count will under-
   estimate the marginal cost of predicting high-value blocks accurately.
2. **Geographic leakage on random splits.** Rows are geographically clustered; a
   train/test split lets the model "see" neighboring census blocks in both folds.
   The reported test performance therefore reflects prediction within California
   geography and **does not promise performance in an unseen region**.
3. **No missing values, by construction** — but the validation layer checks for
   nulls defensively, and failure there stops the pipeline rather than assuming
   cleanliness.

See the project's evaluation protocol in [`src/housing/evaluate.py`](./../src/housing/evaluate.py)
once the core package exists.
