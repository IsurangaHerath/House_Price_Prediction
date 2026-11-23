# Importance stability across split seeds

Generated with:

```powershell
.venv\Scripts\python.exe -m src.housing.stability
```

The raw table is committed as
[`importance_stability.csv`](./importance_stability.csv).

## Why this check matters

Permutation importance is computed on one holdout set. If a different random
split gave a different ranking, the explanation in the model card would be a
property of one split rather than of the model. This workflow repeats the
analysis on five independent 80/20 splits.

## Method

| Item | Value |
|---|---|
| Model | Tuned random forest pipeline |
| Seeds | 42, 7, 123, 2024, 99 |
| Split | 80/20 train/test for each seed |
| Importance | Increase in MAE when a feature is shuffled |
| Permutation repeats | 3 per feature per seed |

## Results

| Feature | Mean importance | Std across seeds | Best rank | Worst rank |
|---|---:|---:|---:|---:|
| MedInc | 0.4253 | 0.0099 | 1 | 1 |
| Latitude | 0.3296 | 0.0099 | 2 | 2 |
| Longitude | 0.2874 | 0.0082 | 3 | 3 |
| AveOccup | 0.1523 | 0.0055 | 4 | 4 |
| HouseAge | 0.0516 | 0.0024 | 5 | 5 |
| AveRooms | 0.0474 | 0.0013 | 6 | 6 |
| AveBedrms | 0.0095 | 0.0008 | 7 | 7 |
| Population | 0.0063 | 0.0012 | 8 | 8 |

The ranking is **identical on all five seeds**. Every one of the 10 pairwise
Spearman correlations between seed rankings is **1.0**. The four leading
features — MedInc, Latitude, Longitude, and AveOccup — account for about **91%**
of the total importance, and their values vary by less than 1% of their mean.

The means are close to the single-split result in
[`explainability.md`](./explainability.md): MedInc **0.417**, Latitude **0.322**,
Longitude **0.280**, and AveOccup **0.153**. The original model-card explanation
is therefore not a split artifact.

## Implications

- The story that income and location dominate is robust to how the data is
  split.
- The weak features stay weak. `Population` and `AveBedrms` contribute almost
  nothing beyond what the other features already capture.
- Nothing in this check changes the model; it only makes the explanation more
  trustworthy.

## Caveats

- **Correlated features.** Permutation importance can split credit between
  related features, so a low value does not prove a feature is irrelevant.
- **Fixed permutation noise.** The permutation random state is 42 for every
  seed, so the measured variation comes from the split itself, not from the
  shuffling.
- **Five seeds.** A perfect rank agreement on five splits is strong evidence,
  not a guarantee for every possible sample.
- **One model.** This applies to the tuned random forest only.
