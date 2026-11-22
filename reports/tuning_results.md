# Random forest tuning results

Generated with:

```powershell
.venv\Scripts\python.exe -m src.housing.tuning
```

## Goal

Check whether different random forest settings beat the default model. The
search runs on the training split only, so the holdout set stays untouched for
the final comparison.

## Search setup

| Item | Value |
|---|---|
| Candidates sampled | 6 |
| Cross-validation | 3 shuffled folds, `random_state=42` |
| Scoring | Negative mean absolute error |
| Trees per candidate | 100 |
| Search space | `max_depth`, `min_samples_leaf`, `max_features` |

The sampled values were:

- `max_depth`: `None`, `20`, `30`
- `min_samples_leaf`: `1`, `2`, `4`
- `max_features`: `1.0`, `0.5`, `"sqrt"`

## Best settings

```text
max_depth=30
max_features=0.5
min_samples_leaf=1
```

Search CV MAE: **0.334601**

These settings are now the project's random forest in
[`src/housing/models.py`](../src/housing/models.py).

## Fair 5-fold comparison

Both configurations were scored on the same five folds used by the main
workflow:

| Model | Mean MAE | Std |
|---|---:|---:|
| Default random forest | 0.335053 | 0.005150 |
| Tuned random forest | 0.330936 | 0.005814 |

The tuned model improved every fold:

| Fold | Default MAE | Tuned MAE | Improvement |
|---:|---:|---:|---:|
| 1 | 0.343966 | 0.340674 | 0.003292 |
| 2 | 0.329565 | 0.328862 | 0.000703 |
| 3 | 0.331576 | 0.323695 | 0.007881 |
| 4 | 0.337423 | 0.333671 | 0.003752 |
| 5 | 0.332737 | 0.327777 | 0.004960 |

Mean paired improvement: **0.004118** MAE (about **1.2%**).

## Holdout check

| Model | MAE | MAE ($) |
|---|---:|---:|
| Default random forest | 0.330565 | 33,057 |
| Tuned random forest | 0.327104 | 32,710 |

The tuned model is better by **0.003461** MAE, or about **$346** per
prediction. The gain is small but consistent with the cross-validation result.

## Conclusion

The tuned settings are kept because they improve both cross-validation and the
holdout set. The improvement is modest, which is expected: the default random
forest was already a strong baseline for this dataset.

## Caveats

- The search space was deliberately small to keep the run fast.
- A wider search or repeated cross-validation could find slightly better
  settings, but the returns are likely small.
- The holdout set still contains all 207 imputed `AveBedrms` rows and the
  geographic leakage described in [`baseline_results.md`](./baseline_results.md).
