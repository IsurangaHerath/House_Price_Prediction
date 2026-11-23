# Subgroup error analysis

Generated with:

```powershell
.venv\Scripts\python.exe -m src.housing.segments
```

## Why this check matters

One holdout MAE hides where the model is reliable and where it is not. This
workflow splits the holdout rows into readable groups and scores each one, so
the model card can describe uneven accuracy with numbers instead of guesses.

## Method

| Item | Value |
|---|---|
| Model | Tuned random forest pipeline |
| Evaluation rows | Holdout test set (4,128 rows) |
| Split | Usual random 80/20 split, `random_state=42` |
| Segments | Fixed bins on income, house age, occupancy, and latitude |
| Bias | Mean of `true - predicted`; positive means under-prediction |

The segment edges are analyst choices, not natural boundaries. The same
workflow can be rerun with different bins.

## Overall result

MAE **0.3252**, RMSE **0.5006**, R² **0.8088**.

## Income

`MedInc` is in `$10,000` units.

| Segment | Rows | MAE | RMSE | R² | Bias |
|---|---:|---:|---:|---:|---:|
| under $20k | 495 | 0.2584 | 0.4683 | 0.6020 | 0.0003 |
| $20k–$35k | 1,566 | 0.2747 | 0.4366 | 0.7056 | -0.0128 |
| $35k–$50k | 1,215 | 0.3494 | 0.5272 | 0.6804 | -0.0052 |
| $50k+ | 852 | 0.4226 | 0.5830 | 0.7195 | -0.0212 |

Error rises steadily with income: MAE is **0.258** in the lowest band and
**0.423** in the highest, a **64%** increase. High-income blocks cover a wider
range of values and include many capped rows, so they are harder in absolute
terms even though their R² stays relatively high.

## House age

| Segment | Rows | MAE | RMSE | R² | Bias |
|---|---:|---:|---:|---:|---:|
| under 10 | 256 | 0.3610 | 0.5698 | 0.6570 | -0.0485 |
| 10–19 | 890 | 0.3300 | 0.4881 | 0.7649 | -0.0153 |
| 20–29 | 952 | 0.3359 | 0.5160 | 0.8035 | 0.0040 |
| 30–39 | 1,180 | 0.2906 | 0.4668 | 0.8360 | -0.0030 |
| 40+ | 850 | 0.3455 | 0.5186 | 0.8309 | -0.0218 |

Age matters less than income. The newest group has the highest MAE (**0.361**)
and the largest negative bias (**-0.049**), so new homes are slightly
over-predicted on average. The 30–39 band is the easiest.

## Occupancy

| Segment | Rows | MAE | RMSE | R² | Bias |
|---|---:|---:|---:|---:|---:|
| 1–1.9 | 311 | 0.5815 | 0.7782 | 0.6674 | -0.0186 |
| 2–2.9 | 2,212 | 0.3513 | 0.5197 | 0.8111 | -0.0092 |
| 3+ | 1,605 | 0.2396 | 0.3936 | 0.8135 | -0.0113 |

The **1–1.9** group stands out with MAE **0.582**, more than double the 3+ group
(**0.240**). It is also the smallest group, with 311 rows, so this estimate is
noisier than the others. These blocks may include unusual household
arrangements, but the data does not say why.

## Latitude

| Segment | Rows | MAE | RMSE | R² | Bias |
|---|---:|---:|---:|---:|---:|
| under 34 | 1,247 | 0.3254 | 0.4896 | 0.7831 | -0.0095 |
| 34–35.9 | 1,151 | 0.3542 | 0.5604 | 0.7663 | 0.0125 |
| 36–37.9 | 1,114 | 0.3426 | 0.5141 | 0.8419 | -0.0178 |
| 38+ | 616 | 0.2394 | 0.3590 | 0.6875 | -0.0437 |

The northern band (38+) has the lowest MAE (**0.239**) but also the lowest R²
(0.688), because its target variance is smaller. The 34–35.9 band is hardest
(**0.354**). This is a geographic association, not a causal effect.

## What stands out

- **Income is the clearest gradient.** Higher-income blocks have larger absolute
  errors, consistent with the target cap and wider value range.
- **Bias is small in every group.** Apart from the newest homes, group means are
  within about `±0.02`, so the differences are mostly variance rather than a
  systematic group-level shift.
- **Small groups are noisy.** Occupancy 1–1.9 has only 311 rows; its high MAE is
  worth noting but not overinterpreting.
- **The cap remains the main bias problem.** Capped rows are under-predicted
  even though the overall subgroup bias is near zero.

## Caveats

- **Random split.** Neighbouring blocks appear in both training and test, so
  these numbers are within-geography estimates. The spatial validation shows
  errors roughly double on unseen regions.
- **Analyst-defined bins.** Different edges would give different group scores.
- **Not a fairness audit.** The segments use housing features, not protected
  classes. They describe where errors are larger, not whether outcomes are fair.
- **Small-group metrics are unstable.** R² and RMSE move a lot with few hundred
  rows.
- **Censored target.** Values at `$500,000` are stored as `5.0`, which limits
  what any group-level score can say about the top end.
