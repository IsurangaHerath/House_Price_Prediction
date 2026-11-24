"""Read committed report files for the dashboard."""

from pathlib import Path

import pandas as pd

from .segments import SEGMENTS, SEGMENT_CSV, add_segment

REPORT_DIR = Path(__file__).resolve().parent.parent.parent / "reports"

# Keep a fixed order so the dashboard menu is predictable.
AVAILABLE_REPORTS = (
    "model_card.md",
    "baseline_results.md",
    "tuning_results.md",
    "spatial_validation.md",
    "error_map.md",
    "explainability.md",
    "importance_stability.md",
    "uncertainty.md",
    "uncertainty_segments.md",
    "conformal.md",
    "error_analysis.md",
    "learning_curve.md",
    "worst_predictions.md",
)


def load_report(name):
    """Return a committed report's text, or None when it is missing."""
    if name not in AVAILABLE_REPORTS:
        raise ValueError(f"Unknown report: {name}")

    path = REPORT_DIR / name
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def load_segment_errors():
    """Return the committed holdout error table, or None when missing."""
    if not SEGMENT_CSV.exists():
        return None
    return pd.read_csv(SEGMENT_CSV)


def typical_segment_error(segment_type, value):
    """Return the band label and its holdout MAE for a feature value."""
    table = load_segment_errors()
    if table is None:
        return None, None

    spec = SEGMENTS[segment_type]
    band = add_segment(
        pd.DataFrame({spec["column"]: [value]}),
        spec["column"],
        spec["bins"],
        spec["labels"],
    ).iloc[0]

    row = table[
        (table["segment_type"] == segment_type)
        & (table["segment"] == str(band))
    ]
    if row.empty:
        return None, None

    return str(band), float(row.iloc[0]["mae"])
