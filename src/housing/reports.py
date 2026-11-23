"""Read committed report files for the dashboard."""

from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent.parent.parent / "reports"

# Keep a fixed order so the dashboard menu is predictable.
AVAILABLE_REPORTS = (
    "model_card.md",
    "baseline_results.md",
    "tuning_results.md",
    "spatial_validation.md",
    "error_map.md",
    "explainability.md",
    "uncertainty.md",
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
