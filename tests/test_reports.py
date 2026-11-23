"""Tests for the report loader."""

import pandas as pd
import pytest

from src.housing import reports as hr


def test_load_report_returns_markdown():
    text = hr.load_report("model_card.md")

    assert text is not None
    assert text.startswith("# Model card")


def test_load_report_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown report"):
        hr.load_report("not_a_report.md")


def test_every_listed_report_exists():
    for name in hr.AVAILABLE_REPORTS:
        assert hr.load_report(name) is not None


def write_segment_table(path):
    pd.DataFrame(
        {
            "segment_type": ["Income", "Income"],
            "segment": ["$20k-$35k", "$50k+"],
            "mae": [0.275, 0.423],
        }
    ).to_csv(path, index=False)


def test_load_segment_errors(monkeypatch, tmp_path):
    path = tmp_path / "segment_errors.csv"
    write_segment_table(path)
    monkeypatch.setattr(hr, "SEGMENT_CSV", path)

    table = hr.load_segment_errors()

    assert table["segment"].tolist() == ["$20k-$35k", "$50k+"]


def test_typical_segment_error_finds_the_income_band(monkeypatch, tmp_path):
    path = tmp_path / "segment_errors.csv"
    write_segment_table(path)
    monkeypatch.setattr(hr, "SEGMENT_CSV", path)

    band, mae = hr.typical_segment_error("Income", 2.5)

    assert band == "$20k-$35k"
    assert mae == pytest.approx(0.275)


def test_typical_segment_error_handles_missing_table(monkeypatch, tmp_path):
    monkeypatch.setattr(hr, "SEGMENT_CSV", tmp_path / "missing.csv")

    assert hr.typical_segment_error("Income", 2.5) == (None, None)
