"""Tests for the report loader."""

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
