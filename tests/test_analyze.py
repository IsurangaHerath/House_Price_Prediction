"""Tests for the residual analysis script."""

from src.housing import analyze as ha


def test_save_residual_plots(tmp_path):
    """The analysis should save two plot files."""
    paths = ha.save_residual_plots(
        [1.0, 2.0, 3.0],
        [1.1, 1.9, 3.2],
        output_dir=tmp_path,
    )

    assert len(paths) == 2
    assert all(path.exists() for path in paths)
