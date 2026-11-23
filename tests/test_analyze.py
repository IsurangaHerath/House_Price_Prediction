"""Tests for the residual analysis script."""

import pandas as pd

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


def test_save_error_map(tmp_path):
    """The error map should be saved as a figure."""
    X = pd.DataFrame(
        {
            "Latitude": [34.0, 35.0, 36.0],
            "Longitude": [-118.0, -119.0, -120.0],
        }
    )

    path = ha.save_error_map(X, [1.0, 2.0, 3.0], [1.1, 1.9, 3.2], output_dir=tmp_path)

    assert path.exists()
