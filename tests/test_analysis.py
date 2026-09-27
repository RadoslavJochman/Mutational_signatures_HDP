"""Tests for src/analysis/analysis.py's read_data_matrix.

Not a full test suite for analysis.py (its other helpers are exercised
indirectly through the scripts that already use them); this covers only
the shared count/spectra loader added alongside TreeHDP's data-matrix
validation (see tests/test_hdp_inference.py).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.analysis import read_data_matrix  # noqa: E402


class TestReadDataMatrix:
    def test_casts_a_numeric_index_to_str(self, tmp_path):
        path = tmp_path / "spectra.csv"
        pd.DataFrame({"ch0": [1, 2]}, index=[3, 7]).to_csv(path)
        # confirm the pitfall exists before checking the fix
        assert pd.read_csv(path, index_col=0).index.dtype == np.int64

        df = read_data_matrix(path)
        assert list(df.index) == ["3", "7"]
        assert df.index.dtype == object

    def test_leaves_an_already_string_index_unchanged(self, tmp_path):
        path = tmp_path / "counts.csv"
        pd.DataFrame({"ch0": [1, 2]}, index=["T1_1", "T1_2"]).to_csv(path)
        df = read_data_matrix(path)
        assert list(df.index) == ["T1_1", "T1_2"]
