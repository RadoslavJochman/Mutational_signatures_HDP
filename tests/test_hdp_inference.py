"""Tests for src/models/hdp_inference.py's data-matrix/tree-node validation.

Not a full test suite for TreeHDP (its sampling behaviour is covered by
tests/test_smoke.py and the statistical tests); this covers only
_BaseTreeHDP._validate_data_matrix, added so that a data matrix which
cannot actually condition the model raises loudly instead of silently
sampling from the prior (see CLAUDE.md's real-data notes and
realdata/scripts/check_tree_inputs.py, which found the silent failure on
real slice D data).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.analysis import DEFAULT_PRIORS, read_data_matrix  # noqa: E402
from src.models.hdp_inference import TreeHDP  # noqa: E402

TINY_NEWICK = "((a,b)g1,c)germline;"
TWO_SIGNATURES = np.array([[0.5, 0.5], [0.2, 0.8]])


def _tiny_spectra() -> pd.DataFrame:
    return pd.DataFrame({"ch0": [5, 3, 4], "ch1": [1, 2, 0]}, index=["a", "b", "c"])


class TestValidateDataMatrix:
    def test_a_well_formed_matrix_builds_without_error(self):
        model = TreeHDP(
            newick_string=TINY_NEWICK,
            data_matrix=_tiny_spectra(),
            priors=DEFAULT_PRIORS,
            fixed_signatures=TWO_SIGNATURES,
        )
        assert model.observed_labels == {"a", "b", "c"}

    def test_a_row_matching_no_tree_node_raises(self):
        spectra = _tiny_spectra().rename(index={"c": "z"})
        with pytest.raises(ValueError, match="match no tree node"):
            TreeHDP(
                newick_string=TINY_NEWICK,
                data_matrix=spectra,
                priors=DEFAULT_PRIORS,
                fixed_signatures=TWO_SIGNATURES,
            )

    def test_every_row_summing_to_zero_raises_no_observed_nodes(self):
        spectra = _tiny_spectra()
        spectra.loc[:] = 0
        with pytest.raises(ValueError, match="no observed nodes"):
            TreeHDP(
                newick_string=TINY_NEWICK,
                data_matrix=spectra,
                priors=DEFAULT_PRIORS,
                fixed_signatures=TWO_SIGNATURES,
            )

    def test_an_unconverted_numeric_index_raises_rather_than_sampling_from_the_prior(
        self, tmp_path
    ):
        path = tmp_path / "spectra.csv"
        pd.DataFrame({"ch0": [1, 2], "ch1": [3, 4]}, index=[3, 7]).to_csv(path)
        raw = pd.read_csv(path, index_col=0)  # deliberately not cast to str
        with pytest.raises(ValueError, match="match no tree node"):
            TreeHDP(
                newick_string="(3,7)germline;",
                data_matrix=raw,
                priors=DEFAULT_PRIORS,
                fixed_signatures=TWO_SIGNATURES,
            )

    def test_read_data_matrix_then_treehdp_loads_an_int_indexed_file(self, tmp_path):
        path = tmp_path / "spectra.csv"
        pd.DataFrame({"ch0": [5, 3, 4], "ch1": [1, 2, 0]}, index=[1, 2, 3]).to_csv(path)
        spectra = read_data_matrix(path)  # the shared fix, not a bare pd.read_csv
        model = TreeHDP(
            newick_string="((1,2)g1,3)germline;",
            data_matrix=spectra,
            priors=DEFAULT_PRIORS,
            fixed_signatures=TWO_SIGNATURES,
        )
        assert model.observed_labels == {"1", "2", "3"}
        assert len(model.model.observed_RVs) == 1
