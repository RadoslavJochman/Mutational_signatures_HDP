"""Tests for realdata/scripts/check_tree_inputs.py, on tiny synthetic trees
and data -- never the real slice D files under realdata/local_tree_input/
(not committed, and not something a test should depend on existing)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "realdata" / "scripts"))

import check_tree_inputs as cti  # noqa: E402

# A tiny synthetic tree: germline -> g1 -> {a, b}, germline -> c. Mirrors the
# real pipeline's own shape (a hidden group node alongside a direct leaf)
# without any of its real data.
TINY_NEWICK = "((a,b)g1,c)germline;"


def _tiny_spectra() -> pd.DataFrame:
    return pd.DataFrame({"ch0": [5, 3, 4], "ch1": [1, 2, 0]}, index=["a", "b", "c"])


class TestReadDataMatrix:
    def test_casts_a_numeric_index_to_str(self, tmp_path):
        path = tmp_path / "spectra.csv"
        pd.DataFrame({"ch0": [1, 2]}, index=[3, 7]).to_csv(path)
        # confirm the pitfall exists before checking the fix
        assert pd.read_csv(path, index_col=0).index.dtype == np.int64
        df = cti.read_data_matrix(path)
        assert list(df.index) == ["3", "7"]
        assert df.index.dtype == object


class TestDescribeTree:
    def test_parent_root_and_observed_flags(self):
        from src.analysis.analysis import DEFAULT_PRIORS
        from src.models.hdp_inference import TreeHDP

        spectra = _tiny_spectra()
        fixed_signatures = np.array([[0.5, 0.5], [0.2, 0.8]])
        model = TreeHDP(
            newick_string=TINY_NEWICK,
            data_matrix=spectra,
            priors=DEFAULT_PRIORS,
            fixed_signatures=fixed_signatures,
        )
        desc = cti.describe_tree(model)

        assert desc.loc["germline", "parent"] is None
        assert desc.loc["germline", "is_root"]
        assert not desc.loc["germline", "observed"]
        assert desc.loc["g1", "parent"] == "germline"
        assert not desc.loc["g1", "observed"]
        assert desc.loc["a", "parent"] == "g1"
        assert desc.loc["a", "observed"]
        assert desc.loc["c", "parent"] == "germline"
        assert desc.loc["c", "observed"]

    def test_a_row_of_all_zero_counts_is_not_observed(self):
        from src.analysis.analysis import DEFAULT_PRIORS
        from src.models.hdp_inference import TreeHDP

        spectra = _tiny_spectra()
        spectra.loc["c"] = [0, 0]
        fixed_signatures = np.array([[0.5, 0.5], [0.2, 0.8]])
        model = TreeHDP(
            newick_string=TINY_NEWICK,
            data_matrix=spectra,
            priors=DEFAULT_PRIORS,
            fixed_signatures=fixed_signatures,
        )
        desc = cti.describe_tree(model)
        assert not desc.loc["c", "observed"]


def _tiny_desc() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "parent": [None, "germline", "g1", "g1", "germline"],
            "observed": [False, False, True, True, True],
            "is_root": [True, False, False, False, False],
        },
        index=["germline", "g1", "a", "b", "c"],
    )


class TestAssertTreeContract:
    def test_passes_on_a_valid_contract(self):
        cti.assert_tree_contract(_tiny_desc(), expected_observed={"a", "b", "c"})

    def test_raises_when_the_root_is_not_germline(self):
        desc = _tiny_desc()
        desc.loc["germline", "is_root"] = False
        desc.loc["g1", "is_root"] = True
        desc.loc["g1", "parent"] = None
        with pytest.raises(AssertionError, match="root is 'g1'"):
            cti.assert_tree_contract(desc, expected_observed={"a", "b", "c"})

    def test_raises_when_the_root_is_observed(self):
        desc = _tiny_desc()
        desc.loc["germline", "observed"] = True
        with pytest.raises(AssertionError, match="must be latent"):
            cti.assert_tree_contract(desc, expected_observed={"a", "b", "c"})

    def test_raises_when_the_observed_set_is_wrong(self):
        with pytest.raises(AssertionError, match="observed nodes"):
            cti.assert_tree_contract(_tiny_desc(), expected_observed={"a", "b"})

    def test_raises_when_a_latent_node_does_not_match_the_hidden_pattern(self):
        desc = _tiny_desc()
        desc = desc.rename(index={"g1": "hidden1"})
        desc.loc["a", "parent"] = "hidden1"
        desc.loc["b", "parent"] = "hidden1"
        with pytest.raises(AssertionError, match="hidden-node pattern"):
            cti.assert_tree_contract(desc, expected_observed={"a", "b", "c"})

    def test_raises_when_an_excluded_id_is_present(self):
        with pytest.raises(AssertionError, match="excluded id"):
            cti.assert_tree_contract(
                _tiny_desc(), expected_observed={"a", "b", "c"}, excluded_ids={"c"}
            )


class TestAssertChannelOrder:
    def test_passes_when_columns_match(self):
        df = pd.DataFrame({"Channel_0": [1], "Channel_1": [2]})
        cti.assert_channel_order(df, df)

    def test_raises_when_columns_differ(self):
        a = pd.DataFrame({"Channel_0": [1], "Channel_1": [2]})
        b = pd.DataFrame({"Channel_1": [2], "Channel_0": [1]})
        with pytest.raises(AssertionError, match="channel"):
            cti.assert_channel_order(a, b)


class TestAssertNonnegativeIntegerCounts:
    def test_passes_on_nonnegative_integers(self):
        cti.assert_nonnegative_integer_counts(pd.DataFrame({"a": [0, 1, 5]}))

    def test_raises_on_a_negative_count(self):
        with pytest.raises(AssertionError, match="negative"):
            cti.assert_nonnegative_integer_counts(pd.DataFrame({"a": [-1, 1]}))

    def test_raises_on_a_non_integer_count(self):
        with pytest.raises(AssertionError, match="non-integer"):
            cti.assert_nonnegative_integer_counts(pd.DataFrame({"a": [1.5, 2]}))


class TestTotalSnvsPerCluster:
    def test_sums_rows(self):
        totals = cti.total_snvs_per_cluster(_tiny_spectra())
        assert totals["a"] == 6
        assert totals["b"] == 5
        assert totals["c"] == 4


class TestSmokeSample:
    def test_builds_and_samples_a_tiny_tree_without_error(self):
        spectra = _tiny_spectra()
        fixed_signatures = np.array([[0.5, 0.5], [0.2, 0.8]])
        model, trace = cti.smoke_sample(
            TINY_NEWICK,
            spectra,
            fixed_signatures,
            draws=10,
            tune=10,
            chains=1,
            cores=1,
        )
        assert "e_level_0" in trace.posterior
        assert "e_level_1" in trace.posterior
