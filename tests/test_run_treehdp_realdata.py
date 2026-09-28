"""Tests for realdata/scripts/run_treehdp_realdata.py's helper logic
(signature selection by name, the comparison tables), on tiny synthetic
inputs -- never the real slice D trees/spectra/traces under
realdata/local_tree_input/ (not committed, and not something a test
should depend on existing). Building and sampling a real model is covered
by check_tree_inputs.py's own smoke test; not repeated here."""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "realdata" / "scripts"))

import run_treehdp_realdata as rtr  # noqa: E402


class TestFullRepertoire:
    def test_has_the_ten_named_signatures_and_excludes_sbs95(self):
        assert set(rtr.FULL_REPERTOIRE) == {
            "SBS1",
            "SBS2",
            "SBS3",
            "SBS5",
            "SBS8",
            "SBS13",
            "SBS17a",
            "SBS17b",
            "SBS18",
            "SBS40a",
        }
        assert "SBS95" not in rtr.FULL_REPERTOIRE


class TestSelectSignatures:
    def test_returns_rows_in_exactly_the_requested_order(self):
        catalogue = pd.DataFrame(
            {"ch0": [1.0, 2.0, 3.0], "ch1": [4.0, 5.0, 6.0]},
            index=["SBS5", "SBS1", "SBS95"],
        )
        matrix = rtr.select_signatures(catalogue, ["SBS1", "SBS5"])
        assert matrix.shape == (2, 2)
        assert list(matrix[0]) == [2.0, 5.0]  # SBS1's row
        assert list(matrix[1]) == [1.0, 4.0]  # SBS5's row

    def test_raises_naming_a_missing_signature_rather_than_dropping_it(self):
        catalogue = pd.DataFrame({"ch0": [1.0]}, index=["SBS1"])
        with pytest.raises(ValueError, match="SBS999"):
            rtr.select_signatures(catalogue, ["SBS1", "SBS999"])


def _activity_df(rows):
    """rows: iterable of (node, kind, signature, mean, p5, p95)."""
    return pd.DataFrame(
        rows, columns=["node", "kind", "signature", "mean", "p5", "p95"]
    )


class TestActivityComparisonTable:
    def test_collects_one_row_per_run_cluster_signature(self):
        runs = {
            "main_snv": {
                "activity": _activity_df(
                    [
                        ("3", "observed", "SBS1", 0.2, 0.1, 0.3),
                        ("3", "observed", "SBS5", 0.7, 0.6, 0.8),
                        ("germline", "latent", "SBS1", 0.5, 0.4, 0.6),
                    ]
                )
            },
            "main_cna": {
                "activity": _activity_df([("3", "observed", "SBS1", 0.25, 0.15, 0.35)])
            },
        }
        table = rtr.activity_comparison_table(runs, signatures=("SBS1", "SBS5"))
        assert set(table["run"]) == {"main_snv", "main_cna"}
        # the latent germline row must never appear.
        assert "germline" not in set(table["node"]) if "node" in table.columns else True
        assert len(table[(table["run"] == "main_snv") & (table["cluster"] == "3")]) == 2
        assert len(table[(table["run"] == "main_cna") & (table["cluster"] == "3")]) == 1

    def test_empty_runs_gives_an_empty_table(self):
        table = rtr.activity_comparison_table({})
        assert len(table) == 0


class TestTreeSensitivityTable:
    def test_reports_the_largest_cross_tree_difference_and_overlap(self):
        runs = {
            "main_snv": {
                "activity": _activity_df([("3", "observed", "SBS1", 0.20, 0.10, 0.30)])
            },
            "main_cna": {
                "activity": _activity_df([("3", "observed", "SBS1", 0.50, 0.40, 0.60)])
            },
            "main_consensus": {
                "activity": _activity_df([("3", "observed", "SBS1", 0.22, 0.12, 0.32)])
            },
        }
        table = rtr.tree_sensitivity_table(runs, "main", signatures=("SBS1",))
        row = table.iloc[0]
        assert row["tree_max"] == "cna"
        assert row["tree_min"] == "snv"
        assert row["max_diff"] == pytest.approx(0.30, abs=1e-9)
        # [0.10, 0.30] and [0.40, 0.60] do not overlap.
        assert bool(row["intervals_overlap"]) is False

    def test_overlapping_intervals_are_reported_as_such(self):
        runs = {
            "main_snv": {
                "activity": _activity_df([("3", "observed", "SBS1", 0.20, 0.10, 0.35)])
            },
            "main_cna": {
                "activity": _activity_df([("3", "observed", "SBS1", 0.25, 0.15, 0.40)])
            },
        }
        table = rtr.tree_sensitivity_table(runs, "main", signatures=("SBS1",))
        assert bool(table.iloc[0]["intervals_overlap"]) is True

    def test_fewer_than_two_trees_gives_an_empty_table(self):
        runs = {
            "main_snv": {
                "activity": _activity_df([("3", "observed", "SBS1", 0.2, 0.1, 0.3)])
            }
        }
        table = rtr.tree_sensitivity_table(runs, "main")
        assert len(table) == 0


class TestSpectraSensitivityTable:
    def test_renormalises_main_to_sbs1_plus_sbs5_and_reports_sbs95_share(self):
        runs = {
            "main_snv": {
                "activity": _activity_df(
                    [
                        ("3", "observed", "SBS1", 0.2, 0.1, 0.3),
                        ("3", "observed", "SBS5", 0.6, 0.5, 0.7),
                        ("3", "observed", "SBS95", 0.2, 0.1, 0.3),
                    ]
                )
            },
            "sensitivity_snv": {
                "activity": _activity_df(
                    [
                        ("3", "observed", "SBS1", 0.3, 0.2, 0.4),
                        ("3", "observed", "SBS5", 0.7, 0.6, 0.8),
                    ]
                )
            },
        }
        table = rtr.spectra_sensitivity_table(runs, "snv")
        row = table.iloc[0]
        # main SBS1=0.2, SBS5=0.6 -> renormalised over their own sum (0.8):
        # 0.25 and 0.75.
        assert row["main_sbs1_renorm"] == pytest.approx(0.25)
        assert row["main_sbs5_renorm"] == pytest.approx(0.75)
        assert row["sensitivity_sbs1"] == pytest.approx(0.3)
        assert row["sensitivity_sbs5"] == pytest.approx(0.7)
        assert row["main_sbs95_share"] == pytest.approx(0.2)

    def test_missing_run_gives_an_empty_table(self):
        runs = {
            "main_snv": {
                "activity": _activity_df([("3", "observed", "SBS1", 0.2, 0.1, 0.3)])
            }
        }
        table = rtr.spectra_sensitivity_table(runs, "snv")
        assert len(table) == 0


class TestActiveSignatureSummary:
    def test_counts_clusters_whose_lower_bound_clears_the_threshold(self):
        activity = _activity_df(
            [
                ("3", "observed", "SBS2", 0.05, 0.03, 0.08),
                ("7", "observed", "SBS2", 0.01, 0.00, 0.02),
                ("8", "observed", "SBS2", 0.10, 0.025, 0.15),
                ("germline", "latent", "SBS2", 0.5, 0.4, 0.6),
            ]
        )
        table = rtr.active_signature_summary(activity, threshold=0.02)
        row = table[table["signature"] == "SBS2"].iloc[0]
        assert row["n_clusters_active"] == 2
        assert row["n_clusters_total"] == 3
        assert row["max_mean"] == pytest.approx(0.10)

    def test_a_signature_with_no_clusters_above_threshold_reports_zero(self):
        activity = _activity_df([("3", "observed", "SBS13", 0.01, 0.0, 0.02)])
        table = rtr.active_signature_summary(activity, threshold=0.02)
        assert table.iloc[0]["n_clusters_active"] == 0


class TestBaselineComparisonTable:
    def test_is_a_no_op_pending_a_baseline_model(self):
        assert rtr.BASELINE_AVAILABLE is False
        table = rtr.baseline_comparison_table()
        assert len(table) == 0


class TestLoadTree:
    def test_consensus_matches_check_tree_inputs_constant(self):
        import check_tree_inputs as cti

        assert rtr.load_tree("consensus", Path(".")) == cti.CONSENSUS_NEWICK

    def test_an_unknown_tree_name_raises(self):
        with pytest.raises(ValueError, match="unknown tree name"):
            rtr.load_tree("not-a-tree", Path("."))
