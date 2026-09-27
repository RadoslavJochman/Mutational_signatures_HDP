"""Tests for realdata/scripts/refit_cosmic.py, on tiny synthetic signatures
and spectra -- never the real slice D files under realdata/local_tree_input/
(not committed, and not something a test should depend on existing)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "realdata" / "scripts"))

import refit_cosmic as rc  # noqa: E402

# Two orthogonal-ish toy signatures over 4 channels, so NNLS has an exact
# solution to check against.
TWO_SIGNATURES = pd.DataFrame(
    {
        "ch0": [0.7, 0.1],
        "ch1": [0.2, 0.1],
        "ch2": [0.05, 0.7],
        "ch3": [0.05, 0.1],
    },
    index=["SIGA", "SIGB"],
)


class TestSplitAvailable:
    def test_splits_present_and_missing_preserving_order(self):
        cosmic = pd.DataFrame({"ch0": [1, 2]}, index=["SBS1", "SBS5"])
        present, missing = rc.split_available(cosmic, ["SBS5", "SBS1", "SBS99"])
        assert present == ["SBS5", "SBS1"]
        assert missing == ["SBS99"]


class TestPairwiseSignatureCosine:
    def test_diagonal_is_one_and_matrix_is_symmetric(self):
        catalogue = pd.DataFrame(
            {"ch0": [1.0, 0.0, 0.6], "ch1": [0.0, 1.0, 0.8]},
            index=["SIGA", "SIGB", "SIGC"],
        )
        matrix = rc.pairwise_signature_cosine(catalogue, ["SIGA", "SIGB", "SIGC"])
        assert np.allclose(np.diag(matrix.to_numpy()), 1.0)
        assert matrix.loc["SIGA", "SIGB"] == pytest.approx(matrix.loc["SIGB", "SIGA"])
        assert matrix.loc["SIGA", "SIGB"] == pytest.approx(0.0, abs=1e-9)

    def test_drops_a_missing_name_rather_than_raising(self):
        catalogue = pd.DataFrame({"ch0": [1.0], "ch1": [0.0]}, index=["SIGA"])
        matrix = rc.pairwise_signature_cosine(catalogue, ["SIGA", "NOT_THERE"])
        assert list(matrix.index) == ["SIGA"]


class TestFlagCollinearPairs:
    def test_flags_only_pairs_above_threshold_once_each(self):
        matrix = pd.DataFrame(
            [[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        flagged = rc.flag_collinear_pairs(matrix, threshold=0.8)
        assert flagged == [("A", "B", pytest.approx(0.9))]

    def test_nothing_flagged_below_threshold(self):
        matrix = pd.DataFrame(
            [[1.0, 0.3], [0.3, 1.0]], index=["A", "B"], columns=["A", "B"]
        )
        assert rc.flag_collinear_pairs(matrix, threshold=0.8) == []


class TestNnlsFit:
    def test_recovers_known_exposures_on_a_pure_mixture(self):
        true_fractions = np.array([0.3, 0.7])
        spectrum = true_fractions @ TWO_SIGNATURES.to_numpy() * 1000
        fractions, similarity = rc.nnls_fit(spectrum, TWO_SIGNATURES)
        assert fractions["SIGA"] == pytest.approx(0.3, abs=1e-3)
        assert fractions["SIGB"] == pytest.approx(0.7, abs=1e-3)
        assert similarity == pytest.approx(1.0, abs=1e-6)

    def test_an_all_zero_spectrum_gives_all_zero_fractions_not_a_crash(self):
        fractions, similarity = rc.nnls_fit(np.zeros(4), TWO_SIGNATURES)
        assert (fractions.to_numpy() == 0).all()
        assert similarity == 0.0


THREE_SIGNATURES = pd.DataFrame(
    {
        "ch0": [0.9, 0.05, 0.05],
        "ch1": [0.05, 0.9, 0.05],
        "ch2": [0.05, 0.05, 0.9],
    },
    index=["SIGA", "SIGB", "SIGC"],
)


class TestForwardSelection:
    def test_adds_a_signature_that_explains_extra_structure(self):
        # spectrum is purely SIGC-shaped: SIGA+SIGB alone cannot explain
        # ch2 at all, so SIGC should be added.
        spectrum = THREE_SIGNATURES.loc["SIGC"].to_numpy() * 100
        selected, order = rc.forward_selection(
            spectrum, THREE_SIGNATURES, start=["SIGA", "SIGB"], min_gain=0.01
        )
        assert selected == ["SIGA", "SIGB", "SIGC"]
        assert len(order) == 1
        assert order[0]["signature"] == "SIGC"
        assert order[0]["gain"] > 0.01
        assert order[0]["cosine_after"] == pytest.approx(1.0, abs=1e-6)

    def test_stops_when_the_start_already_explains_the_spectrum(self):
        spectrum = (
            THREE_SIGNATURES.loc["SIGA"] + THREE_SIGNATURES.loc["SIGB"]
        ).to_numpy() * 100
        selected, order = rc.forward_selection(
            spectrum, THREE_SIGNATURES, start=["SIGA", "SIGB"], min_gain=0.01
        )
        assert selected == ["SIGA", "SIGB"]
        assert order == []

    def test_a_high_min_gain_stops_selection_before_a_real_gain_clears_it(self):
        spectrum = THREE_SIGNATURES.loc["SIGC"].to_numpy() * 100
        selected, order = rc.forward_selection(
            spectrum, THREE_SIGNATURES, start=["SIGA", "SIGB"], min_gain=0.99
        )
        assert selected == ["SIGA", "SIGB"]
        assert order == []


class TestBootstrapStability:
    def test_a_clean_single_signature_spectrum_is_fully_stable(self):
        spectrum = TWO_SIGNATURES.loc["SIGA"].to_numpy() * 2000
        stability = rc.bootstrap_stability(
            spectrum, TWO_SIGNATURES, n_bootstrap=50, threshold=0.05, seed=0
        )
        assert stability["SIGA"] == pytest.approx(1.0, abs=0.05)
        assert stability["SIGB"] < 0.5

    def test_zero_bootstrap_replicates_gives_zero_stability_not_an_error(self):
        spectrum = TWO_SIGNATURES.loc["SIGA"].to_numpy() * 2000
        stability = rc.bootstrap_stability(spectrum, TWO_SIGNATURES, n_bootstrap=0)
        assert (stability.to_numpy() == 0).all()

    def test_a_zero_total_spectrum_gives_zero_stability_not_a_zero_division(self):
        stability = rc.bootstrap_stability(np.zeros(4), TWO_SIGNATURES, n_bootstrap=50)
        assert (stability.to_numpy() == 0).all()


class TestFlagMissingFromCandidate:
    def test_flags_a_full_fit_signature_above_threshold_and_outside_candidates(self):
        exposures = pd.DataFrame(
            [
                {
                    "cluster": "3",
                    "fit_set": "full",
                    "signature": "SIGA",
                    "exposure_fraction": 0.6,
                },
                {
                    "cluster": "3",
                    "fit_set": "full",
                    "signature": "SIGB",
                    "exposure_fraction": 0.4,
                },
                {
                    "cluster": "3",
                    "fit_set": "candidate",
                    "signature": "SIGA",
                    "exposure_fraction": 1.0,
                },
            ]
        )
        flagged = rc.flag_missing_from_candidate(exposures, ["SIGA"], threshold=0.05)
        assert flagged == [("3", "SIGB", 0.4)]

    def test_nothing_flagged_when_the_full_fit_agrees_with_the_candidate_set(self):
        exposures = pd.DataFrame(
            [
                {
                    "cluster": "3",
                    "fit_set": "full",
                    "signature": "SIGA",
                    "exposure_fraction": 1.0,
                },
            ]
        )
        assert rc.flag_missing_from_candidate(exposures, ["SIGA"], threshold=0.05) == []


class TestFlagDeadWeightCandidates:
    def test_flags_a_candidate_signature_never_above_threshold_anywhere(self):
        exposures = pd.DataFrame(
            [
                {
                    "cluster": "3",
                    "fit_set": "candidate",
                    "signature": "SIGA",
                    "exposure_fraction": 0.9,
                },
                {
                    "cluster": "3",
                    "fit_set": "candidate",
                    "signature": "SIGB",
                    "exposure_fraction": 0.02,
                },
                {
                    "cluster": "7",
                    "fit_set": "candidate",
                    "signature": "SIGB",
                    "exposure_fraction": 0.01,
                },
            ]
        )
        assert rc.flag_dead_weight_candidates(exposures, threshold=0.05) == ["SIGB"]

    def test_a_signature_that_clears_the_threshold_once_is_not_flagged(self):
        exposures = pd.DataFrame(
            [
                {
                    "cluster": "3",
                    "fit_set": "candidate",
                    "signature": "SIGB",
                    "exposure_fraction": 0.01,
                },
                {
                    "cluster": "7",
                    "fit_set": "candidate",
                    "signature": "SIGB",
                    "exposure_fraction": 0.5,
                },
            ]
        )
        assert rc.flag_dead_weight_candidates(exposures, threshold=0.05) == []


class TestCompareClusterToOthers:
    def test_an_identical_profile_matches_the_others_perfectly(self):
        spectra = pd.DataFrame(
            {"ch0": [10, 20, 30], "ch1": [10, 20, 30]},
            index=["a", "b", "c"],
        )
        to_mean, baseline = rc.compare_cluster_to_others(spectra, "a")
        assert to_mean == pytest.approx(baseline, abs=1e-9)

    def test_a_distinct_profile_scores_lower_than_the_others_baseline(self):
        # b, c, d are all mutually similar (ch0-dominant); odd is the only
        # ch1-dominant profile, so it should score below their own baseline.
        spectra = pd.DataFrame(
            {
                "ch0": [99, 95, 90, 1],
                "ch1": [1, 5, 10, 99],
            },
            index=["b", "c", "d", "odd"],
        )
        to_mean, baseline = rc.compare_cluster_to_others(spectra, "odd")
        assert to_mean < baseline
