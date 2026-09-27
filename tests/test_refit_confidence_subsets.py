"""Tests for realdata/scripts/refit_confidence_subsets.py, on tiny
synthetic signatures and spectra -- never the real slice D confidence
subsets under realdata/local_tree_input/ (Euler-generated, not committed,
and not something a test should depend on existing)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "realdata" / "scripts"))

import refit_confidence_subsets as rcs  # noqa: E402

CATALOGUE = pd.DataFrame(
    {
        "ch0": [1.0, 0.0, 0.0, 0.5, 0.2],
        "ch1": [0.0, 1.0, 0.0, 0.5, 0.3],
        "ch2": [0.0, 0.0, 1.0, 0.0, 0.5],
    },
    index=["SBS1", "SBS3", "SBS95", "SBS_OTHER", "SBS5"],
)


class TestFixedBasisFit:
    def test_uses_candidate_names_present_in_the_catalogue_plus_extra(
        self, monkeypatch
    ):
        monkeypatch.setattr(rcs, "BREAST_CANDIDATE_SET", ["SBS1", "SBS3", "SBS17a"])
        spectrum = CATALOGUE.loc["SBS1"].to_numpy() * 10
        fractions, similarity = rcs.fixed_basis_fit(spectrum, CATALOGUE, ["SBS95"])
        # SBS17a is not in CATALOGUE and must be silently dropped, not raise.
        assert set(fractions.index) == {"SBS1", "SBS3", "SBS95"}
        assert similarity > 0.99

    def test_does_not_duplicate_an_extra_signature_already_in_the_candidate_set(
        self, monkeypatch
    ):
        monkeypatch.setattr(rcs, "BREAST_CANDIDATE_SET", ["SBS1", "SBS95"])
        spectrum = CATALOGUE.loc["SBS1"].to_numpy() * 10
        fractions, _ = rcs.fixed_basis_fit(spectrum, CATALOGUE, ["SBS95"])
        assert list(fractions.index).count("SBS95") == 1


class TestEndToEnd:
    def test_runs_on_tiny_synthetic_subsets_and_writes_all_three_csvs(self, tmp_path):
        catalogue_path = tmp_path / "catalogue.csv"
        CATALOGUE.to_csv(catalogue_path)

        clusters = ["c1", "c2"]
        for subset, scale in (("a", 1.0), ("b", 0.7), ("c", 0.4)):
            mix = 0.5 * CATALOGUE.loc["SBS1"] + 0.5 * CATALOGUE.loc["SBS3"]
            counts = (mix * 100 * scale).round().astype(int)
            df = pd.DataFrame({cl: counts for cl in clusters}).T
            df.index.name = None
            subset_dir = tmp_path / subset
            subset_dir.mkdir()
            df.to_csv(subset_dir / "spectra.csv")

        outdir = tmp_path / "refit_out"
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "realdata" / "scripts" / "refit_confidence_subsets.py"),
                "--subset-a",
                str(tmp_path / "a" / "spectra.csv"),
                "--subset-b",
                str(tmp_path / "b" / "spectra.csv"),
                "--subset-c",
                str(tmp_path / "c" / "spectra.csv"),
                "--catalogue",
                str(catalogue_path),
                "--outdir",
                str(outdir),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

        exposures = pd.read_csv(outdir / "confidence_exposures.csv")
        shares = pd.read_csv(outdir / "confidence_shares.csv")
        selection_order = pd.read_csv(outdir / "confidence_selection_order.csv")

        assert set(exposures["subset"]) == {"a", "b", "c"}
        assert set(shares["cluster"].astype(str)) == set(clusters)
        assert "SBS95_change_a_to_c" in shares.columns
        # forward selection on a clean SBS1+SBS3 mixture should reach near
        # cosine 1.0 and log at least one entry per cluster per subset.
        assert (selection_order.groupby(["subset", "cluster"]).size() >= 1).all()
        assert np.all(exposures["cosine_similarity"] > 0.99)
