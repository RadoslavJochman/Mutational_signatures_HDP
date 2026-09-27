"""Tests for realdata/scripts/convert_cosmic_catalogue.py, on tiny
synthetic catalogues -- never the downloaded COSMIC file (not committed,
licence pending; see COSMIC_sig/README_full_catalogue.md)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "realdata" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "realdata" / "scripts" / "euler"))

import build_snv_tree as bt  # noqa: E402
import convert_cosmic_catalogue as ccc  # noqa: E402

# Every one of COSMIC's 96 Type labels, built from build_snv_tree.py's own
# axes (public trinucleotide notation, not downloaded data), in an order
# deliberately NOT matching the target channel order -- convert_catalogue
# must key by label, not by row position.
ALL_LABELS = [
    f"{five}[{subtype}]{three}"
    for three in reversed(bt.BASES)
    for subtype in reversed(bt.SUBTYPES)
    for five in reversed(bt.BASES)
]


class TestParseTypeLabel:
    def test_matches_build_snv_tree_s_own_channel_index(self):
        assert ccc.parse_type_label("A[C>A]A") == bt.channel_index("A", "C", "A", "A")
        assert ccc.parse_type_label("T[T>G]C") == bt.channel_index("T", "T", "G", "C")

    def test_raises_on_an_unrecognised_label(self):
        with pytest.raises(ValueError, match="unrecognised COSMIC Type label"):
            ccc.parse_type_label("not-a-type-label")


class TestConvertCatalogue:
    def test_places_each_row_by_its_own_channel_not_its_position(self):
        raw = pd.DataFrame(
            {"SIGX": range(len(ALL_LABELS)), "SIGY": range(len(ALL_LABELS))},
            index=ALL_LABELS,
        )
        converted = ccc.convert_catalogue(raw)
        assert converted.shape == (2, 96)
        assert list(converted.index) == ["SIGX", "SIGY"]

        # A[C>A]A is channel 0 regardless of where it sat in the raw file.
        a_c_a_a_row = ALL_LABELS.index("A[C>A]A")
        assert converted.loc["SIGX", "Channel_0"] == a_c_a_a_row
        # T[T>G]T is the last channel, index 95.
        t_t_g_t_row = ALL_LABELS.index("T[T>G]T")
        assert converted.loc["SIGX", "Channel_95"] == t_t_g_t_row

    def test_raises_when_fewer_than_96_distinct_channels_are_present(self):
        raw = pd.DataFrame({"SIGX": [1, 2, 3]}, index=ALL_LABELS[:3])
        with pytest.raises(ValueError, match="expected 96 distinct channels"):
            ccc.convert_catalogue(raw)

    def test_raises_when_a_label_repeats_a_channel_instead_of_covering_all_96(self):
        labels = ALL_LABELS[:-1] + [ALL_LABELS[0]]  # 96 rows, only 95 channels
        raw = pd.DataFrame({"SIGX": range(96)}, index=labels)
        with pytest.raises(ValueError, match="expected 96 distinct channels"):
            ccc.convert_catalogue(raw)


class TestCompareToExisting:
    def test_reports_cosine_only_for_shared_signatures(self):
        converted = pd.DataFrame(
            {"Channel_0": [1.0, 0.0], "Channel_1": [0.0, 1.0]},
            index=["SBS1", "SBS_ONLY_IN_FULL"],
        )
        existing = pd.DataFrame(
            {"Channel_0": [1.0], "Channel_1": [0.0]}, index=["SBS1"]
        )
        result = ccc.compare_to_existing(converted, existing)
        assert list(result["signature"]) == ["SBS1"]
        assert result["cosine_to_full_catalogue"].iloc[0] == pytest.approx(1.0)

    def test_a_mismatched_shared_signature_scores_below_one(self):
        converted = pd.DataFrame(
            {"Channel_0": [1.0], "Channel_1": [0.0]}, index=["SBS1"]
        )
        existing = pd.DataFrame(
            {"Channel_0": [0.0], "Channel_1": [1.0]}, index=["SBS1"]
        )
        result = ccc.compare_to_existing(converted, existing)
        assert result["cosine_to_full_catalogue"].iloc[0] < 0.5
