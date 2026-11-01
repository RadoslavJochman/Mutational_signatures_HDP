"""Tests for realdata/scripts/euler/cnv_heatmap_data.py.

Pure helpers only: pyscicone and cnv_data.h5 are not needed. They cover the
replay of read_10x's bin filter, block layout (never across chromosomes), the
block means and modes, and the node copy-number rebuild, including the
zero-neutral region remap that pyscicone's learn_tree makes necessary.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "realdata" / "scripts" / "euler"))

import cnv_heatmap_data as hm  # noqa: E402


def _filter_bins(counts, thres=3):
    """pyscicone's scicone.utils.filter_bins, copied for the test."""
    avg = counts.mean(axis=0)
    outliers = np.where(avg > thres * np.median(avg))[0]
    keep = np.ones(counts.shape[1], dtype=bool)
    keep[outliers] = False
    return counts[:, keep], outliers


def test_kept_bin_index_replays_mappable_and_noisy_filters():
    rng = np.random.default_rng(0)
    unfiltered = rng.poisson(10, size=(4, 12)).astype(float)
    unfiltered[:, 5] = 500  # noisy
    mappable = np.ones(12, dtype=bool)
    mappable[[0, 7]] = False
    filtered = unfiltered[:, [1, 2, 3, 4, 6, 8, 9, 10, 11]]
    index = hm.kept_bin_index(unfiltered, mappable, filtered, _filter_bins)
    assert index.tolist() == [1, 2, 3, 4, 6, 8, 9, 10, 11]


def test_kept_bin_index_raises_when_filtering_differs():
    unfiltered = np.arange(20, dtype=float).reshape(2, 10) + 1
    with pytest.raises(ValueError, match="reproduce"):
        hm.kept_bin_index(
            unfiltered, np.ones(10, bool), unfiltered[:, :9], _filter_bins
        )


def test_genomic_blocks_never_cross_chromosomes():
    # bin 20 kb, block 100 kb -> 5 bins per block; chromosomes of 7 and 5 bins
    block, block_chrom, chrom_start = hm.genomic_blocks([7, 5], 20_000, 100_000)
    assert block.tolist() == [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
    assert block_chrom.tolist() == [0, 0, 1]
    assert chrom_start.tolist() == [0, 2]


def test_genomic_blocks_rejects_block_below_bin():
    with pytest.raises(ValueError):
        hm.genomic_blocks([5], 20_000, 10_000)


def test_normalise_cells_centres_each_row_at_one():
    out = hm.normalise_cells(np.array([[1.0, 3.0], [10.0, 30.0]]))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out.mean(axis=1), 1.0)


def test_block_means_and_empty_blocks():
    values = np.array([[1.0, 3.0, 5.0], [2.0, 2.0, 8.0]])
    out = hm.block_means(values, np.array([0, 0, 2]), 3)
    np.testing.assert_allclose(out[:, [0, 2]], [[2.0, 5.0], [2.0, 8.0]])
    assert np.isnan(out[:, 1]).all()


def test_block_modes_majority_and_empty():
    values = np.array([[2, 2, 3, 4, 4], [1, 3, 3, 0, 0]])
    out = hm.block_modes(values, np.array([0, 0, 0, 2, 2]), 3)
    assert out.tolist() == [[2, -1, 4], [3, -1, 0]]


def test_node_copy_number_remaps_past_zero_neutral_regions():
    # regions: 0,1 autosome (2), 2 chrY (0), 3 chrX (2)
    neutral = np.array([2, 2, 0, 2])
    nodes = {
        "0": {"parent": "NULL", "events": {}},
        "1": {"parent": "0", "events": {"0": "1", "2": "-1"}},  # key 2 -> region 3
        "2": {"parent": "1", "events": {"2": "1", "1": "2"}},
    }
    cn = hm.node_copy_number(nodes, neutral)
    assert cn["0"].tolist() == [2, 2, 0, 2]
    assert cn["1"].tolist() == [3, 2, 0, 1]
    assert cn["2"].tolist() == [3, 4, 0, 2]


def test_node_copy_number_rejects_key_past_nonzero_regions():
    nodes = {"0": {"parent": "NULL", "events": {"3": "1"}}}
    with pytest.raises(ValueError, match="past"):
        hm.node_copy_number(nodes, np.array([2, 2, 0, 2]))


def test_node_copy_number_rejects_negative_copy_number():
    nodes = {
        "0": {"parent": "NULL", "events": {}},
        "1": {"parent": "0", "events": {"0": "-3"}},
    }
    with pytest.raises(ValueError, match="negative"):
        hm.node_copy_number(nodes, np.array([2]))


def test_region_of_bin():
    assert hm.region_of_bin(np.array([2, 1, 3])).tolist() == [0, 0, 1, 2, 2, 2]


def test_profile_correlation_detects_alignment():
    node_cn = np.array([[2, 2, 3, 1, -1], [2, 2, 2, 2, 2]], dtype=np.int8)
    depth = np.array([[1.0, 1.0, 1.5, 0.5, np.nan], [1.0, 1.0, 1.4, 0.6, 1.0]])
    corr = hm.profile_correlation(node_cn, depth, np.array(["a", "a"]), ["a", "b"])
    assert corr["a"] > 0.99
    assert "b" not in corr


def test_read_cell_table(tmp_path):
    p = tmp_path / "cells.csv"
    p.write_text("barcode,secedo_cluster,scicone_node\nA-1,4,30\nB-1,7,21\n")
    assert hm.read_cell_table(p) == (["A-1", "B-1"], ["4", "7"], ["30", "21"])
