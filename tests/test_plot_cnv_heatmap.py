"""Tests for realdata/scripts/plot_cnv_heatmap.py: row ordering, runs, and an
end-to-end render of a tiny synthetic npz (shapes and files, not looks)."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "realdata" / "scripts"))

import plot_cnv_heatmap as ph  # noqa: E402


def test_runs():
    labels = np.array(["a", "a", "b", "a"])
    assert ph.runs(labels) == [("a", 0, 2), ("b", 2, 3), ("a", 3, 4)]


def test_row_order_groups_by_node_then_cluster():
    node = np.array(["21", "30", "21", "30", "25"])
    cluster = np.array(["8", "4", "7", "3", "9"])
    depth = np.ones((5, 4))
    order = ph.row_order(node, cluster, ["30", "21", "25"], depth)
    assert node[order].tolist() == ["30", "30", "21", "21", "25"]
    assert cluster[order].tolist() == ["3", "4", "7", "8", "9"]


def test_row_order_rejects_unknown_node():
    with pytest.raises(ValueError):
        ph.row_order(np.array(["1"]), np.array(["3"]), ["2"], np.ones((1, 2)))


def test_within_group_order_is_a_permutation():
    rng = np.random.default_rng(0)
    depth = rng.random((6, 10))
    depth[0, 3] = np.nan
    assert sorted(ph.within_group_order(depth).tolist()) == list(range(6))


def _tiny_npz():
    node_ids = ["0", "21", "25", "30", "32"]
    cells = ["30"] * 4 + ["21"] * 4 + ["25"] * 4
    clusters = ["3", "3", "4", "4", "7", "7", "8", "8", "9", "9", "10", "10"]
    node_cn = np.full((5, 6), 2, dtype=np.int8)
    node_cn[1, 1] = 3
    node_cn[2, 1] = 1
    node_cn[:, 4] = -1
    rng = np.random.default_rng(1)
    depth = (rng.random((12, 6)) + 0.5).astype(np.float32)
    depth[:, 4] = np.nan
    return {
        "depth": depth,
        "node_cn": node_cn,
        "node_ids": np.array(node_ids),
        "node_parent": np.array(["NULL", "32", "21", "32", "0"]),
        "node_n_events": np.array([0, 56, 7, 1, 1]),
        "node_depth_corr": np.array([np.nan, 0.9, 0.8, 0.1, np.nan]),
        "secedo_cluster": np.array(clusters),
        "scicone_node": np.array(cells),
        "chrom_names": np.array(["1", "2", "Y"]),
        "chrom_start_block": np.array([0, 3, 5]),
        "bin_size": np.array(20000),
        "block_bp": np.array(1_000_000),
    }


def test_fig6_renders_and_captions(tmp_path):
    ph.figure_style.apply_style()
    caption = ph.fig6_cnv_heatmap(_tiny_npz(), tmp_path)
    assert (tmp_path / "fig6_cnv_heatmap.pdf").exists()
    assert (tmp_path / "fig6_cnv_heatmap.png").exists()
    assert "12 cells" in caption and "1 Mb" in caption
    assert caption.index("near-diploid") < caption.index("rearranged clone")
    assert caption.index("rearranged clone") < caption.index("subclone")
