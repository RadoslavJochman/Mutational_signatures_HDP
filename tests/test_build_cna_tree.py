"""Tests for realdata/scripts/euler/build_cna_tree.py.

Covers the parts of the CNA-tree pipeline that are fully specified and
testable without a real cnv_data.h5 or a built SCICoNE binary: the h5
dataset-resolution heuristics (against a synthetic h5py file), barcode <->
SECEDO cluster mapping, pseudobulk aggregation, majority-vote and
pseudobulk-label cluster-node resolution, and SCICoNE's plain-text edge/
assignment file parsing (subprocess calls are mocked, matching this repo's
existing convention for external-tool wrappers). The module docstring's
caveat on cnv_data.h5's exact schema and SCICoNE's exact CLI/output format
not being confirmed against a real run applies to the heuristics tested here
too -- these tests check the *documented* behaviour, not real-world fidelity.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "realdata" / "scripts" / "euler"))

import build_cna_tree as ct  # noqa: E402

h5py = pytest.importorskip("h5py")


# --------------------------------------------------------------------------- #
# cnv_data.h5 extraction
# --------------------------------------------------------------------------- #


def _write_fake_cnv_h5(path, counts_path="raw_counts", barcodes_path="cell_barcodes"):
    counts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    barcodes = np.array([b"AAA-1", b"BBB-1", b"CCC-1"], dtype="S6")
    with h5py.File(path, "w") as f:
        # nested group, to exercise the recursive walk
        parts = counts_path.split("/")
        group = f
        for part in parts[:-1]:
            group = group.require_group(part)
        group.create_dataset(parts[-1], data=counts)

        parts = barcodes_path.split("/")
        group = f
        for part in parts[:-1]:
            group = group.require_group(part)
        group.create_dataset(parts[-1], data=barcodes)
    return counts, [b.decode() for b in barcodes]


def test_load_cnv_counts_matrix_via_candidate_paths(tmp_path):
    h5_path = tmp_path / "cnv_data.h5"
    counts, barcodes = _write_fake_cnv_h5(h5_path)
    loaded_counts, loaded_barcodes, counts_path, barcodes_path = (
        ct.load_cnv_counts_matrix(h5_path)
    )
    assert np.array_equal(loaded_counts, counts)
    assert loaded_barcodes == barcodes
    assert counts_path == "raw_counts"
    assert barcodes_path == "cell_barcodes"


def test_load_cnv_counts_matrix_via_heuristic_fallback(tmp_path):
    # Neither dataset uses a candidate path name -- must fall back to the
    # shape/name heuristic (2D dataset with "raw" in its path; 1D dataset of
    # matching length for barcodes).
    h5_path = tmp_path / "cnv_data.h5"
    counts, barcodes = _write_fake_cnv_h5(
        h5_path, counts_path="some_group/raw_bin_counts", barcodes_path="cell_ids"
    )
    loaded_counts, loaded_barcodes, counts_path, barcodes_path = (
        ct.load_cnv_counts_matrix(h5_path)
    )
    assert np.array_equal(loaded_counts, counts)
    assert loaded_barcodes == barcodes
    assert counts_path == "some_group/raw_bin_counts"
    assert barcodes_path == "cell_ids"


def test_find_counts_dataset_raises_when_nothing_2d():
    datasets = {"a": np.array([1, 2, 3])}
    with pytest.raises(ValueError, match="no 2D dataset"):
        ct._find_counts_dataset(datasets)


def test_find_barcode_dataset_raises_when_no_length_match():
    datasets = {"a": np.array([1, 2])}
    with pytest.raises(ValueError, match="no length-3"):
        ct._find_barcode_dataset(datasets, n_cells=3)


# --------------------------------------------------------------------------- #
# Barcode <-> SECEDO cluster mapping
# --------------------------------------------------------------------------- #


def test_barcode_from_bam_name():
    assert ct.barcode_from_bam_name("AAACCTGAGCTAACAA-1.bam") == "AAACCTGAGCTAACAA-1"
    assert ct.barcode_from_bam_name("already_no_suffix") == "already_no_suffix"


def test_build_barcode_to_cluster(tmp_path):
    map_file = tmp_path / "chromosome_1.map"
    map_file.write_text("AAA-1.bam\t0\nBBB-1.bam\t1\nCCC-1.bam\t2\n")
    clustering_file = tmp_path / "clustering"
    clustering_file.write_text("0,1,1\n")  # cell 0 unclustered, cells 1&2 -> cluster 1

    mapping = ct.build_barcode_to_cluster(map_file, clustering_file)
    assert mapping == {"BBB-1": "1", "CCC-1": "1"}
    assert "AAA-1" not in mapping  # cluster 0 excluded


def test_build_barcode_to_cluster_raises_on_length_mismatch(tmp_path):
    map_file = tmp_path / "chromosome_1.map"
    map_file.write_text("AAA-1.bam\t0\n")
    clustering_file = tmp_path / "clustering"
    clustering_file.write_text("0,1\n")
    with pytest.raises(ValueError, match="clustering has"):
        ct.build_barcode_to_cluster(map_file, clustering_file)


# --------------------------------------------------------------------------- #
# Pseudobulk aggregation
# --------------------------------------------------------------------------- #


def test_aggregate_counts_by_cluster():
    counts = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    barcodes = ["a", "b", "c", "d"]
    # "d" has no cluster mapping -- excluded from the aggregate entirely.
    pseudobulk, cluster_order = ct.aggregate_counts_by_cluster(
        counts, barcodes, {"a": "1", "b": "1", "c": "2"}
    )
    assert cluster_order == ["1", "2"]
    assert np.array_equal(pseudobulk[0], [3, 3])  # a + b
    assert np.array_equal(pseudobulk[1], [3, 3])  # c alone


def test_aggregate_counts_by_cluster_skips_unmatched_cells():
    counts = np.array([[1, 1], [2, 2]])
    barcodes = ["a", "unmatched"]
    pseudobulk, cluster_order = ct.aggregate_counts_by_cluster(
        counts, barcodes, {"a": "1"}
    )
    assert cluster_order == ["1"]
    assert np.array_equal(pseudobulk[0], [1, 1])


# --------------------------------------------------------------------------- #
# SCICoNE plumbing (the binary itself is not exercised)
# --------------------------------------------------------------------------- #


def test_find_scicone_binary_precedence(monkeypatch):
    assert ct.find_scicone_binary("explicit/path") == "explicit/path"
    monkeypatch.setenv("SCICONE_BIN", "env/path")
    assert ct.find_scicone_binary(None) == "env/path"
    monkeypatch.delenv("SCICONE_BIN")
    expected = REPO_ROOT / "realdata" / "external" / "SCICoNE" / "build" / "scicone"
    assert Path(ct.find_scicone_binary(None)) == expected


def test_run_scicone_builds_expected_command_and_returns_paths(tmp_path, monkeypatch):
    calls = {}

    def fake_run(cmd, check, stdout, stderr):
        calls["cmd"] = cmd
        out_prefix = cmd[cmd.index("--output") + 1]
        Path(f"{out_prefix}.edges.tsv").write_text("germline\tn1\n")
        Path(f"{out_prefix}.assignment.tsv").write_text("AAA-1\tn1\n")

    monkeypatch.setattr(ct.subprocess, "run", fake_run)

    edges_path, assignment_path = ct.run_scicone(
        tmp_path / "matrix.txt",
        n_rows=3,
        n_bins=10,
        out_prefix=tmp_path / "scicone_out",
        log_path=tmp_path / "scicone.log",
        scicone_bin="scicone",
    )
    assert edges_path.exists()
    assert assignment_path.exists()
    cmd = calls["cmd"]
    assert cmd[0] == "scicone"
    assert cmd[cmd.index("--n_cells") + 1] == "3"
    assert cmd[cmd.index("--n_bins") + 1] == "10"


def test_run_scicone_raises_if_output_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(ct.subprocess, "run", lambda *a, **k: None)
    with pytest.raises(FileNotFoundError):
        ct.run_scicone(
            tmp_path / "matrix.txt",
            n_rows=3,
            n_bins=10,
            out_prefix=tmp_path / "scicone_out",
            log_path=tmp_path / "scicone.log",
            scicone_bin="scicone",
        )


def test_parse_scicone_edges(tmp_path):
    edges_path = tmp_path / "out.edges.tsv"
    edges_path.write_text("# comment\ngermline\tn1\nn1\tn2\n\n")
    parent_of = ct.parse_scicone_edges(edges_path)
    assert parent_of == {"n1": "germline", "n2": "n1"}


def test_parse_scicone_edges_raises_on_empty(tmp_path):
    edges_path = tmp_path / "empty.tsv"
    edges_path.write_text("# just a comment\n")
    with pytest.raises(ValueError, match="no parseable"):
        ct.parse_scicone_edges(edges_path)


def test_parse_scicone_cell_assignment(tmp_path):
    path = tmp_path / "out.assignment.tsv"
    path.write_text("AAA-1\tn1\nBBB-1\tn2\n")
    assignment = ct.parse_scicone_cell_assignment(path)
    assert assignment == {"AAA-1": "n1", "BBB-1": "n2"}


def test_parse_scicone_cell_assignment_raises_on_empty(tmp_path):
    path = tmp_path / "empty.tsv"
    path.write_text("\n")
    with pytest.raises(ValueError, match="no parseable"):
        ct.parse_scicone_cell_assignment(path)


# --------------------------------------------------------------------------- #
# Majority-vote / pseudobulk cluster-node resolution
# --------------------------------------------------------------------------- #


def test_majority_cluster_nodes_picks_the_majority():
    cell_to_node = {"a": "n1", "b": "n1", "c": "n2", "d": "n2", "e": "n2"}
    barcode_to_cluster = {
        "a": "1",
        "b": "1",
        "c": "1",  # cluster 1: n1, n1, n2 -> n1 wins
        "d": "2",
        "e": "2",  # cluster 2: n2, n2 -> n2
    }
    resolved, unmatched = ct.majority_cluster_nodes(cell_to_node, barcode_to_cluster)
    assert resolved == {"1": "n1", "2": "n2"}
    assert unmatched == []


def test_majority_cluster_nodes_reports_unmatched_clusters():
    cell_to_node = {"a": "n1"}
    barcode_to_cluster = {"a": "1", "b": "2"}  # "b" never appears in cell_to_node
    resolved, unmatched = ct.majority_cluster_nodes(cell_to_node, barcode_to_cluster)
    assert resolved == {"1": "n1"}
    assert unmatched == ["2"]


def test_majority_cluster_nodes_deterministic_tie_break():
    # cluster "1"'s cells split evenly between n1 and n2 -- tie broken by
    # sorted node name, so the result is reproducible.
    cell_to_node = {"a": "n2", "b": "n1"}
    barcode_to_cluster = {"a": "1", "b": "1"}
    resolved, _ = ct.majority_cluster_nodes(cell_to_node, barcode_to_cluster)
    assert resolved == {"1": "n1"}


def test_resolve_pseudobulk_cluster_nodes_verbatim():
    parent_of = {"1": "germline", "2": "1"}
    resolved = ct.resolve_pseudobulk_cluster_nodes(parent_of, ["1", "2"])
    assert resolved == {"1": "1", "2": "2"}


def test_resolve_pseudobulk_cluster_nodes_raises_when_unresolved():
    parent_of = {"x": "y"}
    with pytest.raises(ValueError, match="no consistent"):
        ct.resolve_pseudobulk_cluster_nodes(parent_of, ["1", "2"])


# --------------------------------------------------------------------------- #
# End-to-end collapse (reusing build_snv_tree's shared collapsing helper)
# --------------------------------------------------------------------------- #


def test_collapse_via_majority_vote_produces_valid_newick():
    parent_of = {"n1": "germline", "n2": "n1", "n3": "germline"}
    cell_to_node = {"a": "n1", "b": "n2", "c": "n3"}
    barcode_to_cluster = {"a": "7", "b": "8", "c": "9"}
    resolved, unmatched = ct.majority_cluster_nodes(cell_to_node, barcode_to_cluster)
    assert unmatched == []

    tree = ct.collapse_by_nearest_labelled_ancestor(
        parent_of, resolved, ct.GERMLINE_ROOT_ID
    )
    newick = ct.digraph_to_newick(tree, ct.GERMLINE_ROOT_ID)
    ct.verify_newick(newick, {"7", "8", "9"}, ct.GERMLINE_ROOT_ID)  # no raise
    assert set(tree.predecessors("8")) == {"7"}
    assert set(tree.predecessors("9")) == {ct.GERMLINE_ROOT_ID}
