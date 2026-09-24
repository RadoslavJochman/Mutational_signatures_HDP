"""Tests for realdata/scripts/euler/build_cna_tree.py.

The pyscicone wrapper is mocked throughout (no SCICoNE binary is needed): a
small fake stands in for ``scicone.SCICoNE`` and a synthetic h5py file for
``cnv_data.h5``. They cover the filtered-barcode reconstruction, sex-dependent
neutral states, the node_dict topology and root, the majority-vote collapse,
the cluster-4 fold, and the failure paths. They check the documented logic,
not fidelity to a real SCICoNE run.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "realdata" / "scripts" / "euler"))

import build_cna_tree as ct  # noqa: E402

h5py = pytest.importorskip("h5py")


# --------------------------------------------------------------------------- #
# Filtered-cell barcodes
# --------------------------------------------------------------------------- #


def _write_h5(path, barcodes, outlier, barcodes_key=True, outlier_key=True):
    with h5py.File(path, "w") as f:
        if barcodes_key:
            f.create_dataset("cell_barcodes", data=np.array(barcodes, dtype="S16"))
        if outlier_key:
            grp = f.require_group("per_cell_summary_metrics")
            grp.create_dataset("is_high_dimapd", data=np.array(outlier, dtype=np.int8))


def test_filtered_barcodes_applies_the_outlier_mask(tmp_path):
    h5_path = tmp_path / "cnv.h5"
    _write_h5(h5_path, [b"A-1", b"B-1", b"C-1", b"D-1"], [0, 1, 0, 1])
    assert ct.filtered_barcodes(h5_path, 2) == ["A-1", "C-1"]


def test_filtered_barcodes_raises_on_missing_barcodes(tmp_path):
    h5_path = tmp_path / "cnv.h5"
    _write_h5(h5_path, [b"A-1"], [0], barcodes_key=False)
    with pytest.raises(ValueError, match="cell_barcodes"):
        ct.filtered_barcodes(h5_path, 1)


def test_filtered_barcodes_raises_on_missing_outlier_flags(tmp_path):
    h5_path = tmp_path / "cnv.h5"
    _write_h5(h5_path, [b"A-1"], [0], outlier_key=False)
    with pytest.raises(ValueError, match="is_high_dimapd"):
        ct.filtered_barcodes(h5_path, 1)


def test_filtered_barcodes_raises_on_mask_length_mismatch(tmp_path):
    h5_path = tmp_path / "cnv.h5"
    _write_h5(h5_path, [b"A-1", b"B-1"], [0, 0, 1])
    with pytest.raises(ValueError, match="mask cannot be reproduced"):
        ct.filtered_barcodes(h5_path, 2)


def test_filtered_barcodes_raises_when_count_disagrees_with_filtered_rows(tmp_path):
    h5_path = tmp_path / "cnv.h5"
    _write_h5(h5_path, [b"A-1", b"B-1", b"C-1"], [0, 0, 1])
    with pytest.raises(ValueError, match="do not describe the same cells"):
        ct.filtered_barcodes(h5_path, 3)


def test_filtered_barcodes_raises_on_duplicates(tmp_path):
    h5_path = tmp_path / "cnv.h5"
    _write_h5(h5_path, [b"A-1", b"A-1"], [0, 0])
    with pytest.raises(ValueError, match="duplicate"):
        ct.filtered_barcodes(h5_path, 2)


# --------------------------------------------------------------------------- #
# Barcode <-> SECEDO cluster mapping
# --------------------------------------------------------------------------- #


def test_barcode_from_bam_name():
    assert ct.barcode_from_bam_name("AAACCTGAGCTAACAA-1.bam") == "AAACCTGAGCTAACAA-1"
    assert ct.barcode_from_bam_name("already_no_suffix") == "already_no_suffix"


def test_barcode_from_bam_name_strips_the_stage_01_cb_prefix():
    # split_by_CBtag.py (stage 01) names per-cell BAMs CB_<barcode>.bam, so stage
    # 03's pileup .map file carries that prefix; cnv_data.h5's own barcodes never
    # do. All four forms must resolve to the same bare barcode, -1 suffix intact.
    expected = "AAACCTGAGTGCTGCC-1"
    assert ct.barcode_from_bam_name("CB_AAACCTGAGTGCTGCC-1.bam") == expected
    assert ct.barcode_from_bam_name("CB_AAACCTGAGTGCTGCC-1") == expected
    assert ct.barcode_from_bam_name("AAACCTGAGTGCTGCC-1.bam") == expected
    assert ct.barcode_from_bam_name("AAACCTGAGTGCTGCC-1") == expected


def test_build_barcode_to_cluster(tmp_path):
    map_file = tmp_path / "chromosome_1.map"
    map_file.write_text("AAA-1.bam\t0\nBBB-1.bam\t1\nCCC-1.bam\t2\n")
    clustering_file = tmp_path / "clustering"
    clustering_file.write_text("0,1,1\n")  # cell 0 unclustered, cells 1&2 -> cluster 1

    mapping = ct.build_barcode_to_cluster(map_file, clustering_file)
    assert mapping == {"BBB-1": "1", "CCC-1": "1"}
    assert "AAA-1" not in mapping  # cluster 0 excluded


def test_build_barcode_to_cluster_strips_the_cb_prefix_to_join_h5_barcodes(tmp_path):
    # The .map file's names carry stage 01's CB_ prefix; cnv_data.h5's barcodes
    # (bare, as filtered_barcodes returns them) do not. The join must match anyway.
    map_file = tmp_path / "chromosome_1.map"
    map_file.write_text("CB_AAA-1.bam\t0\nCB_BBB-1.bam\t1\nCB_CCC-1.bam\t2\n")
    clustering_file = tmp_path / "clustering"
    clustering_file.write_text("0,1,1\n")

    mapping = ct.build_barcode_to_cluster(map_file, clustering_file)
    assert mapping == {"BBB-1": "1", "CCC-1": "1"}
    # The join check_match_rate performs against cnv_data.h5's own bare barcodes.
    h5_barcodes = ["AAA-1", "BBB-1", "CCC-1"]
    assert ct.check_match_rate(h5_barcodes, mapping, min_fraction=0.5) == 2


def test_build_barcode_to_cluster_raises_on_length_mismatch(tmp_path):
    map_file = tmp_path / "chromosome_1.map"
    map_file.write_text("AAA-1.bam\t0\n")
    clustering_file = tmp_path / "clustering"
    clustering_file.write_text("0,1\n")
    with pytest.raises(ValueError, match="clustering has"):
        ct.build_barcode_to_cluster(map_file, clustering_file)


def test_check_match_rate_returns_the_matched_count():
    assert ct.check_match_rate(["a", "b", "c"], {"a": "1", "b": "1"}, 0.5) == 2


def test_check_match_rate_raises_and_shows_both_sides():
    with pytest.raises(ValueError) as err:
        ct.check_match_rate(["AAA-1", "BBB-1"], {"AAA": "1", "BBB": "1"}, 0.5)
    message = str(err.value)
    assert "0/2" in message
    assert "AAA-1" in message  # SCICoNE side
    assert "'AAA'" in message  # SECEDO side


# --------------------------------------------------------------------------- #
# Chromosomes, neutral states, sex audit
# --------------------------------------------------------------------------- #


def test_neutral_states_female():
    assert ct.chromosome_neutral_states(["1", "22", "X", "Y"], "female") == [2, 2, 2, 0]


def test_neutral_states_male():
    assert ct.chromosome_neutral_states(["1", "X", "Y"], "male") == [2, 1, 1]


def test_neutral_states_accept_chr_prefix():
    assert ct.chromosome_neutral_states(["chr3", "chrX", "chrY"], "female") == [2, 2, 0]


def test_neutral_states_raise_on_unrecognised_chromosome():
    with pytest.raises(ValueError, match="unrecognised chromosome 'MT'"):
        ct.chromosome_neutral_states(["1", "MT"], "female")


def test_neutral_states_raise_on_unknown_sex():
    with pytest.raises(ValueError, match="sex must be"):
        ct.chromosome_neutral_states(["1"], "unknown")


def test_sex_chromosome_depth_female_like():
    # chr1 bins 0-3, chr2 bins 4-7, X bins 8-9, Y bins 10-11.
    stops = {"1": 3, "2": 7, "X": 9, "Y": 11}
    cell = np.array([10.0] * 10 + [0.5] * 2)
    depth = ct.sex_chromosome_depth(np.tile(cell, (5, 1)), stops)
    assert depth["X"] == pytest.approx(1.0, abs=0.05)
    assert depth["Y"] < 0.1


def test_sex_chromosome_depth_none_without_bins():
    stops = {"1": 3, "X": 7}  # no chrY at all
    depth = ct.sex_chromosome_depth(np.ones((3, 8)), stops)
    assert depth["Y"] is None
    assert depth["X"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Region condensing, subsampling, pseudobulk aggregation
# --------------------------------------------------------------------------- #


def test_condense_regions_sums_within_regions():
    counts = np.arange(12).reshape(2, 6)
    out = ct.condense_regions(counts, [2, 1, 3])
    assert np.array_equal(out, [[1, 2, 12], [13, 8, 30]])


def test_condense_regions_raises_when_sizes_do_not_cover_the_bins():
    with pytest.raises(ValueError, match="add up"):
        ct.condense_regions(np.ones((2, 6)), [2, 2])


def test_subsample_rows_is_sorted_deterministic_and_capped():
    a = ct.subsample_rows(100, 10, seed=1)
    assert len(a) == 10 and list(a) == sorted(a)
    assert np.array_equal(a, ct.subsample_rows(100, 10, seed=1))
    assert np.array_equal(ct.subsample_rows(5, 10, seed=1), np.arange(5))


def test_aggregate_counts_by_cluster_takes_the_mean():
    counts = np.array([[1, 1], [3, 3], [4, 4], [9, 9]])
    barcodes = ["a", "b", "c", "d"]
    # "d" has no cluster mapping and is excluded entirely.
    pseudobulk, order, sizes = ct.aggregate_counts_by_cluster(
        counts, barcodes, {"a": "1", "b": "1", "c": "2"}
    )
    assert order == ["1", "2"]
    assert np.array_equal(pseudobulk[0], [2, 2])  # mean of a, b
    assert np.array_equal(pseudobulk[1], [4, 4])  # c alone
    assert list(sizes) == [2, 1]


# --------------------------------------------------------------------------- #
# node_dict topology and cell node ids
# --------------------------------------------------------------------------- #

NODE_DICT = {
    "0": {"parent_id": "NULL"},
    "1": {"parent_id": "0"},
    "2": {"parent_id": "1"},
    "3": {"parent_id": "0"},
}


def test_node_parent_map_and_root():
    parent_of, root = ct.node_parent_map(NODE_DICT)
    assert root == "0"
    assert parent_of == {"1": "0", "2": "1", "3": "0"}


def test_node_parent_map_raises_on_empty():
    with pytest.raises(ValueError, match="empty"):
        ct.node_parent_map({})


def test_node_parent_map_raises_without_a_root():
    with pytest.raises(ValueError, match="exactly one root"):
        ct.node_parent_map({"1": {"parent_id": "2"}, "2": {"parent_id": "1"}})


def test_node_parent_map_raises_on_two_roots():
    with pytest.raises(ValueError, match="exactly one root"):
        ct.node_parent_map({"0": {"parent_id": "NULL"}, "1": {"parent_id": "NULL"}})


def test_node_parent_map_raises_on_unknown_parent():
    with pytest.raises(ValueError, match="not a known node"):
        ct.node_parent_map({"0": {"parent_id": "NULL"}, "1": {"parent_id": "9"}})


def _tree_with_ids(ids):
    arr = np.column_stack([np.arange(len(ids)), np.asarray(ids, dtype=float)])
    return SimpleNamespace(outputs={"cell_node_ids": arr})


def test_cell_node_ids_reads_the_last_column_as_strings():
    tree = _tree_with_ids([0, 2, 3])
    assert ct.cell_node_ids(tree, 3, NODE_DICT) == ["0", "2", "3"]


def test_cell_node_ids_raises_when_output_missing():
    with pytest.raises(ValueError, match="no 'cell_node_ids'"):
        ct.cell_node_ids(SimpleNamespace(outputs={}), 3, NODE_DICT)


def test_cell_node_ids_raises_on_row_count_mismatch():
    with pytest.raises(ValueError, match="expected 4"):
        ct.cell_node_ids(_tree_with_ids([0, 1]), 4, NODE_DICT)


def test_cell_node_ids_raises_on_unknown_node():
    with pytest.raises(ValueError, match="absent from node_dict"):
        ct.cell_node_ids(_tree_with_ids([0, 7]), 2, NODE_DICT)


# --------------------------------------------------------------------------- #
# Majority vote
# --------------------------------------------------------------------------- #


def test_majority_cluster_nodes_picks_the_majority():
    cell_to_node = {"a": "n1", "b": "n1", "c": "n2", "d": "n2", "e": "n2"}
    barcode_to_cluster = {"a": "1", "b": "1", "c": "1", "d": "2", "e": "2"}
    resolved, unmatched = ct.majority_cluster_nodes(cell_to_node, barcode_to_cluster)
    assert resolved == {"1": "n1", "2": "n2"}
    assert unmatched == []


def test_majority_cluster_nodes_reports_unmatched_clusters():
    resolved, unmatched = ct.majority_cluster_nodes({"a": "n1"}, {"a": "1", "b": "2"})
    assert resolved == {"1": "n1"}
    assert unmatched == ["2"]


def test_majority_cluster_nodes_deterministic_tie_break():
    resolved, _ = ct.majority_cluster_nodes(
        {"a": "n2", "b": "n1"}, {"a": "1", "b": "1"}
    )
    assert resolved == {"1": "n1"}


# --------------------------------------------------------------------------- #
# Cluster-4 fold and the collapse
# --------------------------------------------------------------------------- #


def test_fold_normal_cluster_removes_it_and_reports_its_node():
    tumour, node, at_root = ct.fold_normal_cluster(
        {"4": "0", "7": "1", "8": "2"}, "4", "0"
    )
    assert tumour == {"7": "1", "8": "2"}
    assert node == "0" and at_root


def test_fold_normal_cluster_flags_a_non_root_normal():
    _, node, at_root = ct.fold_normal_cluster({"4": "3", "7": "1"}, "4", "0")
    assert node == "3" and not at_root


def test_fold_normal_cluster_raises_when_normal_has_no_cells():
    with pytest.raises(ValueError, match="normal cluster '4'"):
        ct.fold_normal_cluster({"7": "1"}, "4", "0")


def test_build_cna_tree_collapses_onto_the_tumour_clusters():
    parent_of, root = ct.node_parent_map(NODE_DICT)
    tree, at_root = ct.build_cna_tree(parent_of, {"7": "1", "8": "2", "9": "3"}, root)
    assert at_root == []
    newick = ct.digraph_to_newick(tree, ct.GERMLINE_ROOT_ID)
    ct.verify_newick(newick, {"7", "8", "9"}, ct.GERMLINE_ROOT_ID)  # no raise
    assert newick == "((8)7,9)germline;"


def test_build_cna_tree_keeps_a_root_voting_cluster_under_germline():
    parent_of, root = ct.node_parent_map(NODE_DICT)
    tree, at_root = ct.build_cna_tree(parent_of, {"7": "0", "8": "1"}, root)
    assert at_root == ["7"]
    # The root-voting cluster must not become 8's parent.
    assert set(tree.predecessors("8")) == {ct.GERMLINE_ROOT_ID}
    assert set(tree.predecessors("7")) == {ct.GERMLINE_ROOT_ID}


def test_shared_node_groups():
    groups = ct.shared_node_groups({"7": "1", "8": "1", "9": "2"})
    assert groups == [["7", "8"]]


# --------------------------------------------------------------------------- #
# Wrapper calls (mocked)
# --------------------------------------------------------------------------- #


def test_detect_bps_raises_when_the_binary_produced_nothing():
    sci = SimpleNamespace(detect_breakpoints=lambda **kw: {"cmd_output": None})
    with pytest.raises(ValueError, match="segmented_regions"):
        ct.detect_bps(sci, np.ones((3, 4)), {"1": 3}, np.arange(3), 2, 3.0)


OPTS = {"n_reps": 3, "max_tries": 1, "copy_number_limit": 4, "cluster_tree_n_iters": 9}


def test_run_per_cell_passes_no_seed():
    seen = {}

    def learn_tree(*args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace()

    ct.run_per_cell(
        SimpleNamespace(learn_tree=learn_tree), np.ones((2, 3)), [1, 2], [2, 2], OPTS
    )
    assert "seed" not in seen  # would override the per-replicate seeds
    assert seen["cluster"] is True and seen["full"] is False
    assert seen["n_reps"] == 3 and seen["copy_number_limit"] == 4


def test_run_per_cell_raises_on_no_tree():
    sci = SimpleNamespace(learn_tree=lambda *a, **k: None)
    with pytest.raises(ValueError, match="no tree"):
        ct.run_per_cell(sci, np.ones((2, 3)), [1, 2], [2, 2], OPTS)


def test_run_pseudobulk_drops_zero_neutral_regions():
    seen = {}

    def learn_single_tree(data, sizes, **kwargs):
        seen.update(data=data, sizes=sizes, **kwargs)
        return SimpleNamespace()

    ct.run_pseudobulk(
        SimpleNamespace(learn_single_tree=learn_single_tree),
        np.arange(6.0).reshape(2, 3),
        [1, 1, 1],
        [2, 0, 2],
        np.array([5.0, 6.0]),
        OPTS,
    )
    assert seen["data"].shape == (2, 2)
    assert list(seen["sizes"]) == [1, 1]
    assert list(seen["region_neutral_states"]) == [2, 2]


# --------------------------------------------------------------------------- #
# End to end with a fake wrapper
# --------------------------------------------------------------------------- #


class _FakeSCICoNE:
    """Stands in for scicone.SCICoNE: 7 kept cells x 12 bins, 4 regions."""

    def __init__(self, build_dir, out_dir, verbose=False):
        rng = np.random.default_rng(0)
        self.data = {
            "filtered_counts": rng.poisson(20, size=(7, 12)).astype(float),
            "filtered_chromosome_stops": {"1": 5, "X": 11},
        }

    def read_10x(self, path):
        assert Path(path).exists()

    def detect_breakpoints(self, **kwargs):
        return {
            "segmented_regions": np.array([2, 5, 8, 11]),
            "segmented_region_sizes": np.array([3, 3, 3, 3]),
        }

    def learn_tree(self, data, sizes, **kwargs):
        assert data.shape == (7, 4)
        # Kept cells in order: cluster 4, then 7, 7, 8, 8, 9, 9.
        nodes = [0, 1, 1, 2, 2, 3, 3]
        arr = np.column_stack([np.arange(7), np.asarray(nodes, dtype=float)])
        return SimpleNamespace(
            node_dict=NODE_DICT, outputs={"cell_node_ids": arr}, score=-1.0
        )


def _fake_scicone_module():
    utils = SimpleNamespace(
        set_region_neutral_states=lambda regions, stops, states: np.full(
            len(regions), 2
        )
    )
    return SimpleNamespace(SCICoNE=_FakeSCICoNE, utils=utils)


def _write_inputs(tmp_path):
    names = [f"{c}-1" for c in "ABCDEFGH"]
    # Cell B (cluster 4) is a CellRanger outlier and is dropped by read_10x.
    _write_h5(
        tmp_path / "cnv.h5", [n.encode() for n in names], [0, 1, 0, 0, 0, 0, 0, 0]
    )
    (tmp_path / "chromosome_1.map").write_text(
        "".join(f"{n}.bam\t{i}\n" for i, n in enumerate(names))
    )
    (tmp_path / "clustering").write_text("4,4,7,7,8,8,9,9\n")


def _run_main(tmp_path, monkeypatch, extra=()):
    _write_inputs(tmp_path)
    monkeypatch.setattr(ct, "_import_scicone", _fake_scicone_module)
    argv = [
        "build_cna_tree.py",
        "--cnv-h5", str(tmp_path / "cnv.h5"),
        "--map-file", str(tmp_path / "chromosome_1.map"),
        "--clustering-file", str(tmp_path / "clustering"),
        "--out-dir", str(tmp_path / "out"),
        "--scicone-build-dir", str(tmp_path),
        "--sex", "female",
        "--normal-cluster-id", "4",
        *extra,
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    ct.main()
    return tmp_path / "out"


def test_main_end_to_end_folds_the_normal_cluster(tmp_path, monkeypatch):
    out = _run_main(tmp_path, monkeypatch)
    assert (out / "cna_tree.nwk").read_text().strip() == "((8)7,9)germline;"
    diagnostics = (out / "cna_tree_diagnostics.txt").read_text()
    assert "Mode: per_cell" in diagnostics
    assert "voted for SCICoNE node 0 (the root, as expected)" in diagnostics
    assert "--sex female" in diagnostics
    assert "chrX" in diagnostics
    rows = (out / "cna_cell_nodes.csv").read_text().splitlines()
    assert rows[0] == "barcode,secedo_cluster,scicone_node"
    assert len(rows) == 1 + 7  # the outlier cell is absent


def test_main_requires_sex(tmp_path, monkeypatch):
    _write_inputs(tmp_path)
    monkeypatch.setattr(ct, "_import_scicone", _fake_scicone_module)
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_cna_tree.py", "--cnv-h5", "x", "--map-file", "x",
         "--clustering-file", "x", "--out-dir", str(tmp_path / "o"),
         "--scicone-build-dir", "x", "--normal-cluster-id", "4"],
    )  # fmt: skip
    with pytest.raises(SystemExit):
        ct.main()


def test_main_fails_when_barcodes_do_not_match(tmp_path, monkeypatch):
    _write_inputs(tmp_path)
    # SECEDO-side names lose the "-1" suffix the h5 barcodes carry.
    (tmp_path / "chromosome_1.map").write_text(
        "".join(f"{c}.bam\t{i}\n" for i, c in enumerate("ABCDEFGH"))
    )
    monkeypatch.setattr(ct, "_import_scicone", _fake_scicone_module)
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_cna_tree.py", "--cnv-h5", str(tmp_path / "cnv.h5"),
         "--map-file", str(tmp_path / "chromosome_1.map"),
         "--clustering-file", str(tmp_path / "clustering"),
         "--out-dir", str(tmp_path / "o"), "--scicone-build-dir", "x",
         "--sex", "female", "--normal-cluster-id", "4"],
    )  # fmt: skip
    with pytest.raises(ValueError, match="match a SECEDO-clustered barcode"):
        ct.main()
