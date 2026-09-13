"""Tests for realdata/scripts/euler/build_tree.py.

Covers the model-loader contract in build_tree.py's module docstring: tree
construction by mutation-set accumulation (known containment -> expected
parents and exact Newick), the 96-channel binning (known context -> exact
COSMIC-order channel), the Newick round-trip against the model loader's own
parsing, and the channel-order guard against cosmic_signatures.csv.
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "realdata" / "scripts" / "euler"))

import build_tree as bt  # noqa: E402

COSMIC_CSV = REPO_ROOT / "COSMIC_sig" / "cosmic_signatures.csv"


# --------------------------------------------------------------------------- #
# Tree construction by mutation-set accumulation
# --------------------------------------------------------------------------- #


def test_build_clone_tree_known_containment():
    """Clone 2 contains clone 1's mutation; clone 3 is disjoint from both.

    Expected: 1 and 3 attach to the root (0 shared mutations each), 2 attaches
    under 1 (its one shared mutation beats every other candidate).
    """
    mutation_sets = {
        "1": {"m1"},
        "2": {"m1", "m2"},
        "3": {"m3"},
    }
    tree = bt.build_clone_tree(mutation_sets, normal_id="0")

    assert dict(nx.get_edge_attributes(tree, "dummy")) == {}  # no stray attrs
    assert set(tree.predecessors("1")) == {"0"}
    assert set(tree.predecessors("3")) == {"0"}
    assert set(tree.predecessors("2")) == {"1"}
    assert [n for n, d in tree.in_degree() if d == 0] == ["0"]


def test_build_clone_tree_exact_newick():
    mutation_sets = {
        "1": {"m1"},
        "2": {"m1", "m2"},
        "3": {"m3"},
    }
    tree = bt.build_clone_tree(mutation_sets, normal_id="0")
    newick = bt.digraph_to_newick(tree, "0")
    assert newick == "((2)1,3)0;"


def test_build_clone_tree_root_is_always_a_fallback():
    """Two clones with no mutation in common with anything both land on the root."""
    mutation_sets = {"5": {"a", "b"}, "9": {"c"}}
    tree = bt.build_clone_tree(mutation_sets, normal_id="normal")
    assert set(tree.successors("normal")) == {"5", "9"}


def test_containment_fraction():
    mutation_sets = {"1": {"a", "b"}, "2": {"a", "b", "c"}}
    assert bt.containment_fraction(mutation_sets, "1", "2") == pytest.approx(1.0)
    # root is never a key in mutation_sets -- treated as the empty set
    assert bt.containment_fraction(mutation_sets, "root", "1") is None


def test_three_gamete_violations():
    # x: cluster 1 only, y: cluster 3 only, both: cluster 2 -- all three gametes
    # (10, 01, 11) present for the pair (x, y), a perfect-phylogeny violation.
    mutation_sets = {"1": {"x"}, "2": {"x", "y"}, "3": {"y"}}
    assert bt.three_gamete_violations(mutation_sets) == 1

    # nested sets never violate: every mutation in 1 also sits in 2
    nested = {"1": {"x"}, "2": {"x", "y"}}
    assert bt.three_gamete_violations(nested) == 0


# --------------------------------------------------------------------------- #
# Newick round-trip against the model loader's own parsing
# --------------------------------------------------------------------------- #


def test_newick_roundtrip_matches_model_loader():
    mutation_sets = {"1": {"m1"}, "2": {"m1", "m2"}, "3": {"m3"}}
    tree = bt.build_clone_tree(mutation_sets, normal_id="normal")
    newick = bt.digraph_to_newick(tree, "normal")

    bt.verify_newick(newick, cluster_ids={"1", "2", "3"}, normal_id="normal")

    parsed = bt.parse_newick_like_model(newick)
    assert set(parsed.nodes()) == {"normal", "1", "2", "3"}
    roots = [n for n, d in parsed.in_degree() if d == 0]
    assert roots == ["normal"]


def test_verify_newick_rejects_wrong_root():
    with pytest.raises(AssertionError):
        bt.verify_newick("(b,c)a;", cluster_ids={"b", "c"}, normal_id="a_typo")


def test_verify_newick_rejects_missing_cluster():
    with pytest.raises(AssertionError):
        bt.verify_newick("(b)a;", cluster_ids={"b", "c"}, normal_id="a")


# --------------------------------------------------------------------------- #
# 96-channel binning, COSMIC order
# --------------------------------------------------------------------------- #


def test_channel_labels_length_and_names():
    labels = bt.channel_labels()
    assert len(labels) == 96
    assert labels[0] == "Channel_0"
    assert labels[-1] == "Channel_95"


def test_cosmic_signatures_channel_order_guard():
    """Guard: cosmic_signatures.csv's columns equal build_tree's channel axis,
    in order."""
    cols = list(pd.read_csv(COSMIC_CSV, index_col=0, nrows=0).columns)
    assert cols == bt.channel_labels()


def test_pyrimidine_normalise_leaves_pyrimidine_ref_alone():
    assert bt.pyrimidine_normalise("A", "C", "T", "G") == ("A", "C", "T", "G")


def test_pyrimidine_normalise_flips_purine_ref():
    # five=T, ref=G, alt=A, three=C on the reference strand -> reverse
    # complement onto the pyrimidine strand: five=G, ref=C, alt=T, three=A.
    assert bt.pyrimidine_normalise("T", "G", "A", "C") == ("G", "C", "T", "A")


def test_snv_channel_known_context_matches_cosmic_order():
    """A[C>T]G is SBS1's dominant channel (NCG>NTG, CpG deamination) at index 10
    in cosmic_signatures.csv -- confirmed against the file's own SBS1 row."""
    assert bt.snv_channel("A", "C", "T", "G") == 10

    row = pd.read_csv(COSMIC_CSV, index_col=0).loc["SBS1"]
    assert row.idxmax() == "Channel_10"


class _FakeFasta:
    """Duck-types pysam.FastaFile's .fetch(chrom, start, end) for a fixed context."""

    def __init__(self, contexts):
        self.contexts = contexts  # {(chrom, start, end): context_str}

    def fetch(self, chrom, start, end):
        return self.contexts[(chrom, start, end)]


def test_bin_cluster_spectra_known_snv():
    # 1-based VCF pos 100, ref C, alt T -> 0-based fetch window (98, 101) = "ACG"
    fasta = _FakeFasta({("1", 98, 101): "ACG"})
    cluster_to_snvs = {"7": {("1", 100, "C", "T")}}
    spectra, skipped = bt.bin_cluster_spectra(cluster_to_snvs, fasta)

    assert spectra.loc["7", "Channel_10"] == 1
    assert spectra.loc["7"].sum() == 1
    assert skipped == {"ambiguous_context": 0, "ref_mismatch": 0}


def test_bin_cluster_spectra_skips_ambiguous_and_ref_mismatch():
    fasta = _FakeFasta(
        {
            ("1", 98, 101): "ANG",  # ambiguous middle-ish base
            ("1", 198, 201): "AAG",  # middle base A, but VCF ref is C -> mismatch
        }
    )
    cluster_to_snvs = {"7": {("1", 100, "C", "T"), ("1", 200, "C", "T")}}
    spectra, skipped = bt.bin_cluster_spectra(cluster_to_snvs, fasta)

    assert spectra.loc["7"].sum() == 0
    assert skipped == {"ambiguous_context": 1, "ref_mismatch": 1}


# --------------------------------------------------------------------------- #
# VCF and matrix plumbing
# --------------------------------------------------------------------------- #


_VCF_TEXT = """\
##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
1\t100\t.\tC\tT\t.\tPASS\t.
1\t200\t.\tC\tT\t.\tlow_allele_frac\t.
1\t300\t.\tCA\tT\t.\tPASS\t.
1\t400\t.\tC\tT,G\t.\tPASS\t.
"""


def test_parse_vcf_snvs(tmp_path):
    vcf_path = tmp_path / "clone1_1.vcf"
    vcf_path.write_text(_VCF_TEXT)
    snvs = bt.parse_vcf_snvs(vcf_path)
    assert snvs == {
        ("1", 100, "C", "T"),  # PASS SNP, kept
        ("1", 400, "C", "T"),  # multiallelic, single-base alts split and kept
        ("1", 400, "C", "G"),
    }
    # position 200 (non-PASS) and 300 (indel) are excluded


def test_discover_cluster_vcfs(tmp_path):
    for name in ["clone1_1.vcf", "clone1_2.vcf", "clone5_X.vcf", "clone9_1.vcf"]:
        (tmp_path / name).write_text(_VCF_TEXT)
    (tmp_path / "not_a_clone_vcf.txt").write_text("")

    found = bt.discover_cluster_vcfs(tmp_path, normal_id="9")
    assert set(found) == {"1", "5"}
    assert sorted(f.name for f in found["1"]) == ["clone1_1.vcf", "clone1_2.vcf"]


def test_build_snv_presence_matrix_and_mutation_sets_roundtrip():
    cluster_to_snvs = {
        "1": {("1", 100, "C", "T")},
        "2": {("1", 100, "C", "T"), ("1", 200, "C", "A")},
    }
    matrix = bt.build_snv_presence_matrix(cluster_to_snvs)
    assert list(matrix.columns) == ["1", "2"]
    assert set(matrix.index) == {"1:100:C>T", "1:200:C>A"}
    assert matrix.loc["1:100:C>T", "1"] == 1
    assert matrix.loc["1:200:C>A", "1"] == 0
    assert matrix.loc["1:200:C>A", "2"] == 1

    mutation_sets = bt.mutation_sets_from_matrix(matrix)
    assert mutation_sets["1"] == {"1:100:C>T"}
    assert mutation_sets["2"] == {"1:100:C>T", "1:200:C>A"}


# --------------------------------------------------------------------------- #
# SCITE plumbing (primary tree builder; the binary itself is not exercised)
# --------------------------------------------------------------------------- #


def test_write_scite_matrix(tmp_path):
    matrix = pd.DataFrame([[1, 0], [0, 1]], index=["m1", "m2"], columns=["1", "2"])
    out = tmp_path / "geno.txt"
    bt.write_scite_matrix(matrix, out)
    loaded = np.loadtxt(out, dtype=int)
    assert np.array_equal(loaded, matrix.values)


def test_write_scite_mutation_names(tmp_path):
    out = tmp_path / "names.txt"
    bt.write_scite_mutation_names(["1:100:C>T", "1:200:C>A"], out)
    assert out.read_text().splitlines() == ["1:100:C>T", "1:200:C>A"]


def test_build_scite_input_matrix_orientation_and_normal_column():
    cluster_to_snvs = {
        "7": {("1", 100, "C", "T")},
        "8": {("1", 100, "C", "T"), ("1", 200, "C", "A")},
    }
    snv_matrix = bt.build_snv_presence_matrix(cluster_to_snvs)
    full, column_order = bt.build_scite_input_matrix(snv_matrix, normal_id="4")

    assert column_order == ["4", "7", "8"]  # pseudo-normal sorted in by cluster ID
    assert list(full.columns) == column_order
    assert full.shape == (2, 3)  # 2 SNVs (rows) x 3 samples (columns)
    assert (full["4"] == 0).all()  # pseudo-normal is the all-zero reference
    assert list(full["7"]) == list(snv_matrix["7"])
    assert list(full["8"]) == list(snv_matrix["8"])


def test_find_scite_binary_precedence(monkeypatch):
    assert bt.find_scite_binary("explicit/path") == "explicit/path"

    monkeypatch.setenv("SCITE_BIN", "env/path")
    assert bt.find_scite_binary(None) == "env/path"

    monkeypatch.delenv("SCITE_BIN")
    default = Path(bt.find_scite_binary(None))
    assert default == REPO_ROOT / "realdata" / "external" / "scite" / "scite"


def test_run_scite_builds_expected_command_and_returns_newick_path(
    tmp_path, monkeypatch
):
    calls = {}

    def fake_run(cmd, check, stdout, stderr):
        calls["cmd"] = cmd
        out_prefix = cmd[cmd.index("-o") + 1]
        Path(f"{out_prefix}_ml0.newick").write_text("(1,2)root;\n")

    monkeypatch.setattr(bt.subprocess, "run", fake_run)

    newick_path = bt.run_scite(
        tmp_path / "geno.txt",
        n_mutations=2,
        n_samples=2,
        out_prefix=tmp_path / "scite_out",
        log_path=tmp_path / "scite.log",
        scite_bin="scite",
        names_path=tmp_path / "names.txt",
    )
    assert newick_path == tmp_path / "scite_out_ml0.newick"
    assert newick_path.exists()
    cmd = calls["cmd"]
    assert cmd[0] == "scite"
    assert "-s" in cmd  # MAP, always requested
    assert cmd[cmd.index("-seed") + 1] == "42"  # fixed default, for reproducibility
    assert cmd[cmd.index("-names") + 1] == str(tmp_path / "names.txt")
    assert (tmp_path / "scite.log").exists()  # stdout/stderr captured


def test_run_scite_raises_if_newick_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(bt.subprocess, "run", lambda *a, **k: None)
    with pytest.raises(FileNotFoundError):
        bt.run_scite(
            tmp_path / "geno.txt",
            n_mutations=2,
            n_samples=2,
            out_prefix=tmp_path / "scite_out",
            log_path=tmp_path / "scite.log",
            scite_bin="scite",
        )


def test_parse_scite_newick(tmp_path):
    p = tmp_path / "scite_out_ml0.newick"
    p.write_text("(1,2)root;\n")
    tree = bt.parse_scite_newick(p)
    labels = {d.get("label") for _, d in tree.nodes(data=True)}
    assert labels == {"root", "1", "2"}


def test_collapse_scite_tree_to_clones(tmp_path):
    # SCITE mutation tree: root -> sample-leaf "1" (cluster c0) -> sample-leaf
    # "2" (cluster c1); root -> sample-leaf "3" (the pseudo-normal). Neither
    # cluster's own ID appears in the newick (no per-sample naming in this
    # SCITE build), so this also exercises the column-index fallback.
    newick_path = tmp_path / "scite_out_ml0.newick"
    newick_path.write_text("((2)1,3)root;\n")
    scite_tree = bt.parse_scite_newick(newick_path)

    column_order = ["c0", "c1", "n"]
    clone_tree = bt.collapse_scite_tree_to_clones(
        scite_tree, column_order, normal_id="n"
    )

    assert set(clone_tree.predecessors("c0")) == {"n"}  # no sample ancestor -> root
    assert set(clone_tree.predecessors("c1")) == {"c0"}  # nested under c0's leaf
    assert bt.digraph_to_newick(clone_tree, "n") == "((c1)c0)n;"


def test_collapse_scite_tree_to_clones_raises_on_missing_leaf(tmp_path):
    newick_path = tmp_path / "scite_out_ml0.newick"
    newick_path.write_text("(1,3)root;\n")  # cluster c1 (column 2) has no leaf
    scite_tree = bt.parse_scite_newick(newick_path)
    with pytest.raises(ValueError, match="c1"):
        bt.collapse_scite_tree_to_clones(scite_tree, ["c0", "c1", "n"], normal_id="n")


def test_compare_topologies_agrees_on_identical_chains():
    tree_a = nx.DiGraph([("root", "c0"), ("c0", "c1")])
    tree_b = nx.DiGraph([("root2", "c0"), ("c0", "c1")])
    assert bt.compare_topologies(tree_a, tree_b, ["c0", "c1"]) == pytest.approx(1.0)


def test_compare_topologies_disagrees_on_star_vs_chain():
    chain = nx.DiGraph([("root", "c0"), ("c0", "c1")])
    star = nx.DiGraph([("root", "c0"), ("root", "c1")])
    assert bt.compare_topologies(chain, star, ["c0", "c1"]) == pytest.approx(0.0)
