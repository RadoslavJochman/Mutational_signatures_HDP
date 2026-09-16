"""Tests for realdata/scripts/euler/build_snv_tree.py.

Covers the model-loader contract in build_snv_tree.py's module docstring: the
96-channel binning, the Newick round-trip against the model loader's own
parsing, the channel-order guard against cosmic_signatures.csv, VCF/matrix
plumbing, the Camin-Sokal parsimony search (exhaustive topology enumeration
and scoring), and LICHeE input/output plumbing (the binary itself is not
exercised -- subprocess calls are mocked, matching this repo's existing
convention for external-tool wrappers).
"""

import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "realdata" / "scripts" / "euler"))

import build_snv_tree as bt  # noqa: E402

COSMIC_CSV = REPO_ROOT / "COSMIC_sig" / "cosmic_signatures.csv"


# --------------------------------------------------------------------------- #
# Newick round-trip against the model loader's own parsing
# --------------------------------------------------------------------------- #


def test_newick_roundtrip_matches_model_loader():
    tree = nx.DiGraph([("normal", "1"), ("1", "2"), ("normal", "3")])
    newick = bt.digraph_to_newick(tree, "normal")
    assert newick == "((2)1,3)normal;"

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
# Three-gamete compatibility diagnostic (tool-agnostic)
# --------------------------------------------------------------------------- #


def test_three_gamete_violations():
    # x: cluster 1 only, y: cluster 3 only, both: cluster 2 -- all three gametes
    # (10, 01, 11) present for the pair (x, y), a perfect-phylogeny violation.
    mutation_sets = {"1": {"x"}, "2": {"x", "y"}, "3": {"y"}}
    assert bt.three_gamete_violations(mutation_sets) == 1

    # nested sets never violate: every mutation in 1 also sits in 2
    nested = {"1": {"x"}, "2": {"x", "y"}}
    assert bt.three_gamete_violations(nested) == 0


# --------------------------------------------------------------------------- #
# 96-channel binning, COSMIC order
# --------------------------------------------------------------------------- #


def test_channel_labels_length_and_names():
    labels = bt.channel_labels()
    assert len(labels) == 96
    assert labels[0] == "Channel_0"
    assert labels[-1] == "Channel_95"


def test_cosmic_signatures_channel_order_guard():
    """Guard: cosmic_signatures.csv's columns equal build_snv_tree's channel
    axis, in order."""
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


_FORCED_VCF_TEXT = """\
##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE
1\t100\t.\tC\tT\t.\tPASS\t.\tGT:AD:AF:DP\t0/1:8,12:0.600:20
1\t200\t.\tC\tT\t.\tgermline\t.\tGT:AD:AF:DP\t0/0:20,1:0.048:21
1\t300\t.\tCA\tT\t.\tPASS\t.\tGT:AD:AF:DP\t0/1:5,5:0.500:10
1\t400\t.\tC\tT,G\t.\tPASS\t.\tGT:AD:AF:DP\t0/1/2:10,3,2:0.300,0.200:15
1\t500\t.\tC\tT\t.\tPASS\t.\tGT:AD:AF:DP\t./.:.:.:0
"""


def test_parse_format_values():
    assert bt.parse_format_values("GT:AD:AF:DP", "0/1:8,12:0.600:20") == {
        "GT": "0/1",
        "AD": "8,12",
        "AF": "0.600",
        "DP": "20",
    }


def test_parse_forced_vcf_calls(tmp_path):
    vcf_path = tmp_path / "clone7_1.forced.vcf"
    vcf_path.write_text(_FORCED_VCF_TEXT)
    calls = bt.parse_forced_vcf_calls(vcf_path)

    assert calls[("1", 100, "C", "T")] == (pytest.approx(0.6), 12)
    assert calls[("1", 200, "C", "T")] == (
        pytest.approx(0.048),
        1,
    )  # non-PASS still parsed
    assert ("1", 300, "C", "T") not in calls  # multi-base REF (indel) excluded
    assert calls[("1", 400, "C", "T")] == (
        pytest.approx(0.3),
        3,
    )  # multi-allelic, split
    assert calls[("1", 400, "C", "G")] == (pytest.approx(0.2), 2)
    assert calls[("1", 500, "C", "T")] == (0.0, 0)  # "." fields -> zero, not raised


def test_resolve_presence_calls_thresholds_vaf_and_alt_reads():
    cluster_to_calls = {
        "7": {("1", 100, "C", "T"): (0.6, 12), ("1", 200, "C", "T"): (0.048, 1)},
        "8": {("1", 100, "C", "T"): (0.02, 5)},  # enough reads, VAF too low
    }
    presence = bt.resolve_presence_calls(
        cluster_to_calls, min_vaf=0.05, min_alt_reads=2
    )
    assert presence["7"] == {("1", 100, "C", "T")}  # the 200 site fails both thresholds
    assert presence["8"] == set()


def test_discover_forced_cluster_vcfs(tmp_path):
    for name in [
        "clone1_1.forced.vcf",
        "clone1_2.forced.vcf",
        "clone5_X.forced.vcf",
        "clone9_1.forced.vcf",
    ]:
        (tmp_path / name).write_text(_FORCED_VCF_TEXT)
    (tmp_path / "not_a_clone_vcf.txt").write_text("")
    (tmp_path / "clone1_1.vcf").write_text(_FORCED_VCF_TEXT)  # pass-1 output, ignored

    found = bt.discover_forced_cluster_vcfs(tmp_path)
    assert set(found) == {"1", "5", "9"}  # every cluster, none excluded
    assert sorted(f.name for f in found["1"]) == [
        "clone1_1.forced.vcf",
        "clone1_2.forced.vcf",
    ]


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
# Topology classification (tool-agnostic)
# --------------------------------------------------------------------------- #


def test_is_unbranched_chain():
    assert bt.is_unbranched_chain(nx.DiGraph([("a", "b"), ("b", "c")])) is True
    assert bt.is_unbranched_chain(nx.DiGraph([("a", "b"), ("a", "c")])) is False
    single = nx.DiGraph()
    single.add_node("a")
    assert bt.is_unbranched_chain(single) is True


def test_chain_depth_if_linear():
    chain = nx.DiGraph([("a", "b"), ("b", "c")])
    assert bt.chain_depth_if_linear(chain) == 2
    branching = nx.DiGraph([("a", "b"), ("a", "c")])
    assert bt.chain_depth_if_linear(branching) is None
    two_roots = nx.DiGraph([("a", "b")])
    two_roots.add_node("z")  # a second in-degree-0 node -- not a single path
    assert bt.chain_depth_if_linear(two_roots) is None


def test_classify_topology():
    assert bt.classify_topology(None) == "unavailable"
    assert bt.classify_topology(nx.DiGraph([("a", "b"), ("b", "c")])) == (
        "linear chain (depth 2)"
    )
    assert bt.classify_topology(nx.DiGraph([("a", "b"), ("a", "c")])) == "branching"


def test_compare_topologies_agrees_on_identical_chains():
    tree_a = nx.DiGraph([("root", "c0"), ("c0", "c1")])
    tree_b = nx.DiGraph([("root2", "c0"), ("c0", "c1")])
    assert bt.compare_topologies(tree_a, tree_b, ["c0", "c1"]) == pytest.approx(1.0)


def test_compare_topologies_disagrees_on_star_vs_chain():
    chain = nx.DiGraph([("root", "c0"), ("c0", "c1")])
    star = nx.DiGraph([("root", "c0"), ("root", "c1")])
    assert bt.compare_topologies(chain, star, ["c0", "c1"]) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Camin-Sokal parsimony
# --------------------------------------------------------------------------- #


def test_enumerate_rooted_trees_count_and_validity():
    # 2 leaves, root "r": each leaf's parent is r or the other leaf, minus the
    # self-parent and 2-cycle cases -- (1+2)**2 = 9 raw combos, 3 invalid
    # (self-parent x2, mutual-parent cycle x1) leaves 6 valid trees... but a
    # leaf can't be its own parent (already excluded), and "1->2, 2->1" is a
    # 2-cycle with neither reaching r -- invalid. Just check every yielded
    # tree really is acyclic and rooted at r, and count is as expected.
    trees = list(bt.enumerate_rooted_trees(["1", "2"], "r"))
    for parent_of in trees:
        assert bt._is_valid_rooted_tree(parent_of, "r", ["1", "2"])
        assert parent_of["1"] != "1"
        assert parent_of["2"] != "2"
    # Valid trees: (1->r,2->r), (1->r,2->1), (1->2,2->r) = 3
    assert len(trees) == 3


def test_is_valid_rooted_tree_rejects_cycle():
    # 1 -> 2 -> 1, neither reaches root "r"
    assert bt._is_valid_rooted_tree({"1": "2", "2": "1"}, "r", ["1", "2"]) is False


def test_is_valid_rooted_tree_accepts_chain_to_root():
    assert bt._is_valid_rooted_tree({"1": "r", "2": "1"}, "r", ["1", "2"]) is True


def test_camin_sokal_topology_score_counts_gains_and_reversals():
    mutation_sets = {"1": {"m1"}, "2": {"m1", "m2"}, "3": set()}
    # Tree: r -> 1 -> 2, r -> 3. Edge r->1: gain m1 (root empty -> {m1}).
    # Edge 1->2: gain m2 ({m1} -> {m1,m2}). Edge r->3: no change.
    parent_of = {"1": "r", "2": "1", "3": "r"}
    total, n_gains, n_reversals = bt.camin_sokal_topology_score(
        parent_of, mutation_sets
    )
    assert n_gains == 2
    assert n_reversals == 0
    assert total == 2


def test_camin_sokal_topology_score_penalises_reversal():
    mutation_sets = {"1": {"m1"}, "2": set()}
    # Tree: r -> 1 -> 2. Edge r->1: gain m1. Edge 1->2: reversal (loses m1).
    parent_of = {"1": "r", "2": "1"}
    total, n_gains, n_reversals = bt.camin_sokal_topology_score(
        parent_of, mutation_sets
    )
    assert n_gains == 1
    assert n_reversals == 1
    assert total == bt.CAMIN_SOKAL_REVERSAL_PENALTY + 1


def test_camin_sokal_tree_finds_known_optimal_nesting():
    # Cluster 2 contains cluster 1's mutation plus its own; cluster 3 is
    # disjoint. The zero-reversal, minimum-gain tree nests 2 under 1 (shared
    # m1 inherited, only m2 gained) rather than under root (m1 and m2 both
    # gained independently) or any other arrangement.
    mutation_sets = {
        "1": {"m1"},
        "2": {"m1", "m2"},
        "3": {"m3"},
    }
    tree, n_gains, n_reversals = bt.camin_sokal_tree(mutation_sets, "root")
    assert n_reversals == 0
    assert n_gains == 3  # m1 (at 1), m2 (at 2), m3 (at 3)
    assert set(tree.predecessors("2")) == {"1"}
    assert set(tree.predecessors("1")) == {"root"}
    assert set(tree.predecessors("3")) == {"root"}


def test_camin_sokal_tree_deterministic_across_runs():
    mutation_sets = {"1": {"m1"}, "2": {"m1", "m2"}, "3": {"m3"}, "4": {"m3", "m4"}}
    tree_a, _, _ = bt.camin_sokal_tree(mutation_sets, "root")
    tree_b, _, _ = bt.camin_sokal_tree(mutation_sets, "root")
    assert sorted(tree_a.edges()) == sorted(tree_b.edges())


def test_camin_sokal_tree_raises_past_cap():
    mutation_sets = {str(i): {f"m{i}"} for i in range(5)}
    with pytest.raises(ValueError, match="exceeds"):
        bt.camin_sokal_tree(mutation_sets, "root", max_clusters=3)


def test_camin_sokal_tree_handles_incompatible_matrix_without_raising():
    # x: 1 only, y: 3 only, both: 2 -- a genuine three-gamete violation, no
    # fully compatible tree exists. Must still return a well-defined tree.
    mutation_sets = {"1": {"x"}, "2": {"x", "y"}, "3": {"y"}}
    tree, n_gains, n_reversals = bt.camin_sokal_tree(mutation_sets, "root")
    assert set(tree.nodes()) == {"root", "1", "2", "3"}
    # Some arrangement is optimal; whatever it is, it must be a valid tree
    # (every non-root node has exactly one parent, reachable from root).
    assert nx.is_weakly_connected(tree)


# --------------------------------------------------------------------------- #
# LICHeE plumbing (the binary itself is not exercised)
# --------------------------------------------------------------------------- #


def test_write_lichee_input_format_and_germline_column(tmp_path):
    cluster_to_calls = {
        "7": {("1", 100, "C", "T"): (0.6, 12)},
        "8": {("1", 100, "C", "T"): (0.3, 5), ("1", 200, "C", "A"): (0.4, 6)},
    }
    out = tmp_path / "lichee_in.txt"
    columns = bt.write_lichee_input(cluster_to_calls, out)

    assert columns == [bt.GERMLINE_ROOT_ID, "7", "8"]
    lines = out.read_text().splitlines()
    assert lines[0] == "#chr\tposition\tdescription\tgermline\t7\t8"
    # 2 SNV rows, sorted by (chrom, pos, ref, alt)
    assert len(lines) == 3
    row_100 = lines[1].split("\t")
    assert row_100[:4] == ["1", "100", "C>T", "0.0"]  # germline always 0.0
    assert row_100[4] == "0.6000"  # cluster 7's VAF at this site
    assert row_100[5] == "0.3000"  # cluster 8's VAF at this site too
    row_200 = lines[2].split("\t")
    assert row_200[4] == "0.0000"  # cluster 7 has no call here
    assert row_200[5] == "0.4000"


def test_find_lichee_binary_precedence(monkeypatch):
    assert bt.find_lichee_binary("explicit/path") == "explicit/path"

    monkeypatch.setenv("LICHEE_BIN", "env/path")
    assert bt.find_lichee_binary(None) == "env/path"

    monkeypatch.delenv("LICHEE_BIN")
    default = Path(bt.find_lichee_binary(None))
    expected = REPO_ROOT / "realdata" / "external" / "lichee" / "release" / "lichee"
    assert default == expected


def test_run_lichee_builds_expected_command_and_returns_dot_path(tmp_path, monkeypatch):
    calls = {}

    def fake_run(cmd, check, stdout, stderr):
        calls["cmd"] = cmd
        out_prefix = cmd[cmd.index("-o") + 1]
        Path(f"{out_prefix}.dot").write_text('"1" -> "2";\n')

    monkeypatch.setattr(bt.subprocess, "run", fake_run)

    dot_path = bt.run_lichee(
        tmp_path / "in.txt",
        out_prefix=tmp_path / "lichee_out",
        log_path=tmp_path / "lichee.log",
        lichee_bin="lichee",
        normal_index=0,
        min_vaf_present=0.05,
        max_vaf_absent=0.0,
    )
    assert dot_path.exists()
    cmd = calls["cmd"]
    assert cmd[0] == "lichee"
    assert "-build" in cmd
    assert cmd[cmd.index("-n") + 1] == "0"
    assert (tmp_path / "lichee.log").exists()


def test_run_lichee_raises_if_dot_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(bt.subprocess, "run", lambda *a, **k: None)
    with pytest.raises(FileNotFoundError):
        bt.run_lichee(
            tmp_path / "in.txt",
            out_prefix=tmp_path / "lichee_out",
            log_path=tmp_path / "lichee.log",
            lichee_bin="lichee",
        )


def test_parse_lichee_dot_edges_and_labels(tmp_path):
    dot_path = tmp_path / "out.dot"
    dot_path.write_text(
        "digraph G {\n"
        '"n0" [label="germline"];\n'
        '"n1" [label="7"];\n'
        '"n0" -> "n1";\n'
        '"n1" -> "n2";\n'
        "}\n"
    )
    node_labels, edges = bt.parse_lichee_dot(dot_path)
    assert node_labels["n0"] == "germline"
    assert node_labels["n1"] == "7"
    assert {"a": "n0", "b": "n1"} in edges
    assert {"a": "n1", "b": "n2"} in edges


def test_parse_lichee_dot_raises_on_no_edges(tmp_path):
    dot_path = tmp_path / "empty.dot"
    dot_path.write_text("digraph G {\n}\n")
    with pytest.raises(ValueError, match="no parseable"):
        bt.resolve_lichee_clone_tree(dot_path, ["7", "8"], "germline")


def test_resolve_lichee_clone_tree_via_node_ids(tmp_path):
    # DOT nodes named directly after the real cluster IDs (no label attrs) --
    # the verbatim scheme should resolve immediately.
    dot_path = tmp_path / "out.dot"
    dot_path.write_text(
        'digraph G {\n"germline" -> "m1";\n"m1" -> "7";\n"m1" -> "8";\n}\n'
    )
    tree, source = bt.resolve_lichee_clone_tree(dot_path, ["7", "8"], "germline")
    assert set(tree.predecessors("7")) == {"germline"}
    assert set(tree.predecessors("8")) == {"germline"}
    assert "dot" in source


def test_resolve_lichee_clone_tree_via_label_attribute(tmp_path):
    # Node ids are opaque LICHeE-internal ids; real cluster identity only
    # appears in each node's label="..." attribute.
    dot_path = tmp_path / "out.dot"
    dot_path.write_text(
        "digraph G {\n"
        '"n0" [label="germline"];\n'
        '"n1" [label="7"];\n'
        '"n2" [label="8"];\n'
        '"n0" -> "n1";\n'
        '"n1" -> "n2";\n'
        "}\n"
    )
    tree, source = bt.resolve_lichee_clone_tree(dot_path, ["7", "8"], "germline")
    assert set(tree.predecessors("7")) == {"germline"}
    assert set(tree.predecessors("8")) == {"7"}


def test_resolve_lichee_clone_tree_raises_when_unresolved(tmp_path):
    dot_path = tmp_path / "out.dot"
    dot_path.write_text('digraph G {\n"a" -> "b";\n}\n')
    with pytest.raises(ValueError, match="no consistent"):
        bt.resolve_lichee_clone_tree(dot_path, ["7", "8"], "germline")
