"""Tests for realdata/scripts/euler/build_snv_tree.py.

Covers the model-loader contract in build_snv_tree.py's module docstring: the
96-channel binning, the Newick round-trip against the model loader's own
parsing, the channel-order guard against cosmic_signatures.csv, VCF/matrix
plumbing, the Camin-Sokal parsimony search (exhaustive topology enumeration
and scoring), and the LICHeE integration: input writing, the java invocation, the
``.trees.txt`` parser and the tree it builds (the binary itself is not
exercised -- subprocess calls are mocked, matching this repo's existing
convention for external-tool wrappers). The two fixtures under
``tests/fixtures/lichee`` are verbatim LICHeE output from real runs.
"""

import subprocess
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "realdata" / "scripts" / "euler"))

import build_snv_tree as bt  # noqa: E402

COSMIC_CSV = REPO_ROOT / "COSMIC_sig" / "cosmic_signatures.csv"
LICHEE_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "lichee"


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
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tclone4\tclone7
1\t100\t.\tC\tT\t.\tPASS\t.\tGT:AD:AF:DP\t0/0:20,0:0.0:20\t0/1:8,12:0.600:20
1\t200\t.\tC\tT\t.\tgermline\t.\tGT:AD:AF:DP\t0/0:19,1:0.05:20\t0/0:20,1:0.048:21
1\t300\t.\tCA\tT\t.\tPASS\t.\tGT:AD:AF:DP\t0/0:9,0:0.0:9\t0/1:5,5:0.500:10
1\t400\t.\tC\tT,G\t.\tPASS\t.\tGT:AD:AF:DP\t0/0:14,0,0:0.0,0.0:14\t0/1/2:10,3,2:0.300,0.200:15
1\t500\t.\tC\tT\t.\tPASS\t.\tGT:AD:AF:DP\t0/0:10,0:0.0:10\t./.:.:.:0
"""

# Same records, but the tumour's column comes FIRST in the header (GATK writes sample
# columns in sorted name order, so which side the tumour lands on depends on the
# cluster ID and NORMAL_CLUSTER_ID -- clone3/clone10 sort ahead of clone4, clone7/8/9
# sort after it). Column values are swapped to match the swapped header.
_FORCED_VCF_TEXT_TUMOUR_FIRST = """\
##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tclone7\tclone4
1\t100\t.\tC\tT\t.\tPASS\t.\tGT:AD:AF:DP\t0/1:8,12:0.600:20\t0/0:20,0:0.0:20
1\t200\t.\tC\tT\t.\tgermline\t.\tGT:AD:AF:DP\t0/0:20,1:0.048:21\t0/0:19,1:0.05:20
1\t300\t.\tCA\tT\t.\tPASS\t.\tGT:AD:AF:DP\t0/1:5,5:0.500:10\t0/0:9,0:0.0:9
1\t400\t.\tC\tT,G\t.\tPASS\t.\tGT:AD:AF:DP\t0/1/2:10,3,2:0.300,0.200:15\t0/0:14,0,0:0.0,0.0:14
1\t500\t.\tC\tT\t.\tPASS\t.\tGT:AD:AF:DP\t./.:.:.:0\t0/0:10,0:0.0:10
"""


def test_parse_format_values():
    assert bt.parse_format_values("GT:AD:AF:DP", "0/1:8,12:0.600:20") == {
        "GT": "0/1",
        "AD": "8,12",
        "AF": "0.600",
        "DP": "20",
    }


@pytest.mark.parametrize(
    "text",
    [_FORCED_VCF_TEXT, _FORCED_VCF_TEXT_TUMOUR_FIRST],
    ids=["normal-first", "tumour-first"],
)
def test_parse_forced_vcf_calls_reads_the_tumour_column_by_name(tmp_path, text):
    vcf_path = tmp_path / "clone7_1.forced.vcf"
    vcf_path.write_text(text)
    calls = bt.parse_forced_vcf_calls(vcf_path, "7")

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


@pytest.mark.parametrize("text", [_FORCED_VCF_TEXT, _FORCED_VCF_TEXT_TUMOUR_FIRST])
def test_parse_forced_vcf_calls_reads_the_normal_column_for_its_own_id(tmp_path, text):
    # Reading cluster "4" (the pseudo-normal's own column) gets its low/zero values,
    # not the tumour's -- confirms the selection really is by name, not position.
    vcf_path = tmp_path / "clone4_1.forced.vcf"
    vcf_path.write_text(text)
    calls = bt.parse_forced_vcf_calls(vcf_path, "4")
    assert calls[("1", 100, "C", "T")] == (0.0, 0)


def test_parse_forced_vcf_calls_raises_when_the_sample_column_is_absent(tmp_path):
    vcf_path = tmp_path / "clone9_1.forced.vcf"
    vcf_path.write_text(_FORCED_VCF_TEXT)
    with pytest.raises(ValueError, match="clone9") as err:
        bt.parse_forced_vcf_calls(vcf_path, "9")
    assert "clone4" in str(err.value) and "clone7" in str(err.value)


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


# --------------------------------------------------------------------------- #
# Rebuilding stage 06b's union from pass-1's own PASS VCFs
# --------------------------------------------------------------------------- #

_PASS1_VCF_TEXT = """\
##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tclone4\tclone7
1\t100\t.\tC\tT\t.\tPASS\t.\tGT:AD:AF:DP\t0/0:20,0:0.0:20\t0/1:8,12:0.600:20
1\t150\t.\tC\tA\t.\tPASS\t.\tGT:AD:AF:DP\t0/0:20,0:0.0:20\t0/1:8,12:0.600:20
"""


def test_discover_pass1_cluster_vcfs_excludes_forced_vcfs(tmp_path):
    (tmp_path / "clone7_1.vcf").write_text(_PASS1_VCF_TEXT)
    (tmp_path / "clone7_2.vcf").write_text(_PASS1_VCF_TEXT)
    (tmp_path / "clone7_1.forced.vcf").write_text(_FORCED_VCF_TEXT)
    (tmp_path / "clone9_1.vcf").write_text(_PASS1_VCF_TEXT)

    found = bt.discover_pass1_cluster_vcfs(tmp_path)
    assert set(found) == {"7", "9"}
    assert sorted(f.name for f in found["7"]) == ["clone7_1.vcf", "clone7_2.vcf"]


def test_parse_vcf_site_keys_ignores_genotypes_and_indels(tmp_path):
    text = _PASS1_VCF_TEXT.replace("1\t150\t.\tC\tA", "1\t150\t.\tCA\tA")
    vcf_path = tmp_path / "clone7_1.vcf"
    vcf_path.write_text(text)
    assert bt.parse_vcf_site_keys(vcf_path) == {("1", 100, "C", "T")}  # indel excluded


def test_parse_vcf_site_keys(tmp_path):
    vcf_path = tmp_path / "clone7_1.vcf"
    vcf_path.write_text(_PASS1_VCF_TEXT)
    assert bt.parse_vcf_site_keys(vcf_path) == {
        ("1", 100, "C", "T"),
        ("1", 150, "C", "A"),
    }


def test_build_union_sites_unions_across_clusters_and_files(tmp_path):
    (tmp_path / "clone7_1.vcf").write_text(_PASS1_VCF_TEXT)
    other = _PASS1_VCF_TEXT.replace("1\t150", "1\t250")
    (tmp_path / "clone9_1.vcf").write_text(other)
    pass1_vcfs = bt.discover_pass1_cluster_vcfs(tmp_path)

    union = bt.build_union_sites(pass1_vcfs)
    assert union == {
        ("1", 100, "C", "T"),
        ("1", 150, "C", "A"),
        ("1", 250, "C", "A"),
    }


def test_build_union_sites_raises_when_empty():
    with pytest.raises(ValueError, match="no pass-1"):
        bt.build_union_sites({})


def test_restrict_calls_to_union_drops_records_outside_it():
    calls = {
        ("1", 100, "C", "T"): (0.6, 12),  # in the union
        ("1", 999, "C", "A"): (0.9, 20),  # Mutect2's own discovery call, not in it
    }
    union_sites = {("1", 100, "C", "T")}
    kept, n_dropped = bt.restrict_calls_to_union(calls, union_sites)
    assert kept == {("1", 100, "C", "T"): (0.6, 12)}
    assert n_dropped == 1


def test_assert_presence_within_union_passes_when_within_bounds():
    bt.assert_presence_within_union(
        {"7": {("1", 100, "C", "T")}}, {("1", 100, "C", "T"), ("1", 200, "C", "A")}
    )  # no raise


def test_assert_presence_within_union_raises_when_exceeded():
    with pytest.raises(ValueError, match="cluster 7 has 2 present SNVs"):
        bt.assert_presence_within_union(
            {"7": {("1", 100, "C", "T"), ("1", 200, "C", "A")}},
            {("1", 100, "C", "T")},
        )


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
# Dollo parsimony over trees with hidden internal nodes
# --------------------------------------------------------------------------- #


def test_dollo_pattern_cost_hand_checked():
    # ((a,c),b): S={a,b} spans both of the root's children, so the root is the
    # LCA; the (a,c) side needs one loss (c is absent under a present-mixed
    # subtree), the b side needs none (b itself is present) -- 1 gain + 1 loss.
    topology = frozenset([frozenset(["a", "c"]), "b"])
    assert bt.dollo_pattern_cost(topology, frozenset(["a", "b"])) == (2, 1)


def test_dollo_pattern_cost_singleton_and_full_set_are_topology_invariant():
    topologies = bt.enumerate_dollo_topologies(["a", "b", "c"])
    for t in topologies:
        assert bt.dollo_pattern_cost(t, frozenset(["a"])) == (1, 0)
        assert bt.dollo_pattern_cost(t, frozenset(["a", "b", "c"])) == (1, 0)


def test_enumerate_dollo_topologies_count():
    # (2n-3)!! rooted binary topologies: 105 for n=5.
    assert len(bt.enumerate_dollo_topologies(["1", "2", "3", "4", "5"])) == 105
    assert len(bt.enumerate_dollo_topologies(["1", "2"])) == 1
    assert len(bt.enumerate_dollo_topologies(["1"])) == 1


def test_enumerate_dollo_topologies_raises_past_the_cap():
    with pytest.raises(ValueError, match="exceeds the exhaustive search cap"):
        bt.enumerate_dollo_topologies(["1", "2", "3"], max_clusters=2)


def test_enumerate_dollo_topologies_raises_on_no_clusters():
    with pytest.raises(ValueError, match="no clusters"):
        bt.enumerate_dollo_topologies([])


# The real slice D pattern counts over columns [3, 7, 8, 9, 10] (see the
# module docstring's real-run derivation): confirmed by hand and by
# independent script, the unique optimum is (3,(10,(9,(7,8)))), cost 1996
# above the topology-invariant baseline (10609 total), runner-up 2026 above
# (10639 total) -- swapping whether 9 or 10 joins the {7,8} clade first.
REAL_RUN_PATTERNS = {
    frozenset(["10"]): 3931,
    frozenset(["8"]): 1087,
    frozenset(["7"]): 917,
    frozenset(["9"]): 909,
    frozenset(["3"]): 900,
    frozenset(["3", "7", "8", "9", "10"]): 869,
    frozenset(["7", "8", "9", "10"]): 390,
    frozenset(["3", "7", "8", "9"]): 201,
    frozenset(["3", "7", "8", "10"]): 135,
    frozenset(["7", "8", "9"]): 130,
    frozenset(["7", "8"]): 110,
    frozenset(["7", "8", "10"]): 100,
    frozenset(["3", "7", "9", "10"]): 90,
    frozenset(["3", "8", "9", "10"]): 88,
    frozenset(["7", "9", "10"]): 69,
}
REAL_RUN_LEAVES = ["3", "7", "8", "9", "10"]


def test_dollo_best_topologies_matches_the_real_run_derivation():
    topologies = bt.enumerate_dollo_topologies(REAL_RUN_LEAVES)
    table = bt.dollo_cost_table(topologies, REAL_RUN_PATTERNS.keys())
    best_cost, best = bt.dollo_best_topologies(topologies, REAL_RUN_PATTERNS, table)
    assert best_cost == 10609
    assert len(best) == 1
    assert bt.render_topology(best[0]) == "(3,(10,((7,8),9)))"


def test_dollo_tree_real_run_fixture_emits_high_support_784_clade():
    result = bt.dollo_tree(REAL_RUN_PATTERNS, REAL_RUN_LEAVES, n_bootstrap=200, seed=0)
    assert result.best_cost == 10609
    assert result.runner_up_cost == 10639
    assert result.n_ties == 1

    clades_by_leaves = {frozenset(k): v for k, v in result.clade_support.items()}
    support_7_8_9_10 = clades_by_leaves.get(frozenset(["7", "8", "9", "10"]))
    assert support_7_8_9_10 is not None and support_7_8_9_10 >= 0.7
    newick = bt.digraph_to_newick(result.tree, bt.GERMLINE_ROOT_ID)
    bt.verify_newick(
        newick, set(REAL_RUN_LEAVES), bt.GERMLINE_ROOT_ID, result.hidden_ids
    )
    # {7,8} and {9,10} are not asserted -- report only, per the real-run
    # instructions: whether the {7,8,9} split survives depends on its own
    # reported bootstrap support, not on an assumption made here.
    print("clade support:", clades_by_leaves)
    print("top2_win_rate:", result.top2_win_rate)


def test_dollo_tree_ties_emit_the_strict_consensus_as_a_polytomy():
    # Two patterns of equal weight favour opposite pairings among 3 leaves,
    # so no clade beyond the full set (the MRCA) is common to every optimal
    # topology: the consensus is a star under the MRCA, not one pairing
    # picked arbitrarily.
    patterns = {frozenset(["a", "b"]): 10, frozenset(["b", "c"]): 10}
    result = bt.dollo_tree(patterns, ["a", "b", "c"], n_bootstrap=0)
    assert result.n_ties > 1
    assert len(result.hidden_ids) == 1  # just the MRCA, no {a,b}/{b,c} split kept
    mrca = next(iter(result.hidden_ids))
    assert set(result.tree.successors(bt.GERMLINE_ROOT_ID)) == {mrca}
    assert set(result.tree.successors(mrca)) == {"a", "b", "c"}
    newick = bt.digraph_to_newick(result.tree, bt.GERMLINE_ROOT_ID)
    bt.verify_newick(newick, {"a", "b", "c"}, bt.GERMLINE_ROOT_ID, result.hidden_ids)


def test_dollo_tree_all_private_data_gives_a_star():
    patterns = {
        frozenset(["a"]): 5,
        frozenset(["b"]): 5,
        frozenset(["c"]): 5,
        frozenset(["d"]): 5,
        frozenset(["a", "b", "c", "d"]): 5,
    }
    result = bt.dollo_tree(patterns, ["a", "b", "c", "d"], n_bootstrap=0)
    # exactly one hidden node (the MRCA), every leaf a direct child of it.
    assert len(result.hidden_ids) == 1
    mrca = next(iter(result.hidden_ids))
    assert set(result.tree.successors(bt.GERMLINE_ROOT_ID)) == {mrca}
    assert set(result.tree.successors(mrca)) == {"a", "b", "c", "d"}


def test_dollo_tree_single_cluster_is_trivial():
    result = bt.dollo_tree({frozenset(["a"]): 3}, ["a"], n_bootstrap=0)
    assert set(result.tree.edges()) == {(bt.GERMLINE_ROOT_ID, "a")}
    assert result.hidden_ids == set()


def test_dollo_bootstrap_is_deterministic_given_a_seed():
    r1 = bt.dollo_tree(REAL_RUN_PATTERNS, REAL_RUN_LEAVES, n_bootstrap=50, seed=7)
    r2 = bt.dollo_tree(REAL_RUN_PATTERNS, REAL_RUN_LEAVES, n_bootstrap=50, seed=7)
    assert r1.clade_support == r2.clade_support
    assert r1.top2_win_rate == r2.top2_win_rate


def test_dollo_low_support_clade_is_collapsed():
    # {a,b}'s only support is one weak pattern (count 1), heavily outweighed
    # by private mutations on a and b individually: the point estimate still
    # picks {a,b} as the optimal clade (cost 406, unique), but its bootstrap
    # support (confirmed deterministic at seed=0: 0.68) falls just under the
    # default 0.7 threshold, so it must not survive into the emitted tree.
    patterns = {
        frozenset(["a", "b"]): 1,
        frozenset(["a"]): 200,
        frozenset(["b"]): 200,
        frozenset(["a", "b", "c"]): 5,
    }
    result = bt.dollo_tree(
        patterns, ["a", "b", "c"], n_bootstrap=200, min_clade_support=0.7, seed=0
    )
    assert result.n_ties == 1
    support = {frozenset(k): v for k, v in result.clade_support.items()}
    assert support[frozenset(["a", "b"])] < 0.7
    assert frozenset(["a", "b"]) in {frozenset(c) for c in result.collapsed_clades}
    # collapsed: a, b and c all sit directly under the one hidden MRCA node.
    assert len(result.hidden_ids) == 1
    mrca = next(iter(result.hidden_ids))
    assert set(result.tree.successors(mrca)) == {"a", "b", "c"}


def test_dollo_recovers_a_known_tree_under_dropout():
    # Known tree: germline -> mrca -> {e, X}; X -> {d, Y}; Y -> {c, {a,b}}.
    # Each clade gets its own defining mutations; a dropout rate removes some
    # 1s (a present cluster read as absent), the realistic failure mode.
    rng = np.random.default_rng(0)
    leaves = ["a", "b", "c", "d", "e"]
    clade_mutation_counts = {
        frozenset(["a", "b"]): 40,
        frozenset(["a", "b", "c"]): 40,
        frozenset(["a", "b", "c", "d"]): 40,
        frozenset(leaves): 40,
    }
    private_counts = {frozenset([leaf]): 40 for leaf in leaves}

    dropout_rate = 0.07
    resampled_patterns = {}
    for pattern, n in {**clade_mutation_counts, **private_counts}.items():
        for _ in range(n):
            observed = frozenset(
                leaf for leaf in pattern if rng.random() >= dropout_rate
            )
            if observed:
                resampled_patterns[observed] = resampled_patterns.get(observed, 0) + 1

    result = bt.dollo_tree(resampled_patterns, leaves, n_bootstrap=200, seed=1)
    support = {frozenset(k): v for k, v in result.clade_support.items()}
    for true_clade in [
        frozenset(["a", "b"]),
        frozenset(["a", "b", "c"]),
        frozenset(["a", "b", "c", "d"]),
    ]:
        assert support.get(true_clade, 0.0) >= 0.7, (true_clade, support)


# --------------------------------------------------------------------------- #
# --compare-tree: clade-level agreement, ignoring hidden-node names
# --------------------------------------------------------------------------- #


def test_compare_tree_clades_agreement(tmp_path):
    # SNV tree: ((7,8)g1,9,10)germline -- clade {7,8}.
    snv_tree = nx.DiGraph(
        [
            ("germline", "g1"),
            ("g1", "7"),
            ("g1", "8"),
            ("germline", "9"),
            ("germline", "10"),
        ]
    )
    other_path = tmp_path / "other.nwk"
    # CNA tree: ((7,8)h1,(9,10)h2)germline -- clades {7,8} and {9,10}.
    other_path.write_text("((7,8)h1,(9,10)h2)germline;\n")

    result = bt.compare_tree_clades(snv_tree, "germline", other_path)
    assert result["both"] == [["7", "8"]]
    assert result["only_second"] == [["9", "10"]]
    assert result["only_first"] == []


def test_compare_tree_clades_raises_on_multiple_roots(tmp_path):
    other_path = tmp_path / "other.nwk"
    other_path.write_text("(a,b)r1;(c,d)r2;\n")  # two distinctly-labelled roots
    snv_tree = nx.DiGraph([("germline", "a")])
    with pytest.raises(ValueError, match="exactly one root"):
        bt.compare_tree_clades(snv_tree, "germline", other_path)


# --------------------------------------------------------------------------- #
# LICHeE: verify_newick with hidden group nodes
# --------------------------------------------------------------------------- #


def test_verify_newick_accepts_hidden_group_nodes():
    newick = "((7,8)g1)germline;"
    bt.verify_newick(newick, {"7", "8"}, "germline", hidden_ids={"g1"})
    with pytest.raises(AssertionError):  # without hidden_ids the extra node is a defect
        bt.verify_newick(newick, {"7", "8"}, "germline")


def test_verify_newick_rejects_hidden_id_colliding_with_a_cluster():
    with pytest.raises(AssertionError, match="collide"):
        bt.verify_newick("((7)g1)germline;", {"7", "g1"}, "germline", hidden_ids={"g1"})


# --------------------------------------------------------------------------- #
# LICHeE: input, home resolution, invocation
# --------------------------------------------------------------------------- #

COLUMNS = ["germline", "c3", "c7", "c8", "c9", "c10"]  # the fixtures' input header


def test_write_lichee_input_format_and_germline_column(tmp_path):
    cluster_to_calls = {
        "7": {("1", 100, "C", "T"): (0.6, 12)},
        "8": {("1", 100, "C", "T"): (0.3, 5), ("1", 200, "C", "A"): (0.4, 6)},
    }
    out = tmp_path / "lichee_in.txt"
    columns = bt.write_lichee_input(cluster_to_calls, out)

    assert columns == [bt.GERMLINE_ROOT_ID, "c7", "c8"]
    lines = out.read_text().splitlines()
    assert lines[0] == "#chr\tposition\tdescription\tgermline\tc7\tc8"
    # 2 SNV rows, sorted by (chrom, pos, ref, alt)
    assert len(lines) == 3
    row_100 = lines[1].split("\t")
    assert row_100[:4] == ["1", "100", "C>T", "0.0"]  # germline always 0.0
    assert row_100[4] == "0.6000"  # cluster 7's VAF at this site
    assert row_100[5] == "0.3000"  # cluster 8's VAF at this site too
    row_200 = lines[2].split("\t")
    assert row_200[4] == "0.0000"  # cluster 7 has no call here
    assert row_200[5] == "0.4000"


def _fake_home(tmp_path):
    home = tmp_path / "LICHeE"
    (home / "release").mkdir(parents=True)
    (home / "release" / "lichee.jar").write_text("")
    (home / "lib").mkdir()
    return home


def test_resolve_lichee_home_precedence_and_result(tmp_path, monkeypatch):
    monkeypatch.setattr(bt.shutil, "which", lambda name: "/usr/bin/java")
    home = _fake_home(tmp_path)
    jar, lib = bt.resolve_lichee_home(str(home))
    assert jar == home / "release" / "lichee.jar" and lib == home / "lib"

    monkeypatch.setenv("LICHEE_HOME", str(home))
    assert bt.resolve_lichee_home(None) == (jar, lib)


def test_resolve_lichee_home_default_is_the_repo_checkout(monkeypatch):
    monkeypatch.delenv("LICHEE_HOME", raising=False)
    with pytest.raises(FileNotFoundError) as err:
        bt.resolve_lichee_home(None)
    expected = REPO_ROOT / "realdata" / "external" / "lichee" / "LICHeE"
    assert str(expected) in str(err.value)


def test_resolve_lichee_home_names_everything_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(bt.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError) as err:
        bt.resolve_lichee_home(str(tmp_path / "nowhere"))
    message = str(err.value)
    assert "lichee.jar" in message and "lib directory" in message
    assert "java on PATH" in message


def _run_lichee(tmp_path, monkeypatch, write=True, returncode=0, **extra):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if write:
            Path(cmd[cmd.index("-o") + 1]).write_text("Nodes:\n")
        return subprocess.CompletedProcess(cmd, returncode)

    monkeypatch.setattr(bt.subprocess, "run", fake_run)
    out = tmp_path / "lichee_out.trees.txt"
    result = bt.run_lichee(
        tmp_path / "in.txt",
        out_path=out,
        log_path=tmp_path / "lichee.log",
        jar=tmp_path / "release" / "lichee.jar",
        lib=tmp_path / "lib",
        min_vaf_present=0.05,
        max_vaf_absent=0.05,
        **extra,
    )
    return result, calls


def test_run_lichee_command_is_java_cp_with_an_unexpanded_lib_glob(
    tmp_path, monkeypatch
):
    result, calls = _run_lichee(tmp_path, monkeypatch)
    cmd = calls[0][0]
    assert result == tmp_path / "lichee_out.trees.txt"
    assert cmd[0] == "java"
    assert cmd[1] == "-cp"
    # One classpath argument, jar then lib/*, the glob left for Java to expand.
    assert cmd[2] == f"{tmp_path / 'release' / 'lichee.jar'}:{tmp_path / 'lib'}/*"
    assert cmd[3] == "lineage.LineageEngine"
    assert cmd[cmd.index("-s") + 1] == "1"
    assert cmd[cmd.index("-n") + 1] == "0"
    assert cmd[cmd.index("-o") + 1] == str(tmp_path / "lichee_out.trees.txt")
    assert "-dot" not in cmd
    assert "-minClusterSize" not in cmd
    assert (tmp_path / "lichee.log").exists()


def test_run_lichee_uses_the_given_vaf_cutoffs(tmp_path, monkeypatch):
    _, calls = _run_lichee(tmp_path, monkeypatch)
    cmd = calls[0][0]
    assert cmd[cmd.index("-minVAFPresent") + 1] == "0.05"
    assert cmd[cmd.index("-maxVAFAbsent") + 1] == "0.05"


def test_run_lichee_passes_optional_flags_only_when_given(tmp_path, monkeypatch):
    _, calls = _run_lichee(tmp_path, monkeypatch)
    assert "-minClusterSize" not in calls[0][0]
    assert "-e" not in calls[0][0]

    _, calls = _run_lichee(tmp_path, monkeypatch, min_cluster_size=50, error_margin=0.2)
    cmd = calls[0][0]
    assert cmd[cmd.index("-minClusterSize") + 1] == "50"
    assert cmd[cmd.index("-e") + 1] == "0.2"


def test_extract_lichee_verdict_quotes_the_line_verbatim(tmp_path):
    log = tmp_path / "lichee.log"
    log.write_text("some chatter\nFound 0 valid trees\nmore chatter\n")
    assert bt.extract_lichee_verdict(log) == "Found 0 valid trees"


def test_extract_lichee_verdict_none_when_absent(tmp_path):
    log = tmp_path / "lichee.log"
    log.write_text("nothing relevant here\n")
    assert bt.extract_lichee_verdict(log) is None


def test_run_lichee_ignores_exit_status_when_the_file_appears(tmp_path, monkeypatch):
    result, _ = _run_lichee(tmp_path, monkeypatch, returncode=1)
    assert result.exists()


def test_run_lichee_raises_naming_file_and_log_when_no_output(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError) as err:
        _run_lichee(tmp_path, monkeypatch, write=False)
    assert "lichee_out.trees.txt" in str(err.value)
    assert "lichee.log" in str(err.value)


def test_run_lichee_removes_a_stale_output_first(tmp_path, monkeypatch):
    (tmp_path / "lichee_out.trees.txt").write_text("stale")
    with pytest.raises(FileNotFoundError):
        _run_lichee(tmp_path, monkeypatch, write=False)
    assert not (tmp_path / "lichee_out.trees.txt").exists()


# --------------------------------------------------------------------------- #
# LICHeE: parsing .trees.txt
# --------------------------------------------------------------------------- #


def _parse_fixture(name):
    return bt.parse_lichee_trees(LICHEE_FIXTURES / name, COLUMNS)


def test_parse_nested_chain_fixture():
    t = _parse_fixture("nested_chain.trees.txt")
    assert t.profiles == {
        "3": "011111",
        "4": "001111",
        "1": "000111",
        "2": "000001",
        "5": "000010",
    }
    # VAFs cover the present columns only: node 1 (000111) is present in 3.
    assert t.vafs["1"] == [0.4, 0.41, 0.41]
    assert t.vafs["2"] == [0.44]
    # Tree 0's edges are listed out of order in the file.
    assert t.parent_of == {"3": "0", "4": "3", "1": "4", "5": "1", "2": "1"}
    assert t.root == "0" and t.n_trees == 1
    assert t.decomposition["germline"] == []
    assert t.decomposition["c9"] == [
        (1, "011111", 0.415),
        (2, "001111", 0.4),
        (3, "000111", 0.405),
        (4, "000010", 0.43),
    ]


def test_parse_shared_nodes_fixture():
    t = _parse_fixture("shared_nodes.trees.txt")
    assert t.parent_of == {"1": "0", "3": "1", "2": "3"}
    assert t.decomposition["c7"] == [(1, "011111", 0.425), (2, "001111", 0.39)]


def _mutate(name, old, new):
    text = (LICHEE_FIXTURES / name).read_text()
    assert old in text
    return text.replace(old, new)


def test_parse_counts_extra_tree_blocks(tmp_path):
    text = _mutate(
        "shared_nodes.trees.txt",
        "Error score:",
        "Error score:",
    ).replace("Sample decomposition", "****Tree 1****\n0 -> 2\n\nSample decomposition")
    path = tmp_path / "t.txt"
    path.write_text(text)
    t = bt.parse_lichee_trees(path, COLUMNS)
    assert t.n_trees == 2
    assert t.parent_of == {"1": "0", "3": "1", "2": "3"}  # Tree 0 only


def _write(tmp_path, text):
    path = tmp_path / "t.txt"
    path.write_text(text)
    return path


@pytest.mark.parametrize(
    "old,new,match",
    [
        ("Nodes:\n", "", "Nodes"),
        ("****Tree 0****", "****Tree 5****", "Tree 0"),
        ("Sample decomposition: ", "SNV info:", "Sample decomposition"),
        ("1\t011111\t", "1\t01111\t", "bits"),
        ("1\t011111\t", "1\t111111\t", "germline bit"),
        ("[ 0.43 0.42 0.45 0.42 0.45]", "[ 0.43 0.42]", "VAFs for"),
        ("1 -> 3", "1 -> 3\n7 -> 8", "root"),
        ("3 -> 2", "3 -> 2\n0 -> 2", "two parents"),
        ("0 -> 1\n", "9 -> 1\n", "root"),
        (".....011111: 0.435", "...011111: 0.435", "indented"),
    ],
)
def test_parse_failure_paths(tmp_path, old, new, match):
    path = _write(tmp_path, _mutate("shared_nodes.trees.txt", old, new))
    with pytest.raises(ValueError, match=match):
        bt.parse_lichee_trees(path, COLUMNS)


# --------------------------------------------------------------------------- #
# attach_option_a: the tool-agnostic attachment shared with build_cna_tree.py
# --------------------------------------------------------------------------- #

_CHAIN_PARENT_OF = {"1": "0", "2": "1", "3": "0"}  # 0 -> {1, 3}, 1 -> 2


def test_attach_option_a_single_item_per_node_labels_the_node():
    a = bt.attach_option_a(_CHAIN_PARENT_OF, {"a": "1", "b": "2", "c": "3"}, "0")
    assert a.hidden_ids == set() and a.shared == [] and a.collapsed_nodes == []
    assert bt.digraph_to_newick(a.tree, bt.GERMLINE_ROOT_ID) == "((b)a,c)germline;"


def test_attach_option_a_shared_node_makes_a_hidden_group():
    # node 1 carries no children of its own beyond {2, 3}, neither of which
    # gets an item, so both collapse and the group has nothing beneath it.
    a = bt.attach_option_a(_CHAIN_PARENT_OF, {"a": "1", "b": "1"}, "0")
    assert a.hidden_ids == {"g1"}
    assert a.shared == [["a", "b"]]
    assert a.collapsed_nodes == ["2", "3"]
    assert a.group_subtends == {"g1": (2, 2)}  # nothing beyond the 2 direct items
    assert bt.digraph_to_newick(a.tree, bt.GERMLINE_ROOT_ID) == "((a,b)g1)germline;"


def test_attach_option_a_shared_node_carries_its_descendants():
    # node 1 (shared by a, b) has child node 2 (item c): c hangs off the
    # group node too, so the group subtends 2 items directly, 3 in total.
    a = bt.attach_option_a(_CHAIN_PARENT_OF, {"a": "1", "b": "1", "c": "2"}, "0")
    assert a.group_subtends == {"g1": (2, 3)}
    assert set(a.tree.edges()) == {
        (bt.GERMLINE_ROOT_ID, "g1"),
        ("g1", "a"),
        ("g1", "b"),
        ("g1", "c"),
    }


def test_attach_option_a_collapses_nodes_with_no_item():
    # node 1 carries no item; node 2 (its child) does, so 2's item lifts to root.
    a = bt.attach_option_a(_CHAIN_PARENT_OF, {"x": "2"}, "0")
    assert a.collapsed_nodes == ["1", "3"]
    assert set(a.tree.edges()) == {(bt.GERMLINE_ROOT_ID, "x")}


def test_attach_option_a_root_items_attach_directly_under_germline():
    a = bt.attach_option_a(_CHAIN_PARENT_OF, {}, "0", root_items=["z"])
    assert set(a.tree.edges()) == {(bt.GERMLINE_ROOT_ID, "z")}


def test_attach_option_a_raises_on_a_group_label_collision():
    with pytest.raises(ValueError, match="collides"):
        bt.attach_option_a(_CHAIN_PARENT_OF, {"g1": "1", "b": "1"}, "0")


# --------------------------------------------------------------------------- #
# LICHeE: the clone tree
# --------------------------------------------------------------------------- #


def _build(name, tau=0.05):
    trees = _parse_fixture(name)
    return bt.build_lichee_clone_tree(trees, COLUMNS, tau)


def _edges(tree):
    return set(tree.edges())


def test_nested_chain_gives_five_labelled_nodes_and_preserves_ancestry():
    r = _build("nested_chain.trees.txt")
    newick = bt.digraph_to_newick(r.tree, bt.GERMLINE_ROOT_ID)
    assert newick == "((((9,10)8)7)3)germline;"
    assert r.hidden_ids == set()
    assert r.shared == [] and r.absent == [] and r.collapsed_nodes == []
    assert r.node_of_cluster == {"3": "3", "7": "4", "8": "1", "9": "5", "10": "2"}
    # ancestry: 3 above 7 above 8 above 9 and 10
    assert nx.has_path(r.tree, "3", "7") and nx.has_path(r.tree, "7", "8")
    assert set(r.tree.successors("8")) == {"9", "10"}
    bt.verify_newick(newick, {"3", "7", "8", "9", "10"}, "germline")


def test_shared_nodes_fixture_makes_hidden_group_nodes():
    r = _build("shared_nodes.trees.txt")
    assert _edges(r.tree) == {
        ("germline", "3"),
        ("3", "g3"),
        ("g3", "7"),
        ("g3", "8"),
        ("g3", "g2"),
        ("g2", "9"),
        ("g2", "10"),
    }
    newick = bt.digraph_to_newick(r.tree, bt.GERMLINE_ROOT_ID)
    assert newick == "(((7,8,(9,10)g2)g3)3)germline;"
    assert r.hidden_ids == {"g3", "g2"}
    # {c7,c8} share node 3, {c9,c10} share node 2; c3 alone holds node 1.
    assert {tuple(g) for g in r.shared} == {("7", "8"), ("9", "10")}
    assert r.group_subtends == {"g3": (2, 4), "g2": (2, 2)}
    # the labelled cluster set is exactly the five clusters plus the root
    assert set(r.tree.nodes()) - r.hidden_ids == {
        "germline", "3", "7", "8", "9", "10"
    }  # fmt: skip
    bt.verify_newick(newick, {"3", "7", "8", "9", "10"}, "germline", r.hidden_ids)
    # c3 stays above the shared nodes, and g3 above g2
    assert nx.has_path(r.tree, "3", "g3") and nx.has_path(r.tree, "g3", "g2")


def test_shared_nodes_diagnostics_name_the_findings():
    r = _build("shared_nodes.trees.txt")
    text = "".join(bt._lichee_lines(r))
    assert "clusters 7,8 indistinguishable by SNV profile" in text
    assert "clusters 9,10 indistinguishable by SNV profile" in text
    assert "Hidden group node g3: 2 clusters directly, 4 in its subtree" in text
    assert "Hidden group node g2: 2 clusters directly, 2 in its subtree" in text
    assert "Fewer nodes than clusters is expected" in text
    assert "emitted 2 trees" not in text


def test_diagnostics_note_more_than_one_tree():
    r = _build("shared_nodes.trees.txt")
    r.n_trees = 3
    assert "emitted 3 trees; Tree 0" in "".join(bt._lichee_lines(r))


# Small synthetic outputs in the same layout, for the paths the fixtures do not hit.


def _synthetic(columns, nodes, edges, decomposition):
    lines = ["Nodes:"]
    for node, profile, vafs in nodes:
        lines.append(f"{node}\t{profile}\t[ {' '.join(map(str, vafs))}]\tsnv{node}")
    lines += ["", "****Tree 0****"] + [f"{a} -> {b}" for a, b in edges]
    lines += ["Error score: 0.1", "", "Sample decomposition: "]
    for sample, entries in decomposition.items():
        lines += [f"\tSample lineage decomposition: {sample}", "GL"]
        for depth, profile, vaf in entries:
            lines.append("." * (5 * depth) + f"{profile}: {vaf} [0.0]")
        lines.append("")
    return "\n".join(lines) + "\n"


def _build_text(tmp_path, columns, nodes, edges, decomposition, tau=0.05):
    path = _write(tmp_path, _synthetic(columns, nodes, edges, decomposition))
    trees = bt.parse_lichee_trees(path, columns)
    return bt.build_lichee_clone_tree(trees, columns, tau)


def test_absent_cluster_hangs_off_the_germline_root_with_a_finding(tmp_path):
    columns = ["germline", "c3", "c7"]
    r = _build_text(
        tmp_path,
        columns,
        [("1", "010", [0.4])],
        [("0", "1")],
        {"germline": [], "c3": [(1, "010", 0.4)], "c7": []},
    )
    assert _edges(r.tree) == {("germline", "3"), ("germline", "7")}
    assert r.absent == ["7"]
    assert "cluster(s) 7 are in no LICHeE node" in "".join(bt._lichee_lines(r))


def test_node_with_no_cluster_is_collapsed_and_children_lift(tmp_path):
    columns = ["germline", "c3", "c7"]
    r = _build_text(
        tmp_path,
        columns,
        [("1", "011", [0.4, 0.4]), ("2", "010", [0.4]), ("3", "001", [0.4])],
        [("0", "1"), ("1", "2"), ("1", "3")],
        {
            "germline": [],
            "c3": [(1, "011", 0.4), (2, "010", 0.4)],
            "c7": [(1, "011", 0.4), (2, "001", 0.4)],
        },
    )
    assert r.collapsed_nodes == ["1"]
    assert _edges(r.tree) == {("germline", "3"), ("germline", "7")}
    assert "carry no cluster and were collapsed" in "".join(bt._lichee_lines(r))


def test_genuine_tie_attaches_at_the_lowest_common_ancestor(tmp_path):
    columns = ["germline", "c3", "c7", "c8"]
    r = _build_text(
        tmp_path,
        columns,
        [
            ("1", "0111", [0.4, 0.4, 0.4]),
            ("2", "0101", [0.4, 0.4]),
            ("3", "0011", [0.4, 0.4]),
        ],
        [("0", "1"), ("1", "2"), ("1", "3")],
        {
            "germline": [],
            "c3": [(1, "0111", 0.4), (2, "0101", 0.4)],
            "c7": [(1, "0111", 0.4), (2, "0011", 0.4)],
            "c8": [(1, "0111", 0.4), (2, "0101", 0.4), (2, "0011", 0.4)],
        },
    )
    # c8 is in both leaves, so it resolves to their parent, node 1.
    assert r.node_of_cluster["8"] == "1"
    assert _edges(r.tree) == {
        ("germline", "8"),
        ("8", "3"),
        ("8", "7"),
    }


def test_a_tie_only_at_the_root_attaches_to_the_germline_root(tmp_path):
    # c3 is in both root-level branches, whose only common ancestor is the root.
    r = _build_text(
        tmp_path,
        ["germline", "c3", "c7", "c8"],
        [("1", "0110", [0.4, 0.4]), ("2", "0101", [0.4, 0.4])],
        [("0", "1"), ("0", "2")],
        {
            "germline": [],
            "c3": [(1, "0110", 0.4), (1, "0101", 0.4)],
            "c7": [(1, "0110", 0.4)],
            "c8": [(1, "0101", 0.4)],
        },
    )
    assert r.at_root == ["3"]
    assert ("germline", "3") in _edges(r.tree)
    assert ("germline", "7") in _edges(r.tree) and ("germline", "8") in _edges(r.tree)
    assert "cluster(s) 3 tie only at the germline root" in "".join(bt._lichee_lines(r))


LEAK_COLUMNS = ["germline", "c3", "c7"]
LEAK_NODES = [("1", "011", [0.4, 0.4]), ("2", "001", [0.02])]
LEAK_DECOMPOSITION = {
    "germline": [],
    "c3": [(1, "011", 0.4)],
    "c7": [(1, "011", 0.4), (2, "001", 0.02)],
}


def test_tau_gates_out_sub_threshold_leakage(tmp_path):
    # c7's 0.02 in node 2 is below tau=0.05, so c7 stays at node 1 with c3.
    r = _build_text(
        tmp_path, LEAK_COLUMNS, LEAK_NODES, [("0", "1"), ("1", "2")], LEAK_DECOMPOSITION
    )
    assert r.hidden_ids == {"g1"}
    assert _edges(r.tree) == {("germline", "g1"), ("g1", "3"), ("g1", "7")}
    assert r.collapsed_nodes == ["2"]


def test_a_lower_tau_lets_the_same_leakage_count(tmp_path):
    r = _build_text(
        tmp_path,
        LEAK_COLUMNS,
        LEAK_NODES,
        [("0", "1"), ("1", "2")],
        LEAK_DECOMPOSITION,
        tau=0.01,
    )
    assert _edges(r.tree) == {("germline", "3"), ("3", "7")}


# Failure paths of the join.


def test_presence_and_decomposition_disagreement_raises(tmp_path):
    with pytest.raises(ValueError, match="presence puts it at node"):
        _build_text(
            tmp_path,
            ["germline", "c3", "c7"],
            [("1", "011", [0.4, 0.4]), ("2", "001", [0.4])],
            [("0", "1"), ("1", "2")],
            {
                "germline": [],
                "c3": [(1, "011", 0.4)],
                "c7": [(1, "011", 0.4)],  # presence says node 2, this stops at node 1
            },
        )


def test_deepest_decomposition_profile_matching_no_node_raises(tmp_path):
    with pytest.raises(ValueError, match="matches no Nodes: entry"):
        _build_text(
            tmp_path,
            ["germline", "c3"],
            [("1", "01", [0.4])],
            [("0", "1")],
            {"germline": [], "c3": [(1, "01", 0.4), (2, "11", 0.4)]},
        )


def test_in_no_node_by_presence_but_placed_by_decomposition_raises(tmp_path):
    with pytest.raises(ValueError, match="in no node by presence"):
        _build_text(
            tmp_path,
            ["germline", "c3", "c7"],
            [("1", "010", [0.4]), ("2", "001", [0.01])],
            [("0", "1"), ("0", "2")],
            {
                "germline": [],
                "c3": [(1, "010", 0.4)],
                "c7": [(1, "001", 0.4)],  # decomposition contradicts the node VAF
            },
        )


def test_unknown_decomposition_sample_raises(tmp_path):
    with pytest.raises(ValueError, match="neither 'germline' nor a c<id> column"):
        _build_text(
            tmp_path,
            ["germline", "c3"],
            [("1", "01", [0.4])],
            [("0", "1")],
            {"germline": [], "c3": [(1, "01", 0.4)], "x9": []},
        )


def test_cluster_without_a_decomposition_block_raises(tmp_path):
    with pytest.raises(ValueError, match="no 'Sample lineage decomposition: c7'"):
        _build_text(
            tmp_path,
            ["germline", "c3", "c7"],
            [("1", "011", [0.4, 0.4])],
            [("0", "1")],
            {"germline": [], "c3": [(1, "011", 0.4)]},
        )


def test_cluster_column_must_carry_the_c_prefix():
    trees = _parse_fixture("shared_nodes.trees.txt")
    with pytest.raises(ValueError, match="not c<id>"):
        bt.build_lichee_clone_tree(
            trees, ["germline", "3", "c7", "c8", "c9", "c10"], 0.05
        )


def test_group_node_label_colliding_with_a_cluster_raises(tmp_path):
    with pytest.raises(ValueError, match="collides"):
        _build_text(
            tmp_path,
            ["germline", "cg1", "c7"],
            [("1", "011", [0.4, 0.4])],
            [("0", "1")],
            {
                "germline": [],
                "cg1": [(1, "011", 0.4)],
                "c7": [(1, "011", 0.4)],
            },
        )


def test_tree_with_a_cycle_raises(tmp_path):
    path = _write(
        tmp_path,
        _synthetic(
            ["germline", "c3"],
            [("1", "01", [0.4])],
            [("0", "1"), ("2", "3"), ("3", "2")],
            {"germline": [], "c3": [(1, "01", 0.4)]},
        ),
    )
    trees = bt.parse_lichee_trees(path, ["germline", "c3"])
    with pytest.raises(ValueError, match="cycle or a broken chain"):
        bt.build_lichee_clone_tree(trees, ["germline", "c3"], 0.05)


# --------------------------------------------------------------------------- #
# Spectra stay one row per cluster, whatever the tree does with shared nodes
# --------------------------------------------------------------------------- #


def test_clusters_sharing_a_node_keep_separate_spectra_rows():
    r = _build("shared_nodes.trees.txt")
    assert {tuple(g) for g in r.shared} == {("7", "8"), ("9", "10")}
    fasta = _FakeFasta({("1", 98, 101): "ACG", ("1", 198, 201): "ACG"})
    cluster_to_snvs = {  # 7 and 8 share a node but not their SNVs
        "7": {("1", 100, "C", "T")},
        "8": {("1", 100, "C", "T"), ("1", 200, "C", "T")},
    }
    spectra, _ = bt.bin_cluster_spectra(cluster_to_snvs, fasta)
    assert list(spectra.index) == ["7", "8"]
    assert spectra.loc["7"].sum() == 1 and spectra.loc["8"].sum() == 2


def _five_cluster_calls():
    site = ("1", 100, "C", "T")
    return {cid: {site: (0.4, 10)} for cid in ["10", "3", "9", "7", "8"]}


def test_written_header_matches_the_fixtures_column_order(tmp_path):
    columns = bt.write_lichee_input(_five_cluster_calls(), tmp_path / "in.txt")
    assert columns == COLUMNS  # numeric sort: 3, 7, 8, 9, 10 as c<id>


def test_write_run_parse_build_verify_chain_as_main_runs_it(tmp_path, monkeypatch):
    fixture = (LICHEE_FIXTURES / "shared_nodes.trees.txt").read_text()

    def fake_run(cmd, **kwargs):
        Path(cmd[cmd.index("-o") + 1]).write_text(fixture)
        return subprocess.CompletedProcess(cmd, 1)  # exit status is not trusted

    monkeypatch.setattr(bt.subprocess, "run", fake_run)
    columns = bt.write_lichee_input(_five_cluster_calls(), tmp_path / "in.txt")
    trees_path = bt.run_lichee(
        tmp_path / "in.txt",
        out_path=tmp_path / bt.LICHEE_OUT_NAME,
        log_path=tmp_path / "run.log",
        jar=tmp_path / "lichee.jar",
        lib=tmp_path / "lib",
        min_vaf_present=0.05,
        max_vaf_absent=0.05,
    )
    result = bt.build_lichee_clone_tree(
        bt.parse_lichee_trees(trees_path, columns), columns, 0.05
    )
    newick = bt.digraph_to_newick(result.tree, bt.GERMLINE_ROOT_ID)
    bt.verify_newick(
        newick, set(_five_cluster_calls()), bt.GERMLINE_ROOT_ID, result.hidden_ids
    )


def test_write_diagnostics_reports_lichee_as_comparison_only(tmp_path):
    r = _build("shared_nodes.trees.txt")
    dollo = bt.dollo_tree({frozenset(["7", "8"]): 5}, ["7", "8"], n_bootstrap=0)
    out = tmp_path / "diag.txt"
    bt.write_diagnostics(
        out, 0, {}, dollo, lichee_result=r, lichee_verdict="Found 3 valid trees"
    )
    text = out.read_text()
    assert "## Method used: dollo" in text
    assert "LICHeE (comparison only, not used for snv_tree.nwk)" in text
    assert "Found 3 valid trees" in text
    assert "indistinguishable by SNV profile" in text
    assert "Hidden group node g3" in text
