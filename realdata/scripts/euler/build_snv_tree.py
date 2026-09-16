"""Turn stage 06/06b's per-cluster somatic VCFs into the two SNV-side inputs
TreeHDP consumes: a rooted Newick tree and a per-cluster 96-channel spectra
matrix. (The CNA-side tree -- SCICoNE over copy-number bins, collapsed onto
the same SECEDO clusters -- is built separately by build_cna_tree.py; the two
trees share the Newick contract below and are run through TreeHDP
independently for comparison.)

Ours: neither secedo nor secedo-evaluation builds a phylogeny over SECEDO's
clusters or bins VCFs into COSMIC's 96 SBS channels. This is that step (recipe
stage 8, `realdata/recipe/euler_slice_d_plan.md`).

Contract (from `src/models/hdp_inference.py`'s `_BaseTreeHDP`, do not "fix" these)
    - Newick is labelled-internal-node form, e.g. ``((c2,c3)c1)germline;``.
      Observed clusters sit at internal nodes carrying their SECEDO cluster-ID
      label; the ancestor-as-tip idiom is not used, and every internal node
      (other than the germline root) is a real, spectrum-bearing cluster --
      there are no hidden/unlabelled Steiner nodes anywhere in this tree.
    - The tree is rooted at ``GERMLINE_ROOT_ID``, an implicit, spectrum-less
      latent node standing for the germline/empty-mutation state -- not a
      pseudo-normal cluster (see below). It carries no spectrum and is never a
      column in the presence matrix or the spectra table, which the model
      handles natively.
    - Node labels in the Newick, the spectra index, and the SECEDO cluster IDs
      are one ID system throughout.
    - The 96 spectra columns are in cosmic_signatures.csv's exact channel
      order, because the model does ``dot(activities, signatures)`` and aligns
      observed counts to signature columns positionally. ``main`` asserts this
      before doing anything else and fails loudly if it does not hold.

Calling design: tumour-vs-pseudo-normal (reverted from Attempt 2's tumour-only
+ gnomAD design, which made 85% of SNVs cluster-private and could not support
a tree). One SECEDO cluster (``NORMAL_CLUSTER_ID``, config.sh) is Mutect2's
-normal and never called as a tumour cluster itself -- it has no VCF, is
absent from ``spectra.csv``, and is not a column in the presence matrix.
Because independently-assembled clusters can each miss a variant another
cluster's own assembly supports, stage 06b unions every tumour cluster's
pass-1 PASS SNP sites per chromosome and force-calls every cluster at that
union, so presence (and VAF) at every site comes from the same read-support
test everywhere. This script reads that force-called output
(``*.forced.vcf``), not stage 06's pass-1 VCFs directly. A site counts as
PRESENT in a cluster if its force-called record has VAF >=
``--presence-min-vaf`` and ALT read depth >= ``--presence-min-alt-reads``
(defaults from config.sh's PRESENCE_MIN_VAF/PRESENCE_MIN_ALT_READS); ABSENT
otherwise.

Tree construction: LICHeE (Popic et al. 2015) is the primary method, with an
in-house Camin-Sokal parsimony search as the automatic fallback -- not a
debug-only escape hatch, a real second method -- if LICHeE fails to run, fails
to parse, or its cluster attachments cannot be resolved from its output.
Neither SCITE nor the mutation-set containment heuristic this module used to
carry is used any more: SCITE is designed for single-cell genotype matrices,
mismatched to SECEDO's pseudobulk clusters, and the containment heuristic
assumes a perfect phylogeny real calls do not satisfy (581k three-gamete
violations, 0.27-0.36 edge containment, and an implausible linear chain on the
earlier tumour-vs-pseudo-normal differential calls -- see git history for the
retired ``build_clone_tree``/``containment_fraction`` code). Three-gamete
(perfect-phylogeny) violations are still reported in snv_tree_diagnostics.txt as a
general compatibility diagnostic, independent of which tree method is used.

LICHeE takes a per-sample VAF table and a required baseline/normal column
(``-n``, 0-based). This pipeline has no cluster standing in for that baseline
(the pseudo-normal cluster is never itself genotyped; it is Mutect2's -normal,
not a row in the VAF matrix), so ``write_lichee_input`` synthesises one: an
all-zero VAF column named after ``GERMLINE_ROOT_ID``, which is exactly what a
true germline baseline would show at every somatic site and lines up
semantically with this tree's own latent root. LICHeE's own tree/network
output (``.dot``, GraphViz edges) is parsed the same way this module used to
parse SCITE's companion ``.gv`` file -- ``A -> B;`` edge lines, format-agnostic
regardless of node-labelling specifics. Cluster attachment is resolved by
matching DOT node labels against the real SECEDO cluster IDs used as LICHeE's
own input sample-column headers (verbatim, or via a 0-/1-based column-index
scheme as a fallback) -- SCITE's own attachment ambiguity is what motivated
this same multi-scheme approach originally.

CAVEAT, stated plainly rather than smoothed over: LICHeE's exact ``.dot``
node-labelling scheme is inferred from its documented CLI/algorithm
description, not confirmed against a real LICHeE run (no LICHeE binary
available while writing this). ``resolve_lichee_clone_tree`` raises loudly,
naming what it tried, if attachment cannot be resolved -- it never fabricates
a tree -- and ``main`` falls back to Camin-Sokal automatically in that case.
Confirm the real ``.dot`` output shape against ``lichee_out/*.dot`` on the
first real Euler run and extend ``_sample_label_schemes``/the DOT parser if
LICHeE's actual node labels don't match what is assumed here (see the plan
doc's pre-flight checklist for the parallel precedent: stage 06's read-group
assumption was flagged the same way before its first real run).

Camin-Sokal parsimony here is not the retired containment heuristic renamed:
every node in this pipeline's tree is an OBSERVED cluster with a fully known
presence/absence call at every mutation (unlike classical parsimony's usual
setting, where only leaves are observed and internal-node states are
inferred), so scoring a candidate topology needs no latent-state DP -- it is
a fixed count, per edge, of gain events (parent absent, child present, cost 1
each -- multiple independent gains, i.e. homoplasy, are allowed) and reversal
events (parent present, child absent -- forbidden under strict Camin-Sokal,
penalised heavily rather than made literally infinite so a well-defined
minimum-cost tree always exists even when the data has a genuine
incompatibility, the same "always-succeeding" property the retired
containment method had). ``camin_sokal_tree`` searches EVERY rooted topology
over the cluster set exhaustively (every node's parent drawn from {germline}
union the other clusters, filtered to acyclic trees) -- tractable at the
handful-of-clusters scale SECEDO produces for one slice, not a general-purpose
phylogenetics tool; ``--camin-sokal-max-clusters`` refuses to run past a
configurable cap rather than hang.

Channel ordering was recovered empirically (cosmic_signatures.csv carries no
channel labels, only ``Channel_0..Channel_95``): SBS1's four dominant channels
sit 24 apart, at the position matching NCG>NTG (C>T at CpG, all four 5' flanks),
and SBS4/SBS5/SBS92's dominant channels match their known C>A/T>C aetiology
under the same hypothesis. That fixes the axis as index = five*24 + subtype*4 +
three, five/three in A,C,G,T order and subtype in C>A,C>G,C>T,T>A,T>C,T>G order
-- the alphabetical sort of COSMIC's own ``A[C>A]A`` .. ``T[T>G]T`` labels.
``main`` still asserts cosmic_signatures.csv's columns match before trusting it.
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Hashable, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import phylox

N_CHANNELS = 96
BASES = ["A", "C", "G", "T"]
SUBTYPES = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
BASE_IDX = {b: i for i, b in enumerate(BASES)}
SUBTYPE_IDX = {s: i for i, s in enumerate(SUBTYPES)}
COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}

# The tree's root: an implicit, spectrum-less latent node for the germline/
# empty-mutation state. Not a real cluster (no VCF, no BAM, no column in the
# presence matrix or spectra table) -- the pseudo-normal cluster (config.sh's
# NORMAL_CLUSTER_ID) is Mutect2's -normal, never itself called, so this plays
# the tree's unique root without being an actual SECEDO cluster.
GERMLINE_ROOT_ID = "germline"

SNVKey = Tuple[str, int, str, str]  # chrom, 1-based pos, ref, alt

_FORCED_VCF_NAME_RE = re.compile(
    r"^clone(?P<cluster>[^_]+)_(?P<chrom>.+)\.forced\.vcf$"
)


# --------------------------------------------------------------------------- #
# Channel axis
# --------------------------------------------------------------------------- #


def channel_labels() -> List[str]:
    """The 96 column names as they appear in cosmic_signatures.csv."""
    return [f"Channel_{i}" for i in range(N_CHANNELS)]


def channel_index(five: str, ref: str, alt: str, three: str) -> int:
    """Index into the 96-channel axis for a pyrimidine-normalised substitution.

    index = five*24 + subtype*4 + three, matching the alphabetical order of
    COSMIC's own "A[C>A]A" .. "T[T>G]T" trinucleotide labels (see module
    docstring for how this was recovered).
    """
    try:
        return BASE_IDX[five] * 24 + SUBTYPE_IDX[f"{ref}>{alt}"] * 4 + BASE_IDX[three]
    except KeyError as exc:
        raise ValueError(
            f"unrecognised base or substitution: five={five!r} ref={ref!r} "
            f"alt={alt!r} three={three!r}"
        ) from exc


def pyrimidine_normalise(
    five: str, ref: str, alt: str, three: str
) -> Tuple[str, str, str, str]:
    """Reverse-complement onto the pyrimidine strand when ref is a purine (A/G)."""
    if ref in ("A", "G"):
        return COMPLEMENT[three], COMPLEMENT[ref], COMPLEMENT[alt], COMPLEMENT[five]
    return five, ref, alt, three


def snv_channel(five: str, ref: str, alt: str, three: str) -> int:
    """Channel index for one SNV, normalising to the pyrimidine strand first."""
    five, ref, alt, three = pyrimidine_normalise(five, ref, alt, three)
    return channel_index(five, ref, alt, three)


# --------------------------------------------------------------------------- #
# VCF parsing
# --------------------------------------------------------------------------- #


def _safe_float(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(x: str) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def parse_format_values(format_str: str, sample_str: str) -> Dict[str, str]:
    """{FORMAT key: raw sample value}, as found in one VCF genotype column pair
    (the ``FORMAT`` and single-sample columns of a single-sample VCF record)."""
    return dict(zip(format_str.split(":"), sample_str.split(":")))


def parse_forced_vcf_calls(vcf_path: Path) -> Dict[SNVKey, Tuple[float, int]]:
    """Parse one force-called, single-sample VCF (stage 06b's pass 2) into
    ``{snv_key: (vaf, alt_reads)}`` for every single-base-substitution ALT
    allele at every record.

    Every record, PASS or not: pass 2 force-calls every cluster at every
    union site regardless of whether that cluster independently supports it,
    and presence is decided in Python directly off VAF/ALT-read depth (see
    ``resolve_presence_calls``), not off the FILTER column.

    SUB-DECISION: AF/AD extraction assumes GATK4 Mutect2's own FORMAT layout
    -- AD is ``ref_depth,alt_depth_1[,alt_depth_2...]`` (one more entry than
    ALT alleles, ref first) and AF is ``alt_af_1[,alt_af_2...]`` (one entry
    per ALT allele, no ref entry) -- paired positionally with the record's own
    (possibly multi-allelic) ALT list. A record whose AD or AF cannot be
    parsed this way (wrong field count, non-numeric, "." for no coverage) is
    treated as zero VAF / zero ALT reads rather than raising: force-calling a
    site with no supporting reads at all is an expected "absent here"
    outcome, not a malformed file.
    """
    calls: Dict[SNVKey, Tuple[float, int]] = {}
    with open(vcf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                continue
            chrom, pos, _id, ref, alt_field = fields[:5]
            format_str, sample_str = fields[8], fields[9]
            ref = ref.upper()
            if len(ref) != 1 or ref not in "ACGT":
                continue
            fmt = parse_format_values(format_str, sample_str)
            ad_raw = fmt.get("AD", "").split(",")
            af_raw = fmt.get("AF", "").split(",")
            for i, alt in enumerate(alt_field.split(",")):
                alt = alt.upper()
                if len(alt) != 1 or alt not in "ACGT":
                    continue
                vaf = _safe_float(af_raw[i]) if i < len(af_raw) else 0.0
                alt_reads = _safe_int(ad_raw[i + 1]) if i + 1 < len(ad_raw) else 0
                calls[(chrom, int(pos), ref, alt)] = (vaf, alt_reads)
    return calls


def resolve_presence_calls(
    cluster_to_calls: Dict[str, Dict[SNVKey, Tuple[float, int]]],
    min_vaf: float,
    min_alt_reads: int,
) -> Dict[str, Set[SNVKey]]:
    """A force-called site is PRESENT in a cluster if its VAF >= ``min_vaf``
    AND ALT read depth >= ``min_alt_reads``; ABSENT otherwise -- including
    sites the cluster's own record never reached these thresholds at.
    """
    return {
        cluster: {
            key
            for key, (vaf, alt_reads) in calls.items()
            if vaf >= min_vaf and alt_reads >= min_alt_reads
        }
        for cluster, calls in cluster_to_calls.items()
    }


def discover_forced_cluster_vcfs(vcf_dir: Path) -> Dict[str, List[Path]]:
    """Group ``clone<cluster>_<chrom>.forced.vcf`` files (stage 06b's pass-2
    force-call output) by cluster. The pseudo-normal cluster is never a key
    here: it has no VCF (excluded from stage 05's tasks.tsv, never called).
    """
    out: Dict[str, List[Path]] = defaultdict(list)
    for f in sorted(vcf_dir.glob("clone*_*.forced.vcf")):
        m = _FORCED_VCF_NAME_RE.match(f.name)
        if not m:
            continue
        out[m.group("cluster")].append(f)
    return dict(out)


# --------------------------------------------------------------------------- #
# clone x SNV matrix
# --------------------------------------------------------------------------- #


def _sort_key(cluster_id: Hashable) -> Tuple[int, object]:
    """Numeric sort for plain-integer cluster IDs, lexicographic fallback otherwise."""
    try:
        return (0, int(cluster_id))
    except (TypeError, ValueError):
        return (1, str(cluster_id))


def snv_key_str(key: SNVKey) -> str:
    chrom, pos, ref, alt = key
    return f"{chrom}:{pos}:{ref}>{alt}"


def build_snv_presence_matrix(cluster_to_snvs: Dict[str, Set[SNVKey]]) -> pd.DataFrame:
    """clone x SNV binary presence matrix: rows the union of SNVs, columns the
    clusters."""
    clusters = sorted(cluster_to_snvs, key=_sort_key)
    all_snvs = sorted(
        {snv_key_str(k) for snvs in cluster_to_snvs.values() for k in snvs}
    )
    matrix = pd.DataFrame(0, index=all_snvs, columns=clusters, dtype=int)
    for cluster, snvs in cluster_to_snvs.items():
        rows = [snv_key_str(k) for k in snvs]
        matrix.loc[rows, cluster] = 1
    return matrix


def mutation_sets_from_matrix(matrix: pd.DataFrame) -> Dict[str, Set[str]]:
    """Column -> set of row labels present (value == 1), read back off the
    presence matrix."""
    return {col: set(matrix.index[matrix[col] == 1]) for col in matrix.columns}


def three_gamete_violations(mutation_sets: Dict[str, Set[Hashable]]) -> int:
    """Count SNV pairs violating the perfect-phylogeny (three-gamete) test.

    Two SNVs are incompatible with a single mutation history if, among the
    clusters, all three of "only i", "only j", and "both" occur (the fourth
    gamete, "neither", is irrelevant to infinite-sites compatibility). Each
    SNV's presence across clusters is bit-packed into one integer so a pair
    check is O(1); overall cost is O(n_snvs^2), fine at the SNV counts this
    pipeline expects (sparse tens-to-hundreds of mutations per cluster).

    A general presence-matrix compatibility diagnostic, independent of which
    tree method is used -- reported in snv_tree_diagnostics.txt regardless.
    """
    clusters = sorted(mutation_sets)
    cluster_bit = {c: i for i, c in enumerate(clusters)}
    snv_masks: Dict[Hashable, int] = {}
    for cluster, muts in mutation_sets.items():
        bit = 1 << cluster_bit[cluster]
        for m in muts:
            snv_masks[m] = snv_masks.get(m, 0) | bit

    masks = list(snv_masks.values())
    violations = 0
    for i in range(len(masks)):
        mi = masks[i]
        for j in range(i + 1, len(masks)):
            mj = masks[j]
            if (mi & ~mj) and (mj & ~mi) and (mi & mj):
                violations += 1
    return violations


# --------------------------------------------------------------------------- #
# Newick I/O and the model loader's own contract
# --------------------------------------------------------------------------- #


def digraph_to_newick(tree: nx.DiGraph, root: Hashable) -> str:
    """Rooted, labelled-internal-node Newick string for a clone tree."""

    def _recurse(node: Hashable) -> str:
        children = sorted(tree.successors(node), key=_sort_key)
        if not children:
            return str(node)
        inner = ",".join(_recurse(c) for c in children)
        return f"({inner}){node}"

    return _recurse(root) + ";"


def parse_newick_like_model(newick_str: str) -> nx.DiGraph:
    """Parse a Newick string exactly the way ``_BaseTreeHDP.__init__`` does.

    Splits on ";", parses each tree with phylox, and relabels every node to
    its Newick label -- the same two steps the model loader runs before it
    ever looks at node IDs.
    """
    graph = nx.DiGraph()
    for s in newick_str.split(";"):
        if not s.strip():
            continue
        tree = phylox.DiNetwork.from_newick(s)
        mapping = {n: tree.nodes[n].get("label", str(n)) for n in tree.nodes()}
        graph = nx.compose(graph, nx.relabel_nodes(tree, mapping))
    return graph


def verify_newick(newick_str: str, cluster_ids: Set[str], normal_id: str) -> None:
    """Round-trip check: parse the way the model does and confirm the contract holds.

    ``normal_id`` is the tree-root label (``GERMLINE_ROOT_ID`` in ``main``),
    not a real cluster. Raises AssertionError if it is not the unique root, or
    if the parsed node labels are not exactly the cluster IDs plus the root,
    each appearing once.
    """
    graph = parse_newick_like_model(newick_str)
    expected = set(cluster_ids) | {normal_id}
    labels = list(graph.nodes())
    if len(labels) != len(expected):
        raise AssertionError(
            f"parsed {len(labels)} node labels, expected {len(expected)} "
            f"(cluster IDs plus the germline root); a label collision or "
            f"missing cluster is likely: {sorted(labels)} vs {sorted(expected)}"
        )
    if set(labels) != expected:
        raise AssertionError(
            f"parsed labels {sorted(labels)} != expected {sorted(expected)}"
        )
    roots = [n for n, d in graph.in_degree() if d == 0]
    if len(roots) != 1:
        raise AssertionError(f"expected exactly one root, got {roots}")
    if roots[0] != normal_id:
        raise AssertionError(
            f"root is {roots[0]!r}, expected germline root {normal_id!r}"
        )


# --------------------------------------------------------------------------- #
# 96-channel binning
# --------------------------------------------------------------------------- #


def bin_cluster_spectra(
    cluster_to_snvs: Dict[str, Set[SNVKey]], fasta
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Bin each cluster's SNVs into the 96 SBS channels using trinucleotide context.

    ``fasta`` is an open pysam.FastaFile over REF_FASTA. Skips (and counts) any
    SNV whose trinucleotide context hits an ambiguous base, or whose middle
    base disagrees with the VCF's REF allele (a reference-build mismatch).
    """
    clusters = sorted(cluster_to_snvs, key=_sort_key)
    labels = channel_labels()
    counts = np.zeros((len(clusters), N_CHANNELS), dtype=int)
    skip_counts = {"ambiguous_context": 0, "ref_mismatch": 0}

    for row, cluster in enumerate(clusters):
        for chrom, pos, ref, alt in cluster_to_snvs[cluster]:
            context = fasta.fetch(chrom, pos - 2, pos + 1).upper()
            if len(context) != 3 or any(b not in "ACGT" for b in context):
                skip_counts["ambiguous_context"] += 1
                continue
            five, mid, three = context[0], context[1], context[2]
            if mid != ref:
                skip_counts["ref_mismatch"] += 1
                continue
            try:
                idx = snv_channel(five, ref, alt, three)
            except ValueError:
                skip_counts["ambiguous_context"] += 1
                continue
            counts[row, idx] += 1

    spectra = pd.DataFrame(counts, index=clusters, columns=labels)
    return spectra, skip_counts


# --------------------------------------------------------------------------- #
# Camin-Sokal parsimony (fallback tree builder)
# --------------------------------------------------------------------------- #

# Reversal (parent present, child absent) is forbidden under strict
# Camin-Sokal; penalised heavily rather than made literally infinite so a
# well-defined minimum-cost topology always exists even on genuinely
# incompatible data (mirrors the retired containment method's
# always-succeeding property). Large enough that any topology needing zero
# reversals is always preferred over one needing even a single reversal, for
# any plausible gain count at this pipeline's SNV-per-cluster scale.
CAMIN_SOKAL_REVERSAL_PENALTY = 10_000


def enumerate_rooted_trees(leaves: List[str], root: str) -> Iterable[Dict[str, str]]:
    """Yield every rooted tree over ``leaves`` as a ``{node: parent}`` dict,
    each node's parent drawn from ``{root}`` union the other leaves, root
    fixed as the unique ultimate ancestor.

    Brute force over every ``len(choices) ** len(leaves)`` parent-choice
    combination, filtered to the acyclic ones. Every internal node in this
    pipeline's Newick contract is itself an observed cluster (no hidden
    Steiner nodes), so this really is the full search space, not an
    approximation -- see ``camin_sokal_tree`` for the size cap that keeps it
    tractable.
    """
    choices = [root] + list(leaves)
    for parents in itertools.product(choices, repeat=len(leaves)):
        parent_of = dict(zip(leaves, parents))
        if any(parent_of[node] == node for node in leaves):
            continue
        if _is_valid_rooted_tree(parent_of, root, leaves):
            yield parent_of


def _is_valid_rooted_tree(
    parent_of: Dict[str, str], root: str, leaves: List[str]
) -> bool:
    """True if following ``parent_of`` from every leaf reaches ``root`` in at
    most ``len(leaves)`` steps without revisiting a node (no cycles)."""
    for start in leaves:
        seen: Set[str] = set()
        cur = start
        steps = 0
        while cur != root:
            if cur in seen or steps > len(leaves):
                return False
            seen.add(cur)
            cur = parent_of.get(cur, root)
            steps += 1
    return True


def camin_sokal_topology_score(
    parent_of: Dict[str, str], mutation_sets: Dict[str, Set[Hashable]]
) -> Tuple[int, int, int]:
    """Score one topology: every node's state is directly observed (this
    pipeline's internal nodes are real clusters, not inferred ancestors -- see
    module docstring for why this differs from classical Camin-Sokal's usual
    latent-internal-state DP). For every (parent, child) edge: a mutation
    gained (absent in parent, present in child) costs 1 -- independent gains
    in different lineages (homoplasy) are allowed, each paying its own cost;
    a mutation reversed (present in parent, absent in child) is forbidden
    under strict Camin-Sokal, penalised by ``CAMIN_SOKAL_REVERSAL_PENALTY``.
    The germline root's own mutation set is always empty.

    Returns ``(total_score, n_gains, n_reversals)``.
    """
    n_gains = 0
    n_reversals = 0
    for child, parent in parent_of.items():
        parent_muts = mutation_sets.get(parent, frozenset())
        child_muts = mutation_sets[child]
        n_gains += len(child_muts - parent_muts)
        n_reversals += len(parent_muts - child_muts)
    total = n_reversals * CAMIN_SOKAL_REVERSAL_PENALTY + n_gains
    return total, n_gains, n_reversals


def camin_sokal_tree(
    mutation_sets: Dict[str, Set[Hashable]],
    root: str,
    max_clusters: int = 8,
) -> Tuple[nx.DiGraph, int, int]:
    """Exhaustive-search Camin-Sokal parsimony tree: the minimum-score
    topology over every rooted tree ``enumerate_rooted_trees`` yields.

    Deterministic: ``enumerate_rooted_trees``' iteration order is fixed given
    a fixed, sorted leaf order, and ties keep the first (strictly-better-only)
    topology found, so re-running on the same input reproduces the same tree.

    Raises ValueError if ``len(mutation_sets)`` exceeds ``max_clusters`` --
    brute force here is ``(n+1)**n``, tractable at the handful-of-clusters
    scale one SECEDO run produces, not in general.

    Returns ``(tree, n_gains, n_reversals)`` for the winning topology.
    """
    leaves = sorted(mutation_sets, key=_sort_key)
    if len(leaves) > max_clusters:
        raise ValueError(
            f"camin_sokal_tree: {len(leaves)} clusters exceeds the exhaustive "
            f"search cap of {max_clusters} (see --camin-sokal-max-clusters)"
        )
    if not leaves:
        raise ValueError("camin_sokal_tree: no clusters to build a tree over")

    best_key: Optional[Tuple[int, Tuple[Tuple[str, str], ...]]] = None
    best_parent_of: Optional[Dict[str, str]] = None
    best_stats = (0, 0)
    for parent_of in enumerate_rooted_trees(leaves, root):
        score, n_gains, n_reversals = camin_sokal_topology_score(
            parent_of, mutation_sets
        )
        # Tie-break deterministically on the sorted edge list so equally-good
        # topologies still pick one reproducible winner.
        tie_break = tuple(sorted(parent_of.items()))
        key = (score, tie_break)
        if best_key is None or key < best_key:
            best_key = key
            best_parent_of = parent_of
            best_stats = (n_gains, n_reversals)

    tree = nx.DiGraph()
    tree.add_node(root)
    for child, parent in best_parent_of.items():
        tree.add_edge(parent, child)
    n_gains, n_reversals = best_stats
    return tree, n_gains, n_reversals


# --------------------------------------------------------------------------- #
# LICHeE (primary tree builder)
# --------------------------------------------------------------------------- #


def find_lichee_binary(explicit: Optional[str] = None) -> str:
    """Resolve the LICHeE launcher: ``explicit`` (the ``--lichee-bin`` CLI
    arg), else ``$LICHEE_BIN``, else the repo's own build at
    ``realdata/external/lichee/release/lichee``.

    Always returns a path string (never None) so the caller can report a
    clear "not found" error against a concrete path rather than an absent
    binary.
    """
    if explicit:
        return explicit
    if os.environ.get("LICHEE_BIN"):
        return os.environ["LICHEE_BIN"]
    return str(
        Path(__file__).resolve().parents[2]
        / "external"
        / "lichee"
        / "release"
        / "lichee"
    )


def write_lichee_input(
    cluster_to_calls: Dict[str, Dict[SNVKey, Tuple[float, int]]],
    path: Path,
    germline_label: str = GERMLINE_ROOT_ID,
) -> List[str]:
    """Write LICHeE's tab-separated VAF input: ``#chr position description
    <samples...>``, one row per SNV present in at least one cluster's calls,
    VAF from ``cluster_to_calls`` (0.0 for a cluster with no record at that
    site).

    Column order is this module's usual numeric-then-lexicographic cluster
    sort, real SECEDO cluster IDs as headers (not LICHeE's example "S1,S2,..."
    placeholders) -- deliberately, so its own output has the best chance of
    preserving real cluster identity in node/edge labels; see
    ``resolve_lichee_clone_tree``. A synthetic all-zero ``germline_label``
    column is prepended as LICHeE's required baseline/normal sample (``-n
    0``) -- see module docstring for why this pipeline has no real one.

    Returns the column order actually written (germline column first, then
    clusters), for the caller to pass to ``-n``/attachment resolution.
    """
    clusters = sorted(cluster_to_calls, key=_sort_key)
    columns = [germline_label] + clusters
    all_sites = sorted({key for calls in cluster_to_calls.values() for key in calls})

    lines = ["#chr\tposition\tdescription\t" + "\t".join(columns)]
    for chrom, pos, ref, alt in all_sites:
        description = f"{ref}>{alt}"
        row = [chrom, str(pos), description, "0.0"]  # germline VAF always 0
        for cluster in clusters:
            vaf, _alt_reads = cluster_to_calls[cluster].get(
                (chrom, pos, ref, alt), (0.0, 0)
            )
            row.append(f"{vaf:.4f}")
        lines.append("\t".join(row))

    Path(path).write_text("\n".join(lines) + "\n")
    return columns


def run_lichee(
    input_path: Path,
    out_prefix: Path,
    log_path: Path,
    lichee_bin: str,
    normal_index: int = 0,
    min_vaf_present: float = 0.05,
    max_vaf_absent: float = 0.0,
    min_cluster_size: int = 2,
    log_header: str = "",
) -> Path:
    """Run LICHeE's ``build`` step and return the path to its ``.dot`` tree
    export (``-dot``).

    Raises ``subprocess.CalledProcessError`` if LICHeE exits non-zero, and
    ``FileNotFoundError`` if it exits cleanly but no ``.dot`` file appears --
    both are the caller's cue to fall back to Camin-Sokal rather than trust a
    missing or partial result.
    """
    trees_out = Path(f"{out_prefix}.trees.txt")
    cmd = [
        lichee_bin,
        "-build",
        "-i", str(input_path),
        "-n", str(normal_index),
        "-minVAFPresent", str(min_vaf_present),
        "-maxVAFAbsent", str(max_vaf_absent),
        "-minClusterSize", str(min_cluster_size),
        "-dot",
        "-o", str(trees_out),
    ]  # fmt: skip
    with open(log_path, "w") as log:
        if log_header:
            log.write(log_header)
        log.write("command: " + " ".join(cmd) + "\n\n")
        log.flush()
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)

    # LICHeE's own naming for the -dot export is not pinned down without a
    # real run (see module docstring); look for the documented default
    # (<output>.dot) and fall back to any *.dot LICHeE wrote alongside it.
    candidates = [Path(f"{trees_out}.dot")] + sorted(
        out_prefix.parent.glob(f"{out_prefix.name}*.dot")
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"LICHeE exited cleanly but no .dot file was found near {trees_out} "
        f"(tried {[str(c) for c in candidates]}); see {log_path}"
    )


_DOT_EDGE_RE = re.compile(
    r'^\s*"?(?P<a>[^"\s\->]+)"?\s*->\s*"?(?P<b>[^"\s\->]+)"?\s*(\[[^\]]*\])?\s*;?\s*$'
)
_DOT_NODE_LABEL_RE = re.compile(
    r'^\s*"?(?P<node>[^"\s\[\->]+)"?\s*\[[^\]]*label\s*=\s*"(?P<label>[^"]*)"'
)


def parse_lichee_dot(dot_path: Path) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    """Parse a GraphViz ``.dot`` file into ``({node: label_or_node}, [{"a":
    parent, "b": child}, ...])``.

    Deliberately minimal and format-tolerant, the same regex-based approach
    already proven here for SCITE's ``.gv`` output (edge lines and a
    ``label="..."`` node attribute are close to universal GraphViz syntax
    regardless of what a specific tool puts in them): does not depend on
    LICHeE's exact node-naming scheme, only on it emitting standard ``A ->
    B;`` edges and optional ``node [label="..."]`` declarations.
    """
    node_labels: Dict[str, str] = {}
    edges: List[Dict[str, str]] = []
    for line in Path(dot_path).read_text().splitlines():
        edge_match = _DOT_EDGE_RE.match(line)
        if edge_match:
            edges.append({"a": edge_match.group("a"), "b": edge_match.group("b")})
            continue
        label_match = _DOT_NODE_LABEL_RE.match(line)
        if label_match:
            node_labels[label_match.group("node")] = label_match.group("label")
    return node_labels, edges


def _sample_label_schemes(column_order: List[str]) -> List[Dict[str, str]]:
    """Ordered, whole-column candidate labelling schemes for a tool's sample
    nodes: cluster_id -> the label that scheme predicts for it.

    Tried as complete, self-consistent schemes rather than per-sample
    candidates independently, so a sample's own identity is not expected to
    appear verbatim in every tool's output, and 0-based/1-based column-index
    spellings share overlapping label spaces (index 1 in one scheme is index
    0 in the shifted one) -- picking candidates per sample independently
    could match different samples under mutually inconsistent schemes and
    silently misassign a leaf instead of failing. Matching one scheme against
    every column at once avoids that.
    """
    n = len(column_order)
    schemes = [dict(zip(column_order, column_order))]  # cluster ID, verbatim
    for start in (1, 0):
        indices = range(start, start + n)
        schemes.append({cid: str(i) for cid, i in zip(column_order, indices)})
        schemes.append({cid: f"s{i}" for cid, i in zip(column_order, indices)})
        schemes.append({cid: f"S{i}" for cid, i in zip(column_order, indices)})
    return schemes


def resolve_lichee_clone_tree(
    dot_path: Path, column_order: List[str], normal_id: str
) -> Tuple[nx.DiGraph, str]:
    """Build the SECEDO-cluster clone tree from LICHeE's ``.dot`` export.

    LICHeE's tree nodes are SSNV clusters (subclone groups), not directly our
    SECEDO clusters, so this resolves each real cluster's attachment point in
    the DOT graph (via node id, or a ``label="..."`` node attribute, matched
    against ``column_order`` through ``_sample_label_schemes``) and collapses
    to the nearest such ancestor -- walking up through unlabelled/unmatched
    LICHeE nodes -- the same idiom this module used for SCITE's mutation
    tree before SCITE was retired. Raises ValueError, naming what it tried,
    if no scheme resolves every column: the caller must fall back to
    Camin-Sokal rather than fabricate a tree (see module docstring's caveat
    on LICHeE's unconfirmed exact output schema).
    """
    node_labels, edges = parse_lichee_dot(dot_path)
    if not edges:
        raise ValueError(f"{dot_path} contains no parseable 'A -> B;' edges")

    all_nodes = {n for e in edges for n in (e["a"], e["b"])} | set(node_labels)
    # Candidate strings for each DOT node: its own id, and its label attribute
    # if it has one.
    node_candidates: Dict[str, Set[str]] = {n: {n} for n in all_nodes}
    for node, label in node_labels.items():
        node_candidates.setdefault(node, set()).add(label)

    resolved: Optional[Dict[str, str]] = None
    for scheme in _sample_label_schemes(column_order):
        target_to_cid = {v: k for k, v in scheme.items()}
        candidate = {
            target_to_cid[cand]: node
            for node, cands in node_candidates.items()
            for cand in cands
            if cand in target_to_cid
        }
        if len(candidate) == len(column_order):
            resolved = candidate
            break

    if resolved is None:
        raise ValueError(
            "no consistent sample-node labelling scheme (cluster ID, or a "
            "0-/1-based column index with an optional s/S prefix) covers all "
            f"of {column_order}; DOT nodes found: {sorted(all_nodes)}"
        )

    parent_of: Dict[str, str] = {}
    for e in edges:
        parent_of[e["b"]] = e["a"]

    tree = collapse_by_nearest_labelled_ancestor(parent_of, resolved, normal_id)
    return tree, "the .dot export's node ids/labels"


def collapse_by_nearest_labelled_ancestor(
    parent_of: Dict[Hashable, Hashable],
    node_of_cluster: Dict[str, Hashable],
    root: str,
) -> nx.DiGraph:
    """Collapse an arbitrary tree (given as a ``{child: parent}`` dict over
    some tool's own node space) onto a small set of labelled clusters, each
    already resolved to one node in that space (``node_of_cluster``): every
    cluster's parent in the result is the nearest OTHER cluster's node found
    by walking up ``parent_of`` from its own node, or ``root`` if none is
    found before the top.

    Shared by every "collapse a tool's raw tree onto our observed clusters"
    case in this pipeline (LICHeE here, via ``resolve_lichee_clone_tree``;
    SCICoNE's cell tree in ``build_cna_tree.py``) -- the same idiom this
    module used for SCITE's mutation tree before SCITE was retired.
    """
    node_to_cluster = {node: cid for cid, node in node_of_cluster.items()}
    tree = nx.DiGraph()
    tree.add_node(root)
    for cid, node in node_of_cluster.items():
        ancestor = parent_of.get(node)
        while ancestor is not None and ancestor not in node_to_cluster:
            ancestor = parent_of.get(ancestor)
        parent_cluster = node_to_cluster.get(ancestor, root)
        tree.add_edge(parent_cluster, cid)
    return tree


# --------------------------------------------------------------------------- #
# Topology classification (chain vs branching)
# --------------------------------------------------------------------------- #


def is_unbranched_chain(tree: nx.DiGraph) -> bool:
    """True if every node in ``tree`` has at most one child -- a straight
    line down from the root, no branching anywhere."""
    return all(tree.out_degree(n) <= 1 for n in tree.nodes())


def chain_depth_if_linear(tree: nx.DiGraph) -> Optional[int]:
    """If ``tree`` is a single unbranched path from a unique root, return its
    depth (edge count from root to the tip); otherwise None."""
    if not is_unbranched_chain(tree):
        return None
    roots = [n for n, d in tree.in_degree() if d == 0]
    if len(roots) != 1:
        return None
    depth = 0
    node = roots[0]
    seen = {node}
    while True:
        children = list(tree.successors(node))
        if not children:
            return depth
        node = children[0]
        if node in seen:  # malformed/cyclic input guard; not expected from a tree
            return None
        seen.add(node)
        depth += 1


def classify_topology(tree: Optional[nx.DiGraph]) -> str:
    """'linear chain (depth N)', 'branching', or 'unavailable' (``tree`` is
    None) -- the one-line topology summary used throughout the diagnostics."""
    if tree is None:
        return "unavailable"
    depth = chain_depth_if_linear(tree)
    return f"linear chain (depth {depth})" if depth is not None else "branching"


def compare_topologies(
    tree_a: nx.DiGraph, tree_b: nx.DiGraph, cluster_ids: List[str]
) -> float:
    """Fraction of cluster pairs whose ancestor/descendant relation agrees
    between two trees -- a pairwise-relation summary, not a full tree-edit
    distance."""

    def relation(tree: nx.DiGraph, a: str, b: str) -> str:
        if a in tree and b in tree:
            if a in nx.ancestors(tree, b):
                return "ancestor"
            if b in nx.ancestors(tree, a):
                return "descendant"
        return "unrelated"

    ids = list(cluster_ids)
    total = agree = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            total += 1
            if relation(tree_a, ids[i], ids[j]) == relation(tree_b, ids[i], ids[j]):
                agree += 1
    return agree / total if total else float("nan")


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


def write_diagnostics(
    path: Path,
    mutation_sets: Dict[str, Set[Hashable]],
    skip_counts: Dict[str, int],
    n_violations: int,
    method_used: str,
    lichee_error: Optional[str],
    camin_sokal_stats: Optional[Tuple[int, int]],
    primary_tree: nx.DiGraph,
) -> None:
    """LICHeE is primary when it yields a resolvable tree; Camin-Sokal
    parsimony is the automatic fallback otherwise -- both are real tree
    sources, not a primary/debug-only-cross-check split. Numbers only, plus
    the one plain-language finding line a degenerate result calls for.
    """
    lines = [
        "# Stage 08 (SNV tree) diagnostics\n\n",
        f"## Method used: {method_used}\n",
    ]
    if method_used == "lichee":
        lines.append("LICHeE ran and its cluster attachments resolved cleanly.\n")
    else:
        lines.append(
            f"LICHeE unavailable or unresolved ({lichee_error}); fell back to "
            "Camin-Sokal parsimony (a real second method, not a debug-only "
            "escape hatch -- see module docstring).\n"
        )
        if camin_sokal_stats is not None:
            n_gains, n_reversals = camin_sokal_stats
            lines.append(
                f"Camin-Sokal search: {n_gains} gain events, {n_reversals} "
                "reversal events in the winning topology "
                f"(reversal penalty {CAMIN_SOKAL_REVERSAL_PENALTY} per event).\n"
            )
            if n_reversals > 0:
                lines.append(
                    "FINDING: the winning topology still needed reversal "
                    "events -- no fully Camin-Sokal-compatible tree exists "
                    "for this presence matrix; treat tree.nwk as the "
                    "least-violating tree, not a clean phylogeny.\n"
                )

    lines.append(f"\nThree-gamete (perfect-phylogeny) violations: {n_violations}\n")
    total_skipped = sum(skip_counts.values())
    lines.append(f"SNVs skipped in binning: {total_skipped} {dict(skip_counts)}\n")

    lines.append("\n## Topology result\n")
    lines.append(f"Emitted tree: {classify_topology(primary_tree)}\n")
    if classify_topology(primary_tree).startswith("linear chain"):
        lines.append(
            "FINDING: the emitted tree is a non-branching chain -- the "
            "force-called SNVs do not support a branching phylogeny under "
            f"the {method_used} method.\n"
        )

    Path(path).write_text("".join(lines))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--vcf-dir",
        required=True,
        type=Path,
        help="directory of clone<c>_<chrom>.forced.vcf files (stage 06b's "
        "pass-2 force-call output, copied out by stage 07)",
    )
    p.add_argument("--ref-fasta", required=True, type=Path)
    p.add_argument(
        "--cosmic-signatures",
        required=True,
        type=Path,
        help="cosmic_signatures.csv, used only to assert the channel order",
    )
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--presence-min-vaf",
        type=float,
        default=0.05,
        help="a force-called site counts as present in a cluster only at or "
        "above this VAF (see config.sh's PRESENCE_MIN_VAF)",
    )
    p.add_argument(
        "--presence-min-alt-reads",
        type=int,
        default=2,
        help="a force-called site counts as present in a cluster only at or "
        "above this many ALT reads (see config.sh's PRESENCE_MIN_ALT_READS)",
    )
    p.add_argument(
        "--lichee-bin",
        default=None,
        help="path to the lichee launcher; defaults to $LICHEE_BIN, then "
        "realdata/external/lichee/release/lichee",
    )
    p.add_argument(
        "--lichee-max-vaf-absent",
        type=float,
        default=0.0,
        help="LICHeE's own -maxVAFAbsent (its internal SSNV clustering step, "
        "separate from --presence-min-vaf, which controls this script's own "
        "presence matrix)",
    )
    p.add_argument(
        "--lichee-min-cluster-size",
        type=int,
        default=2,
        help="LICHeE's own -minClusterSize",
    )
    p.add_argument(
        "--camin-sokal-max-clusters",
        type=int,
        default=8,
        help="refuse the exhaustive Camin-Sokal search past this many "
        "clusters rather than hang (see camin_sokal_tree)",
    )
    p.add_argument(
        "--skip-lichee",
        action="store_true",
        help="go straight to Camin-Sokal parsimony without attempting "
        "LICHeE -- for environments without the LICHeE binary",
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    expected_cols = channel_labels()
    cosmic_cols = list(
        pd.read_csv(args.cosmic_signatures, index_col=0, nrows=0).columns
    )
    if cosmic_cols != expected_cols:
        sys.exit(
            f"{args.cosmic_signatures} channel order does not match the expected "
            f"{N_CHANNELS}-channel axis -- got {cosmic_cols[:4]}..., "
            f"expected {expected_cols[:4]}.... Refusing to emit spectra that would "
            "misalign against the model's signature columns."
        )

    cluster_vcfs = discover_forced_cluster_vcfs(args.vcf_dir)
    if not cluster_vcfs:
        sys.exit(f"no clone*_*.forced.vcf files found in {args.vcf_dir}")

    cluster_to_calls: Dict[str, Dict[SNVKey, Tuple[float, int]]] = {}
    for cluster, files in cluster_vcfs.items():
        calls: Dict[SNVKey, Tuple[float, int]] = {}
        for f in files:
            calls.update(parse_forced_vcf_calls(f))
        cluster_to_calls[cluster] = calls

    # Presence, per cluster: VAF and ALT-read thresholds applied directly to the
    # force-called genotype, not the FILTER column. This same call set backs
    # the presence matrix and the 96-channel spectra below.
    cluster_to_snvs = resolve_presence_calls(
        cluster_to_calls, args.presence_min_vaf, args.presence_min_alt_reads
    )

    snv_matrix = build_snv_presence_matrix(cluster_to_snvs)
    snv_matrix.to_csv(args.out_dir / "clone_snv_matrix.csv")

    mutation_sets = mutation_sets_from_matrix(snv_matrix)
    n_violations = three_gamete_violations(mutation_sets)

    method_used = "camin_sokal"
    lichee_error: Optional[str] = None
    camin_sokal_stats: Optional[Tuple[int, int]] = None
    primary_tree: Optional[nx.DiGraph] = None

    if not args.skip_lichee:
        lichee_bin = find_lichee_binary(args.lichee_bin)
        lichee_jar_ok = os.access(f"{lichee_bin}.jar", os.F_OK)
        if not os.access(lichee_bin, os.X_OK) and not lichee_jar_ok:
            lichee_error = (
                f"LICHeE launcher not found or not executable: {lichee_bin!r}"
            )
        else:
            try:
                lichee_input_path = args.out_dir / "lichee_input.txt"
                column_order = write_lichee_input(cluster_to_calls, lichee_input_path)
                out_prefix = args.out_dir / "lichee_out"
                log_path = args.out_dir / "lichee_run.log"
                dot_path = run_lichee(
                    lichee_input_path,
                    out_prefix=out_prefix,
                    log_path=log_path,
                    lichee_bin=lichee_bin,
                    normal_index=0,
                    min_vaf_present=args.presence_min_vaf,
                    max_vaf_absent=args.lichee_max_vaf_absent,
                    min_cluster_size=args.lichee_min_cluster_size,
                )
                lichee_tree, _source = resolve_lichee_clone_tree(
                    dot_path, column_order[1:], GERMLINE_ROOT_ID
                )
                primary_tree = lichee_tree
                method_used = "lichee"
            except (
                subprocess.CalledProcessError,
                FileNotFoundError,
                ValueError,
            ) as exc:
                lichee_error = str(exc)

    if primary_tree is None:
        primary_tree, n_gains, n_reversals = camin_sokal_tree(
            mutation_sets, GERMLINE_ROOT_ID, max_clusters=args.camin_sokal_max_clusters
        )
        camin_sokal_stats = (n_gains, n_reversals)

    newick_str = digraph_to_newick(primary_tree, GERMLINE_ROOT_ID)
    verify_newick(newick_str, set(cluster_to_snvs), GERMLINE_ROOT_ID)
    (args.out_dir / "snv_tree.nwk").write_text(newick_str + "\n")

    import pysam

    with pysam.FastaFile(str(args.ref_fasta)) as fasta:
        spectra, skip_counts = bin_cluster_spectra(cluster_to_snvs, fasta)
    spectra.to_csv(args.out_dir / "spectra.csv")

    write_diagnostics(
        args.out_dir / "snv_tree_diagnostics.txt",
        mutation_sets,
        skip_counts,
        n_violations,
        method_used,
        lichee_error,
        camin_sokal_stats,
        primary_tree,
    )

    print(
        "Tumour clusters (tree tips + internal observed clones): "
        f"{len(cluster_to_snvs)}"
    )
    print(f"Tree method used: {method_used}")
    print("Somatic SNV count per cluster:")
    for cid in sorted(cluster_to_snvs, key=_sort_key):
        print(f"  clone{cid}: {len(cluster_to_snvs[cid])}")


if __name__ == "__main__":
    main()
