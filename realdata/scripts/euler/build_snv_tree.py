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
semantically with this tree's own latent root. The clusters are the remaining
columns, headed ``c<cluster id>``, which LICHeE echoes back verbatim.

LICHeE is run as ``java -cp <LICHEE_HOME>/release/lichee.jar:<LICHEE_HOME>/lib/*
lineage.LineageEngine -build ... -s 1 -o <out_dir>/lichee_out.trees.txt`` (the
``release/lichee`` launcher cannot find its dependencies on JDK 11; ``lib/*`` is
one classpath entry that Java expands itself). ``-dot`` is never used: it needs
a display and throws HeadlessException on compute nodes, and the ``.trees.txt``
carries the whole topology. Its exit status is unreliable, so success is that
file existing. ``--presence-min-vaf`` (tau) is passed as both ``-minVAFPresent``
and ``-maxVAFAbsent``, one cutoff for "absent" everywhere, so sub-threshold
mixture leakage cannot manufacture ties; LICHeE's default clustering flags are
used as they are.

``parse_lichee_trees`` reads the ``.trees.txt``'s three blocks: ``Nodes:`` (a
presence profile over the input columns, left to right, bit 0 the germline, and
the VAFs of the present columns), ``****Tree 0****`` (parent -> child edges over
node ids, node 0 the germline root) and ``Sample decomposition:``. A cluster is
"in" a node when its profile bit is set and its VAF there exceeds tau, and it
attaches to the deepest such node (the lowest common ancestor of a genuine tie),
cross-checked against its deepest decomposition line.

LICHeE clusters SSNVs by cross-sample presence pattern and builds a tree over
those mutation-presence GROUPS, not one node per sample, so several SECEDO
clusters routinely share a node and the tree can have fewer nodes than there
are clusters. This is intrinsic to LICHeE, not a tuning problem, and is treated
as the normal case. ``build_lichee_clone_tree`` keeps LICHeE's own topology as
the skeleton and hangs the clusters on it, one labelled leaf per cluster: a node
holding one cluster is labelled by it (an internal node if it has descendants);
a node holding several becomes a hidden group node ``g<id>`` with those clusters
as sibling leaves (an unresolved polytomy) and its descendants beneath it; a
node holding none is collapsed. The labelled set is thus exactly the SECEDO
clusters plus the germline root, plus the hidden ``g<id>`` nodes, which have no
spectra row and which the model treats as latent (each adds one random-walk step
for the clusters below it, so they are named in the diagnostics).
``spectra.csv`` is unaffected: one row per cluster from that cluster's own
SNVs, whatever its position in the tree.

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
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
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


def verify_newick(
    newick_str: str,
    cluster_ids: Set[str],
    normal_id: str,
    hidden_ids: Optional[Set[str]] = None,
) -> None:
    """Round-trip check: parse the way the model does and confirm the contract holds.

    ``normal_id`` is the tree-root label (``GERMLINE_ROOT_ID`` in ``main``),
    not a real cluster. ``hidden_ids`` are the hidden group nodes LICHeE's
    shared nodes need (``g<id>``): extra labelled nodes with no spectra row,
    which the model treats as latent, like the root. They must not collide with
    a cluster ID. Raises AssertionError if the root is not the unique root, or
    if the parsed node labels are not exactly the cluster IDs, the root and the
    hidden IDs, each appearing once.
    """
    hidden = set(hidden_ids or ())
    clash = hidden & (set(cluster_ids) | {normal_id})
    if clash:
        raise AssertionError(
            f"hidden node IDs collide with real labels: {sorted(clash)}"
        )
    graph = parse_newick_like_model(newick_str)
    expected = set(cluster_ids) | {normal_id} | hidden
    labels = list(graph.nodes())
    if len(labels) != len(expected):
        raise AssertionError(
            f"parsed {len(labels)} node labels, expected {len(expected)} "
            f"(cluster IDs plus the germline root and any hidden group nodes); "
            f"a label collision or missing cluster is likely: {sorted(labels)} "
            f"vs {sorted(expected)}"
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


LICHEE_MAIN_CLASS = "lineage.LineageEngine"
LICHEE_OUT_NAME = "lichee_out.trees.txt"
_CLUSTER_COLUMN_PREFIX = "c"
_GROUP_NODE_PREFIX = "g"
_DECOMPOSITION_DOTS_PER_LEVEL = 5


def resolve_lichee_home(explicit: Optional[str] = None) -> Tuple[Path, Path]:
    """Locate LICHeE and return ``(jar, lib_dir)``: ``explicit`` (the
    ``--lichee-home`` CLI arg), else ``$LICHEE_HOME``, else the repo's own
    checkout at ``realdata/external/lichee/LICHeE``.

    The ``release/lichee`` launcher is not used (it cannot find its
    dependencies on JDK 11), so what is needed is ``release/lichee.jar``, the
    ``lib/`` directory and ``java`` on ``PATH``. Raises FileNotFoundError,
    naming every one that is missing.
    """
    home = Path(
        explicit
        or os.environ.get("LICHEE_HOME")
        or Path(__file__).resolve().parents[2] / "external" / "lichee" / "LICHeE"
    )
    jar = home / "release" / "lichee.jar"
    lib = home / "lib"
    missing = []
    if not jar.is_file():
        missing.append(f"jar {jar}")
    if not lib.is_dir():
        missing.append(f"lib directory {lib}")
    if shutil.which("java") is None:
        missing.append("java on PATH")
    if missing:
        raise FileNotFoundError(
            f"LICHeE home {home} is unusable: missing {'; '.join(missing)}"
        )
    return jar, lib


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
    sort, headed ``c<cluster id>`` (LICHeE echoes a sample's header verbatim in
    its output, so ``parse_lichee_trees`` joins on it after stripping the
    ``c``). A synthetic all-zero ``germline_label`` column is prepended as
    LICHeE's required baseline/normal sample (``-n 0``) -- see module
    docstring for why this pipeline has no real one.

    Returns the column headers actually written (germline first, then the
    ``c<id>`` clusters), for ``parse_lichee_trees``.
    """
    clusters = sorted(cluster_to_calls, key=_sort_key)
    columns = [germline_label] + [f"{_CLUSTER_COLUMN_PREFIX}{c}" for c in clusters]
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
    out_path: Path,
    log_path: Path,
    jar: Path,
    lib: Path,
    tau: float,
    normal_index: int = 0,
    log_header: str = "",
) -> Path:
    """Run LICHeE's ``build`` step and return ``out_path``, its ``.trees.txt``.

    Invoked as ``java -cp <jar>:<lib>/* lineage.LineageEngine``: the ``lib/*``
    is ONE classpath entry that Java expands itself, so it is passed unglobbed.
    ``-s 1`` is passed and ``-dot`` never is (it needs a display and throws
    HeadlessException on compute nodes); the topology is read from the
    ``.trees.txt`` instead. ``tau`` is both ``-minVAFPresent`` and
    ``-maxVAFAbsent``: one cutoff for "absent", so sub-threshold mixture
    leakage cannot manufacture a band of uncertain calls. LICHeE's exit status
    is not reliable, so it is ignored: success is ``out_path`` existing
    afterwards (any stale copy is removed first). Raises FileNotFoundError,
    naming the file and the log, if it does not.
    """
    out_path = Path(out_path)
    out_path.unlink(missing_ok=True)
    cmd = [
        "java",
        "-cp", f"{jar}:{lib}/*",
        LICHEE_MAIN_CLASS,
        "-build",
        "-i", str(input_path),
        "-n", str(normal_index),
        "-s", "1",
        "-o", str(out_path),
        "-minVAFPresent", str(tau),
        "-maxVAFAbsent", str(tau),
    ]  # fmt: skip
    with open(log_path, "w") as log:
        if log_header:
            log.write(log_header)
        log.write("command: " + " ".join(cmd) + "\n\n")
        log.flush()
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)

    if not out_path.exists():
        raise FileNotFoundError(
            f"LICHeE wrote no {out_path}; see {log_path} for its output"
        )
    return out_path


@dataclass
class LicheeTrees:
    """The three blocks of a LICHeE ``.trees.txt``, as parsed."""

    profiles: Dict[str, str]  # node id -> presence profile over the columns
    vafs: Dict[str, List[float]]  # node id -> VAFs of its PRESENT columns, in order
    parent_of: Dict[str, str]  # Tree 0's {child: parent}, root excluded
    root: str  # Tree 0's root node id
    n_trees: int
    # sample name -> [(depth, profile, vaf)]; empty for a GL-only sample
    decomposition: Dict[str, List[Tuple[int, str, float]]]


_TREE_HEADER_RE = re.compile(r"^\*{4}Tree (\d+)\*{4}$")
_TREE_EDGE_RE = re.compile(r"^(\S+)\s*->\s*(\S+)$")
_DECOMP_HEADER_RE = re.compile(r"^Sample lineage decomposition:\s*(\S+)$")
_DECOMP_LINE_RE = re.compile(r"^(\.*)([01]+):\s*(\S+)\s*\[[^\]]*\]$")


def parse_lichee_trees(path: Path, column_order: List[str]) -> LicheeTrees:
    """Parse LICHeE's ``.trees.txt``: the ``Nodes:`` block, ``****Tree 0****``
    and ``Sample decomposition:``.

    ``column_order`` is the input header (germline first, then the ``c<id>``
    clusters); a node's profile is over those columns left to right, bit 0
    being the germline. A node's VAF vector covers only its PRESENT columns.
    Tree 0's edges come in no particular order, so they are read into a
    parent map. Only Tree 0 is used; ``n_trees`` says how many were written.
    Raises ValueError, naming what was expected, for a missing block, a
    profile of the wrong length or with the germline bit set, a VAF vector
    that does not match its profile, or a tree without a unique root ``0``.
    """
    profiles: Dict[str, str] = {}
    vafs: Dict[str, List[float]] = {}
    edges: List[Tuple[str, str]] = []
    decomposition: Dict[str, List[Tuple[int, str, float]]] = {}
    n_trees = 0
    section = None
    sample: Optional[str] = None

    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "Nodes:":
            section = "nodes"
            continue
        tree_header = _TREE_HEADER_RE.match(line)
        if tree_header:
            n_trees += 1
            section = "tree0" if tree_header.group(1) == "0" else "other_tree"
            continue
        if line.startswith("Sample decomposition"):
            section = "decomposition"
            continue
        if line.startswith("SNV info"):
            section = "ignored"
            continue

        if section == "nodes":
            fields = line.split("\t")
            if len(fields) < 3:
                raise ValueError(f"Nodes: line has fewer than 3 tab fields: {raw!r}")
            node, profile, vaf_text = (f.strip() for f in fields[:3])
            try:
                vaf_values = [float(v) for v in vaf_text.strip("[]").split()]
            except ValueError as e:
                raise ValueError(f"unreadable VAF vector in {raw!r}") from e
            if len(profile) != len(column_order):
                raise ValueError(
                    f"node {node} profile {profile!r} has {len(profile)} bits, "
                    f"expected {len(column_order)} (columns {column_order})"
                )
            if profile[0] != "0":
                raise ValueError(
                    f"node {node} profile {profile!r} has the germline bit set"
                )
            if len(vaf_values) != profile.count("1"):
                raise ValueError(
                    f"node {node}: {len(vaf_values)} VAFs for {profile.count('1')} "
                    f"present columns in profile {profile!r}"
                )
            profiles[node] = profile
            vafs[node] = vaf_values
        elif section == "tree0":
            edge = _TREE_EDGE_RE.match(line)
            if edge:
                edges.append((edge.group(1), edge.group(2)))
        elif section == "decomposition":
            header = _DECOMP_HEADER_RE.match(line)
            if header:
                sample = header.group(1)
                decomposition[sample] = []
            elif line == "GL":
                continue
            else:
                m = _DECOMP_LINE_RE.match(raw.rstrip())
                if m is None or sample is None:
                    raise ValueError(f"unreadable decomposition line: {raw!r}")
                dots = len(m.group(1))
                if dots == 0 or dots % _DECOMPOSITION_DOTS_PER_LEVEL:
                    raise ValueError(
                        f"decomposition line {raw!r} is indented {dots} dots, not a "
                        f"multiple of {_DECOMPOSITION_DOTS_PER_LEVEL}"
                    )
                decomposition[sample].append(
                    (
                        dots // _DECOMPOSITION_DOTS_PER_LEVEL,
                        m.group(2),
                        float(m.group(3)),
                    )
                )

    if not profiles:
        raise ValueError(f"{path} has no 'Nodes:' block with any nodes")
    if not edges:
        raise ValueError(f"{path} has no '****Tree 0****' block with any edges")
    if not decomposition:
        raise ValueError(f"{path} has no 'Sample decomposition:' block")

    parent_of: Dict[str, str] = {}
    for parent, child in edges:
        if child in parent_of:
            raise ValueError(f"Tree 0 gives node {child} two parents")
        parent_of[child] = parent
    roots = sorted({p for p in parent_of.values() if p not in parent_of})
    if roots != ["0"]:
        raise ValueError(
            f"Tree 0 must have the germline node 0 as its one root; roots found: "
            f"{roots}"
        )
    return LicheeTrees(profiles, vafs, parent_of, "0", n_trees, decomposition)


@dataclass
class LicheeResult:
    """The clone tree plus everything the diagnostics report."""

    tree: nx.DiGraph
    hidden_ids: Set[str]
    node_of_cluster: Dict[str, str]  # cluster -> LICHeE node it resolved to
    shared: List[List[str]]  # clusters sharing one node, one list per node
    absent: List[str]  # in no node: attached to the germline root
    at_root: List[str]  # tied only at the germline root: attached to it
    collapsed_nodes: List[str]  # LICHeE nodes carrying no cluster
    group_subtends: Dict[str, Tuple[int, int]]  # g<v> -> (direct, in subtree)
    n_trees: int
    n_nodes: int  # LICHeE nodes in Tree 0, germline excluded


def _lca(nodes: List[str], parent_of: Dict[str, str], root: str) -> str:
    def chain(n: str) -> List[str]:
        out = [n]
        while out[-1] != root:
            if out[-1] not in parent_of:
                raise ValueError(f"node {n} does not reach the root {root} in Tree 0")
            out.append(parent_of[out[-1]])
        return out

    chains = [chain(n) for n in nodes]
    common = set(chains[0]).intersection(*chains[1:])
    if not common:
        raise ValueError(f"nodes {nodes} share no ancestor in Tree 0")
    return next(a for a in chains[0] if a in common)  # nearest to the nodes


def build_lichee_clone_tree(
    trees: LicheeTrees,
    column_order: List[str],
    tau: float,
    germline_id: str = GERMLINE_ROOT_ID,
) -> LicheeResult:
    """Build the SECEDO-cluster clone tree from a parsed ``.trees.txt``.

    Join. A cluster is "in" a node when the node's profile has the cluster's
    column bit set AND its entry in the node's VAF vector (matched by the bit's
    rank among the set bits) exceeds ``tau``. It attaches to the DEEPEST such
    node; if several tie for deepest, to their lowest common ancestor. That is
    cross-checked against the cluster's deepest decomposition line above
    ``tau`` (the same set of nodes must come out) and a disagreement raises.

    Attachment. ``C(v)`` is the set of clusters resolving to node ``v``.
    One cluster: it labels ``v`` (a leaf if ``v`` has no children, else the
    internal node its child subtrees hang under). Several: a hidden group node
    ``g<v>`` stands in for ``v``, the clusters are its sibling leaves and
    ``v``'s child subtrees hang under it. None: ``v`` is collapsed and its
    children lift to the nearest emitted ancestor. A cluster in no node (its
    decomposition is GL-only) or tied only at the germline root hangs directly
    off the germline root. Several clusters sharing a node is the normal case,
    not an error: LICHeE groups SNVs by presence pattern, so it emits fewer
    nodes than there are clusters.
    """
    if column_order[0] != germline_id:
        raise ValueError(f"column 0 is {column_order[0]!r}, expected {germline_id!r}")
    cluster_of_column: Dict[str, str] = {}
    for column in column_order[1:]:
        if not column.startswith(_CLUSTER_COLUMN_PREFIX) or len(column) == 1:
            raise ValueError(f"cluster column {column!r} is not c<id>")
        cluster_of_column[column] = column[len(_CLUSTER_COLUMN_PREFIX) :]
    clusters = list(cluster_of_column.values())

    for name in trees.decomposition:
        if name != germline_id and name not in cluster_of_column:
            raise ValueError(
                f"decomposition sample {name!r} is neither {germline_id!r} nor a "
                f"c<id> column; known clusters {sorted(clusters, key=_sort_key)}"
            )

    parent_of, root = trees.parent_of, trees.root
    children_of: Dict[str, List[str]] = defaultdict(list)
    for child, parent in parent_of.items():
        children_of[parent].append(child)
    depth: Dict[str, int] = {root: 0}

    for start in parent_of:
        path, node = [], start
        while node not in depth:
            if node in path or node not in parent_of:
                raise ValueError(f"Tree 0 has a cycle or a broken chain at node {node}")
            path.append(node)
            node = parent_of[node]
        for step in reversed(path):
            depth[step] = depth[parent_of[step]] + 1

    tree_nodes = [n for n in parent_of if n in trees.profiles]
    profile_to_node: Dict[str, str] = {}
    for node in tree_nodes:
        if trees.profiles[node] in profile_to_node:
            raise ValueError(
                f"nodes {profile_to_node[trees.profiles[node]]} and {node} share "
                f"profile {trees.profiles[node]!r}; the join would be ambiguous"
            )
        profile_to_node[trees.profiles[node]] = node

    node_of_cluster: Dict[str, str] = {}
    absent: List[str] = []
    at_root: List[str] = []
    for j, column in enumerate(column_order):
        if j == 0:
            continue
        cid = cluster_of_column[column]
        in_nodes = []
        for node in tree_nodes:
            profile = trees.profiles[node]
            if profile[j] == "1":
                rank = profile[:j].count("1")
                if trees.vafs[node][rank] > tau:
                    in_nodes.append(node)

        if column not in trees.decomposition:
            raise ValueError(f"no 'Sample lineage decomposition: {column}' block")
        lines = [ln for ln in trees.decomposition[column] if ln[2] > tau]
        decomp_nodes: Set[str] = set()
        if lines:
            deepest_level = max(ln[0] for ln in lines)
            for _, profile, _ in (ln for ln in lines if ln[0] == deepest_level):
                if profile not in profile_to_node:
                    raise ValueError(
                        f"{column}: deepest decomposition profile {profile!r} matches "
                        f"no Nodes: entry (profiles {sorted(profile_to_node)})"
                    )
                decomp_nodes.add(profile_to_node[profile])

        if not in_nodes:
            if decomp_nodes:
                raise ValueError(
                    f"{column} is in no node by presence (bit set and VAF > {tau}) "
                    f"but its decomposition places it at node(s) {sorted(decomp_nodes)}"
                )
            absent.append(cid)
            continue
        top = max(depth[n] for n in in_nodes)
        deepest = {n for n in in_nodes if depth[n] == top}
        if deepest != decomp_nodes:
            raise ValueError(
                f"{column}: presence puts it at node(s) {sorted(deepest)} but its "
                f"deepest decomposition line puts it at {sorted(decomp_nodes)}"
            )
        node = (
            next(iter(deepest))
            if len(deepest) == 1
            else _lca(sorted(deepest), parent_of, root)
        )
        if node == root:
            at_root.append(cid)
        else:
            node_of_cluster[cid] = node

    hidden_ids: Set[str] = set()
    for node in {n for n in node_of_cluster.values()}:
        group = f"{_GROUP_NODE_PREFIX}{node}"
        if group in clusters or group == germline_id:
            raise ValueError(f"group node label {group!r} collides with a real label")

    by_node: Dict[str, List[str]] = defaultdict(list)
    for cid, node in node_of_cluster.items():
        by_node[node].append(cid)
    for cids in by_node.values():
        cids.sort(key=_sort_key)

    tree = nx.DiGraph()
    tree.add_node(germline_id)
    collapsed: List[str] = []
    direct_count: Dict[str, int] = {}

    def place(node: str, attach_to: str) -> None:
        here = by_node.get(node, [])
        kids = sorted(children_of.get(node, []), key=_sort_key)
        if not here:
            collapsed.append(node)
            for kid in kids:
                place(kid, attach_to)
        elif len(here) == 1:
            tree.add_edge(attach_to, here[0])
            for kid in kids:
                place(kid, here[0])
        else:
            group = f"{_GROUP_NODE_PREFIX}{node}"
            hidden_ids.add(group)
            direct_count[group] = len(here)
            tree.add_edge(attach_to, group)
            for cid in here:
                tree.add_edge(group, cid)
            for kid in kids:
                place(kid, group)

    for kid in sorted(children_of.get(root, []), key=_sort_key):
        place(kid, germline_id)
    for cid in sorted(absent + at_root, key=_sort_key):
        tree.add_edge(germline_id, cid)

    placed = set(tree.nodes()) - hidden_ids - {germline_id}
    if placed != set(clusters):
        raise ValueError(
            f"emitted clusters {sorted(placed)} != input clusters {sorted(clusters)}"
        )
    subtends = {
        g: (direct_count[g], len(nx.descendants(tree, g) - hidden_ids))
        for g in hidden_ids
    }
    return LicheeResult(
        tree=tree,
        hidden_ids=hidden_ids,
        node_of_cluster=node_of_cluster,
        shared=[cids for _, cids in sorted(by_node.items()) if len(cids) > 1],
        absent=sorted(absent, key=_sort_key),
        at_root=sorted(at_root, key=_sort_key),
        collapsed_nodes=sorted(collapsed, key=_sort_key),
        group_subtends=subtends,
        n_trees=trees.n_trees,
        n_nodes=len(parent_of),
    )


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
    case in this pipeline (SCICoNE's cell tree in ``build_cna_tree.py``) --
    the same idiom this module used for SCITE's mutation tree before SCITE
    was retired.
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


def _lichee_lines(r: LicheeResult) -> List[str]:
    """Findings from a LICHeE run: node/cluster counts, shared nodes, absent
    clusters and the hidden group nodes (which add a random-walk step for the
    clusters below them in TreeHDP, so are named here to be auditable against
    the CNA-tree comparison)."""
    n_clusters = len(set(r.tree.nodes()) - r.hidden_ids - {GERMLINE_ROOT_ID})
    lines = [
        f"LICHeE Tree 0: {r.n_nodes} nodes carrying {n_clusters} clusters "
        f"({len(set(r.node_of_cluster.values()))} distinct nodes hold a cluster). "
        "Fewer nodes than clusters is expected: LICHeE groups SNVs by "
        "presence pattern, so it cannot tell some clusters apart.\n"
    ]
    if r.n_trees > 1:
        lines.append(
            f"FINDING: LICHeE emitted {r.n_trees} trees; Tree 0 (top-ranked) used.\n"
        )
    for cids in r.shared:
        lines.append(
            f"FINDING: clusters {','.join(cids)} indistinguishable by SNV profile, "
            "attached as siblings.\n"
        )
    if r.absent:
        lines.append(
            f"FINDING: cluster(s) {','.join(r.absent)} are in no LICHeE node (no SNV "
            f"above the VAF cutoff); attached as direct leaves of {GERMLINE_ROOT_ID}.\n"
        )
    if r.at_root:
        lines.append(
            f"FINDING: cluster(s) {','.join(r.at_root)} tie only at the germline "
            f"root; attached as direct leaves of {GERMLINE_ROOT_ID}.\n"
        )
    if r.collapsed_nodes:
        lines.append(
            f"LICHeE node(s) {','.join(r.collapsed_nodes)} carry no cluster and "
            "were collapsed.\n"
        )
    for group, (direct, below) in sorted(r.group_subtends.items()):
        lines.append(
            f"Hidden group node {group}: {direct} clusters directly, {below} in its "
            "subtree. It adds one random-walk step for them in TreeHDP.\n"
        )
    return lines


def write_diagnostics(
    path: Path,
    mutation_sets: Dict[str, Set[Hashable]],
    skip_counts: Dict[str, int],
    n_violations: int,
    method_used: str,
    lichee_error: Optional[str],
    camin_sokal_stats: Optional[Tuple[int, int]],
    primary_tree: nx.DiGraph,
    lichee_result: Optional[LicheeResult] = None,
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
        if lichee_result is not None:
            lines.extend(_lichee_lines(lichee_result))
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
        "--lichee-home",
        default=None,
        help="LICHeE checkout (release/lichee.jar and lib/); defaults to "
        "$LICHEE_HOME, then realdata/external/lichee/LICHeE",
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

    lichee_result: Optional[LicheeResult] = None
    if not args.skip_lichee:
        try:
            jar, lib = resolve_lichee_home(args.lichee_home)
            lichee_input_path = args.out_dir / "lichee_input.txt"
            column_order = write_lichee_input(cluster_to_calls, lichee_input_path)
            trees_path = run_lichee(
                lichee_input_path,
                out_path=args.out_dir / LICHEE_OUT_NAME,
                log_path=args.out_dir / "lichee_run.log",
                jar=jar,
                lib=lib,
                tau=args.presence_min_vaf,
            )
            lichee_result = build_lichee_clone_tree(
                parse_lichee_trees(trees_path, column_order),
                column_order,
                args.presence_min_vaf,
            )
            primary_tree = lichee_result.tree
            method_used = "lichee"
        except (OSError, ValueError) as exc:
            lichee_error = str(exc)

    if primary_tree is None:
        primary_tree, n_gains, n_reversals = camin_sokal_tree(
            mutation_sets, GERMLINE_ROOT_ID, max_clusters=args.camin_sokal_max_clusters
        )
        camin_sokal_stats = (n_gains, n_reversals)

    newick_str = digraph_to_newick(primary_tree, GERMLINE_ROOT_ID)
    verify_newick(
        newick_str,
        set(cluster_to_snvs),
        GERMLINE_ROOT_ID,
        hidden_ids=lichee_result.hidden_ids if lichee_result else None,
    )
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
        lichee_result,
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
