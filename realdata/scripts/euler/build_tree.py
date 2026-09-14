"""Turn stage 07's per-cluster somatic VCFs into the two inputs TreeHDP consumes:
a rooted Newick tree and a per-cluster 96-channel spectra matrix.

Ours: neither secedo nor secedo-evaluation builds a phylogeny over SECEDO's
clusters or bins VCFs into COSMIC's 96 SBS channels. This is that step (recipe
stage 8, `realdata/recipe/breast_cancer_plan.md`).

Contract (from `src/models/hdp_inference.py`'s `_BaseTreeHDP`, do not "fix" these)
    - Newick is labelled-internal-node form, e.g. ``((c2,c3)c1)normal;``. Observed
      ancestral clones sit at internal nodes carrying their cluster-ID label; the
      ancestor-as-tip idiom is not used.
    - The tree is rooted at ``GERMLINE_ROOT_ID``, an implicit, spectrum-less latent
      node standing for the germline/empty-mutation state -- not a pseudo-normal
      cluster (see below). It carries no spectrum and is never a column in the
      presence matrix or the spectra table, which the model handles natively.
    - Node labels in the Newick, the spectra index, and the SECEDO cluster IDs are one
      ID system throughout.
    - The 96 spectra columns are in cosmic_signatures.csv's exact channel order,
      because the model does ``dot(activities, signatures)`` and aligns observed
      counts to signature columns positionally. ``main`` asserts this before
      doing anything else and fails loudly if it does not hold.

Calling design: tumour-only + gnomAD germline-resource, not tumour-vs-pseudo-normal.
Every SECEDO cluster is a tumour cluster now; there is no cluster standing in for the
matched normal (stage 06's old design), because a sibling cluster is not guaranteed
diploid/background and that design made 85% of SNVs cluster-private -- unable to support
a tree. Absolute somatic status from gnomAD subtraction alone still does not make six
independently-assembled clusters' calls consistent with each other, so stage 06b
force-calls every cluster at the union of every cluster's pass-1 discovery sites, per
chromosome; this script reads that force-called output (``*.forced.vcf``), not the
pass-1 discovery VCFs directly. A site counts as PRESENT in a cluster if its
force-called record has VAF >= ``--presence-min-vaf`` and ALT read depth >=
``--presence-min-alt-reads`` (defaults from config.sh's PRESENCE_MIN_VAF/
PRESENCE_MIN_ALT_READS); ABSENT otherwise. All six clusters carry a 96-channel
spectrum (binned from their own present sites) and a column in the presence matrix --
none is held back as a pseudo-normal.

Tree construction is via SCITE (Jahn et al.), the intended method for this
pipeline: SCITE samples a mutation history from the clone x SNV presence
matrix over all six clusters (no extra reference column -- SCITE roots its
own mutation tree; there is no pseudo-normal cluster to give it a dedicated
attachment point) and writes its MAP mutation tree as Newick. Not every SCITE
build puts sample identity in that newick, though -- on the real differential
calls it held only mutation indices, with sample attachments recorded in
SCITE's other output instead -- so ``resolve_scite_clone_tree`` tries, in
order, the newick's own leaf labels, a companion ``.gv`` file (SCITE's
classic '-a' integer node numbering), and a companion ``.samples`` file,
reporting whichever one actually resolved every cluster. However attachments
are found, the clone tree is collapsed the same way: each tumour clone's
parent is the nearest sample-leaf ancestor in SCITE's tree (walking up
through the unlabelled mutation nodes), falling back to the germline root
when no such ancestor exists before the top of SCITE's tree. The clone tree
is always re-rooted at the germline root explicitly -- SCITE's own mutation
tree has no node standing for it at all, so nothing about where SCITE rooted
its own tree is ever consulted -- so the result is always a single tree
rooted at ``GERMLINE_ROOT_ID``.

SCITE's mutation tree on the real differential calls was also a single
unbranched chain of ~1434 nodes, which overflowed Python's default recursion
limit when parsed (phylox's Newick parser recurses once per nesting level).
``parse_scite_newick`` raises the limit before parsing, and every tree walk
this module does over SCITE's output (attachment resolution, degeneracy
classification) is an explicit loop, never recursion, so tree depth cannot
crash the process.

SCITE is required, not optional, but a degenerate result is a documented
finding, not a bug: if the binary never produces a parseable mutation tree at
all, ``main`` exits non-zero rather than emit the accumulation tree below as
tree.nwk (``--allow-containment-fallback`` overrides this for debugging only).
If SCITE does parse but the induced clone tree is an unbranched chain, its raw
mutation tree is itself a chain, or sample attachments cannot be resolved from
any output, that is classified as degenerate: whatever tree can be formed
(SCITE's, chain-shaped, or the containment tree if attachments were wholly
unresolvable) is still written to tree.nwk, tree_diagnostics.txt records the
finding plainly, and ``main`` exits non-zero regardless, pointing at those
diagnostics -- unless ``--allow-degenerate-tree`` is passed, for downstream
plumbing tests that need exit 0. spectra.csv is written either way: the
spectra are valid independent of whether the tree resolved.

Accumulation by mutation-set containment (below) is a cross-check, not the
primary construction: under the old tumour-vs-pseudo-normal design it produced
581k three-gamete violations, 0.27-0.36 edge containment, and an implausible
linear chain on the real differential calls -- untrustworthy as the emitted
tree. Those numbers are from that design and not necessarily representative
of the force-called, tumour-only + gnomAD calls this script now reads; the
containment cross-check and its diagnostics stay regardless, since a
consistently-genotyped presence matrix can still fail the perfect-phylogeny
test for genuine biological reasons (parallel/convergent mutation, allelic
dropout), and that is exactly what this cross-check is for. It is still built
every run and reported in tree_diagnostics.txt, alongside its topology
agreement with SCITE's tree, as a sanity signal.

Tree construction by mutation-set containment (cross-check only, see above): a
total, always-succeeding perfect-phylogeny approximation, not an error-aware
caller. Process tumour clones from fewest to most mutations; each clone's parent
is the already-placed node (root included, with the empty set) maximising shared
mutations, tied-broken by fewest parent-only mutations, then fewest parent
mutations, then cluster ID. This puts observed clones at internal nodes wherever
containment holds and degrades to a star under the root when it does not.

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
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, Hashable, Iterable, List, Optional, Set, Tuple

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
# presence matrix or spectra table) -- there is no pseudo-normal under
# tumour-only + gnomAD calling, so this plays the same structural role the
# pseudo-normal cluster ID used to (the tree's unique root, the "parent"
# containment measures every founder mutation against) without being an
# actual SECEDO cluster.
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
    force-call output) by cluster. No cluster is excluded: tumour-only +
    gnomAD calling has no pseudo-normal cluster to leave out.
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


# --------------------------------------------------------------------------- #
# Tree construction by mutation-set accumulation
# --------------------------------------------------------------------------- #


def build_clone_tree(
    mutation_sets: Dict[str, Set[Hashable]], normal_id: str
) -> nx.DiGraph:
    """Perfect-phylogeny-by-containment clone tree, total over noisy calls.

    Root is the germline state (the empty mutation set, ``normal_id`` here
    being the tree-root label, not a real cluster). Tumour clones are placed in
    ascending order of mutation count; each clone's parent is the already-placed
    node maximising shared mutations, tie-broken by fewest parent-only
    mutations, then fewest parent mutations, then cluster ID. The root is
    always a candidate (intersection 0 with the empty set), so placement never
    fails.
    """
    tree = nx.DiGraph()
    tree.add_node(normal_id)
    placed: Dict[Hashable, FrozenSet] = {normal_id: frozenset()}

    tumour_ids = sorted(
        (cid for cid in mutation_sets if cid != normal_id),
        key=lambda cid: (len(mutation_sets[cid]), _sort_key(cid)),
    )
    for cid in tumour_ids:
        muts = frozenset(mutation_sets[cid])
        best_parent = None
        best_key = None
        for parent, parent_muts in placed.items():
            intersection = len(parent_muts & muts)
            key = (
                -intersection,
                len(parent_muts - muts),
                len(parent_muts),
                _sort_key(parent),
            )
            if best_key is None or key < best_key:
                best_key, best_parent = key, parent
        tree.add_node(cid)
        tree.add_edge(best_parent, cid)
        placed[cid] = muts
    return tree


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
# Diagnostics
# --------------------------------------------------------------------------- #


def containment_fraction(
    mutation_sets: Dict[str, Set[Hashable]], parent: Hashable, child: Hashable
) -> Optional[float]:
    """|muts(parent) ∩ muts(child)| / |muts(parent)|, or None if parent has no
    mutations.

    ``parent`` may be the germline root, which is never a key in
    ``mutation_sets`` (it is not a real cluster); treated as the empty set.
    """
    parent_muts = mutation_sets.get(parent, set())
    if not parent_muts:
        return None
    return len(parent_muts & mutation_sets[child]) / len(parent_muts)


def three_gamete_violations(mutation_sets: Dict[str, Set[Hashable]]) -> int:
    """Count SNV pairs violating the perfect-phylogeny (three-gamete) test.

    Two SNVs are incompatible with a single mutation history if, among the
    clusters, all three of "only i", "only j", and "both" occur (the fourth
    gamete, "neither", is irrelevant to infinite-sites compatibility). Each
    SNV's presence across clusters is bit-packed into one integer so a pair
    check is O(1); overall cost is O(n_snvs^2), fine at the SNV counts this
    pipeline expects (sparse tens-to-hundreds of mutations per cluster).
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


def write_diagnostics(
    path: Path,
    containment_tree: nx.DiGraph,
    mutation_sets: Dict[str, Set[Hashable]],
    skip_counts: Dict[str, int],
    n_violations: int,
    scite_error: Optional[str],
    topology_agreement: Optional[float],
    filter_stats: Optional[Dict[str, int]] = None,
    scite_tree: Optional[nx.DiGraph] = None,
    scite_clone_tree: Optional[nx.DiGraph] = None,
    attachment_source: Optional[str] = None,
    attachment_error: Optional[str] = None,
    optimal_fraction: Optional[float] = None,
) -> None:
    """SCITE is primary (tree.nwk) when it yields a trustworthy, branching
    clone tree; mutation-set containment is always reported as the
    cross-check. Numbers only, plus the one plain-language finding line the
    degenerate case calls for (see the module docstring) -- everything else
    stays a number, not a verdict.

    ``scite_tree`` is SCITE's raw mutation newick (set once it parses, even
    if attachments could not be resolved from it); ``scite_clone_tree`` is
    the induced 6-cluster tree (set only once attachments were resolved from
    some source, named in ``attachment_source``).
    """
    lines = ["# Stage 08 tree diagnostics\n\n", "## Primary tree: SCITE\n"]
    if scite_tree is None:
        lines.append(
            f"SCITE failed to produce a parseable mutation tree: {scite_error}\n"
        )
    elif scite_clone_tree is None:
        lines.append(
            "SCITE's mutation tree parsed, but sample attachments could not "
            f"be resolved from any of its output: {attachment_error}\n"
        )
    else:
        lines.append(
            f"SCITE ran and parsed cleanly (attachments from {attachment_source}).\n"
        )

    lines.append("\n## SNV filtering for SCITE\n")
    if filter_stats is None:
        lines.append("Not computed (SCITE was not attempted).\n")
    else:
        lines.extend(format_filter_stats(filter_stats))

    lines.append("\n## Cross-check: mutation-set containment\n")
    for parent, child in sorted(
        containment_tree.edges(), key=lambda e: (_sort_key(e[0]), _sort_key(e[1]))
    ):
        frac = containment_fraction(mutation_sets, parent, child)
        frac_str = (
            f"{frac:.3f}" if frac is not None else "n/a (parent has no mutations)"
        )
        lines.append(f"{parent} -> {child}: {frac_str}\n")
    lines.append(f"\nThree-gamete (perfect-phylogeny) violations: {n_violations}\n")
    total_skipped = sum(skip_counts.values())
    lines.append(f"SNVs skipped in binning: {total_skipped} {dict(skip_counts)}\n")
    if topology_agreement is None:
        lines.append(
            "SCITE-vs-containment topology agreement: not computed "
            "(SCITE clone tree unavailable).\n"
        )
    else:
        lines.append(
            "SCITE-vs-containment topology agreement (pairwise ancestor/descendant): "
            f"{topology_agreement:.3f}\n"
        )

    if scite_tree is not None:
        lines.append("\n## Topology result\n")
        lines.append(f"SCITE mutation tree: {classify_topology(scite_tree)}\n")
        lines.append(
            f"SCITE-induced clone tree: {classify_topology(scite_clone_tree)}\n"
        )
        if optimal_fraction is not None:
            lines.append(f"SCITE MCMC optimal-step fraction: {optimal_fraction:.3f}\n")
        containment_topology = classify_topology(containment_tree)
        lines.append(f"Containment cross-check: {containment_topology}\n")

        if is_degenerate_result(scite_tree, scite_clone_tree):
            if is_unbranched_chain(containment_tree):
                conclusion = (
                    "both methods produced non-branching topologies; the "
                    "per-cluster tumour-only + gnomAD force-called SNVs do "
                    "not support a branching phylogeny."
                )
            else:
                conclusion = (
                    "SCITE produced a non-branching (or unresolved) topology, "
                    "though the containment cross-check did branch; treat "
                    "tree.nwk as unresolved rather than corroborated."
                )
            lines.append(f"\nFINDING: {conclusion}\n")
            lines.append(
                "tree.nwk below is written for pipeline completeness but is "
                "DEGENERATE/UNTRUSTWORTHY -- a straight chain (or the "
                "containment fallback), not a resolved phylogeny.\n"
            )
    Path(path).write_text("".join(lines))


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
# SCITE (primary tree builder)
# --------------------------------------------------------------------------- #


def find_scite_binary(explicit: Optional[str] = None) -> str:
    """Resolve the SCITE binary: ``explicit`` (the ``--scite-bin`` CLI arg), else
    ``$SCITE_BIN``, else the repo's own build at ``realdata/external/scite/scite``.

    Always returns a path string (never None) so the caller can report a clear
    "not found" error against a concrete path rather than an absent binary.
    """
    if explicit:
        return explicit
    if os.environ.get("SCITE_BIN"):
        return os.environ["SCITE_BIN"]
    return str(Path(__file__).resolve().parents[2] / "external" / "scite" / "scite")


def build_scite_input_matrix(
    snv_matrix: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """SCITE's mutations (rows) x samples (columns) matrix: ``snv_matrix``'s
    own columns, in this module's usual numeric-then-lexicographic cluster
    sort.

    No extra reference column is added: unlike the old tumour-vs-pseudo-normal
    design, there is no cluster standing in for the germline state to give
    SCITE a dedicated all-zero attachment point -- SCITE roots its own
    mutation tree however it likes, and ``resolve_scite_clone_tree`` /
    ``collapse_scite_tree_to_clones`` always re-root the induced clone tree at
    ``GERMLINE_ROOT_ID`` explicitly, never consulting SCITE's own root.
    """
    column_order = sorted(snv_matrix.columns, key=_sort_key)
    return snv_matrix[column_order], column_order


def filter_informative_snvs(
    snv_matrix: pd.DataFrame, min_informative: int = 10
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Keep only SNVs with tree-topology signal: present in at least 2 and at
    most (n_tumour_clusters - 1) of ``snv_matrix``'s columns.

    SCITE's MCMC cost scales with mutation count, and on the real differential
    calls (~7500 SNVs) a run at full size took over 8h. Most of that bulk
    carries no branching signal: a mutation present in every tumour cluster
    sits above all of them alike (uninformative for resolving their order), and
    one private to a single cluster is a leaf with nothing left to split.
    Neither constrains the tree SCITE has to search over.

    ``snv_matrix`` must hold cluster columns only, as ``build_snv_presence_
    matrix`` produces (the germline root is not a real cluster and so never
    appears here) -- this is what keeps it out of the prevalence count, per
    the module contract.

    Returns the filtered matrix and a stats dict (``total``, ``informative``,
    ``all_present``, ``singleton``) for the SCITE log and diagnostics. Warns to
    stderr, but does not fail, if the informative count drops below
    ``min_informative`` -- a small set makes for a poorly resolved tree, not a
    broken one.
    """
    n_clusters = snv_matrix.shape[1]
    prevalence = snv_matrix.sum(axis=1)
    all_present = int((prevalence == n_clusters).sum())
    singleton = int((prevalence == 1).sum())
    informative = (prevalence >= 2) & (prevalence <= n_clusters - 1)
    filtered = snv_matrix.loc[informative]

    stats = {
        "total": int(snv_matrix.shape[0]),
        "informative": int(filtered.shape[0]),
        "all_present": all_present,
        "singleton": singleton,
    }
    if stats["informative"] < min_informative:
        print(
            f"WARNING: only {stats['informative']} topology-informative SNVs "
            f"(< {min_informative}); SCITE's tree may be poorly resolved. "
            "Proceeding anyway.",
            file=sys.stderr,
        )
    return filtered, stats


def subsample_top_variance(matrix: pd.DataFrame, max_mutations: int) -> pd.DataFrame:
    """Safety valve on top of ``filter_informative_snvs``: cap at
    ``max_mutations`` rows, keeping those with the highest presence variance
    across columns (prevalence closest to half the samples is most
    informative for splitting them). A no-op if already at or under the cap.

    Ties are broken by original row order (a stable sort), so the result is
    deterministic for a given input.
    """
    if matrix.shape[0] <= max_mutations:
        return matrix
    variance = matrix.var(axis=1, ddof=0)
    keep = variance.sort_values(ascending=False, kind="mergesort").index[:max_mutations]
    return matrix.loc[matrix.index.isin(keep)]


def format_filter_stats(stats: Dict[str, int]) -> List[str]:
    """Render ``filter_informative_snvs``' (plus an optional subsampling) stats
    dict as text lines, shared between the SCITE run log and tree_diagnostics.txt."""
    lines = [
        f"Total SNVs: {stats['total']}\n",
        f"Topology-informative (kept): {stats['informative']}\n",
        f"Dropped, present in all clusters: {stats['all_present']}\n",
        f"Dropped, private to one cluster (singleton): {stats['singleton']}\n",
    ]
    if "subsampled_from" in stats:
        lines.append(
            f"Subsampled by presence variance: {stats['subsampled_from']} -> "
            f"{stats['used']}\n"
        )
    return lines


def write_scite_matrix(snv_matrix: pd.DataFrame, path: Path) -> None:
    """SCITE genotype format: mutations (rows) x samples (columns), 0/1,
    whitespace-separated. This is format (a), so no ``-transpose`` is passed."""
    np.savetxt(path, snv_matrix.values, fmt="%d")


def write_scite_mutation_names(snv_ids: Iterable[str], path: Path) -> None:
    """One SNV id per line, in row order -- SCITE's optional ``-names`` file.

    Passing this doubles as disambiguation: it makes SCITE's mutation-node
    labels real SNV ids rather than plain integers, so they cannot collide
    with the column-index-based sample-leaf labels ``collapse_scite_tree_to_
    clones`` looks for.
    """
    Path(path).write_text("\n".join(snv_ids) + "\n")


def run_scite(
    matrix_path: Path,
    n_mutations: int,
    n_samples: int,
    out_prefix: Path,
    log_path: Path,
    scite_bin: str,
    fd: float = 1e-3,
    ad: float = 0.15,
    restarts: int = 3,
    chain_length: int = 100_000,
    seed: int = 42,
    names_path: Optional[Path] = None,
    log_header: str = "",
) -> Path:
    """Run SCITE and return the path to its ``<outbase>_ml0.newick`` output.

    Always passes ``-s`` (MAP): the pipeline needs one point-estimate tree,
    not a posterior sample of trees. ``-seed`` is fixed by default for
    reproducibility. The defaults for ``restarts``/``chain_length`` are sized
    for the topology-informative subset ``filter_informative_snvs`` produces,
    not the full SNV set -- SCITE's MCMC cost scales with mutation count, and
    the full ~7500-SNV set at the old defaults (5 restarts x 1,000,000 steps)
    took over 8h. ``log_header`` is written to ``log_path`` before the command
    line, for the informative-SNV filtering stats. Raises
    ``subprocess.CalledProcessError`` if SCITE exits non-zero, and
    ``FileNotFoundError`` if it exits cleanly but the expected Newick file is
    missing -- both are the caller's cue to fail the whole stage rather than
    fall back to the containment tree.
    """
    cmd = [
        scite_bin,
        "-i", str(matrix_path),
        "-n", str(n_mutations),
        "-m", str(n_samples),
        "-r", str(restarts),
        "-l", str(chain_length),
        "-fd", str(fd),
        "-ad", str(ad),
        "-s",
        "-seed", str(seed),
        "-o", str(out_prefix),
    ]  # fmt: skip
    if names_path is not None:
        cmd += ["-names", str(names_path)]
    with open(log_path, "w") as log:
        if log_header:
            log.write(log_header)
        log.write("command: " + " ".join(cmd) + "\n\n")
        log.flush()
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    newick_path = Path(f"{out_prefix}_ml0.newick")
    if not newick_path.exists():
        raise FileNotFoundError(
            f"SCITE exited cleanly but {newick_path} was not written; see {log_path}"
        )
    return newick_path


def parse_scite_newick(newick_path: Path) -> nx.DiGraph:
    """Parse a SCITE mutation-tree Newick file into a DiGraph.

    This is SCITE's own mutation tree, not the model's labelled-internal-node
    clone tree: most internal nodes are unlabelled mutations, and the sample
    columns are attached as leaves anywhere in it. Uses phylox's general
    Newick parser directly (unlike ``parse_newick_like_model``, which enforces
    the model loader's stricter contract) since this tree's shape is
    arbitrary -- on the real differential calls it was a single unbranched
    chain of ~1434 nodes.

    phylox's parser recurses once per nesting level, and Python's default
    1000-frame recursion limit is smaller than that real chain's depth, so
    parsing it raised RecursionError. The number of "(" characters is a safe
    upper bound on nesting depth for any Newick string regardless of shape, so
    the limit is raised before parsing rather than caught after.
    """
    newick_str = Path(newick_path).read_text().strip()
    depth_estimate = newick_str.count("(") + 1
    needed = max(10000, 5 * depth_estimate)
    if sys.getrecursionlimit() < needed:
        sys.setrecursionlimit(needed)
    return phylox.DiNetwork.from_newick(newick_str)


def _scite_sample_label_schemes(column_order: List[str]) -> List[Dict[str, str]]:
    """Ordered, whole-column candidate labelling schemes for SCITE's sample
    leaves: cluster_id -> the leaf label that scheme predicts for it.

    Tried as complete, self-consistent schemes rather than per-sample
    candidates independently: this SCITE build's CLI has no per-sample naming
    flag (only ``-names`` for mutations), so a sample's own cluster identity
    is not expected to appear in the Newick, and the 0-based and 1-based
    column-index spellings share overlapping label spaces (index 1 in one
    scheme is index 0 in the shifted one) -- picking candidates per sample
    independently could match different samples under different, mutually
    inconsistent schemes and silently misassign a leaf instead of failing.
    Matching one scheme against every column at once avoids that.
    """
    n = len(column_order)
    schemes = [dict(zip(column_order, column_order))]  # cluster ID, verbatim
    for start in (1, 0):  # 1-based first: SCITE's own samples are 1..m
        indices = range(start, start + n)
        schemes.append({cid: str(i) for cid, i in zip(column_order, indices)})
        schemes.append({cid: f"s{i}" for cid, i in zip(column_order, indices)})
        schemes.append({cid: f"S{i}" for cid, i in zip(column_order, indices)})
    return schemes


def _sample_nodes_from_newick_labels(
    scite_tree: nx.DiGraph, column_order: List[str]
) -> Dict[str, Hashable]:
    """Resolve each ``column_order`` cluster's attachment node from the
    mutation newick's own leaf labels, via ``_scite_sample_label_schemes``.
    Raises ValueError if no single scheme covers every column.
    """
    label_to_nodes: Dict[str, List[Hashable]] = defaultdict(list)
    for node, data in scite_tree.nodes(data=True):
        label = data.get("label")
        if label is not None:
            label_to_nodes[label].append(node)

    for scheme in _scite_sample_label_schemes(column_order):
        candidate = {
            cid: label_to_nodes[label][0]
            for cid, label in scheme.items()
            if label in label_to_nodes
        }
        if len(candidate) == len(column_order):
            return candidate
    raise ValueError(
        "no consistent sample-leaf labelling scheme (cluster ID, or a "
        "0-/1-based column index with an optional s/S prefix) covers all "
        f"of {column_order}; leaf labels found in the SCITE newick: "
        f"{sorted(label_to_nodes)}"
    )


def _walk_to_nearest_sample_or_root(
    parent_of: Dict[Hashable, Hashable],
    sample_node: Dict[str, Hashable],
    normal_id: str,
) -> nx.DiGraph:
    """Shared tail of every collapse path: given each cluster's already-
    resolved attachment node, plus a parent-lookup over that same node space,
    walk up -- iteratively, a plain while loop, so a long chain costs
    iterations, never stack depth -- to the nearest other sample-leaf
    ancestor, or the germline root.

    ``normal_id`` (``GERMLINE_ROOT_ID`` in ``main``) has no attachment node of
    its own to consult -- it is added as the root outright -- so the result
    is always a single tree rooted there, regardless of where (or whether)
    SCITE's own mutation tree has anything corresponding to it at all.
    """
    node_to_cluster = {node: cid for cid, node in sample_node.items()}
    tree = nx.DiGraph()
    tree.add_node(normal_id)
    for cid, node in sample_node.items():
        if cid == normal_id:
            continue
        ancestor = parent_of.get(node)
        while ancestor is not None and ancestor not in node_to_cluster:
            ancestor = parent_of.get(ancestor)
        parent_cluster = node_to_cluster.get(ancestor, normal_id)
        tree.add_edge(parent_cluster, cid)
    return tree


def _collapse_from_sample_nodes(
    scite_tree: nx.DiGraph, sample_node: Dict[str, Hashable], normal_id: str
) -> nx.DiGraph:
    """``_walk_to_nearest_sample_or_root`` starting from a parsed newick
    DiGraph rather than an already-built parent-lookup dict."""
    parent_of: Dict[Hashable, Hashable] = {}
    for u, v in scite_tree.edges():
        parent_of[v] = u
    return _walk_to_nearest_sample_or_root(parent_of, sample_node, normal_id)


def collapse_scite_tree_to_clones(
    scite_tree: nx.DiGraph, column_order: List[str], normal_id: str
) -> nx.DiGraph:
    """Collapse SCITE's mutation tree to the clone tree over ``column_order``,
    rooted at the germline root, using sample identity found in the newick's
    own leaf labels. Raises ValueError if that identity is not there at all
    (some SCITE builds only put mutation indices in the newick and record
    attachments elsewhere) -- see ``resolve_scite_clone_tree`` for the
    fallback chain over SCITE's other outputs.
    """
    sample_node = _sample_nodes_from_newick_labels(scite_tree, column_order)
    return _collapse_from_sample_nodes(scite_tree, sample_node, normal_id)


_GV_EDGE_RE = re.compile(r"^\s*(\d+)\s*->\s*(\d+)\s*;?\s*$")


def parse_scite_gv_edges(gv_path: Path) -> Dict[int, int]:
    """Parse a SCITE '.gv' (GraphViz) tree into {child_node_id: parent_node_id},
    both SCITE's own integer node ids -- unrelated to phylox's parsed-newick
    node ids, so this is never mixed with ``scite_tree``'s node identities.
    Line-based, so there is nothing here that could recurse.
    """
    parent_of: Dict[int, int] = {}
    for line in Path(gv_path).read_text().splitlines():
        m = _GV_EDGE_RE.match(line)
        if m:
            u, v = int(m.group(1)), int(m.group(2))
            parent_of[v] = u
    return parent_of


def collapse_attachment_to_clone_tree(
    parent_of: Dict[int, int],
    column_order: List[str],
    n_mutations: int,
    normal_id: str,
) -> nx.DiGraph:
    """Collapse a SCITE '-a'-style attachment tree (SCITE's classic integer
    node numbering: mutations 1..n_mutations, the mutation-tree root at
    n_mutations+1, and sample i -- 0-based, in ``column_order``'s order -- at
    node n_mutations+2+i) to the clone tree, rooted explicitly at the
    germline root exactly as ``_collapse_from_sample_nodes`` does.

    Used when the mutation newick itself carries no sample identity and a
    companion ``.gv`` file is the only source of attachments (see
    ``resolve_scite_clone_tree``).
    """
    root_id = n_mutations + 1
    sample_node = {cid: root_id + 1 + i for i, cid in enumerate(column_order)}
    return _walk_to_nearest_sample_or_root(parent_of, sample_node, normal_id)


def _sample_nodes_from_samples_file(
    samples_path: Path, scite_tree: nx.DiGraph, column_order: List[str]
) -> Dict[str, Hashable]:
    """Resolve attachments from a companion ``<out_prefix>*.samples`` file:
    one ``<sample> <attachment label>`` pair per line, whitespace-separated.
    ``<sample>`` is matched against ``column_order`` the same way a newick
    leaf would be (cluster ID, or a 0-/1-based index with an optional s/S
    prefix); ``<attachment label>`` is looked up directly among the mutation
    newick's own node labels (meaningful text, since ``-names`` replaces
    SCITE's numeric mutation ids with the real SNV ids).

    This file's exact format is not documented anywhere this pipeline has
    access to, so this is a best-effort reader: any line that does not
    resolve raises ValueError, which the caller treats the same as the file
    not existing at all.
    """
    label_to_nodes: Dict[str, List[Hashable]] = defaultdict(list)
    for node, data in scite_tree.nodes(data=True):
        label = data.get("label")
        if label is not None:
            label_to_nodes[label].append(node)

    token_to_cid: Dict[str, str] = {}
    for scheme in _scite_sample_label_schemes(column_order):
        for cid, token in scheme.items():
            token_to_cid.setdefault(token, cid)

    sample_node: Dict[str, Hashable] = {}
    for line in Path(samples_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"malformed line in {samples_path}: {line!r}")
        sample_token, attach_label = parts
        cid = token_to_cid.get(sample_token)
        if cid is None or attach_label not in label_to_nodes:
            raise ValueError(
                f"line {line!r} in {samples_path} does not resolve to a "
                "known cluster and a known attachment label"
            )
        sample_node[cid] = label_to_nodes[attach_label][0]

    if set(sample_node) != set(column_order):
        raise ValueError(
            f"{samples_path} did not cover every cluster in {column_order}: "
            f"got {sorted(sample_node)}"
        )
    return sample_node


def collapse_from_samples_file(
    samples_path: Path, scite_tree: nx.DiGraph, column_order: List[str], normal_id: str
) -> nx.DiGraph:
    """Collapse using attachments read from a companion ``.samples`` file (see
    ``_sample_nodes_from_samples_file``)."""
    sample_node = _sample_nodes_from_samples_file(
        samples_path, scite_tree, column_order
    )
    return _collapse_from_sample_nodes(scite_tree, sample_node, normal_id)


def resolve_scite_clone_tree(
    scite_tree: nx.DiGraph,
    out_prefix: Path,
    n_mutations: int,
    column_order: List[str],
    normal_id: str,
) -> Tuple[nx.DiGraph, str]:
    """Build the clone tree from whichever SCITE output actually carries
    sample attachments -- not always the mutation newick itself. On the real
    differential calls, SCITE's newick held only mutation indices; sample
    attachments live in its other output instead.

    Tries, in order: (1) the newick's own leaf labels; (2) a companion
    ``<out_prefix>*.gv`` file, using SCITE's classic '-a' integer node
    numbering; (3) a companion ``<out_prefix>*.samples`` file. Returns
    ``(clone_tree, source)`` on the first that resolves every column in
    ``column_order``. Raises ValueError, naming every source tried, if none
    do -- the caller must treat that as a degenerate/unresolved result and
    never fabricate a tree (see the module docstring).
    """
    errors: List[str] = []

    try:
        return (
            collapse_scite_tree_to_clones(scite_tree, column_order, normal_id),
            "the mutation newick's own leaf labels",
        )
    except ValueError as exc:
        errors.append(f"newick leaf labels: {exc}")

    gv_paths = sorted(out_prefix.parent.glob(f"{out_prefix.name}*.gv"))
    for gv_path in gv_paths:
        try:
            parent_of = parse_scite_gv_edges(gv_path)
            clone_tree = collapse_attachment_to_clone_tree(
                parent_of, column_order, n_mutations, normal_id
            )
            return clone_tree, f"the GraphViz side file ({gv_path.name})"
        except (KeyError, ValueError) as exc:
            errors.append(f"{gv_path.name}: {exc}")
    if not gv_paths:
        errors.append(f"no {out_prefix.name}*.gv side file found")

    samples_paths = sorted(out_prefix.parent.glob(f"{out_prefix.name}*.samples"))
    for samples_path in samples_paths:
        try:
            clone_tree = collapse_from_samples_file(
                samples_path, scite_tree, column_order, normal_id
            )
            return clone_tree, f"the samples side file ({samples_path.name})"
        except (OSError, ValueError) as exc:
            errors.append(f"{samples_path.name}: {exc}")
    if not samples_paths:
        errors.append(f"no {out_prefix.name}*.samples side file found")

    raise ValueError(
        f"sample attachments unresolved for {column_order} in any SCITE "
        "output; tried " + "; ".join(errors)
    )


# --------------------------------------------------------------------------- #
# Topology classification (chain vs branching, iterative)
# --------------------------------------------------------------------------- #


def is_unbranched_chain(tree: nx.DiGraph) -> bool:
    """True if every node in ``tree`` has at most one child -- a straight
    line down from the root, no branching anywhere. A plain out-degree scan,
    so no depth limit of any kind applies."""
    return all(tree.out_degree(n) <= 1 for n in tree.nodes())


def chain_depth_if_linear(tree: nx.DiGraph) -> Optional[int]:
    """If ``tree`` is a single unbranched path from a unique root, return its
    depth (edge count from root to the tip); otherwise None.

    Walks root -> child -> child -> ... in an explicit while loop, never
    recursion, so this is safe on a chain of any length -- including the
    ~1434-node one seen on the real differential calls.
    """
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


def is_degenerate_result(
    scite_tree: nx.DiGraph, scite_clone_tree: Optional[nx.DiGraph]
) -> bool:
    """True if SCITE did not yield a trustworthy, branching clone tree, given
    it at least produced a parseable mutation tree (``scite_tree``; a harder
    failure than this -- SCITE never even ran or parsed -- is handled
    separately in ``main``, before this is called): sample attachments could
    not be resolved from any output at all, the raw mutation tree is itself a
    straight chain, or the resolved clone tree is.
    """
    if scite_clone_tree is None:
        return True
    if is_unbranched_chain(scite_tree):
        return True
    return is_unbranched_chain(scite_clone_tree)


_OPTIMAL_FRACTION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%[^\n]*optimal", re.IGNORECASE)


def parse_scite_optimal_fraction(log_path: Path) -> Optional[float]:
    """Best-effort extraction of an "X% ... optimal" figure from SCITE's own
    stdout/stderr (captured in ``run_scite``'s log). Returns None, not an
    error, if no such line is found -- the exact wording is undocumented and
    may not appear in every SCITE build or version.
    """
    if not Path(log_path).exists():
        return None
    match = _OPTIMAL_FRACTION_RE.search(Path(log_path).read_text())
    return float(match.group(1)) / 100.0 if match else None


def compare_topologies(
    tree_a: nx.DiGraph, tree_b: nx.DiGraph, cluster_ids: List[str]
) -> float:
    """Fraction of cluster pairs whose ancestor/descendant relation agrees
    between two trees.

    A pairwise-relation summary, not a full tree-edit distance: 1.0 means every
    pair of clusters is ordered the same way (ancestor, descendant, or
    unrelated) in both trees.
    """

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
        "--scite-bin",
        default=None,
        help="path to the scite binary; defaults to $SCITE_BIN, then "
        "realdata/external/scite/scite",
    )
    p.add_argument("--scite-fd", type=float, default=1e-3, help="false positive rate")
    p.add_argument("--scite-ad", type=float, default=0.15, help="dropout rate")
    p.add_argument(
        "--scite-restarts",
        type=int,
        default=3,
        help="sized for the topology-informative subset, not the full SNV set",
    )
    p.add_argument(
        "--scite-chain-length",
        type=int,
        default=100_000,
        help="sized for the topology-informative subset, not the full SNV set",
    )
    p.add_argument(
        "--scite-seed", type=int, default=42, help="fixed for reproducibility"
    )
    p.add_argument(
        "--scite-min-informative",
        type=int,
        default=10,
        help="warn (not fail) if fewer topology-informative SNVs survive filtering",
    )
    p.add_argument(
        "--scite-max-mutations",
        type=int,
        default=None,
        help="safety valve: cap the informative SNV set to this many, keeping "
        "the highest presence-variance rows, before feeding SCITE",
    )
    p.add_argument(
        "--allow-containment-fallback",
        action="store_true",
        help="debugging only: emit the containment tree as tree.nwk if SCITE "
        "never produces a parseable mutation tree at all, instead of exiting "
        "non-zero (does not apply to a degenerate-but-complete SCITE result; "
        "see --allow-degenerate-tree)",
    )
    p.add_argument(
        "--allow-degenerate-tree",
        action="store_true",
        help="exit 0 (for downstream plumbing tests) even when the result is "
        "degenerate -- a non-branching topology or unresolved sample "
        "attachments -- instead of exiting non-zero after writing "
        "tree_diagnostics.txt and tree.nwk",
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
    # force-called genotype, not the FILTER column (see 06b_forcecall.sbatch and
    # parse_forced_vcf_calls). This same call set backs the presence matrix,
    # the containment cross-check, and the 96-channel spectra below -- all six
    # clusters carry a spectrum now, none held back as a pseudo-normal.
    cluster_to_snvs = resolve_presence_calls(
        cluster_to_calls, args.presence_min_vaf, args.presence_min_alt_reads
    )

    snv_matrix = build_snv_presence_matrix(cluster_to_snvs)
    snv_matrix.to_csv(args.out_dir / "clone_snv_matrix.csv")

    mutation_sets = mutation_sets_from_matrix(snv_matrix)

    # Cross-check, always built (see write_diagnostics), never emitted as
    # tree.nwk unless --allow-containment-fallback is passed and SCITE fails.
    containment_tree = build_clone_tree(mutation_sets, GERMLINE_ROOT_ID)
    n_violations = three_gamete_violations(mutation_sets)

    # SCITE's MCMC cost scales with mutation count; only the topology-
    # informative SNVs go to SCITE. The full matrix above still backs
    # clone_snv_matrix.csv and the containment cross-check.
    informative_matrix, filter_stats = filter_informative_snvs(
        snv_matrix, min_informative=args.scite_min_informative
    )
    if args.scite_max_mutations is not None:
        filter_stats["subsampled_from"] = informative_matrix.shape[0]
        informative_matrix = subsample_top_variance(
            informative_matrix, args.scite_max_mutations
        )
    filter_stats["used"] = informative_matrix.shape[0]

    scite_bin = find_scite_binary(args.scite_bin)
    scite_error: Optional[str] = None
    scite_tree: Optional[nx.DiGraph] = None  # SCITE's raw mutation newick
    scite_clone_tree: Optional[nx.DiGraph] = None  # induced tree over column_order
    attachment_source: Optional[str] = None
    attachment_error: Optional[str] = None
    optimal_fraction: Optional[float] = None

    if not os.access(scite_bin, os.X_OK):
        scite_error = f"SCITE binary not found or not executable: {scite_bin!r}"
    else:
        scite_log_path = args.out_dir / "scite_run.log"
        try:
            full_matrix, column_order = build_scite_input_matrix(informative_matrix)
            matrix_path = args.out_dir / "scite_genotype_matrix.txt"
            write_scite_matrix(full_matrix, matrix_path)
            names_path = args.out_dir / "scite_mutation_names.txt"
            write_scite_mutation_names(informative_matrix.index, names_path)
            out_prefix = args.out_dir / "scite_out"
            newick_path = run_scite(
                matrix_path,
                n_mutations=full_matrix.shape[0],
                n_samples=full_matrix.shape[1],
                out_prefix=out_prefix,
                log_path=scite_log_path,
                scite_bin=scite_bin,
                fd=args.scite_fd,
                ad=args.scite_ad,
                restarts=args.scite_restarts,
                chain_length=args.scite_chain_length,
                seed=args.scite_seed,
                names_path=names_path,
                log_header="".join(format_filter_stats(filter_stats)) + "\n",
            )
            # Iterative from here on: parse_scite_newick raises its own
            # recursion limit before calling phylox, and every walk over the
            # result (resolve_scite_clone_tree, is_degenerate_result) is a
            # plain loop, so a fully linear mutation tree (as seen on the real
            # differential calls, ~1434 nodes) cannot overflow the stack.
            scite_tree = parse_scite_newick(newick_path)
            optimal_fraction = parse_scite_optimal_fraction(scite_log_path)
        except Exception as exc:
            scite_error = str(exc)

        if scite_tree is not None:
            try:
                scite_clone_tree, attachment_source = resolve_scite_clone_tree(
                    scite_tree,
                    out_prefix,
                    n_mutations=full_matrix.shape[0],
                    column_order=column_order,
                    normal_id=GERMLINE_ROOT_ID,
                )
            except ValueError as exc:
                attachment_error = str(exc)

    degenerate = False
    if scite_tree is None:
        # SCITE never produced even a parseable mutation tree -- a harder
        # failure than a degenerate-but-complete result (see the module
        # docstring); --allow-degenerate-tree does not apply here.
        if args.allow_containment_fallback:
            print(
                f"WARNING: SCITE failed ({scite_error}); emitting the containment "
                "tree as tree.nwk because --allow-containment-fallback was passed. "
                "This is a known-bad topology on real data -- debugging only.",
                file=sys.stderr,
            )
            primary_tree = containment_tree
        else:
            sys.exit(
                f"SCITE tree construction failed: {scite_error}\n"
                "Refusing to fall back to the mutation-set containment tree, which "
                "is known untrustworthy on this data (see the module docstring). "
                "Pass --allow-containment-fallback to override for debugging."
            )
    else:
        degenerate = is_degenerate_result(scite_tree, scite_clone_tree)
        # Attachments unresolved leaves nothing SCITE-derived to write, so the
        # containment tree is the only tree left to form at all; a resolved
        # (even chain-shaped) SCITE clone tree is still SCITE's real result.
        primary_tree = (
            scite_clone_tree if scite_clone_tree is not None else containment_tree
        )

    newick_str = digraph_to_newick(primary_tree, GERMLINE_ROOT_ID)
    verify_newick(newick_str, set(cluster_to_snvs), GERMLINE_ROOT_ID)
    (args.out_dir / "tree.nwk").write_text(newick_str + "\n")

    # Written regardless of the tree outcome above -- the spectra are valid
    # even when the tree is degenerate or SCITE failed outright.
    import pysam

    with pysam.FastaFile(str(args.ref_fasta)) as fasta:
        spectra, skip_counts = bin_cluster_spectra(cluster_to_snvs, fasta)
    spectra.to_csv(args.out_dir / "spectra.csv")

    topology_agreement = None
    if scite_clone_tree is not None:
        clusters_sorted = sorted(cluster_to_snvs, key=_sort_key)
        topology_agreement = compare_topologies(
            scite_clone_tree, containment_tree, clusters_sorted
        )

    write_diagnostics(
        args.out_dir / "tree_diagnostics.txt",
        containment_tree,
        mutation_sets,
        skip_counts,
        n_violations,
        scite_error,
        topology_agreement,
        filter_stats,
        scite_tree=scite_tree,
        scite_clone_tree=scite_clone_tree,
        attachment_source=attachment_source,
        attachment_error=attachment_error,
        optimal_fraction=optimal_fraction,
    )

    print(
        "Tumour clusters (tree tips + internal observed clones): "
        f"{len(cluster_to_snvs)}"
    )
    print("Somatic SNV count per cluster:")
    for cid in sorted(cluster_to_snvs, key=_sort_key):
        print(f"  clone{cid}: {len(cluster_to_snvs[cid])}")

    if scite_tree is not None and degenerate and not args.allow_degenerate_tree:
        diagnostics_path = args.out_dir / "tree_diagnostics.txt"
        sys.exit(
            "Degenerate result: neither SCITE nor the containment cross-check "
            "found a branching topology (or SCITE's sample attachments could "
            f"not be resolved); see {diagnostics_path}. tree.nwk was still "
            "written there, marked degenerate. Pass --allow-degenerate-tree "
            "to exit 0 anyway."
        )


if __name__ == "__main__":
    main()
