"""Turn stage 07's per-cluster somatic VCFs into the two inputs TreeHDP consumes:
a rooted Newick tree and a per-cluster 96-channel spectra matrix.

Ours: neither secedo nor secedo-evaluation builds a phylogeny over SECEDO's
clusters or bins VCFs into COSMIC's 96 SBS channels. This is that step (recipe
stage 8, `realdata/recipe/breast_cancer_plan.md`).

Contract (from `src/models/hdp_inference.py`'s `_BaseTreeHDP`, do not "fix" these)
    - Newick is labelled-internal-node form, e.g. ``((c2,c3)c1)normal;``. Observed
      ancestral clones sit at internal nodes carrying their cluster-ID label; the
      ancestor-as-tip idiom is not used.
    - The pseudo-normal is the single root of every tree and is deliberately
      absent from the spectra matrix -- it has no VCF (excluded from stage 05's
      tasks.tsv), so it becomes a spectrum-less latent root, which the model
      handles natively.
    - Node labels in the Newick, the spectra index, the SECEDO cluster IDs, and
      Mutect2's ``-normal`` are one ID system throughout.
    - The 96 spectra columns are in cosmic_signatures.csv's exact channel order,
      because the model does ``dot(activities, signatures)`` and aligns observed
      counts to signature columns positionally. ``main`` asserts this before
      doing anything else and fails loudly if it does not hold.

Tree construction is via SCITE (Jahn et al.), the intended method for this
pipeline: SCITE samples a mutation history from the clone x SNV presence
matrix (plus an all-zero pseudo-normal reference column, so the model's
spectrum-less root has somewhere to attach) and writes its MAP mutation tree,
samples attached as leaves, as Newick. That tree is collapsed to the clone
tree this script emits: each tumour clone's parent is the nearest sample-leaf
ancestor in SCITE's tree (walking up through the unlabelled mutation nodes),
falling back to the pseudo-normal when no such ancestor exists before the top
of SCITE's tree. The clone tree is always re-rooted at the pseudo-normal
explicitly -- its exact attachment point inside SCITE's own tree is never used
to derive edges -- so the result is a single tree rooted at the pseudo-normal
regardless of exactly where SCITE placed the all-zero column.

SCITE is required, not optional: if the binary is unavailable, or its run or
Newick parse fails, ``main`` exits non-zero rather than emit the accumulation
tree below as tree.nwk. ``--allow-containment-fallback`` overrides this for
debugging only.

Accumulation by mutation-set containment (below) is now a cross-check, not the
primary construction, because on the real differential calls it produced 581k
three-gamete violations, 0.27-0.36 edge containment, and an implausible linear
chain -- untrustworthy as the emitted tree. It is still built every run and
reported in tree_diagnostics.txt, alongside its topology agreement with
SCITE's tree, as a sanity signal.

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

SNVKey = Tuple[str, int, str, str]  # chrom, 1-based pos, ref, alt

_VCF_NAME_RE = re.compile(r"^clone(?P<cluster>[^_]+)_(?P<chrom>.+)\.vcf$")


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


def parse_vcf_snvs(vcf_path: Path) -> Set[SNVKey]:
    """Parse PASS, single-base substitutions out of one VCF into (chrom, pos, ref,
    alt) keys.

    Defensive re-check of PASS/SNP-only, not a new filter: stage 06 already
    restricts to FilterMutectCalls PASS plus SelectVariants SNP-type.
    Multi-allelic ALT fields are split; only single-base alleles are kept.
    """
    snvs: Set[SNVKey] = set()
    with open(vcf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                continue
            chrom, pos, _id, ref, alt_field = fields[:5]
            filt = fields[6] if len(fields) > 6 else "PASS"
            if filt not in ("PASS", "."):
                continue
            ref = ref.upper()
            if len(ref) != 1 or ref not in "ACGT":
                continue
            for alt in alt_field.split(","):
                alt = alt.upper()
                if len(alt) == 1 and alt in "ACGT":
                    snvs.add((chrom, int(pos), ref, alt))
    return snvs


def discover_cluster_vcfs(vcf_dir: Path, normal_id: str) -> Dict[str, List[Path]]:
    """Group ``clone<cluster>_<chrom>.vcf`` files by cluster, excluding the
    pseudo-normal."""
    out: Dict[str, List[Path]] = defaultdict(list)
    for f in sorted(vcf_dir.glob("clone*_*.vcf")):
        m = _VCF_NAME_RE.match(f.name)
        if not m:
            continue
        cluster = m.group("cluster")
        if cluster == str(normal_id):
            continue
        out[cluster].append(f)
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

    Root is the pseudo-normal with the empty set. Tumour clones are placed in
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

    Raises AssertionError if the pseudo-normal is not the unique root, or if
    the parsed node labels are not exactly the tumour cluster IDs plus the
    pseudo-normal, each appearing once.
    """
    graph = parse_newick_like_model(newick_str)
    expected = set(cluster_ids) | {normal_id}
    labels = list(graph.nodes())
    if len(labels) != len(expected):
        raise AssertionError(
            f"parsed {len(labels)} node labels, expected {len(expected)} "
            f"(cluster IDs plus the pseudo-normal); a label collision or "
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
            f"root is {roots[0]!r}, expected pseudo-normal {normal_id!r}"
        )


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


def containment_fraction(
    mutation_sets: Dict[str, Set[Hashable]], parent: Hashable, child: Hashable
) -> Optional[float]:
    """|muts(parent) ∩ muts(child)| / |muts(parent)|, or None if parent has no
    mutations.

    ``parent`` may be the pseudo-normal root, which is never a key in
    ``mutation_sets`` (it has no VCF); treated as the empty set.
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
    scite_ok: bool,
    scite_error: Optional[str],
    topology_agreement: Optional[float],
    filter_stats: Optional[Dict[str, int]] = None,
) -> None:
    """SCITE is primary (tree.nwk); mutation-set containment is the cross-check
    reported here, not what gets emitted. Numbers only, no verdicts.
    """
    lines = ["# Stage 08 tree diagnostics\n\n", "## Primary tree: SCITE\n"]
    if scite_ok:
        lines.append(
            "SCITE ran and parsed cleanly; tree.nwk is its collapsed clone tree.\n"
        )
    else:
        lines.append(f"SCITE failed: {scite_error}\n")

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
            "(SCITE tree unavailable).\n"
        )
    else:
        lines.append(
            "SCITE-vs-containment topology agreement (pairwise ancestor/descendant): "
            f"{topology_agreement:.3f}\n"
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
    snv_matrix: pd.DataFrame, normal_id: str
) -> Tuple[pd.DataFrame, List[str]]:
    """SCITE's mutations (rows) x samples (columns) matrix: ``snv_matrix``'s
    tumour columns plus an all-zero pseudo-normal reference column, in one
    combined column order (this module's usual numeric-then-lexicographic
    cluster sort).

    The pseudo-normal has no VCF and so no column in ``snv_matrix``; adding it
    as an all-zero column gives SCITE somewhere to attach it. Downstream,
    ``collapse_scite_tree_to_clones`` re-roots the clone tree at the
    pseudo-normal explicitly rather than trusting where SCITE attaches an
    all-zero genotype, so this need not land exactly at SCITE's own tree root.
    """
    column_order = sorted(list(snv_matrix.columns) + [normal_id], key=_sort_key)
    tumour_cols = [c for c in column_order if c != normal_id]
    full = snv_matrix.reindex(columns=tumour_cols).copy()
    full[normal_id] = 0
    return full[column_order], column_order


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

    ``snv_matrix`` must hold tumour-cluster columns only, as
    ``build_snv_presence_matrix`` produces (the pseudo-normal has no VCF and so
    never appears here) -- this is what keeps the pseudo-normal out of the
    prevalence count, per the module contract.

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
    arbitrary.
    """
    newick_str = Path(newick_path).read_text().strip()
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


def collapse_scite_tree_to_clones(
    scite_tree: nx.DiGraph, column_order: List[str], normal_id: str
) -> nx.DiGraph:
    """Collapse SCITE's mutation tree to the clone tree over ``column_order``,
    rooted at the pseudo-normal.

    Each non-normal cluster's parent is the nearest sample-leaf ancestor,
    walking up through the unlabelled mutation nodes; a cluster with no
    sample-leaf ancestor before the top of SCITE's tree attaches directly
    under the pseudo-normal. The pseudo-normal's own position inside SCITE's
    tree is never consulted -- it is added as the root outright -- so the
    result is always a single tree rooted at the pseudo-normal regardless of
    exactly where SCITE attached the all-zero reference column.
    """
    label_to_nodes: Dict[str, List[Hashable]] = defaultdict(list)
    for node, data in scite_tree.nodes(data=True):
        label = data.get("label")
        if label is not None:
            label_to_nodes[label].append(node)

    sample_node: Optional[Dict[str, Hashable]] = None
    for scheme in _scite_sample_label_schemes(column_order):
        candidate = {
            cid: label_to_nodes[label][0]
            for cid, label in scheme.items()
            if label in label_to_nodes
        }
        if len(candidate) == len(column_order):
            sample_node = candidate
            break
    if sample_node is None:
        raise ValueError(
            "no consistent sample-leaf labelling scheme (cluster ID, or a "
            "0-/1-based column index with an optional s/S prefix) covers all "
            f"of {column_order}; leaf labels found in the SCITE newick: "
            f"{sorted(label_to_nodes)}"
        )

    node_to_cluster = {node: cid for cid, node in sample_node.items()}
    parent_of: Dict[Hashable, Hashable] = {}
    for u, v in scite_tree.edges():
        parent_of[v] = u

    tree = nx.DiGraph()
    tree.add_node(normal_id)
    for cid in column_order:
        if cid == normal_id:
            continue
        ancestor = parent_of.get(sample_node[cid])
        while ancestor is not None and ancestor not in node_to_cluster:
            ancestor = parent_of.get(ancestor)
        parent_cluster = node_to_cluster.get(ancestor, normal_id)
        tree.add_edge(parent_cluster, cid)
    return tree


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
        help="directory of clone<c>_<chrom>.vcf files (stage 07's output)",
    )
    p.add_argument("--normal-id", required=True, help="pseudo-normal cluster ID")
    p.add_argument("--ref-fasta", required=True, type=Path)
    p.add_argument(
        "--cosmic-signatures",
        required=True,
        type=Path,
        help="cosmic_signatures.csv, used only to assert the channel order",
    )
    p.add_argument("--out-dir", required=True, type=Path)
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
        help="debugging only: emit the known-bad containment tree as tree.nwk "
        "if SCITE is unavailable or fails, instead of exiting non-zero",
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

    cluster_vcfs = discover_cluster_vcfs(args.vcf_dir, args.normal_id)
    if not cluster_vcfs:
        sys.exit(f"no clone*_*.vcf files found in {args.vcf_dir}")

    cluster_to_snvs: Dict[str, Set[SNVKey]] = {}
    for cluster, files in cluster_vcfs.items():
        snvs: Set[SNVKey] = set()
        for f in files:
            snvs |= parse_vcf_snvs(f)
        cluster_to_snvs[cluster] = snvs

    snv_matrix = build_snv_presence_matrix(cluster_to_snvs)
    snv_matrix.to_csv(args.out_dir / "clone_snv_matrix.csv")

    mutation_sets = mutation_sets_from_matrix(snv_matrix)

    # Cross-check, always built (see write_diagnostics), never emitted as
    # tree.nwk unless --allow-containment-fallback is passed and SCITE fails.
    containment_tree = build_clone_tree(mutation_sets, args.normal_id)
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
    scite_ok = False
    scite_error: Optional[str] = None
    scite_tree: Optional[nx.DiGraph] = None
    if not os.access(scite_bin, os.X_OK):
        scite_error = f"SCITE binary not found or not executable: {scite_bin!r}"
    else:
        try:
            full_matrix, column_order = build_scite_input_matrix(
                informative_matrix, args.normal_id
            )
            matrix_path = args.out_dir / "scite_genotype_matrix.txt"
            write_scite_matrix(full_matrix, matrix_path)
            names_path = args.out_dir / "scite_mutation_names.txt"
            write_scite_mutation_names(informative_matrix.index, names_path)
            newick_path = run_scite(
                matrix_path,
                n_mutations=full_matrix.shape[0],
                n_samples=full_matrix.shape[1],
                out_prefix=args.out_dir / "scite_out",
                log_path=args.out_dir / "scite_run.log",
                scite_bin=scite_bin,
                fd=args.scite_fd,
                ad=args.scite_ad,
                restarts=args.scite_restarts,
                chain_length=args.scite_chain_length,
                seed=args.scite_seed,
                names_path=names_path,
                log_header="".join(format_filter_stats(filter_stats)) + "\n",
            )
            scite_mutation_tree = parse_scite_newick(newick_path)
            scite_tree = collapse_scite_tree_to_clones(
                scite_mutation_tree, column_order, args.normal_id
            )
            scite_ok = True
        except Exception as exc:
            scite_error = str(exc)

    if scite_ok:
        primary_tree = scite_tree
    elif args.allow_containment_fallback:
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

    newick_str = digraph_to_newick(primary_tree, args.normal_id)
    verify_newick(newick_str, set(cluster_to_snvs), args.normal_id)
    (args.out_dir / "tree.nwk").write_text(newick_str + "\n")

    import pysam

    with pysam.FastaFile(str(args.ref_fasta)) as fasta:
        spectra, skip_counts = bin_cluster_spectra(cluster_to_snvs, fasta)
    spectra.to_csv(args.out_dir / "spectra.csv")

    topology_agreement = None
    if scite_ok:
        clusters_sorted = sorted(cluster_to_snvs, key=_sort_key)
        topology_agreement = compare_topologies(
            scite_tree, containment_tree, clusters_sorted
        )

    write_diagnostics(
        args.out_dir / "tree_diagnostics.txt",
        containment_tree,
        mutation_sets,
        skip_counts,
        n_violations,
        scite_ok,
        scite_error,
        topology_agreement,
        filter_stats,
    )

    print(
        "Tumour clusters (tree tips + internal observed clones): "
        f"{len(cluster_to_snvs)}"
    )
    print("Somatic SNV count per cluster:")
    for cid in sorted(cluster_to_snvs, key=_sort_key):
        print(f"  clone{cid}: {len(cluster_to_snvs[cid])}")


if __name__ == "__main__":
    main()
