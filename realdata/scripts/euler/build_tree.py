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

Tree construction is accumulation by mutation-set containment: a total, always-
succeeding perfect-phylogeny approximation, not an error-aware caller. Process
tumour clones from fewest to most mutations; each clone's parent is the already-
placed node (root included, with the empty set) maximising shared mutations,
tied-broken by fewest parent-only mutations, then fewest parent mutations, then
cluster ID. This puts observed clones at internal nodes wherever containment
holds and degrades to a star under the root when it does not.

Channel ordering was recovered empirically (cosmic_signatures.csv carries no
channel labels, only ``Channel_0..Channel_95``): SBS1's four dominant channels
sit 24 apart, at the position matching NCG>NTG (C>T at CpG, all four 5' flanks),
and SBS4/SBS5/SBS92's dominant channels match their known C>A/T>C aetiology
under the same hypothesis. That fixes the axis as index = five*24 + subtype*4 +
three, five/three in A,C,G,T order and subtype in C>A,C>G,C>T,T>A,T>C,T>G order
-- the alphabetical sort of COSMIC's own ``A[C>A]A`` .. ``T[T>G]T`` labels.
``main`` still asserts cosmic_signatures.csv's columns match before trusting it.

SCITE cross-check (optional, non-blocking): if a SCITE binary is on ``PATH`` or
named by ``$SCITE_BIN``, run it on the same genotype matrix and compare its
attachment tree's topology to the accumulation tree's; otherwise log one line
and continue. Never blocks the primary outputs.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, Hashable, List, Optional, Set, Tuple

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
    tree: nx.DiGraph,
    mutation_sets: Dict[str, Set[Hashable]],
    skip_counts: Dict[str, int],
    n_violations: int,
    scite_agreement: Optional[float],
) -> None:
    """Per-edge containment, three-gamete violations, and binning skip counts.

    Numbers only, no verdicts -- whether the clean accumulation construction is
    trustworthy, or the calls need an error-aware method, is a judgement call
    for whoever reads this, not something this script decides.
    """
    lines = ["# Stage 08 tree diagnostics\n\n", "## Edge containment\n"]
    for parent, child in sorted(
        tree.edges(), key=lambda e: (_sort_key(e[0]), _sort_key(e[1]))
    ):
        frac = containment_fraction(mutation_sets, parent, child)
        frac_str = (
            f"{frac:.3f}" if frac is not None else "n/a (parent has no mutations)"
        )
        lines.append(f"{parent} -> {child}: {frac_str}\n")
    lines.append(f"\nThree-gamete (perfect-phylogeny) violations: {n_violations}\n")
    total_skipped = sum(skip_counts.values())
    lines.append(f"SNVs skipped in binning: {total_skipped} {dict(skip_counts)}\n")
    if scite_agreement is None:
        lines.append("SCITE cross-check: not run (binary unavailable or run failed).\n")
    else:
        lines.append(
            "SCITE topology agreement (pairwise ancestor/descendant): "
            f"{scite_agreement:.3f}\n"
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
# SCITE cross-check (optional, non-blocking)
# --------------------------------------------------------------------------- #


def find_scite_binary(explicit: Optional[str] = None) -> Optional[str]:
    return explicit or os.environ.get("SCITE_BIN") or shutil.which("scite")


def write_scite_matrix(snv_matrix: pd.DataFrame, path: Path) -> None:
    """SCITE genotype format: mutations (rows) x samples (columns), 0/1,
    whitespace-separated."""
    np.savetxt(path, snv_matrix.values, fmt="%d")


_GV_EDGE_RE = re.compile(r"^\s*(\d+)\s*->\s*(\d+)\s*;?\s*$")


def parse_scite_gv_edges(gv_path: Path) -> Dict[int, int]:
    """Parse a SCITE .gv tree into {child_node_id: parent_node_id}."""
    parent_of: Dict[int, int] = {}
    for line in Path(gv_path).read_text().splitlines():
        m = _GV_EDGE_RE.match(line)
        if m:
            u, v = int(m.group(1)), int(m.group(2))
            parent_of[v] = u
    return parent_of


def collapse_attachment_to_clone_tree(
    parent_of: Dict[int, int], cluster_ids: List[str], n_mutations: int
) -> nx.DiGraph:
    """Collapse a SCITE '-a' sample-attachment tree to a clone tree over cluster_ids.

    SCITE's default '-a' numbering: nodes 1..n_mutations are mutations,
    n_mutations+1 is the mutation-tree root, and n_mutations+1+i (0-based i) is
    the attachment point of the i-th genotype-matrix column. Each sample leaf
    walks up to the nearest ancestor that is itself a sample leaf, or the
    root, giving a parent assignment comparable to build_clone_tree's output.
    """
    root_id = n_mutations + 1
    sample_node = {i: root_id + 1 + i for i in range(len(cluster_ids))}
    node_to_cluster = {
        node: cid for cid, node in zip(cluster_ids, sample_node.values())
    }

    tree = nx.DiGraph()
    tree.add_node("__root__")
    for i, cid in enumerate(cluster_ids):
        node = sample_node[i]
        ancestor = parent_of.get(node)
        while (
            ancestor is not None
            and ancestor not in node_to_cluster
            and ancestor != root_id
        ):
            ancestor = parent_of.get(ancestor)
        parent_cluster = node_to_cluster.get(ancestor, "__root__")
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


def run_scite(
    matrix_path: Path,
    n_mutations: int,
    n_samples: int,
    out_prefix: Path,
    fd: float,
    ad: float,
    scite_bin: str,
) -> Optional[Path]:
    """Run SCITE and return the '-a' attachment .gv path, or None if it
    produced none."""
    cmd = [
        scite_bin,
        "-i",
        str(matrix_path),
        "-n",
        str(n_mutations),
        "-m",
        str(n_samples),
        "-r",
        "1",
        "-l",
        "100000",
        "-fd",
        str(fd),
        "-ad",
        str(ad),
        "-a",
        "-o",
        str(out_prefix),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    candidates = sorted(out_prefix.parent.glob(f"{out_prefix.name}*ml0.gv"))
    return candidates[0] if candidates else None


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
        help="path to the scite binary; defaults to $SCITE_BIN or PATH",
    )
    p.add_argument("--scite-fd", type=float, default=1e-3)
    p.add_argument("--scite-ad", type=float, default=0.1)
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
    tree = build_clone_tree(mutation_sets, args.normal_id)
    newick_str = digraph_to_newick(tree, args.normal_id)
    verify_newick(newick_str, set(cluster_to_snvs), args.normal_id)
    (args.out_dir / "tree.nwk").write_text(newick_str + "\n")

    import pysam

    with pysam.FastaFile(str(args.ref_fasta)) as fasta:
        spectra, skip_counts = bin_cluster_spectra(cluster_to_snvs, fasta)
    spectra.to_csv(args.out_dir / "spectra.csv")

    n_violations = three_gamete_violations(mutation_sets)

    scite_agreement = None
    scite_bin = find_scite_binary(args.scite_bin)
    if scite_bin is None:
        print(
            "SCITE binary not found (set --scite-bin, $SCITE_BIN, or put scite on "
            "PATH); skipping the topology cross-check.",
            file=sys.stderr,
        )
    else:
        try:
            clusters_sorted = sorted(cluster_to_snvs, key=_sort_key)
            matrix_path = args.out_dir / "scite_genotype_matrix.txt"
            write_scite_matrix(snv_matrix.reindex(columns=clusters_sorted), matrix_path)
            gv_path = run_scite(
                matrix_path,
                n_mutations=snv_matrix.shape[0],
                n_samples=len(clusters_sorted),
                out_prefix=args.out_dir / "scite_out",
                fd=args.scite_fd,
                ad=args.scite_ad,
                scite_bin=scite_bin,
            )
            if gv_path is not None:
                parent_of = parse_scite_gv_edges(gv_path)
                scite_tree = collapse_attachment_to_clone_tree(
                    parent_of, clusters_sorted, snv_matrix.shape[0]
                )
                scite_agreement = compare_topologies(tree, scite_tree, clusters_sorted)
        except Exception as exc:
            # SCITE is a non-blocking cross-check, not a dependency: any failure
            # here (binary missing, parse error, ...) is logged and skipped.
            print(
                f"SCITE cross-check failed, continuing without it: {exc}",
                file=sys.stderr,
            )

    write_diagnostics(
        args.out_dir / "tree_diagnostics.txt",
        tree,
        mutation_sets,
        skip_counts,
        n_violations,
        scite_agreement,
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
