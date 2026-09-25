"""Build the CNA-side tree TreeHDP consumes: SCICoNE's copy-number event
history over single cells, collapsed onto the same SECEDO clusters the SNV tree
(build_snv_tree.py) uses, in the same Newick contract, so the two trees are
directly comparable TreeHDP runs -- not merged into one tree.

Ours: neither SECEDO nor 10x's own CellRanger DNA pipeline produces a tree
over SECEDO's clusters. SCICoNE (cbg-ethz) infers a copy-number event history
from a cells x genomic-bins read-depth matrix; this script drives it through
its official Python wrapper (``scicone``, from ``SCICoNE/pyscicone``), which
runs the ``scicone-*`` binaries in the build directory, and collapses the
resulting cell-level tree onto SECEDO's clusters.

Contract (shared with build_snv_tree.py -- see its module docstring and
``_BaseTreeHDP`` in ``src/models/hdp_inference.py``)
    - Newick is labelled-internal-node form, rooted at ``GERMLINE_ROOT_ID``
      (imported from build_snv_tree, the same latent, spectrum-less root both
      trees share). Every other node is a real SECEDO cluster ID -- no hidden
      Steiner nodes.
    - The node set equals the SNV tree's: the tumour clusters, i.e. every
      SECEDO cluster except the pseudo-normal one (``--normal-cluster-id``,
      ``NORMAL_CLUSTER_ID`` in config.sh). See "Cluster-4 fold" below.
    - This tree carries no spectra: TreeHDP is run on it independently,
      comparing its recovered activities/signatures against the SNV tree's.

Flow (per-cell mode, the primary path)
    1. ``sci.read_10x(cnv_data.h5)``: GC-corrected counts, unmappable-bin
       removal, chromosome stops and CellRanger's outlier-cell filtering.
    2. ``sci.detect_breakpoints`` on a random subsample of cells
       (``--bp-max-cells``) with the chromosome stops as fixed breakpoints and
       a window of ``--bp-window-size`` bins (default 100, as in the
       notebook). The tree is then learnt on ALL filtered cells.
    3. ``sci.learn_tree(cluster=True, full=False)``: SCICoNE clusters the
       cells itself (PhenoGraph) and searches a cluster-level tree.
    4. Topology from ``tree.node_dict[node]['parent_id']`` (the wrapper's
       structured parent map), root = the one node with no parent. Each cell's
       node is ``tree.outputs['cell_node_ids'][:, -1]``, rows aligned to the
       filtered cells.
    5. Each SECEDO cluster majority-votes its cells onto a SCICoNE node
       (``majority_cluster_nodes``), then the tree is collapsed onto the
       clusters (``collapse_by_nearest_labelled_ancestor``, shared with
       build_snv_tree.py) and written as ``cna_tree.nwk``.

Three things this stage audits rather than assumes; each fails loudly, naming
what it tried, and is recorded in ``cna_tree_diagnostics.txt``:

    Filtered-cell barcodes. ``read_10x`` drops CellRanger's outlier cells
    (``is_high_dimapd``) and stores NO barcodes, so the rows of
    ``sci.data['filtered_counts']`` (and of the cell node ids) are not in raw
    h5 order. ``filtered_barcodes`` rebuilds the aligned barcodes from the two
    h5 datasets the wrapper itself uses (``cell_barcodes`` and
    ``per_cell_summary_metrics/is_high_dimapd``) and the same mask, and raises
    if either is missing or any length disagrees. A cell-to-cluster match rate
    below ``--min-match-fraction`` also raises, printing raw barcodes from both
    sides so a suffix mismatch (e.g. a trailing ``-1``) is legible.
    ``tree.cell_node_labels`` is deliberately not used: it holds letters for
    occupied nodes only and silently stays empty past 26 of them.

    Region neutral states. SCICoNE needs each chromosome's neutral copy
    number, which depends on sex. ``--sex`` is required with no default (it
    sets the tree root): female gives chrX 2 and chrY 0, male gives 1 and 1,
    autosomes 2. Chromosomes are mapped by name and an unrecognised one
    raises. As an audit, not a gate, the diagnostics report chrX and chrY
    depth relative to the autosomes.

    Cluster-4 fold. The pseudo-normal cluster's cells stay IN the SCICoNE
    inference (they anchor the diploid root and inform the breakpoints), but
    the cluster is dropped from the emitted node set, matching the SNV tree.
    The diagnostics record which SCICoNE node its cells voted for and whether
    that was SCICoNE's own root, the expected outcome. A tumour cluster that
    votes for SCICoNE's root has no CNA of its own, so it becomes a direct
    child of the germline root instead of displacing it. Clusters that share a
    SCICoNE node are reported as a finding, since the CNA tree cannot tell
    them apart.

Wrapper behaviours worth knowing (read from pyscicone's source):
    - ``learn_tree`` forwards keyword arguments to every replicate and lets
      them override its per-replicate seeds (42, 43, ...), so no ``seed`` is
      passed to it; that keeps the ``n_reps`` replicates independent, which
      the robustness score depends on.
    - The wrapper ignores non-zero exit codes from the binaries and then
      finds no output. Every result is therefore checked (breakpoints present,
      a non-empty ``node_dict``, one node id per row) rather than trusted.
    - ``learn_tree(cluster=False)`` returns ``None``, so the pseudobulk
      fallback calls ``learn_single_tree`` directly.
    - The wrapper changes directory while it runs; the work directory is
      resolved to an absolute path and used as the working directory.
    - ``detect_breakpoints`` unconditionally calls
      ``scicone.utils.get_region_gene_map`` afterwards, which queries Ensembl
      BioMart over the network (confirmed: compute nodes have none, so the
      job dies with ``ConnectionRefusedError``) and is GRCh38 besides, wrong
      for our GRCh37 data. There is no parameter to skip it (confirmed
      against the installed source: ``detect_breakpoints`` takes no such
      argument). It feeds only ``Tree.set_gene_event_dicts``, a purely
      additive annotation this script never reads (``node_dict``'s
      ``parent_id``/``region_event_dict``, ``outputs['cell_node_ids']`` and
      ``.score`` are untouched by it, confirmed by reading every use of
      ``region_gene_map`` in the package), so ``detect_bps`` stubs
      ``scicone.utils.get_region_gene_map`` to a no-op for the duration of
      the call (``_no_gene_mapping``), rather than routing it anywhere real.

Runtime. Breakpoint detection over all ~2000 cells at a default window is
impractical, so it sees ``--bp-max-cells`` cells (default 200, as in
pyscicone's 10x notebook). ``learn_tree`` uses ``--n-reps`` parallel workers
(default 10, so ask for that many CPUs), ``--copy-number-limit`` (default 4)
and ``--cluster-tree-n-iters`` (default 40000), the notebook's values, as is
the 100-bin window. The window is fixed rather than scaled with the bin count,
so if the diagnostics' region count looks off on a first run, adjust it. The
region-condensing step is done here with ``np.add.reduceat``, equivalent to
the wrapper's pure-Python loop but fast at this size.

``--pseudobulk-fallback`` is a last-resort escape hatch for when per-cell
inference proves impractical: filtered counts are averaged (not summed, which
would inject cluster-size-proportional depth) to one row per SECEDO cluster and
SCICoNE's single-tree search runs over those rows, so each cluster's node is
read off by row order. It is a real, flagged degradation (coarser than
cell-level history), never a silent substitution: the diagnostics record which
mode ran.

Breakpoint detection is the slow step (hours on the full ``cnv_data.h5``,
confirmed on a real run), so its result is persisted to
``<out_dir>/scicone_breakpoints.npz`` (``save_breakpoints``) once it succeeds.
``--reuse-breakpoints`` loads that file instead of recomputing
(``load_breakpoints``), raising if its saved bin count does not match the
current filtered counts -- a sign ``cnv_data.h5`` or its filtering changed
since it was saved, so the saved breakpoints no longer apply.
``cna_tree_diagnostics.txt`` records whether breakpoints were computed or
reused, how many were found, and ``--bp-limit`` (pyscicone's own cap on the
breakpoint count, exposed here rather than left at its default so a run that
hits it is visible rather than silently truncated).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Hashable, Iterator, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cluster_bams import read_clustering, read_map  # noqa: E402
from build_snv_tree import (  # noqa: E402
    GERMLINE_ROOT_ID,
    _sort_key,
    classify_topology,
    collapse_by_nearest_labelled_ancestor,
    digraph_to_newick,
    verify_newick,
)

# --------------------------------------------------------------------------- #
# Filtered-cell barcodes
# --------------------------------------------------------------------------- #

_BARCODES_DATASET = "cell_barcodes"
_OUTLIER_DATASET = "per_cell_summary_metrics/is_high_dimapd"


def filtered_barcodes(cnv_h5: Path, n_filtered: int) -> List[str]:
    """Barcodes aligned to the rows of ``sci.data['filtered_counts']``.

    ``read_10x`` keeps ``counts[:n_cells][~is_high_dimapd]`` with ``n_cells =
    len(cell_barcodes)`` and stores no barcodes, so they are rebuilt here from
    the same two datasets with the same mask. Raises ValueError if either
    dataset is missing, if the mask length differs from the barcode count, or
    if the masked length differs from ``n_filtered`` (the filtered matrix's
    row count).
    """
    import h5py

    with h5py.File(cnv_h5, "r") as f:
        for path in (_BARCODES_DATASET, _OUTLIER_DATASET):
            if path not in f:
                raise ValueError(
                    f"{cnv_h5} has no dataset {path!r}, needed to recover the "
                    "barcodes of the cells read_10x kept; top-level keys: "
                    f"{sorted(f.keys())}"
                )
        raw = f[_BARCODES_DATASET][()]
        outlier = np.asarray(f[_OUTLIER_DATASET][()]).astype(bool)

    barcodes = [b.decode() if isinstance(b, bytes) else str(b) for b in raw]
    if outlier.shape != (len(barcodes),):
        raise ValueError(
            f"{_OUTLIER_DATASET} has shape {outlier.shape} but {_BARCODES_DATASET} "
            f"has {len(barcodes)} entries; read_10x's mask cannot be reproduced"
        )
    kept = [b for b, drop in zip(barcodes, outlier) if not drop]
    if len(kept) != n_filtered:
        raise ValueError(
            f"masked barcodes number {len(kept)} but sci.data['filtered_counts'] "
            f"has {n_filtered} rows; the two do not describe the same cells"
        )
    if len(set(kept)) != len(kept):
        raise ValueError("duplicate barcodes among the filtered cells")
    return kept


_CB_PREFIX = "CB_"


def barcode_from_bam_name(name: str) -> str:
    """Recover the bare 10x barcode (with its ``-1`` suffix intact) from a
    per-cell BAM name: strip a trailing ``.bam``, then a single leading
    ``CB_`` if present. Stage 01's ``split_by_CBtag.py`` names per-cell BAMs
    ``CB_<barcode>.bam``, so stage 03's pileup ``.map`` file carries that
    prefix; ``cnv_data.h5``'s own barcodes never do.
    """
    if name.endswith(".bam"):
        name = name[:-4]
    if name.startswith(_CB_PREFIX):
        name = name[len(_CB_PREFIX) :]
    return name


def build_barcode_to_cluster(map_file: Path, clustering_file: Path) -> Dict[str, str]:
    """``{barcode: cluster_id}`` for every cell SECEDO actually clustered
    (cluster 0, "no cluster", is excluded -- same convention as
    build_cluster_bams.py). Reuses stage 05's own map/clustering readers
    rather than reimplementing that join.
    """
    idx_to_name = read_map(map_file)
    clusters = read_clustering(clustering_file)
    if len(clusters) != len(idx_to_name):
        raise ValueError(
            f"clustering has {len(clusters)} cells but map file has "
            f"{len(idx_to_name)} -- these must come from the same secedo run."
        )
    out: Dict[str, str] = {}
    for idx, cluster_id in enumerate(clusters):
        if cluster_id == 0:
            continue
        out[barcode_from_bam_name(idx_to_name[idx])] = str(cluster_id)
    return out


def check_match_rate(
    barcodes: List[str],
    barcode_to_cluster: Dict[str, str],
    min_fraction: float,
    n_show: int = 5,
) -> int:
    """Number of ``barcodes`` present in ``barcode_to_cluster``. Raises
    ValueError, printing raw sample barcodes from both sides, if the matched
    fraction is below ``min_fraction`` (a suffix or naming mismatch is the
    likely cause)."""
    n_matched = sum(1 for b in barcodes if b in barcode_to_cluster)
    fraction = n_matched / len(barcodes) if barcodes else 0.0
    if fraction < min_fraction:
        raise ValueError(
            f"only {n_matched}/{len(barcodes)} filtered cnv_data.h5 cells "
            f"({fraction:.1%}) match a SECEDO-clustered barcode, below "
            f"{min_fraction:.0%}. SCICoNE-side barcodes (cnv_data.h5): "
            f"{barcodes[:n_show]}; SECEDO-side barcodes (stage 03 map): "
            f"{list(barcode_to_cluster)[:n_show]}"
        )
    return n_matched


# --------------------------------------------------------------------------- #
# Chromosomes, neutral states, sex audit
# --------------------------------------------------------------------------- #


def _chrom_key(name: str) -> str:
    key = str(name).strip().upper()
    return key[3:] if key.startswith("CHR") else key


def chromosome_neutral_states(chrom_names: List[str], sex: str) -> List[int]:
    """Neutral copy number per chromosome, in ``chrom_names`` order: 2 for
    chr1-22; chrX 2 (female) or 1 (male); chrY 0 (female) or 1 (male). Names
    match with or without a ``chr`` prefix. Raises ValueError for an
    unrecognised chromosome or sex rather than defaulting to 2.
    """
    if sex not in ("female", "male"):
        raise ValueError(f"sex must be 'female' or 'male', got {sex!r}")
    sex_states = {"X": 2, "Y": 0} if sex == "female" else {"X": 1, "Y": 1}
    autosomes = {str(i) for i in range(1, 23)}
    states = []
    for name in chrom_names:
        key = _chrom_key(name)
        if key in autosomes:
            states.append(2)
        elif key in sex_states:
            states.append(sex_states[key])
        else:
            raise ValueError(
                f"unrecognised chromosome {name!r}: expected 1-22, X or Y (with "
                f"or without a 'chr' prefix); chromosomes found: {list(chrom_names)}"
            )
    return states


def sex_chromosome_depth(
    counts: np.ndarray, chr_stops: Dict[str, int]
) -> Dict[str, Optional[float]]:
    """chrX and chrY depth relative to the autosomes, as an audit of ``--sex``.

    Counts are library-normalised per cell (scaled so a flat cell is 1), then
    each bin's mean over cells is taken, and the result is the median over the
    chromosome's bins divided by the median over autosomal bins. A female
    sample should give X near 1 and Y near 0; male, both near 0.5. None when
    the chromosome has no bins after filtering.
    """
    n_bins = counts.shape[1]
    lib = counts.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1
    per_bin = (counts / lib * n_bins).mean(axis=0)

    names = list(chr_stops)
    starts = [0] + [int(chr_stops[n]) + 1 for n in names[:-1]]
    spans = {n: (s, int(chr_stops[n]) + 1) for n, s in zip(names, starts)}
    autosomes = {str(i) for i in range(1, 23)}
    auto_cols = np.concatenate(
        [
            np.arange(*spans[n])
            for n in names
            if _chrom_key(n) in autosomes and spans[n][1] > spans[n][0]
        ]
        or [np.array([], dtype=int)]
    )
    if auto_cols.size == 0:
        return {"X": None, "Y": None}
    auto_median = float(np.median(per_bin[auto_cols]))

    out: Dict[str, Optional[float]] = {"X": None, "Y": None}
    for name in names:
        key = _chrom_key(name)
        lo, hi = spans[name]
        if key in out and hi > lo and auto_median > 0:
            out[key] = float(np.median(per_bin[lo:hi]) / auto_median)
    return out


# --------------------------------------------------------------------------- #
# Region condensing and pseudobulk
# --------------------------------------------------------------------------- #


def condense_regions(counts: np.ndarray, region_sizes) -> np.ndarray:
    """Sum bins within each breakpoint-delimited region: ``(n_rows,
    n_regions)``. Equivalent to ``SCICoNE.condense_regions``, vectorised.
    Raises ValueError if the region sizes are not positive or do not add up to
    the bin count."""
    sizes = np.asarray(region_sizes).ravel().astype(int)
    if sizes.size == 0 or np.any(sizes <= 0) or sizes.sum() != counts.shape[1]:
        raise ValueError(
            f"region sizes (n={sizes.size}, sum={sizes.sum()}) must be positive "
            f"and add up to the {counts.shape[1]} bins"
        )
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    return np.add.reduceat(counts, starts, axis=1)


def subsample_rows(n_rows: int, max_rows: int, seed: int) -> np.ndarray:
    """Sorted random row indices, at most ``max_rows`` of ``n_rows`` (all rows
    if ``max_rows`` is not positive or already covers them)."""
    if max_rows <= 0 or max_rows >= n_rows:
        return np.arange(n_rows)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=max_rows, replace=False))


def aggregate_counts_by_cluster(
    counts: np.ndarray, barcodes: List[str], barcode_to_cluster: Dict[str, str]
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Average per-cell counts into one row per SECEDO cluster: ``(mean counts
    (n_clusters, n_bins), cluster_order, cells per cluster)``. The mean, not the
    sum, so a cluster's depth does not scale with its size. Cells with no
    cluster match (barcode not in ``barcode_to_cluster``) are skipped.
    """
    cluster_rows: Dict[str, List[int]] = defaultdict(list)
    for row, barcode in enumerate(barcodes):
        cluster_id = barcode_to_cluster.get(barcode)
        if cluster_id is not None:
            cluster_rows[cluster_id].append(row)

    cluster_order = sorted(cluster_rows, key=_sort_key)
    pseudobulk = np.zeros((len(cluster_order), counts.shape[1]), dtype=float)
    sizes = np.zeros(len(cluster_order))
    for i, cluster_id in enumerate(cluster_order):
        pseudobulk[i] = counts[cluster_rows[cluster_id]].mean(axis=0)
        sizes[i] = len(cluster_rows[cluster_id])
    return pseudobulk, cluster_order, sizes


# --------------------------------------------------------------------------- #
# SCICoNE tree -> topology and cell assignments
# --------------------------------------------------------------------------- #

_NO_PARENT = {"", "null", "none", "root"}


def node_parent_map(node_dict: Dict[str, dict]) -> Tuple[Dict[str, str], str]:
    """``({child: parent}, root_node)`` from a SCICoNE ``tree.node_dict``.

    The root is the one node whose ``parent_id`` is empty or NULL. Raises
    ValueError if ``node_dict`` is empty, there is not exactly one such node,
    or a parent id names no node.
    """
    if not node_dict:
        raise ValueError(
            "tree.node_dict is empty: SCICoNE produced no readable tree (the "
            "wrapper ignores failing binaries, so check the inference logs)"
        )
    roots = [
        str(n)
        for n, d in node_dict.items()
        if str(d.get("parent_id")).strip().lower() in _NO_PARENT
    ]
    if len(roots) != 1:
        raise ValueError(
            f"expected exactly one root node (parent_id NULL), found {roots}; "
            f"node_dict keys: {sorted(node_dict)}"
        )
    parent_of: Dict[str, str] = {}
    for node, d in node_dict.items():
        node = str(node)
        if node == roots[0]:
            continue
        parent = str(d["parent_id"])
        if parent not in {str(k) for k in node_dict}:
            raise ValueError(f"node {node!r} has parent {parent!r}, not a known node")
        parent_of[node] = parent
    return parent_of, roots[0]


def cell_node_ids(tree, n_rows: int, known_nodes) -> List[str]:
    """Node id (string, matching ``node_dict`` keys) of every input row, from
    ``tree.outputs['cell_node_ids'][:, -1]``. Raises ValueError if the output
    is missing, has the wrong row count, or names a node not in
    ``known_nodes``."""
    if "cell_node_ids" not in tree.outputs:
        raise ValueError(
            "tree.outputs has no 'cell_node_ids': SCICoNE wrote no cell "
            f"assignment; outputs found: {sorted(tree.outputs)}"
        )
    raw = np.asarray(tree.outputs["cell_node_ids"])
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if raw.shape[0] != n_rows:
        raise ValueError(
            f"cell_node_ids has {raw.shape[0]} rows, expected {n_rows} (one per "
            "input row)"
        )
    nodes = [str(int(v)) for v in raw[:, -1]]
    known = {str(k) for k in known_nodes}
    unknown = sorted(set(nodes) - known)
    if unknown:
        raise ValueError(f"cell_node_ids names nodes absent from node_dict: {unknown}")
    return nodes


# --------------------------------------------------------------------------- #
# Majority-vote collapse, cluster-4 fold
# --------------------------------------------------------------------------- #


def majority_cluster_nodes(
    cell_to_node: Dict[str, str], barcode_to_cluster: Dict[str, str]
) -> Tuple[Dict[str, str], List[str]]:
    """``{cluster_id: majority SCICoNE node}`` -- among a cluster's own
    cells (barcodes both SECEDO-clustered and SCICoNE-assigned), the node
    the largest number of them share; ties broken by the node's own sort
    key for determinism.

    Returns ``(resolved, unmatched_clusters)`` -- clusters with no cell
    present in ``cell_to_node`` are reported, not silently dropped.
    """
    cluster_nodes: Dict[str, List[str]] = defaultdict(list)
    for barcode, cluster_id in barcode_to_cluster.items():
        node = cell_to_node.get(barcode)
        if node is not None:
            cluster_nodes[cluster_id].append(node)

    resolved: Dict[str, str] = {}
    for cluster_id, nodes in cluster_nodes.items():
        counts = Counter(nodes)
        max_count = max(counts.values())
        winners = sorted(n for n, c in counts.items() if c == max_count)
        resolved[cluster_id] = winners[0]

    all_clusters = set(barcode_to_cluster.values())
    unmatched = sorted(all_clusters - set(resolved), key=_sort_key)
    return resolved, unmatched


def fold_normal_cluster(
    node_of_cluster: Dict[str, str], normal_id: str, root_node: str
) -> Tuple[Dict[str, str], str, bool]:
    """Drop the pseudo-normal cluster from the node set (it folds into the
    germline root). Returns ``(tumour node_of_cluster, the SCICoNE node the
    normal cluster voted for, whether that node is SCICoNE's root)``. Raises
    ValueError if the normal cluster placed no cell, since then the fold
    cannot be audited.
    """
    if normal_id not in node_of_cluster:
        raise ValueError(
            f"normal cluster {normal_id!r} has no cell in the SCICoNE output "
            f"(clusters resolved: {sorted(node_of_cluster, key=_sort_key)}); "
            "check --normal-cluster-id against stage 04/05's clustering"
        )
    normal_node = node_of_cluster[normal_id]
    tumour = {c: n for c, n in node_of_cluster.items() if c != normal_id}
    return tumour, normal_node, normal_node == root_node


def build_cna_tree(
    parent_of: Dict[str, str],
    tumour_nodes: Dict[str, str],
    root_node: str,
):
    """Collapse SCICoNE's tree onto the tumour clusters. A cluster whose
    majority node is SCICoNE's own root has no CNA of its own: it hangs
    directly off ``GERMLINE_ROOT_ID`` and is never an ancestor of others
    (which would displace the root). Returns ``(tree, clusters_at_root)``.
    """
    at_root = sorted(
        (c for c, n in tumour_nodes.items() if n == root_node), key=_sort_key
    )
    placed = {c: n for c, n in tumour_nodes.items() if n != root_node}
    tree = collapse_by_nearest_labelled_ancestor(parent_of, placed, GERMLINE_ROOT_ID)
    for cid in at_root:
        tree.add_edge(GERMLINE_ROOT_ID, cid)
    return tree, at_root


def shared_node_groups(node_of_cluster: Dict[str, str]) -> List[List[str]]:
    """Groups of clusters that voted for the same SCICoNE node."""
    by_node: Dict[Hashable, List[str]] = defaultdict(list)
    for cid, node in node_of_cluster.items():
        by_node[node].append(cid)
    return [
        sorted(cids, key=_sort_key)
        for _, cids in sorted(by_node.items(), key=lambda kv: str(kv[0]))
        if len(cids) > 1
    ]


# --------------------------------------------------------------------------- #
# SCICoNE runs (wrapper mocked in tests)
# --------------------------------------------------------------------------- #


def _import_scicone():
    try:
        import scicone
    except ImportError as e:
        raise RuntimeError(
            "cannot import scicone (pyscicone). Install it from the clone: "
            "pip install realdata/external/SCICoNE/pyscicone/ (it needs "
            f"PhenoGraph too): {e}"
        ) from e
    return scicone


@contextlib.contextmanager
def _working_dir(path: Path) -> Iterator[None]:
    """The wrapper writes intermediate files relative to the current
    directory and changes directory itself, so run it from a known one."""
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextlib.contextmanager
def _no_gene_mapping(scicone_module) -> Iterator[None]:
    """``detect_breakpoints`` unconditionally calls
    ``scicone.utils.get_region_gene_map``, which queries Ensembl BioMart over
    the network -- unreachable from a compute node, and GRCh38 besides, wrong
    for our GRCh37 data (see module docstring). It feeds only
    ``Tree.set_gene_event_dicts``, an annotation this script never reads, so
    it is stubbed to a no-op for the scope of this context manager and
    restored afterwards, rather than routed anywhere real.
    """
    original = scicone_module.utils.get_region_gene_map
    scicone_module.utils.get_region_gene_map = lambda *args, **kwargs: None
    try:
        yield
    finally:
        scicone_module.utils.get_region_gene_map = original


def detect_bps(
    sci, scicone_module, counts, stops, subset, window_size, threshold, bp_limit
):
    """Breakpoints from a cell subsample, with the chromosome stops as fixed
    breakpoints, run under ``_no_gene_mapping`` so the unconditional Ensembl
    query never fires. Raises ValueError if the wrapper returned none (it
    ignores a failing binary).
    """
    with _no_gene_mapping(scicone_module):
        bps = sci.detect_breakpoints(
            data=counts[subset],
            window_size=window_size,
            threshold=threshold,
            input_breakpoints=list(stops.values()),
            bp_limit=bp_limit,
        )
    for key in ("segmented_regions", "segmented_region_sizes"):
        if key not in bps:
            raise ValueError(
                f"detect_breakpoints returned no {key!r}; keys: {sorted(bps)}. "
                "The breakpoint binary probably failed; see its output above."
            )
    return bps


_BREAKPOINTS_FILE = "scicone_breakpoints.npz"


def save_breakpoints(out_dir: Path, bps: Dict[str, np.ndarray], n_bins: int) -> Path:
    """Persist ``detect_breakpoints``' result to ``out_dir`` (the slow step,
    hours on the full ``cnv_data.h5``), so a rerun can load it via
    ``load_breakpoints`` instead of recomputing. ``n_bins`` (the current
    filtered bin count) is saved alongside so a rerun can tell whether it
    still applies. Returns the path written.
    """
    path = out_dir / _BREAKPOINTS_FILE
    np.savez(
        path,
        segmented_regions=np.asarray(bps["segmented_regions"]),
        segmented_region_sizes=np.asarray(bps["segmented_region_sizes"]),
        n_bins=np.asarray(n_bins),
    )
    return path


def load_breakpoints(out_dir: Path, n_bins: int) -> Dict[str, np.ndarray]:
    """Load a breakpoint result saved by ``save_breakpoints``. Raises
    FileNotFoundError, naming the path, if ``--reuse-breakpoints`` was passed
    but nothing was saved yet, and ValueError if the saved bin count does not
    match ``n_bins`` -- the filtered counts have changed (a different
    ``cnv_data.h5``, or different outlier filtering) and the saved
    breakpoints no longer apply.
    """
    path = out_dir / _BREAKPOINTS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"--reuse-breakpoints was passed but {path} does not exist; run "
            "once without it first"
        )
    saved = np.load(path)
    saved_n_bins = int(saved["n_bins"])
    if saved_n_bins != n_bins:
        raise ValueError(
            f"{path} was saved for {saved_n_bins} bins, but the current "
            f"filtered counts have {n_bins} -- cnv_data.h5 or its filtering "
            "must have changed; rerun without --reuse-breakpoints"
        )
    return {
        "segmented_regions": saved["segmented_regions"],
        "segmented_region_sizes": saved["segmented_region_sizes"],
    }


def run_per_cell(sci, seg_counts, region_sizes, neutral_states, opts):
    """Cell-level tree via ``learn_tree(cluster=True, full=False)``. No seed
    is passed: it would override the per-replicate seeds (module docstring)."""
    tree = sci.learn_tree(
        seg_counts,
        region_sizes,
        region_neutral_states=neutral_states,
        cluster=True,
        full=False,
        n_reps=opts["n_reps"],
        max_tries=opts["max_tries"],
        copy_number_limit=opts["copy_number_limit"],
        cluster_tree_n_iters=opts["cluster_tree_n_iters"],
    )
    if tree is None:
        raise ValueError("learn_tree returned no tree")
    return tree


def run_pseudobulk(sci, seg_counts, region_sizes, neutral_states, cluster_sizes, opts):
    """Single tree over one row per cluster. Zero-neutral regions (chrY in a
    female) are dropped, as ``learn_tree`` does for the per-cell path."""
    states = np.asarray(neutral_states)
    keep = states != 0
    tree = sci.learn_single_tree(
        seg_counts[:, keep],
        np.asarray(region_sizes).ravel()[keep],
        n_iters=opts["cluster_tree_n_iters"],
        region_neutral_states=states[keep],
        copy_number_limit=opts["copy_number_limit"],
        cluster_sizes=cluster_sizes,
    )
    if tree is None:
        raise ValueError("learn_single_tree returned no tree")
    return tree


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cnv-h5", required=True, type=Path, help="cnv_data.h5")
    p.add_argument(
        "--map-file", required=True, type=Path, help="stage 03's chromosome_1.map"
    )
    p.add_argument(
        "--clustering-file", required=True, type=Path, help="stage 04's clustering"
    )
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--scicone-build-dir",
        required=True,
        type=Path,
        help="directory holding the scicone-* binaries",
    )
    p.add_argument(
        "--sex",
        required=True,
        choices=["female", "male"],
        help="patient sex, sets chrX/chrY neutral copy number; no default",
    )
    p.add_argument(
        "--normal-cluster-id",
        required=True,
        help="pseudo-normal SECEDO cluster (NORMAL_CLUSTER_ID in config.sh)",
    )
    p.add_argument("--pseudobulk-fallback", action="store_true")
    p.add_argument("--bp-max-cells", type=int, default=200)
    p.add_argument("--bp-window-size", type=int, default=100)
    p.add_argument("--bp-threshold", type=float, default=3.0)
    p.add_argument(
        "--bp-limit",
        type=int,
        default=300,
        help="pyscicone's own cap on the number of breakpoints detect_breakpoints "
        "reports (its own default); exposed here so a run that hits it is visible",
    )
    p.add_argument(
        "--reuse-breakpoints",
        action="store_true",
        help="load a previously saved breakpoint result from --out-dir instead of "
        "recomputing (see save_breakpoints/load_breakpoints); raises if none was "
        "saved or if the bin count no longer matches",
    )
    p.add_argument("--n-reps", type=int, default=10)
    p.add_argument("--copy-number-limit", type=int, default=4)
    p.add_argument("--cluster-tree-n-iters", type=int, default=40000)
    p.add_argument("--max-tries", type=int, default=1)
    p.add_argument("--min-match-fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42, help="breakpoint cell subsample")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = args.out_dir.resolve()
    work_dir = (args.out_dir / "scicone_tmp").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    scicone = _import_scicone()
    sci = scicone.SCICoNE(str(args.scicone_build_dir), str(work_dir), verbose=False)
    with _working_dir(work_dir):
        sci.read_10x(str(args.cnv_h5.resolve()))
    counts = sci.data["filtered_counts"]
    stops = sci.data["filtered_chromosome_stops"]
    n_cells, n_bins = counts.shape

    barcodes = filtered_barcodes(args.cnv_h5, n_cells)
    barcode_to_cluster = build_barcode_to_cluster(args.map_file, args.clustering_file)
    n_matched = check_match_rate(barcodes, barcode_to_cluster, args.min_match_fraction)
    normal_id = str(args.normal_cluster_id)

    chrom_states = chromosome_neutral_states(list(stops), args.sex)
    depth = sex_chromosome_depth(counts, stops)

    mode = "pseudobulk" if args.pseudobulk_fallback else "per_cell"
    window = args.bp_window_size
    subset = subsample_rows(n_cells, args.bp_max_cells, args.seed)
    opts = {
        "n_reps": args.n_reps,
        "max_tries": args.max_tries,
        "copy_number_limit": args.copy_number_limit,
        "cluster_tree_n_iters": args.cluster_tree_n_iters,
    }

    with _working_dir(work_dir):
        if args.reuse_breakpoints:
            bps = load_breakpoints(args.out_dir, n_bins)
            breakpoints_reused = True
        else:
            bps = detect_bps(
                sci, scicone, counts, stops, subset, window, args.bp_threshold,
                args.bp_limit,
            )  # fmt: skip
            save_breakpoints(args.out_dir, bps, n_bins)
            breakpoints_reused = False
        n_breakpoints_found = len(np.asarray(bps["segmented_regions"]).ravel())
        region_sizes = np.asarray(bps["segmented_region_sizes"]).ravel()
        neutral_states = scicone.utils.set_region_neutral_states(
            bps["segmented_regions"], list(stops.values()), chrom_states
        )

        if mode == "pseudobulk":
            pseudo, row_order, cluster_sizes = aggregate_counts_by_cluster(
                counts, barcodes, barcode_to_cluster
            )
            tree = run_pseudobulk(
                sci,
                condense_regions(pseudo, region_sizes),
                region_sizes,
                neutral_states,
                cluster_sizes,
                opts,
            )
        else:
            tree = run_per_cell(
                sci,
                condense_regions(counts, region_sizes),
                region_sizes,
                neutral_states,
                opts,
            )

    parent_of, root_node = node_parent_map(tree.node_dict)
    if mode == "pseudobulk":
        row_nodes = cell_node_ids(tree, len(row_order), tree.node_dict)
        node_of_cluster = dict(zip(row_order, row_nodes))
        unmatched: List[str] = []
    else:
        row_nodes = cell_node_ids(tree, n_cells, tree.node_dict)
        node_of_cluster, unmatched = majority_cluster_nodes(
            dict(zip(barcodes, row_nodes)), barcode_to_cluster
        )

    tumour_nodes, normal_node, normal_at_root = fold_normal_cluster(
        node_of_cluster, normal_id, root_node
    )
    cna_tree, at_root = build_cna_tree(parent_of, tumour_nodes, root_node)
    newick_str = digraph_to_newick(cna_tree, GERMLINE_ROOT_ID)
    verify_newick(newick_str, set(tumour_nodes), GERMLINE_ROOT_ID)
    (args.out_dir / "cna_tree.nwk").write_text(newick_str + "\n")

    if mode == "per_cell":
        rows = ["barcode,secedo_cluster,scicone_node"] + [
            f"{b},{barcode_to_cluster.get(b, '')},{n}"
            for b, n in zip(barcodes, row_nodes)
        ]
        (args.out_dir / "cna_cell_nodes.csv").write_text("\n".join(rows) + "\n")

    def _fmt(x: Optional[float]) -> str:
        return "n/a (no bins after filtering)" if x is None else f"{x:.2f}"

    shared = shared_node_groups(tumour_nodes)
    normal_where = "the root, as expected" if normal_at_root else "NOT the root"
    diagnostics = [
        "# Stage 09 (CNA tree) diagnostics\n\n",
        f"## Mode: {mode}\n",
        f"Filtered cells (read_10x): {n_cells} x {n_bins} bins\n",
        f"Cells matched to a SECEDO cluster: {n_matched}/{n_cells}\n",
        "\n## Sex and neutral states\n",
        f"--sex {args.sex}; neutral state per chromosome: "
        f"{dict(zip(stops, chrom_states))}\n",
        f"Depth relative to autosome median: chrX {_fmt(depth['X'])}, "
        f"chrY {_fmt(depth['Y'])} (female expects X near 1, Y near 0; male, "
        "both near 0.5)\n",
        "\n## Breakpoints and tree search\n",
        f"Breakpoints: {'reused' if breakpoints_reused else 'computed'} "
        f"({args.out_dir / _BREAKPOINTS_FILE}); {n_breakpoints_found} found "
        f"(bp_limit {args.bp_limit})\n",
        f"Breakpoint cells: {len(subset)}/{n_cells}, window {window}, "
        f"threshold {args.bp_threshold}; {len(region_sizes)} regions\n",
        f"n_reps {args.n_reps}, copy_number_limit {args.copy_number_limit}, "
        f"cluster_tree_n_iters {args.cluster_tree_n_iters}, "
        f"max_tries {args.max_tries}\n",
        f"SCICoNE tree: {len(tree.node_dict)} nodes, root node {root_node}, "
        f"score {tree.score}\n",
        "\n## Cluster placement\n",
        f"Clusters resolved to a SCICoNE node: {len(node_of_cluster)}\n",
        f"Cluster-4 fold: normal cluster {normal_id} voted for SCICoNE node "
        f"{normal_node} ({normal_where}); folded into {GERMLINE_ROOT_ID}, "
        "not emitted\n",
    ]
    if n_breakpoints_found >= args.bp_limit:
        diagnostics.append(
            f"FINDING: breakpoint detection hit its cap of {args.bp_limit} -- "
            "there may be more breakpoints than reported; rerun with a higher "
            "--bp-limit if that matters here.\n"
        )
    if not normal_at_root:
        diagnostics.append(
            "FINDING: the pseudo-normal cluster did not vote for SCICoNE's root, "
            "so the root may not be diploid-like; inspect before trusting.\n"
        )
    if at_root:
        diagnostics.append(
            f"FINDING: tumour cluster(s) {at_root} voted for SCICoNE's root (no "
            f"CNA of their own); placed directly under {GERMLINE_ROOT_ID}.\n"
        )
    if shared:
        diagnostics.append(
            f"FINDING: cluster group(s) {shared} share one SCICoNE node, so the "
            "CNA tree cannot tell them apart; their placement among siblings is "
            "arbitrary.\n"
        )
    if unmatched:
        diagnostics.append(
            f"FINDING: {len(unmatched)} cluster(s) had no cell present in "
            f"SCICoNE's own output, so could not be placed: {unmatched}\n"
        )
    diagnostics.append(f"\nEmitted tree: {classify_topology(cna_tree)}\n")
    if classify_topology(cna_tree).startswith("linear chain"):
        diagnostics.append("FINDING: the emitted CNA tree is a non-branching chain.\n")
    (args.out_dir / "cna_tree_diagnostics.txt").write_text("".join(diagnostics))

    print(
        f"Mode: {mode}. Wrote cna_tree.nwk and cna_tree_diagnostics.txt "
        f"to {args.out_dir}."
    )


if __name__ == "__main__":
    main()
