"""Build the CNA-side tree TreeHDP consumes: SCICoNE's copy-number event
history over single cells, collapsed onto the same six SECEDO clusters the
SNV tree (build_snv_tree.py) uses, in the same Newick contract, so the two
trees are directly comparable TreeHDP runs -- not merged into one tree.

Ours: neither SECEDO nor 10x's own CellRanger DNA pipeline produces a tree
over SECEDO's clusters. SCICoNE (Kuipers/Jahn lab, cbg-ethz) infers a
copy-number event history from a cells x genomic-bins read-depth matrix;
this script gets that matrix, runs SCICoNE, and collapses its cell-level tree
onto SECEDO's clusters.

Contract (shared with build_snv_tree.py -- see its module docstring and
``_BaseTreeHDP`` in ``src/models/hdp_inference.py``)
    - Newick is labelled-internal-node form, rooted at ``GERMLINE_ROOT_ID``
      (imported from build_snv_tree, the same latent, spectrum-less root both
      trees share). Every other node is a real SECEDO cluster ID -- no hidden
      Steiner nodes.
    - This tree carries no spectra (unlike the SNV tree): TreeHDP is run on
      it independently, comparing its recovered activities/signatures
      against the SNV tree's, not merging the two.

Input path: 10x's own CellRanger DNA per-cell CNV output (``cnv_data.h5``),
not bins derived from the per-cell BAMs ourselves -- confirmed hosted at the
same download host as breast_tissue_D's BAM (HTTP 200, ~2.5 GB), and it is
CellRanger DNA's own already-binned, GC/mappability-aware read-depth matrix,
so this avoids re-implementing that correction from scratch. ``config.sh``'s
``00_download.sbatch`` fetches it.

CAVEAT, stated plainly rather than smoothed over: this script was written
without access to a real ``cnv_data.h5`` file or a built SCICoNE binary (10x's
own schema documentation page could not be reached while writing this, and no
SCICoNE output sample was available either). Three places this matters,
each handled the same way -- try documented/plausible defaults first, but
never guess silently:

    1. ``load_cnv_counts_matrix`` tries a short list of plausible dataset
       paths first, then falls back to a shape/name heuristic over every
       leaf dataset in the file, and raises loudly (listing every dataset
       path and shape found) if nothing matches. Confirm the real
       ``cnv_data.h5``'s actual layout with ``h5py`` on the first real Euler
       run and extend ``_COUNTS_CANDIDATE_PATHS``/``_BARCODE_CANDIDATE_PATHS``
       if the heuristic ever fires.
    2. ``write_scicone_matrix``/``run_scicone`` write a plain whitespace-
       delimited integer matrix and invoke the SCICoNE binary by the CLI
       shape its README documents (cells x bins counts in, a tree plus a
       per-cell node assignment out) -- confirm the exact flags against
       ``scicone --help`` (or the SCICoNE python package, if that is the
       preferred interface once built) on Euler before trusting a real run.
    3. ``parse_scicone_edges``/``parse_scicone_cell_assignment`` assume a
       plain edge-list (``parent<TAB>child`` per line) and a two-column
       ``cell<TAB>node`` assignment file -- SCICoNE's actual output file
       names and exact format need confirming the same way.

None of these raise silently on a mismatch: every parser here fails loudly,
naming what it tried, rather than fabricate a tree from a guessed format --
same discipline as build_snv_tree.py's LICHeE integration.

Cell-level inference may simply prove impractical (too few informative bins
after GC/mappability filtering, SCICoNE failing to converge at ~2000 cells,
runtime past the SLURM budget). ``--pseudobulk-fallback`` is the documented
escape hatch: aggregate the per-cell matrix to one row per SECEDO cluster
(summing raw counts) and run SCICoNE over that pseudobulk matrix instead --
SCICoNE's own tree nodes then correspond directly to SECEDO clusters (via
the same label-matching scheme build_snv_tree.py's LICHeE integration uses),
skipping the majority-vote collapse. This is a real, flagged degradation
(coarser than cell-level history), not a silent substitution: ``main``
prints and records in tree_diagnostics.txt which mode actually ran.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cluster_bams import read_clustering, read_map  # noqa: E402
from build_snv_tree import (  # noqa: E402
    GERMLINE_ROOT_ID,
    _sample_label_schemes,
    _sort_key,
    classify_topology,
    collapse_by_nearest_labelled_ancestor,
    digraph_to_newick,
    verify_newick,
)

# --------------------------------------------------------------------------- #
# cnv_data.h5 extraction
# --------------------------------------------------------------------------- #

_COUNTS_CANDIDATE_PATHS = [
    "raw_counts",
    "constants/raw_counts",
    "cnvs/raw_counts",
    "normalized_counts",
]
_BARCODE_CANDIDATE_PATHS = [
    "cell_barcodes",
    "barcodes",
    "constants/cell_barcodes",
]


def _walk_h5_datasets(group, prefix: str = "") -> Iterable[Tuple[str, object]]:
    """Yield ``(full_path, h5py.Dataset)`` for every leaf dataset under
    ``group``, recursing through subgroups."""
    import h5py

    for key in group.keys():
        item = group[key]
        path = f"{prefix}{key}"
        if isinstance(item, h5py.Group):
            yield from _walk_h5_datasets(item, prefix=f"{path}/")
        else:
            yield path, item


def _find_counts_dataset(datasets: Dict[str, object]) -> str:
    """Resolve the cells x bins raw-counts dataset path: an exact candidate
    from ``_COUNTS_CANDIDATE_PATHS`` first, else the largest 2D numeric
    dataset whose path mentions "raw" (case-insensitive), else the largest
    2D numeric dataset at all. Raises ValueError, listing every dataset path
    and shape found, if nothing 2D exists.
    """
    for candidate in _COUNTS_CANDIDATE_PATHS:
        if candidate in datasets:
            return candidate

    two_d = {p: d for p, d in datasets.items() if getattr(d, "ndim", 0) == 2}
    if not two_d:
        shapes = {p: getattr(d, "shape", None) for p, d in datasets.items()}
        raise ValueError(
            "no 2D dataset found in cnv_data.h5 to use as a cells x bins "
            f"counts matrix (see module docstring's caveat); datasets found: {shapes}"
        )
    raw_named = {p: d for p, d in two_d.items() if "raw" in p.lower()}
    pool = raw_named if raw_named else two_d
    return max(pool, key=lambda p: pool[p].shape[1])


def _find_barcode_dataset(datasets: Dict[str, object], n_cells: int) -> str:
    """Resolve the per-cell barcode dataset path: an exact candidate from
    ``_BARCODE_CANDIDATE_PATHS`` first, else any 1D dataset of length
    ``n_cells``. Raises ValueError, listing every dataset path and shape
    found, if nothing matches.
    """
    for candidate in _BARCODE_CANDIDATE_PATHS:
        if candidate in datasets and datasets[candidate].shape[0] == n_cells:
            return candidate

    one_d = {
        p: d
        for p, d in datasets.items()
        if getattr(d, "ndim", 0) == 1 and d.shape[0] == n_cells
    }
    if not one_d:
        shapes = {p: getattr(d, "shape", None) for p, d in datasets.items()}
        raise ValueError(
            f"no length-{n_cells} 1D dataset found in cnv_data.h5 to use as "
            f"cell barcodes (see module docstring's caveat); datasets found: {shapes}"
        )
    return sorted(one_d)[0]


def load_cnv_counts_matrix(
    h5_path: Path,
) -> Tuple[np.ndarray, List[str], str, str]:
    """Load the cells x bins raw-counts matrix and per-cell barcodes from a
    CellRanger DNA ``cnv_data.h5``. Returns ``(counts, barcodes,
    counts_path, barcodes_path)`` -- the resolved dataset paths are returned
    too so callers can log exactly what was used (see module docstring's
    caveat on this being schema-inferred, not confirmed).
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        datasets = dict(_walk_h5_datasets(f))
        counts_path = _find_counts_dataset(datasets)
        counts = np.asarray(datasets[counts_path][()])
        barcodes_path = _find_barcode_dataset(datasets, counts.shape[0])
        raw_barcodes = datasets[barcodes_path][()]
        barcodes = [
            b.decode() if isinstance(b, bytes) else str(b) for b in raw_barcodes
        ]
    return counts, barcodes, counts_path, barcodes_path


# --------------------------------------------------------------------------- #
# Barcode <-> SECEDO cluster mapping
# --------------------------------------------------------------------------- #


def barcode_from_bam_name(name: str) -> str:
    """Strip a trailing ``.bam`` extension from a per-cell BAM name to
    recover its 10x barcode, matching build_cluster_bams.py's own per-cell
    BAM naming (and hence cnv_data.h5's raw barcode strings)."""
    return name[:-4] if name.endswith(".bam") else name


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


# --------------------------------------------------------------------------- #
# Pseudobulk aggregation (--pseudobulk-fallback)
# --------------------------------------------------------------------------- #


def aggregate_counts_by_cluster(
    counts: np.ndarray, barcodes: List[str], barcode_to_cluster: Dict[str, str]
) -> Tuple[np.ndarray, List[str]]:
    """Sum raw per-cell counts into one row per SECEDO cluster: ``(pseudobulk
    counts (n_clusters, n_bins), cluster_order)``. Cells with no cluster
    match (barcode not in ``barcode_to_cluster``) are skipped.
    """
    cluster_rows: Dict[str, List[int]] = defaultdict(list)
    for row, barcode in enumerate(barcodes):
        cluster_id = barcode_to_cluster.get(barcode)
        if cluster_id is not None:
            cluster_rows[cluster_id].append(row)

    cluster_order = sorted(cluster_rows, key=_sort_key)
    pseudobulk = np.zeros((len(cluster_order), counts.shape[1]), dtype=counts.dtype)
    for i, cluster_id in enumerate(cluster_order):
        pseudobulk[i] = counts[cluster_rows[cluster_id]].sum(axis=0)
    return pseudobulk, cluster_order


# --------------------------------------------------------------------------- #
# SCICoNE invocation (schema not confirmed against a real run -- see caveat)
# --------------------------------------------------------------------------- #


def find_scicone_binary(explicit: Optional[str] = None) -> str:
    """Resolve the SCICoNE binary: ``explicit`` (the ``--scicone-bin`` CLI
    arg), else ``$SCICONE_BIN``, else the repo's own build at
    ``realdata/external/SCICoNE/build/scicone``."""
    if explicit:
        return explicit
    if os.environ.get("SCICONE_BIN"):
        return os.environ["SCICONE_BIN"]
    return str(
        Path(__file__).resolve().parents[2]
        / "external"
        / "SCICoNE"
        / "build"
        / "scicone"
    )


def write_scicone_matrix(counts: np.ndarray, path: Path) -> None:
    """SCICoNE's read-counts input: a plain whitespace-delimited integer
    matrix, cells (or pseudobulk clusters) x bins -- see module docstring's
    caveat."""
    np.savetxt(path, counts, fmt="%d")


def run_scicone(
    matrix_path: Path,
    n_rows: int,
    n_bins: int,
    out_prefix: Path,
    log_path: Path,
    scicone_bin: str,
) -> Tuple[Path, Path]:
    """Run SCICoNE and return ``(edges_path, assignment_path)`` -- see module
    docstring's caveat on this CLI shape not being confirmed. Raises
    ``subprocess.CalledProcessError`` if SCICoNE exits non-zero, and
    ``FileNotFoundError`` if it exits cleanly but the expected output files
    are missing -- both are the caller's cue to fail the stage rather than
    trust a partial result.
    """
    edges_path = Path(f"{out_prefix}.edges.tsv")
    assignment_path = Path(f"{out_prefix}.assignment.tsv")
    cmd = [
        scicone_bin,
        "--input", str(matrix_path),
        "--n_cells", str(n_rows),
        "--n_bins", str(n_bins),
        "--output", str(out_prefix),
    ]  # fmt: skip
    with open(log_path, "w") as log:
        log.write("command: " + " ".join(cmd) + "\n\n")
        log.flush()
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)

    missing = [p for p in (edges_path, assignment_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"SCICoNE exited cleanly but expected output missing: {missing}; "
            f"see {log_path} and confirm the real output filenames/flags "
            "against the built binary (module docstring's caveat)."
        )
    return edges_path, assignment_path


_EDGE_LINE_RE = re.compile(r"^\s*(?P<a>\S+)\s+(?P<b>\S+)\s*$")


def parse_scicone_edges(edges_path: Path) -> Dict[str, str]:
    """Parse a plain ``parent<whitespace>child`` edge-list file into
    ``{child: parent}``. Raises ValueError if no edges parse."""
    parent_of: Dict[str, str] = {}
    for line in Path(edges_path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _EDGE_LINE_RE.match(line)
        if m:
            parent_of[m.group("b")] = m.group("a")
    if not parent_of:
        raise ValueError(f"{edges_path} contains no parseable edge lines")
    return parent_of


def parse_scicone_cell_assignment(assignment_path: Path) -> Dict[str, str]:
    """Parse a plain ``cell<whitespace>node`` assignment file into ``{cell_id
    (barcode or row index as a string): node_id}``. Raises ValueError if no
    assignments parse."""
    assignment: Dict[str, str] = {}
    for line in Path(assignment_path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        assignment[parts[0]] = parts[1]
    if not assignment:
        raise ValueError(f"{assignment_path} contains no parseable assignment lines")
    return assignment


# --------------------------------------------------------------------------- #
# Majority-vote collapse (per-cell mode) / direct label match (pseudobulk mode)
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


def resolve_pseudobulk_cluster_nodes(
    parent_of: Dict[str, str], cluster_order: List[str]
) -> Dict[str, str]:
    """Resolve each pseudobulk cluster's own SCICoNE node directly by label
    (SCICoNE's row/sample identity for a pseudobulk run should be the
    cluster IDs passed in as row order, or a 0-/1-based index -- the same
    ambiguity build_snv_tree.py's LICHeE integration handles via
    ``_sample_label_schemes``). Raises ValueError, naming what it tried, if
    no scheme covers every cluster.
    """
    all_nodes = set(parent_of) | set(parent_of.values())
    for scheme in _sample_label_schemes(cluster_order):
        target_to_cid = {v: k for k, v in scheme.items()}
        candidate = {
            target_to_cid[node]: node for node in all_nodes if node in target_to_cid
        }
        if len(candidate) == len(cluster_order):
            return candidate
    raise ValueError(
        "no consistent node labelling scheme covers all pseudobulk clusters "
        f"{cluster_order}; SCICoNE nodes found: {sorted(all_nodes)}"
    )


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
    p.add_argument("--scicone-bin", default=None)
    p.add_argument(
        "--pseudobulk-fallback",
        action="store_true",
        help="aggregate to one row per SECEDO cluster before running SCICoNE, "
        "instead of running it over individual cells -- see module docstring",
    )
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    counts, barcodes, counts_path, barcodes_path = load_cnv_counts_matrix(args.cnv_h5)
    print(
        f"cnv_data.h5: counts from {counts_path!r} {counts.shape}, "
        f"barcodes from {barcodes_path!r}"
    )

    barcode_to_cluster = build_barcode_to_cluster(args.map_file, args.clustering_file)
    n_matched = sum(1 for b in barcodes if b in barcode_to_cluster)
    print(
        f"{n_matched}/{len(barcodes)} cnv_data.h5 cells matched a SECEDO cluster "
        f"({len(set(barcode_to_cluster.values()))} clusters total)"
    )

    scicone_bin = find_scicone_binary(args.scicone_bin)
    mode = "pseudobulk" if args.pseudobulk_fallback else "per_cell"

    if mode == "pseudobulk":
        matrix, row_order = aggregate_counts_by_cluster(
            counts, barcodes, barcode_to_cluster
        )
        row_ids = row_order
    else:
        matrix, row_ids = counts, barcodes

    matrix_path = args.out_dir / "scicone_input.txt"
    write_scicone_matrix(matrix, matrix_path)
    out_prefix = args.out_dir / "scicone_out"
    log_path = args.out_dir / "scicone_run.log"
    edges_path, assignment_path = run_scicone(
        matrix_path,
        n_rows=matrix.shape[0],
        n_bins=matrix.shape[1],
        out_prefix=out_prefix,
        log_path=log_path,
        scicone_bin=scicone_bin,
    )
    parent_of = parse_scicone_edges(edges_path)

    if mode == "pseudobulk":
        node_of_cluster = resolve_pseudobulk_cluster_nodes(parent_of, row_ids)
        unmatched: List[str] = []
    else:
        cell_to_node = parse_scicone_cell_assignment(assignment_path)
        node_of_cluster, unmatched = majority_cluster_nodes(
            cell_to_node, barcode_to_cluster
        )

    tree = collapse_by_nearest_labelled_ancestor(
        parent_of, node_of_cluster, GERMLINE_ROOT_ID
    )
    newick_str = digraph_to_newick(tree, GERMLINE_ROOT_ID)
    verify_newick(newick_str, set(node_of_cluster), GERMLINE_ROOT_ID)
    (args.out_dir / "cna_tree.nwk").write_text(newick_str + "\n")

    diagnostics = [
        "# Stage 09 (CNA tree) diagnostics\n\n",
        f"## Mode: {mode}\n",
        f"cnv_data.h5 counts dataset: {counts_path}\n",
        f"cnv_data.h5 barcodes dataset: {barcodes_path}\n",
        f"Cells matched to a SECEDO cluster: {n_matched}/{len(barcodes)}\n",
        f"Clusters resolved to a SCICoNE node: {len(node_of_cluster)}\n",
    ]
    if unmatched:
        diagnostics.append(
            f"FINDING: {len(unmatched)} cluster(s) had no cell present in "
            f"SCICoNE's own output, so could not be placed: {unmatched}\n"
        )
    diagnostics.append(f"\nEmitted tree: {classify_topology(tree)}\n")
    if classify_topology(tree).startswith("linear chain"):
        diagnostics.append("FINDING: the emitted CNA tree is a non-branching chain.\n")
    (args.out_dir / "tree_diagnostics.txt").write_text("".join(diagnostics))

    print(
        f"Mode: {mode}. Wrote cna_tree.nwk and tree_diagnostics.txt to {args.out_dir}."
    )


if __name__ == "__main__":
    main()
