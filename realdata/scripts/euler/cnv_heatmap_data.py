"""Data for the slice D copy-number heatmap (Figure 6): per-cell normalised
read depth and per-node inferred copy number, both on ~1 Mb genomic blocks.

Runs on Euler, where ``cnv_data.h5`` lives; the plot itself is local
(``realdata/scripts/plot_cnv_heatmap.py``) and reads only the npz written
here. Nothing is re-inferred: SCICoNE is not rerun, and the copy number is
rebuilt from stage 09's own saved outputs.

Cells. ``cnv_data.h5`` is read with pyscicone's ``read_10x``, exactly as stage
09 does, and the rows are labelled with ``build_cna_tree.filtered_barcodes``.
The barcode set must equal ``cna_cell_nodes.csv``'s or the script raises; rows
are then put in that file's order, so each cell keeps its SECEDO cluster and
SCICoNE node. Each cell is divided by its mean count over the filtered bins,
so a flat diploid cell sits at 1.

Bins to blocks. ``read_10x`` drops unmappable bins and then "noisy" bins
(``scicone.utils.filter_bins`` on the mappable columns) and keeps no record of
which original bins survive. ``kept_bin_index`` recovers them by rerunning
that same filter and checks that indexing the unfiltered matrix with the
result reproduces ``filtered_counts`` exactly. Each surviving bin then has a
genomic position (chromosome, offset times ``bin_size``) and falls into a
block of ``--block-mb`` within its own chromosome; blocks never span a
chromosome boundary. A block with no surviving bin (centromeres, gaps) is NaN.

Copy number. Region boundaries are ``segmented_region_sizes`` from
``scicone_breakpoints.npz``; neutral states come from
``scicone.utils.set_region_neutral_states`` with the same chromosome states
stage 09 used (female: autosomes and X 2, Y 0). Node events and parents are
parsed from ``cna_tree_diagnostics.txt``. One subtlety, read from pyscicone's
``learn_tree``: regions with neutral state 0 (chrY here) are deleted before
inference, so the region keys in the node events index the remaining regions,
not the full list. ``node_copy_number`` maps them back. A node's profile is the
neutral state plus the events summed along its root-to-node path. Each block
takes the most common copy number among its bins.

As a check that the rebuild lines up with the data, the Pearson correlation
between each occupied node's copy-number profile and the mean normalised depth
of its cells is printed and saved; a misaligned region map would show up as a
low value.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Callable

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from build_cna_tree import (  # noqa: E402
    _BREAKPOINTS_FILE,
    _working_dir,
    chromosome_neutral_states,
    filtered_barcodes,
    load_breakpoints,
)
from cna_nodes import parse_scicone_events  # noqa: E402

# --------------------------------------------------------------------------- #
# Bins
# --------------------------------------------------------------------------- #


def kept_bin_index(
    unfiltered: np.ndarray,
    is_mappable: np.ndarray,
    filtered: np.ndarray,
    filter_bins: Callable,
) -> np.ndarray:
    """Original (unfiltered) column index of every column of ``filtered``.

    Mirrors ``read_10x``: keep the mappable columns, then drop what
    ``filter_bins`` (pyscicone's noisy-bin filter, passed in) drops from
    them. Raises ValueError unless ``unfiltered[:, index]`` equals
    ``filtered`` exactly, so a change in pyscicone's filtering cannot pass
    silently.
    """
    mappable_idx = np.flatnonzero(np.asarray(is_mappable, dtype=bool))
    _, noisy = filter_bins(unfiltered[:, mappable_idx])
    keep = np.ones(mappable_idx.size, dtype=bool)
    keep[np.asarray(noisy, dtype=int)] = False
    index = mappable_idx[keep]
    if index.size != filtered.shape[1] or not np.array_equal(
        unfiltered[:, index], filtered
    ):
        raise ValueError(
            f"could not reproduce read_10x's bin filtering: {index.size} bins "
            f"recovered against {filtered.shape[1]} filtered"
        )
    return index


def bin_positions(bins_per_chrom: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """For every original bin: ``(chromosome index, offset in bins from the
    chromosome start)``, chromosomes in the given order."""
    sizes = np.asarray(bins_per_chrom, dtype=int)
    chrom = np.repeat(np.arange(sizes.size), sizes)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    offset = np.arange(sizes.sum()) - starts[chrom]
    return chrom, offset


def genomic_blocks(
    bins_per_chrom: list[int], bin_size: int, block_bp: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Block layout over the genome: ``(block of every original bin, chromosome
    index of every block, first block of every chromosome)``.

    Each chromosome is cut into ``ceil(n_bins * bin_size / block_bp)`` blocks
    counted from its own start, so no block spans two chromosomes. Raises
    ValueError if ``block_bp`` is smaller than ``bin_size``.
    """
    if block_bp < bin_size:
        raise ValueError(f"block size {block_bp} is below the bin size {bin_size}")
    sizes = np.asarray(bins_per_chrom, dtype=int)
    bins_per_block = block_bp / bin_size
    n_blocks = np.ceil(sizes / bins_per_block).astype(int)
    chrom_start = np.concatenate([[0], np.cumsum(n_blocks)[:-1]])
    chrom, offset = bin_positions(bins_per_chrom)
    block = chrom_start[chrom] + np.floor(offset / bins_per_block).astype(int)
    block_chrom = np.repeat(np.arange(sizes.size), n_blocks)
    return block, block_chrom, chrom_start


def normalise_cells(counts: np.ndarray) -> np.ndarray:
    """Each row divided by its own mean, as float32. Raises ValueError for a
    row whose mean is not positive."""
    counts = np.asarray(counts, dtype=np.float32)
    means = counts.mean(axis=1, keepdims=True)
    if np.any(means <= 0):
        raise ValueError("a cell has no counts over the filtered bins")
    return counts / means


def block_means(values: np.ndarray, block: np.ndarray, n_blocks: int) -> np.ndarray:
    """Mean of ``values`` (rows x bins) over the bins of each block, float32,
    NaN for a block with no bin."""
    from scipy import sparse

    n_bins = values.shape[1]
    member = sparse.csr_matrix(
        (np.ones(n_bins, dtype=np.float32), (np.arange(n_bins), block)),
        shape=(n_bins, n_blocks),
    )
    sums = np.asarray(values @ member, dtype=np.float32)
    n = np.bincount(block, minlength=n_blocks).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = sums / n
    out[:, n == 0] = np.nan
    return out


def block_modes(values: np.ndarray, block: np.ndarray, n_blocks: int) -> np.ndarray:
    """Most common integer value over the bins of each block, per row, as
    int8; -1 for a block with no bin. Ties go to the smaller value."""
    values = np.asarray(values, dtype=int)
    levels = np.arange(values.min(), values.max() + 1)
    out = np.full((values.shape[0], n_blocks), -1, dtype=np.int8)
    n = np.bincount(block, minlength=n_blocks)
    for r, row in enumerate(values):
        tallies = np.stack(
            [np.bincount(block, weights=row == v, minlength=n_blocks) for v in levels]
        )
        out[r, n > 0] = levels[tallies.argmax(axis=0)][n > 0]
    return out


# --------------------------------------------------------------------------- #
# Copy number per node
# --------------------------------------------------------------------------- #


def node_copy_number(
    nodes: dict[str, dict], neutral_states: np.ndarray
) -> dict[str, np.ndarray]:
    """Copy number per region for every SCICoNE node.

    ``nodes`` is ``parse_scicone_events`` output. Event keys index the regions
    left after dropping zero-neutral ones (pyscicone's ``learn_tree`` deletes
    them before inference), so key ``k`` is full region
    ``flatnonzero(neutral_states)[k]``. A node's profile is the neutral state
    plus the events on its root-to-node path. Raises ValueError for an event
    key past the non-zero regions, for a missing parent, or for a negative
    copy number.
    """
    neutral = np.asarray(neutral_states, dtype=int)
    kept = np.flatnonzero(neutral != 0)
    delta = {}
    for node, info in nodes.items():
        d = np.zeros(neutral.size, dtype=int)
        for key, value in info["events"].items():
            k = int(key)
            if k >= kept.size:
                raise ValueError(
                    f"node {node}: event region {k} is past the {kept.size} "
                    "regions with non-zero neutral state"
                )
            d[kept[k]] += int(value)
        delta[node] = d

    out = {}
    for node in nodes:
        cn = neutral + delta[node]
        parent = nodes[node]["parent"]
        while parent != "NULL":
            if parent not in nodes:
                raise ValueError(f"node {node}: parent {parent} is not a known node")
            cn = cn + delta[parent]
            parent = nodes[parent]["parent"]
        if np.any(cn < 0):
            raise ValueError(f"node {node}: negative copy number after events")
        out[node] = cn
    return out


def region_of_bin(region_sizes: np.ndarray) -> np.ndarray:
    """Region index of every filtered bin, from the region sizes."""
    sizes = np.asarray(region_sizes, dtype=int).ravel()
    return np.repeat(np.arange(sizes.size), sizes)


def profile_correlation(
    node_cn: np.ndarray, depth: np.ndarray, cell_node: np.ndarray, node_ids: list
) -> dict[str, float]:
    """Pearson correlation, over blocks with data, between each occupied
    node's copy-number profile and the mean normalised depth of its cells."""
    out = {}
    for i, node in enumerate(node_ids):
        rows = cell_node == node
        if not rows.any():
            continue
        with warnings.catch_warnings():  # all-NaN blocks are expected
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_depth = np.nanmean(depth[rows], axis=0)
        ok = (node_cn[i] >= 0) & np.isfinite(mean_depth)
        a, b = node_cn[i][ok].astype(float), mean_depth[ok]
        out[node] = (
            float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
        )
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def read_cell_table(path: Path) -> tuple[list[str], list[str], list[str]]:
    """``(barcodes, SECEDO clusters, SCICoNE nodes)`` from cna_cell_nodes.csv."""
    lines = Path(path).read_text().splitlines()
    if lines[0].strip() != "barcode,secedo_cluster,scicone_node":
        raise ValueError(f"unexpected header in {path}: {lines[0]!r}")
    rows = [line.split(",") for line in lines[1:] if line.strip()]
    return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cnv-h5", required=True, type=Path)
    p.add_argument(
        "--tree-dir",
        required=True,
        type=Path,
        help="stage 09 output: cna_cell_nodes.csv, cna_tree_diagnostics.txt, "
        "scicone_breakpoints.npz",
    )
    p.add_argument("--scicone-build-dir", required=True, type=Path)
    p.add_argument("--sex", required=True, choices=["female", "male"])
    p.add_argument("--block-mb", type=float, default=1.0)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    import h5py
    import scicone

    work_dir = (args.out.parent / "scicone_tmp").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    sci = scicone.SCICoNE(str(args.scicone_build_dir), str(work_dir), verbose=False)
    with _working_dir(work_dir):
        sci.read_10x(str(args.cnv_h5.resolve()))
    sci.data.pop("filtered_cnvs", None)  # unused, ~2 GB
    filtered = sci.data["filtered_counts"]
    unfiltered = sci.data.pop("unfiltered_counts")
    stops = sci.data["filtered_chromosome_stops"]
    bin_size = int(sci.data["bin_size"])
    n_cells, n_bins = filtered.shape

    # Rows: same cells as stage 09, put in cna_cell_nodes.csv's order.
    barcodes = filtered_barcodes(args.cnv_h5, n_cells)
    csv_barcodes, clusters, cell_nodes = read_cell_table(
        args.tree_dir / "cna_cell_nodes.csv"
    )
    if set(barcodes) != set(csv_barcodes) or len(barcodes) != len(csv_barcodes):
        raise ValueError(
            f"cnv_data.h5 filtered cells ({len(barcodes)}) and cna_cell_nodes.csv "
            f"({len(csv_barcodes)}) are not the same barcode set; "
            f"{len(set(barcodes) ^ set(csv_barcodes))} differ"
        )
    row = {b: i for i, b in enumerate(barcodes)}
    order = np.array([row[b] for b in csv_barcodes])

    # Bins: recover each filtered bin's genomic position.
    with h5py.File(args.cnv_h5, "r") as f:
        chroms = scicone.utils.sort_chromosomes(
            f["constants"]["chroms"][()].astype(str)
        )
        mappable = [f["genome_tracks"]["is_mappable"][ch][()] for ch in chroms]
    bins_per_chrom = [m.size for m in mappable]
    if sum(bins_per_chrom) != unfiltered.shape[1]:
        raise ValueError("is_mappable tracks do not cover the unfiltered bins")
    kept = kept_bin_index(
        unfiltered, np.concatenate(mappable), filtered, scicone.utils.filter_bins
    )
    del unfiltered
    block_bp = int(round(args.block_mb * 1e6))
    block_of_bin, block_chrom, chrom_start = genomic_blocks(
        bins_per_chrom, bin_size, block_bp
    )
    block = block_of_bin[kept]
    n_blocks = block_chrom.size

    # SCICoNE's own chromosome stops, against the recovered positions.
    stop_chrom = np.searchsorted(np.asarray(list(stops.values())), np.arange(n_bins))
    true_chrom, _ = bin_positions(bins_per_chrom)
    stop_names = np.asarray(list(stops))[np.minimum(stop_chrom, len(stops) - 1)]
    n_stop_mismatch = int(np.sum(stop_names != np.asarray(chroms)[true_chrom[kept]]))

    depth = block_means(normalise_cells(filtered)[order], block, n_blocks)
    del filtered

    # Copy number per node, on the filtered bins, then on blocks.
    bps = load_breakpoints(args.tree_dir, n_bins)
    region_sizes = np.asarray(bps["segmented_region_sizes"]).ravel().astype(int)
    neutral = np.asarray(
        scicone.utils.set_region_neutral_states(
            bps["segmented_regions"],
            list(stops.values()),
            chromosome_neutral_states(list(stops), args.sex),
        )
    )
    if neutral.size != region_sizes.size:
        raise ValueError(
            f"{neutral.size} neutral states for {region_sizes.size} regions"
        )
    nodes = parse_scicone_events(args.tree_dir / "cna_tree_diagnostics.txt")
    node_ids = sorted(nodes, key=int)
    cn_regions = node_copy_number(nodes, neutral)
    bin_region = region_of_bin(region_sizes)
    cn_bins = np.stack([cn_regions[n][bin_region] for n in node_ids])
    node_cn = block_modes(cn_bins, block, n_blocks)

    cell_node = np.asarray(cell_nodes)
    corr = profile_correlation(node_cn, depth, cell_node, node_ids)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        depth=depth,
        node_cn=node_cn,
        node_ids=np.asarray(node_ids),
        node_parent=np.asarray([nodes[n]["parent"] for n in node_ids]),
        node_n_events=np.asarray([len(nodes[n]["events"]) for n in node_ids]),
        node_depth_corr=np.asarray([corr.get(n, np.nan) for n in node_ids]),
        barcodes=np.asarray(csv_barcodes),
        secedo_cluster=np.asarray(clusters),
        scicone_node=cell_node,
        block_chrom=np.asarray(chroms)[block_chrom],
        block_start_bp=(np.arange(n_blocks) - chrom_start[block_chrom]) * block_bp,
        chrom_names=np.asarray(chroms),
        chrom_start_block=chrom_start,
        bin_size=np.asarray(bin_size),
        block_bp=np.asarray(block_bp),
        n_bins=np.asarray(n_bins),
        n_stop_mismatch=np.asarray(n_stop_mismatch),
        breakpoints_file=np.asarray(str(args.tree_dir / _BREAKPOINTS_FILE)),
    )

    print("key,value")
    print(f"cells,{depth.shape[0]}")
    print(f"filtered_bins,{n_bins}")
    print(f"bin_size_bp,{bin_size}")
    print(f"blocks,{n_blocks}")
    print(f"blocks_without_bins,{int(np.isnan(depth[0]).sum())}")
    print(f"regions,{region_sizes.size}")
    print(f"bins_where_scicone_stops_disagree,{n_stop_mismatch}")
    for node, r in corr.items():
        print(f"node_{node}_cn_depth_corr,{r:.3f}")
    print(f"npz,{args.out}")
    print(f"npz_mb,{args.out.stat().st_size / 1e6:.1f}")


if __name__ == "__main__":
    main()
