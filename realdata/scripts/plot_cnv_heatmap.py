"""Figure 6: per-cell copy-number heatmap of slice D, cells grouped by
SCICoNE node.

Reads only ``realdata/local_tree_input/figures_data/cnv_heatmap.npz``, written
on Euler by ``realdata/scripts/euler/cnv_heatmap_data.py`` (see its docstring
for how the blocks and the node copy number are built), and writes
``fig6_cnv_heatmap.pdf``/``.png`` plus the Figure 6 caption in captions.md to
``realdata/local_tree_input/figures/`` (gitignored, real data).

Rows run by SCICoNE node in tree order (near-diploid, rearranged clone,
subclone; named by ``cna_nodes.node_roles``, as in Figure 1), then by SECEDO
cluster, then by average-linkage hierarchical clustering on the cell's block
profile so structure within a group is visible. Two strips on the left give
the SCICoNE node and the SECEDO cluster, with the pseudo-normal cluster
hatched. The top panel is depth normalised to each cell's mean (1 is the
cell's average), on a diverging map centred at 1 and clipped to 0..2. The
bottom panel gives each cell its node's inferred copy number, on a discrete
map centred at 2 (0, 1, 2, 3, 4+). Grey marks blocks with no filtered bin.

Usage (from the repo root)
--------------------------
    python realdata/scripts/plot_cnv_heatmap.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from captions import upsert_captions  # noqa: E402
from cna_nodes import node_roles  # noqa: E402

from src.plotting import figure_style  # noqa: E402
from src.plotting.figure_style import PALETTE  # noqa: E402

INPUT_DIR = REPO_ROOT / "realdata" / "local_tree_input"
PSEUDO_NORMAL = "4"
MISSING = "#d9d9d9"
NODE_COLOUR = {
    "near-diploid": PALETTE["grey"],
    "rearranged clone": PALETTE["stiff"],
    "subclone": PALETTE["accent"],
}
CN_LEVELS = [0, 1, 2, 3, 4]


def blend(colour: str, weight: float) -> tuple:
    """``colour`` mixed with white; ``weight`` 1 is the colour itself."""
    rgb = np.array(plt.matplotlib.colors.to_rgb(colour))
    return tuple(1 - weight * (1 - rgb))


DEPTH_CMAP = LinearSegmentedColormap.from_list(
    "depth", [PALETTE["stiff"], "white", PALETTE["soft"]]
)
DEPTH_CMAP.set_bad(MISSING)
CN_CMAP = ListedColormap(
    [
        PALETTE["stiff"],
        blend(PALETTE["stiff"], 0.4),
        "white",
        blend(PALETTE["soft"], 0.5),
        PALETTE["soft"],
    ]
)
CN_CMAP.set_bad(MISSING)
CN_NORM = BoundaryNorm(np.arange(-0.5, 5.5), CN_CMAP.N)


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def within_group_order(depth: np.ndarray) -> np.ndarray:
    """Leaf order of average-linkage clustering (correlation distance) on the
    block profiles, NaN blocks filled with 1; identity for fewer than 3 rows
    or when a row is constant."""
    from scipy.cluster.hierarchy import leaves_list, linkage

    if depth.shape[0] < 3:
        return np.arange(depth.shape[0])
    x = np.nan_to_num(depth, nan=1.0)
    if np.any(x.std(axis=1) == 0):
        return np.arange(depth.shape[0])
    return leaves_list(linkage(x, method="average", metric="correlation"))


def row_order(
    cell_node: np.ndarray,
    cluster: np.ndarray,
    node_order: list[str],
    depth: np.ndarray,
) -> np.ndarray:
    """Row indices sorted by node (in ``node_order``), then SECEDO cluster
    (numeric), then ``within_group_order`` inside each node-cluster group.
    Raises ValueError if a cell's node is not in ``node_order``."""
    unknown = set(cell_node) - set(node_order)
    if unknown:
        raise ValueError(f"cells on nodes outside the order: {sorted(unknown)}")
    out = []
    for node in node_order:
        for c in sorted(set(cluster[cell_node == node]), key=int):
            idx = np.flatnonzero((cell_node == node) & (cluster == c))
            out.append(idx[within_group_order(depth[idx])])
    return np.concatenate(out)


def runs(labels: np.ndarray) -> list[tuple[str, int, int]]:
    """Contiguous runs of equal labels as ``(label, start, stop)``."""
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [labels.size]])
    return [(labels[a], int(a), int(b)) for a, b in zip(starts, stops)]


def draw_strips(ax_node, ax_cluster, node_labels, cluster_labels, names):
    n = node_labels.size
    for ax in (ax_node, ax_cluster):
        ax.set_xlim(0, 1)
        ax.set_ylim(n, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for node, a, b in runs(node_labels):
        ax_node.add_patch(
            plt.Rectangle((0, a), 1, b - a, color=NODE_COLOUR[names[node]], lw=0)
        )
    for k, (c, a, b) in enumerate(runs(cluster_labels)):
        shade = "white" if c == PSEUDO_NORMAL else blend("black", 0.18 + 0.2 * (k % 2))
        ax_cluster.add_patch(
            plt.Rectangle(
                (0, a),
                1,
                b - a,
                fc=shade,
                ec=PALETTE["grey"] if c == PSEUDO_NORMAL else "none",
                hatch="////" if c == PSEUDO_NORMAL else None,
                lw=0.3,
            )
        )
        if b - a >= 0.03 * n:
            ax_cluster.text(
                0.5,
                (a + b) / 2,
                c,
                ha="center",
                va="center",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8)
                if c == PSEUDO_NORMAL
                else None,
            )
    ax_node.set_title("node", fontsize=7.5, fontweight="normal", rotation=90)
    ax_cluster.set_title("cluster", fontsize=7.5, fontweight="normal", rotation=90)


def draw_heatmap(ax, matrix, cmap, norm, data, separators, label_x):
    im = ax.imshow(
        matrix,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
        rasterized=True,
    )
    starts = data["chrom_start_block"]
    for s in starts[1:]:
        ax.axvline(s - 0.5, color="black", lw=0.4, ls=(0, (3, 3)), alpha=0.6)
    for y in separators:
        ax.axhline(y - 0.5, color="black", lw=0.8)
    ends = np.concatenate([starts[1:], [matrix.shape[1]]])
    names = [str(c).removeprefix("chr") for c in data["chrom_names"]]
    if label_x:
        ax.set_xticks((starts + ends - 1) / 2, names, fontsize=6.5)
        ax.tick_params(axis="x", length=0)
        ax.set_xlabel("chromosome")
    else:
        ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def fig6_cnv_heatmap(data: dict, outdir: Path) -> str:
    cell_node = data["scicone_node"].astype(str)
    cluster = data["secedo_cluster"].astype(str)
    node_ids = data["node_ids"].astype(str).tolist()
    parent = dict(zip(node_ids, data["node_parent"].astype(str)))
    n_events = dict(zip(node_ids, data["node_n_events"].astype(int)))
    occupied = sorted(set(cell_node), key=int)
    order_nodes, names = node_roles(occupied, parent, n_events)

    rows = row_order(cell_node, cluster, order_nodes, data["depth"])
    depth = data["depth"][rows]
    node_sorted, cluster_sorted = cell_node[rows], cluster[rows]
    node_index = {n: i for i, n in enumerate(node_ids)}
    cn = data["node_cn"][[node_index[n] for n in node_sorted]].astype(float)
    cn[cn < 0] = np.nan
    cn = np.minimum(cn, 4)
    separators = [a for _, a, _ in runs(node_sorted)[1:]]

    fig = plt.figure(figsize=(12, 9.5))
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[0.25, 0.25, 30, 0.45],
        height_ratios=[1, 1],
        wspace=0.04,
        hspace=0.12,
    )
    for r, (matrix, cmap, norm, title) in enumerate(
        [
            (
                depth,
                DEPTH_CMAP,
                plt.Normalize(0, 2),
                "Normalised read depth (cell mean = 1)",
            ),
            (cn, CN_CMAP, CN_NORM, "Inferred copy number (SCICoNE node profile)"),
        ]
    ):
        ax_node, ax_cl = fig.add_subplot(gs[r, 0]), fig.add_subplot(gs[r, 1])
        ax = fig.add_subplot(gs[r, 2])
        draw_strips(ax_node, ax_cl, node_sorted, cluster_sorted, names)
        for y in separators:
            for a in (ax_node, ax_cl):
                a.axhline(y, color="black", lw=0.8)
        im = draw_heatmap(ax, matrix, cmap, norm, data, separators, label_x=r == 1)
        ax.set_title(title, fontsize=10, loc="left")
        cax = fig.add_subplot(gs[r, 3])
        if r == 0:
            cb = fig.colorbar(im, cax=cax, extend="max")
            cb.set_ticks([0, 0.5, 1, 1.5, 2], labels=["0", "0.5", "1", "1.5", "2+"])
        else:
            cb = fig.colorbar(im, cax=cax, ticks=CN_LEVELS)
            cb.set_ticklabels(["0", "1", "2", "3", "4+"])
        cb.ax.tick_params(labelsize=7.5)
        cb.outline.set_linewidth(0.4)

    counts = {n: int((cell_node == n).sum()) for n in order_nodes}
    handles = [
        Patch(
            color=NODE_COLOUR[names[n]],
            label=f"{names[n]}: node {n}, {counts[n]} cells",
        )
        for n in order_nodes
    ]
    handles.append(
        Patch(
            fc="white",
            ec=PALETTE["grey"],
            hatch="////",
            label=f"cluster {PSEUDO_NORMAL} (pseudo-normal)",
        )
    )
    handles.append(Patch(color=MISSING, label="no filtered bin"))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.subplots_adjust(bottom=0.11)
    figure_style.save(fig, outdir, "fig6_cnv_heatmap")
    plt.close(fig)

    block_mb = int(data["block_bp"]) / 1e6
    n_missing = int(np.isnan(data["depth"]).all(axis=0).sum())
    corr = dict(zip(node_ids, data["node_depth_corr"]))
    node_desc = []
    for n in order_nodes:
        ev = int(n_events[n])
        cl = cluster[cell_node == n]
        top = sorted(set(cl), key=lambda c: -(cl == c).sum())[:2]
        node_desc.append(
            f"{names[n]} (node {n}, {ev} region event{'s' * (ev != 1)}, "
            f"{counts[n]} cells, mostly cluster{'s' * (len(top) > 1)} "
            f"{' and '.join(top)}; correlation of its profile with its cells' "
            f"mean depth {corr[n]:.2f})"
        )
    return (
        f"Copy-number heatmap of {depth.shape[0]} cells over "
        f"{data['depth'].shape[1]} genomic blocks of {block_mb:g} Mb "
        f"({int(data['bin_size']) // 1000} kb bins averaged within a "
        f"chromosome, never across one; {n_missing} blocks with no filtered bin "
        f"in grey). Rows are grouped by SCICoNE node in tree order, then by "
        f"SECEDO cluster, then by hierarchical clustering; the left strips give "
        f"the node and the cluster, with the pseudo-normal cluster "
        f"{PSEUDO_NORMAL} hatched. Top: read depth divided by each cell's mean, "
        f"centred at 1 and clipped at 2. Bottom: each cell shows its node's "
        f"inferred copy number (neutral state plus the region events on its "
        f"root-to-node path, majority over the block), centred at 2; chrY is 0 "
        f"in this female sample. Nodes: "
        f"{'; '.join(node_desc)}."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--npz",
        type=Path,
        default=INPUT_DIR / "figures_data" / "cnv_heatmap.npz",
    )
    p.add_argument("--outdir", type=Path, default=INPUT_DIR / "figures")
    args = p.parse_args()
    figure_style.apply_style()
    caption = fig6_cnv_heatmap(load(args.npz), args.outdir)
    upsert_captions(args.outdir / "captions.md", {"Figure 6. CNV heatmap": caption})


if __name__ == "__main__":
    main()
