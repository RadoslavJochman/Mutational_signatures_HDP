"""
generate_walk_figure.py

Activity-walk figure (Figure 1) for the report, generated from the real simulator.

Shows the shared cohort baseline e_0 on top, two tumour trees descending from it
(with an ellipsis for the rest of the forest), each node's activity e_j drawn as
a pie over K signatures, and the mixed spectrum e_j S for the labelled leaf j.

Usage
    python scripts/generate_walk_figure.py    # writes activity_walk.pdf
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Wedge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.hdp_simulator import TreeSignatureGenerator


def main():
    SEED = 42
    K = 5
    NEWICK = "((A1:400,A2:400)A0:600,A3:500)RA:0;(B1:400,B2:400)RB:0;"

    gen = TreeSignatureGenerator(
        newick_forest=NEWICK,
        n_signatures=K,
        signature_correlation=0.0,
        alpha=25.0,
        alpha_0=8,
        activity_sparsity=0.0,
        signature_dropout=0.0,
        seed=SEED,
    )

    activities = gen.get_true_activities()  # tree nodes only
    S = gen.get_true_signatures()
    edge_df = gen.get_tree_edges()
    E0 = "e0"
    activities[E0] = gen.e_0.copy()  # add cohort baseline as a node

    children = {}
    all_children = set()
    for _, r in edge_df.iterrows():
        children.setdefault(r["parent"], []).append(r["child"])
        all_children.add(r["child"])
    roots = [n for n in activities if n != E0 and n not in all_children]
    children[E0] = sorted(roots)

    draw_edges = [(r["parent"], r["child"]) for _, r in edge_df.iterrows()]
    draw_edges += [(E0, r) for r in roots]

    depth = {}

    def set_depth(n, d):
        """Assign depth d to node n and recurse into its children."""
        depth[n] = d
        for c in sorted(children.get(n, [])):
            set_depth(c, d + 1)

    set_depth(E0, 0)
    max_depth = max(depth.values())

    pos = {}
    _leaf_x = [0.0]

    def layout(n):
        """Assign an x position to node n (mean of its children, or the next free
        leaf slot) and return it, filling the pos dict."""
        kids = sorted(children.get(n, []))
        if not kids:
            x = _leaf_x[0]
            _leaf_x[0] += 1.0
        else:
            x = float(np.mean([layout(c) for c in kids]))
        pos[n] = (x, depth[n])
        return x

    layout(E0)

    xs = [p[0] for p in pos.values()]
    xmin, xmax = min(xs), max(xs)
    X_LO, X_HI = 0.08, 1.18
    Y_TOP, Y_BOT = 0.92, 0.16
    for n in pos:
        x, d = pos[n]
        nx_ = X_LO + (x - xmin) / (xmax - xmin + 1e-9) * (X_HI - X_LO)
        ny_ = Y_TOP - d / max_depth * (Y_TOP - Y_BOT)
        pos[n] = (nx_, ny_)

    sig_colors = [
        "mediumpurple",  # Sig 1
        "seagreen",  # Sig 2
        "coral",  # Sig 3
        "orange",  # Sig 4
        "steelblue",  # Sig 5
    ]
    # spectrum blocks: COSMIC six-class convention, 16 channels each
    spec_block_colors = [
        "deepskyblue",  # C>A
        "black",  # C>G
        "red",  # C>T
        "silver",  # T>A
        "yellowgreen",  # T>C
        "pink",  # T>G
    ]

    def draw_pie(ax, center, vec, radius=0.05):
        """Draw an activity vector as a pie of signature wedges centred at center."""
        cx, cy = center
        start = 90.0
        for i, frac in enumerate(vec):
            if frac <= 0:
                continue
            theta = frac * 360.0
            ax.add_patch(
                Wedge(
                    (cx, cy),
                    radius,
                    start - theta,
                    start,
                    facecolor=sig_colors[i],
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=3,
                )
            )
            start -= theta
        ax.add_patch(
            plt.Circle(
                (cx, cy),
                radius,
                fill=False,
                edgecolor="dimgray",
                linewidth=0.5,
                zorder=4,
            )
        )

    fig = plt.figure(figsize=(12.0, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.10)

    axL = fig.add_subplot(gs[0, 0])
    axL.set_xlim(0, 1.5)
    axL.set_ylim(0, 1)
    axL.set_aspect("equal")
    axL.axis("off")

    for a, b in draw_edges:
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        axL.plot([x1, x2], [y1, y2], color="gray", lw=0.8, zorder=1)

    for n, c in pos.items():
        draw_pie(axL, c, activities[n])

    ex, ey = pos[E0]
    axL.text(
        ex, ey - 0.095, r"$e_0$", fontsize=12, color="black", ha="center", va="top"
    )

    def subtree_nodes(root):
        """All node ids in the subtree rooted at root (iterative DFS)."""
        out, stack = [], [root]
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(children.get(n, []))
        return out

    t1 = subtree_nodes(roots[0])
    t2 = subtree_nodes(roots[1])
    gap_x = (max(pos[n][0] for n in t1) + min(pos[n][0] for n in t2)) / 2
    for k in (-1, 0, 1):
        axL.add_patch(
            plt.Circle((gap_x + k * 0.05, 0.42), 0.011, color="gray", zorder=2)
        )

    # labelled leaf j, and the x S link to the spectrum
    leaves = [n for n in pos if n != E0 and not children.get(n)]
    leaf = max(leaves, key=lambda n: pos[n][0])
    lx, ly = pos[leaf]
    axL.text(lx, ly - 0.095, r"$j$", fontsize=13, color="black", ha="center", va="top")
    con = FancyArrowPatch(
        (lx + 0.05, ly),
        (lx + 0.47, ly),
        transform=axL.transData,
        mutation_scale=12,
        connectionstyle="arc3,rad=0.0",
        lw=1.0,
        color="dimgray",
        clip_on=False,
        zorder=5,
    )
    axL.add_patch(con)
    axL.text(lx + 0.2, ly + 0.02, r"$\times\,S$", fontsize=12, color="black")

    # signature legend
    for i, col in enumerate(sig_colors):
        cy = 0.96 - i * 0.06
        axL.add_patch(
            plt.Rectangle(
                (0.02, cy - 0.014),
                0.022,
                0.028,
                facecolor=col,
                edgecolor="none",
                transform=axL.transAxes,
                clip_on=False,
            )
        )
        axL.text(
            0.052,
            cy,
            f"Sig {i + 1}",
            fontsize=8.5,
            color="dimgray",
            transform=axL.transAxes,
            va="center",
            ha="left",
        )

    # spectrum panel
    axR = fig.add_subplot(gs[0, 1])
    spec = activities[leaf] @ S
    axR.bar(np.arange(96), spec, color=np.repeat(spec_block_colors, 16), width=0.85)
    axR.set_xlim(-1, 96)
    axR.set_ylim(0, spec.max() * 1.18)
    axR.set_xticks([])
    axR.set_yticks([])
    for s in ["top", "right", "left"]:
        axR.spines[s].set_visible(False)
    axR.spines["bottom"].set_color("gray")
    axR.set_xlabel("96 mutation channels", fontsize=10, color="dimgray")
    axR.set_title(r"$e_j\,S$  (mixed spectrum of node $j$)", fontsize=11, color="black")

    fig.savefig("activity_walk.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
