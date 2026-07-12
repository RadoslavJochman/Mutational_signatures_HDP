"""
plot_sweep_comparison.py

Tree-HDP versus NMF recovery across the two sweeps.

Reads the aggregated_summary.csv files and draws a 2x2 panel: rows are signature
recovery (Hellinger to truth) and activity recovery (L1, twice the total
variation), columns are the forest-size sweep and the signature-overlap sweep.
Lower is better in every panel.

The size column is the 20-tree size sweep, Tree-HDP against NMF with a +/- SD
band. The overlap column overlays the correlation sweep at 20 trees (dashed) and
40 trees (solid), so the effect of more data shows directly: doubling the forest
lowers Tree-HDP's error at moderate overlap and pushes the crossover with NMF to
higher rho, while at rho = 0.9 both methods are unmoved. SD bands in the overlap
column are drawn for the 40-tree curves only, to keep four lines legible.

Inputs
    --agg     20-tree aggregate dir with size_{treehdp,nmf}/ and
              corr_{treehdp,nmf}/ subdirs, each holding aggregated_summary.csv.
    --agg40   40-tree aggregate dir with corr_{treehdp,nmf}/ (the overlay).
    --outdir  directory for the figure (default ../plots).
    --name    output figure basename (default 'fig_sweep_comparison').

Usage
    python scripts/plot_sweep_comparison.py \\
        --agg ../results/agg --agg40 ../results/agg40 --outdir ../plots
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.plotting.figure_style as fs

TREE = fs.PALETTE["stiff"]
NMF = fs.PALETTE["grey"]


def _series(base, sweep, method, base_metric, scale):
    """One curve (x, mean, sd) for a metric, from one sweep aggregated_summary.csv."""
    df = pd.read_csv(base / f"{sweep}_{method}" / "aggregated_summary.csv").sort_values("setting")
    x = df["setting"].to_numpy()
    v = f"{base_metric}_mean_mean" if method == "treehdp" else f"{base_metric}_mean"
    s = f"{base_metric}_mean_std" if method == "treehdp" else f"{base_metric}_std"
    return x, scale * df[v].to_numpy(), scale * df[s].to_numpy()


def _grid(ax):
    """Shared y-grid styling, axis drawn below the data, y starting at zero."""
    ax.grid(axis="y", color=fs.PALETTE["grey"], lw=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)


def panel_size(ax, base, base_metric, scale):
    """Size-sweep panel, Tree-HDP and NMF at 20 trees, each with an SD band."""
    for method, col in [("treehdp", TREE), ("nmf", NMF)]:
        x, y, sd = _series(base, "size", method, base_metric, scale)
        ax.plot(x, y, "-o", color=col, lw=1.8, ms=5, zorder=3)
        ax.fill_between(x, y - sd, y + sd, color=col, alpha=0.16, lw=0, zorder=2)
    _grid(ax)


def panel_corr(ax, base, base40, base_metric, scale):
    """Overlap-sweep panel, 20 trees (dashed) over 40 trees (solid), SD band on
    the 40-tree curves only."""
    for agg, ls, band in [(base, "--", False), (base40, "-", True)]:
        for method, col in [("treehdp", TREE), ("nmf", NMF)]:
            x, y, sd = _series(agg, "corr", method, base_metric, scale)
            ax.plot(x, y, ls, color=col, lw=2.0 if band else 1.6, marker="o", ms=5, zorder=3)
            if band:
                ax.fill_between(x, y - sd, y + sd, color=col, alpha=0.14, lw=0, zorder=2)
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    _grid(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default="../results/agg",
                    help="20-tree aggregate dir (size_* and corr_* subdirs)")
    ap.add_argument("--agg40", default="../results/agg40",
                    help="40-tree aggregate dir (corr_* subdirs)")
    ap.add_argument("--outdir", default="../plots/")
    ap.add_argument("--name", default="fig_sweep_comparison",
                    help="output figure basename")
    a = ap.parse_args()
    base, base40 = Path(a.agg), Path(a.agg40)

    fs.apply_style()
    fig, ax = plt.subplots(2, 2, figsize=(8.4, 6.0))
    panel_size(ax[0, 0], base, "sig_hellinger_mean", 1.0)
    panel_corr(ax[0, 1], base, base40, "sig_hellinger_mean", 1.0)
    panel_size(ax[1, 0], base, "act_tv_median", 2.0)
    panel_corr(ax[1, 1], base, base40, "act_tv_median", 2.0)

    ax[0, 0].set_title("forest size")
    ax[0, 1].set_title("signature overlap")
    ax[0, 0].set_ylabel("signature Hellinger to truth")
    ax[1, 0].set_ylabel(r"activity $L_1$ to truth")
    ax[1, 0].set_xlabel("number of trees")
    ax[1, 1].set_xlabel(r"signature correlation $\rho$")

    method_keys = [Line2D([0], [0], color=TREE, lw=2, marker="o", ms=5, label="Tree-HDP"),
                   Line2D([0], [0], color=NMF, lw=2, marker="o", ms=5, label="NMF")]
    ax[0, 0].legend(handles=method_keys, frameon=False, loc="upper right")
    tree_keys = [Line2D([0], [0], color="0.35", lw=1.6, ls="--", label="20 trees"),
                 Line2D([0], [0], color="0.35", lw=2.0, ls="-", label="40 trees")]
    ax[0, 1].legend(handles=tree_keys, frameon=False, loc="upper left")

    fig.tight_layout()
    fs.save(fig, a.outdir, a.name)
    print(f"saved {a.name} to {a.outdir}")


if __name__ == "__main__":
    main()