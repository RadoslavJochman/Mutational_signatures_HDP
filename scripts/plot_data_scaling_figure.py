"""
plot_data_scaling_figure.py

Two-panel scaling figure for the report, from scaling_results.csv.

Left  : convergence, smallest bulk ESS and worst r-hat against the number of trees.
Right : accuracy, the median per-node L1 to truth against the number of trees.

Uses the shared figure_style palette so the look matches the other figures.

Inputs
    --scaling-csv  scaling_results.csv, one row per run (n_trees, min_ess,
                   max_rhat, median_L1), as written by scaling_metrics.py.
    --outdir       directory for the figure (default ../plots).
    --name         output figure basename (default 'scaling').

Usage
    python scripts/plot_data_scaling_figure.py \\
        --scaling-csv ../results/scaling_results.csv --outdir ../plots
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.plotting.figure_style as fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaling-csv", default="../results/scaling_results.csv",
                    help="scaling_results.csv written by scaling_metrics.py")
    ap.add_argument("--outdir", default="../plots/")
    ap.add_argument("--name", default="scaling", help="output figure basename")
    a = ap.parse_args()

    fs.apply_style()
    P = fs.PALETTE
    df = pd.read_csv(a.scaling_csv).sort_values("n_trees")
    t = df["n_trees"].values

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 3.7))

    axA.plot(t, df["min_ess"], "-o", color=P["stiff"], lw=1.6, ms=6, zorder=3)
    axA.set_ylim(0, 3800)
    axA.set_ylabel("min bulk ESS", color=P["stiff"])
    axA.tick_params(axis="y", colors=P["stiff"])
    axA.set_xlabel("number of trees")
    axA.set_xticks(t)
    axA.set_title("Convergence")
    axA.grid(axis="y", color=P["grey"], lw=0.6, alpha=0.35)
    axA.set_axisbelow(True)

    axA2 = axA.twinx()
    axA2.spines["top"].set_visible(False)
    axA2.plot(t, df["max_rhat"], "-s", color=P["soft"], lw=1.4, ms=5, zorder=3)
    axA2.set_ylim(0.99, 1.015)
    axA2.set_ylabel(r"max $\hat{r}$", color=P["soft"])
    axA2.tick_params(axis="y", colors=P["soft"])

    axB.plot(t, df["median_L1"], "-o", color=P["accent"], lw=1.6, ms=6, zorder=3)
    axB.set_ylim(0, 0.22)
    axB.set_ylabel(r"median $L_1$ to truth", color=P["accent"])
    axB.tick_params(axis="y", colors=P["accent"])
    axB.set_xlabel("number of trees")
    axB.set_xticks(t)
    axB.set_title("Activity recovery")
    axB.grid(axis="y", color=P["grey"], lw=0.6, alpha=0.35)
    axB.set_axisbelow(True)

    fig.tight_layout()
    fs.save(fig, a.outdir, a.name)
    print(f"saved {a.name} figure to {a.outdir}")


if __name__ == "__main__":
    main()