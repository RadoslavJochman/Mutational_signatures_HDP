"""
plot_convergence_exposure.py

Plotting half. Reads convergence_exposure.csv and shows that the convergence of both
signatures and activities tracks how much each signature is used. Two panels, r-hat
and effective sample size, each against true exposure on a log axis, with a series
for the signatures and one for the activities.

Usage
    python scripts/plot_convergence_exposure.py --indir <run>/conv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plotting.figure_style import PALETTE, apply_style, save  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    indir = Path(args.indir)
    outdir = Path(args.outdir) if args.outdir else indir

    df = pd.read_csv(indir / "convergence_exposure.csv").sort_values("exposure")
    x = df["exposure"].values

    import matplotlib

    matplotlib.use("Agg")
    apply_style()
    import matplotlib.pyplot as plt

    P = PALETTE

    def _panel(ax, ycol_sig, ycol_act, ylabel, hline=None):
        ax.plot(
            x,
            df[ycol_sig],
            "-o",
            color=P["stiff"],
            ms=6,
            lw=1.2,
            label="signatures",
            zorder=3,
        )
        ax.plot(
            x,
            df[ycol_act],
            "-s",
            color=P["accent"],
            ms=6,
            lw=1.2,
            label="activities",
            zorder=3,
        )
        if hline is not None:
            ax.axhline(hline, color=P["grey"], lw=1, ls="--", zorder=1)
        ax.set_xscale("log")
        ax.set_xlabel("true signature exposure (log scale)")
        ax.set_ylabel(ylabel)
        ax.grid(color=P["grey"], lw=0.6, alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    _panel(axes[0], "rhat_sig", "rhat_act", "worst-case $\\hat{r}$", hline=1.01)
    axes[0].set_title("Mixing degrades at low exposure")
    _panel(axes[1], "ess_sig", "ess_act", "minimum bulk ESS")
    axes[1].set_yscale("log")
    axes[1].set_title("Sampling efficiency degrades at low exposure")
    fig.tight_layout()
    save(fig, outdir, "fig_convergence_exposure")
    plt.close(fig)
    print(f"wrote fig_convergence_exposure to {outdir}")


if __name__ == "__main__":
    main()
