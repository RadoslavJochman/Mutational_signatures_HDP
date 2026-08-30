"""
plot_spread_across_runs.py

Overlay the across-chain spread against exposure for several runs. Now that the
signature set is frozen across runs (the rho=0.3 set), a signature index means the
same spectrum in every run, so the points that share an index are the same
signature seen at different cohort sizes; the index annotated on each marker lets a
signature be followed across runs without drawing connectors. Reads the
mode_summary.csv written by diagnose_modes.py for each run.

The cloud as a whole shows whether the exposure relation holds across independent
forests rather than within one. The splitting threshold is left to the text rather
than marked on the axes.    

Usage
    python scripts/plot_spread_across_runs.py \
        --paths run20/modes/mode_summary.csv run30/modes/mode_summary.csv \
                run40/modes/mode_summary.csv \
        --labels "20 trees" "30 trees" "40 trees" \
        --outdir figs
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plotting.figure_style import PALETTE, apply_style, save  # noqa: E402


def _run_order(label: str, fallback: int) -> int:
    """Numeric key to order the runs in the legend; first integer in the label
    (so '20 trees' sorts before '30 trees'), else the input position."""
    m = re.search(r"\d+", label)
    return int(m.group()) if m else fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    if len(args.paths) != len(args.labels):
        raise SystemExit("need one label per path")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runs = sorted(
        ((lab, pd.read_csv(p)) for lab, p in zip(args.labels, args.paths)),
        key=lambda lr: _run_order(lr[0], args.labels.index(lr[0])),
    )

    import matplotlib

    matplotlib.use("Agg")
    apply_style()
    import matplotlib.pyplot as plt

    P = PALETTE
    colours = [P["stiff"], P["soft"], P["accent"], P["grey"]]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    has_act = all("act_spread" in df.columns for _, df in runs)

    def _panel(ax, col, ylabel):
        for i, (lab, df) in enumerate(runs):
            ax.scatter(
                df["exposure"],
                df[col],
                s=100,
                color=colours[i % len(colours)],
                marker=markers[i % len(markers)],
                edgecolor="black",
                lw=0.6,
                label=lab,
                zorder=3,
            )
            for _, r in df.iterrows():
                ax.annotate(
                    str(int(r["signature"])),
                    (r["exposure"], r[col]),
                    fontsize=7,
                    color="white",
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=4,
                )
        ax.set_xscale("log")
        ax.set_xlabel("true signature exposure (log scale)")
        ax.set_ylabel(ylabel)
        ax.grid(color=P["grey"], lw=0.6, alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9, title="run")

    fig, axes = plt.subplots(
        1, 2 if has_act else 1, figsize=(9.6 if has_act else 5.4, 4.0), squeeze=False
    )
    _panel(axes[0][0], "spread", "across-chain spread of signature cosine")
    axes[0][0].set_title("Signatures")
    if has_act:
        _panel(axes[0][1], "act_spread", "relative across-chain spread of activity")
        axes[0][1].set_title("Activities")
    fig.tight_layout()
    save(fig, outdir, "fig_spread_across_runs")
    plt.close(fig)
    print(f"wrote fig_spread_across_runs to {outdir} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
