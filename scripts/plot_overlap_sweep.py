"""
Plot the signature-overlap sweep overview.

Purpose
    Render the four-panel overview of the de novo sweep: signature recovery
    against overlap with the three regimes marked, activity recovery against
    overlap, the smallest singular value that drives both, and the
    convergence diagnostics that look best where recovery is worst.

Method
    Every value is read from the assembled sweep table written by
    sweep_aggregate.py; no trace is touched. The regime bands are descriptive
    overlays drawn at fixed correlation boundaries and carry no analysis.

Inputs
    sweep_master.csv  one row per correlation, with at least the columns
                      correlation, smallest_sv, eff_num_sigs, ess_median,
                      sig_recovery_best, sig_recovery_worst, L1_best_median

Outputs
    a PDF and a PNG written to the output directory

Interpretation hint
    Read the effective signatures and the convergence columns together: a high
    ess with a low effective signature count is a collapsed run, not a clean
    one. Recovery against truth, not convergence, separates the regimes.

Usage:
    python scripts/plot_overlap_sweep.py \
        --sweep sweep_master.csv \
        [--outdir figures] [--name denovo_overlap_sweep]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plotting.figure_style import PALETTE, apply_style, save

# descriptive regime overlays, as (rho_low, rho_high, colour, label)
REGIMES = [
    (0.04, 0.60, "#e9f3ec", "soft ridge\n(activity-limited)"),
    (0.60, 0.80, "#fdeede", "multimodal $S$\n(sampling-limited)"),
    (0.80, 0.96, "#fae3e3", "collapse\n(non-identifiable)"),
]


def _bands(ax, label=True):
    """Shade the three regimes behind the data and label them at the top."""
    y0, y1 = ax.get_ylim()
    for lo, hi, col, name in REGIMES:
        ax.axvspan(lo, hi, color=col, zorder=0)
        if label:
            ax.text((lo + hi) / 2, y1 - (y1 - y0) * 0.04, name, ha="center",
                    va="top", fontsize=7.6, color="#555555")
    ax.set_xlim(0.04, 0.96)


def panel_recovery(ax, df):
    """Best- and worst-camp signature cosine to truth against correlation."""
    r = df["correlation"].values
    ax.set_ylim(0.55, 1.03)
    _bands(ax)
    ax.plot(r, df["sig_recovery_best"], "-o", color=PALETTE["stiff"], lw=2,
            ms=7, label="best camp", zorder=3)
    ax.plot(r, df["sig_recovery_worst"], "--s", color=PALETTE["soft"], lw=1.8,
            ms=6, label="worst camp", zorder=3)
    ax.annotate("camps reconstruct\ndifferent $S$", (0.7, 0.738), (0.55, 0.66),
                fontsize=8, color=PALETTE["soft"], ha="center",
                arrowprops=dict(arrowstyle="->", color=PALETTE["soft"], lw=1))
    ax.set_xlabel(r"signature correlation $\rho$")
    ax.set_ylabel("signature cosine to truth")
    ax.set_title("A.  Signature recovery and the three regimes")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.set_xticks(r)


def panel_activity(ax, df):
    """Median best-chain activity L1 to truth against correlation."""
    r = df["correlation"].values
    ax.set_ylim(0, 2.02)
    _bands(ax)
    ax.plot(r, df["L1_best_median"], "-o", color=PALETTE["stiff"], lw=2, ms=7,
            zorder=3)
    ax.axhline(0.1, color=PALETTE["grey"], ls=":", lw=1)
    ax.text(0.05, 0.13, "close threshold 0.1", fontsize=7.8,
            color=PALETTE["grey"])
    ax.axhline(2.0, color=PALETTE["grey"], ls=":", lw=1)
    ax.text(0.05, 1.93, "max possible $L_1$ = 2", fontsize=7.8,
            color=PALETTE["grey"], va="top")
    for xi, yi in zip(r, df["L1_best_median"]):
        ax.annotate(f"{yi:.2f}", (xi, yi), (0, 8), textcoords="offset points",
                    fontsize=8, ha="center", color=PALETTE["stiff"],
                    fontweight="bold")
    ax.set_xlabel(r"signature correlation $\rho$")
    ax.set_ylabel("median best-chain activity $L_1$ to truth")
    ax.set_title("B.  Activity recovery: smooth decay, then a cliff")
    ax.set_xticks(r)


def panel_conditioning(ax, df):
    """Smallest singular value of S on the sum-zero subspace, the gain that
    drives recovery, against correlation."""
    r = df["correlation"].values
    ax.plot(r, df["smallest_sv"], "-o", color=PALETTE["accent"], lw=2, ms=7,
            zorder=3)
    ax.set_ylim(0, df["smallest_sv"].max() * 1.15)
    _bands(ax, label=False)
    ax.axhspan(0.03, 0.05, color="#dddddd", alpha=0.7, zorder=0.5)
    ax.text(0.93, 0.04, "identifiability\nboundary", fontsize=7.6,
            color="#555555", ha="right", va="center")
    for xi, yi in zip(r, df["smallest_sv"]):
        ax.annotate(f"{yi:.3f}", (xi, yi), (0, 8), textcoords="offset points",
                    fontsize=8, ha="center", color="#3a7d4f")
    ax.set_xlabel(r"signature correlation $\rho$")
    ax.set_ylabel(r"$\sigma_{min}$ of $S$ on sum-zero subspace")
    ax.set_title("C.  The driver: the softest direction's gain collapses")
    ax.set_xticks(r)


def panel_trap(ax, df):
    """Median activity ess against correlation with effective signatures on a
    second axis: the cleanest convergence sits on the collapsed run."""
    r = df["correlation"].values
    ax.bar(r, df["ess_median"], width=0.07, color="#b8cce4",
           edgecolor=PALETTE["stiff"], zorder=3)
    ax.set_ylabel("median ess$_{bulk}$", color=PALETTE["stiff"])
    ax.tick_params(axis="y", labelcolor=PALETTE["stiff"])
    ax.set_ylim(0, df["ess_median"].max() * 1.12)
    for xi, yi in zip(r, df["ess_median"]):
        ax.annotate(f"{yi:.0f}", (xi, yi), (0, 4), textcoords="offset points",
                    fontsize=8, ha="center", color=PALETTE["stiff"])
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(r, df["eff_num_sigs"], "-D", color=PALETTE["soft"], lw=1.8, ms=6,
             zorder=4)
    ax2.set_ylabel("effective # signatures", color=PALETTE["soft"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["soft"])
    ax2.set_ylim(0, 7)
    ax.set_xlabel(r"signature correlation $\rho$")
    ax.set_title("D.  The trap: best diagnostics at the most-wrong run")
    ax.set_xticks(r)
    ax.set_xlim(0.02, 0.98)
    worst = df.loc[df["L1_best_median"].idxmax()]
    ax.annotate("ess best but collapsed to ~1 signature",
                (worst["correlation"], worst["ess_median"]),
                (0.55, worst["ess_median"] * 0.82), fontsize=7.8,
                color="#333333",
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1))


def build_figure(df):
    df = df.sort_values("correlation")
    fig, ax = plt.subplots(2, 2, figsize=(11.4, 8.8))
    fig.suptitle("De novo signature discovery across the signature-overlap "
                 "sweep", fontsize=13, fontweight="bold", y=0.985)
    panel_recovery(ax[0, 0], df)
    panel_activity(ax[0, 1], df)
    panel_conditioning(ax[1, 0], df)
    panel_trap(ax[1, 1], df)
    fig.text(0.5, 0.005,
             "Recovery is clean up to $\\rho=0.5$ (camps agree on $S$, only "
             "soft activity ridges); the signature posterior fragments at "
             "$\\rho\\approx0.7$ and collapses at $\\rho=0.9$, where "
             "convergence looks best.",
             ha="center", fontsize=9, style="italic", color="#222222")
    fig.tight_layout(rect=[0, 0.03, 1, 0.965])
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True,
                    help="sweep_master.csv from sweep_aggregate.py.")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--name", default="denovo_overlap_sweep")
    args = ap.parse_args()

    apply_style()
    df = pd.read_csv(args.sweep)
    fig = build_figure(df)
    save(fig, args.outdir, args.name)
    print(f"Written: {Path(args.outdir) / (args.name + '.pdf')}, "
          f"{Path(args.outdir) / (args.name + '.png')}")


if __name__ == "__main__":
    main()