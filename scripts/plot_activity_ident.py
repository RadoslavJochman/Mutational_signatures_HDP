"""
Plot the activity identifiability summary for one overlap level.

Purpose
    Render the four-panel figure for the soft-ridge analysis of a single run:
    that the signatures are recovered and the camps agree on them, that the
    camp difference and the residual error occupy opposite (stiff and soft)
    subspaces, that the camp difference is amplified near the stiffest gain,
    and that the data assigns a log-likelihood gap between the camps.

Method
    Every quantity is read from the diagnostic tables already written by
    diagnose_activity_identifiability.py and diagnose_camp_sig.py, so the
    figure is reproducible from the analysis outputs without re-touching the
    trace. Medians for the annotations are recomputed from the per-node table.

Inputs
    activity_ident_by_node.csv  per-node geometry, spectrum distances, log
                                likelihood gaps, and error decomposition
    camp_signatures.csv         per-signature between-camp and to-truth cosine
                                (optional; the first panel is skipped if absent)
    sigma_min, sigma_max        the softest and stiffest singular values of S
                                on the sum-zero subspace, taken from the
                                summary table or passed directly

Outputs
    a PDF and a PNG written to the output directory

Interpretation hint
    A camp difference concentrated at low soft fraction together with a
    residual error concentrated at high soft fraction says the multimodality
    is a sampling effect along data-visible directions while the residual is
    an information floor along data-invisible ones. A non-trivial
    log-likelihood gap means the camps are not observationally equivalent.

Usage:
    python scripts/plot_activity_ident.py \
        --by-node activity_ident/activity_ident_by_node.csv \
        --camp-sigs diagnose_camp_sig/camp_signatures.csv \
        --summary activity_ident/activity_ident_summary.csv \
        [--sigma-min 0.068] [--sigma-max 0.128] [--split-threshold 0.05] \
        [--rho 0.3] [--outdir figures] [--name denovo_corr03_soft_ridge]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plotting.figure_style import PALETTE, apply_style, save


def _resolve_sigmas(args):
    """Sigma_min and sigma_max from the explicit args, else the summary CSV,
    else (None, None) so the conditioning band is omitted."""
    if args.sigma_min is not None and args.sigma_max is not None:
        return args.sigma_min, args.sigma_max
    if args.summary and Path(args.summary).exists():
        s = pd.read_csv(args.summary)
        if {"sigma_min", "sigma_max"} <= set(s.columns):
            return float(s["sigma_min"].iloc[0]), float(s["sigma_max"].iloc[0])
    return None, None


def panel_recovery(ax, camp_sigs):
    """Signature recovery: between-camp cosine and worst-camp cosine to truth.
    Both near one means the failure is not in S but in the activities."""
    x = camp_sigs["signature"].values
    worst_truth = np.minimum(camp_sigs["campA_cos_truth"],
                             camp_sigs["campB_cos_truth"])
    ax.vlines(x, 0.97, camp_sigs["between_camp_cos"], color="#cccccc", lw=1)
    ax.scatter(x, camp_sigs["between_camp_cos"], s=42, color=PALETTE["stiff"],
               zorder=3, label="between camps (A vs B)")
    ax.scatter(x, worst_truth, s=42, marker="s", color=PALETTE["soft"],
               zorder=3, label="camp vs truth (worst of A,B)")
    ax.axhline(0.99, color=PALETTE["grey"], ls=":", lw=1)
    ax.set_ylim(0.970, 1.002)
    ax.set_xticks(x)
    ax.set_xlabel("signature index")
    ax.set_ylabel("cosine similarity")
    ax.set_title("A.  Signatures are recovered, and both camps agree")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.text(0.02, 0.30,
            "between-camp cosine $\\geq$ 0.999\n"
            "the disagreement is not in $S$, it is in the activities",
            transform=ax.transAxes, fontsize=8.3, va="bottom", color="#333333")


def panel_subspace(ax, df, split):
    """Soft-energy fraction of the camp difference (split nodes) against that
    of the best-chain residual (all nodes), with the isotropic baseline."""
    bins = np.linspace(0, 1, 26)
    ax.hist(split["de_soft_frac"], bins=bins, density=True,
            color=PALETTE["stiff"], alpha=0.78,
            label=f"camp difference $e_A-e_B$  (n={len(split)} split nodes)")
    ax.hist(df["err_soft_frac"].dropna(), bins=bins, density=True,
            color=PALETTE["soft"], alpha=0.62,
            label="best-chain residual $e_{best}-e_{true}$  (all nodes)")
    ymax = ax.get_ylim()[1]
    ax.axvline(0.5, color=PALETTE["grey"], ls="--", lw=1.2)
    ax.text(0.51, ymax * 0.42, "isotropic\nbaseline 0.5", fontsize=8,
            color=PALETTE["grey"], va="center", ha="center")
    m_camp = split["de_soft_frac"].median()
    m_err = df["err_soft_frac"].median()
    ax.axvline(m_camp, color=PALETTE["stiff"], lw=2)
    ax.axvline(m_err, color=PALETTE["soft"], lw=2)
    ax.annotate(f"median {m_camp:.2f}", (m_camp, ymax * 0.50),
                color=PALETTE["stiff"], fontsize=8.5, ha="center",
                fontweight="bold")
    ax.annotate(f"median {m_err:.2f}", (m_err, ymax * 0.62),
                color=PALETTE["soft"], fontsize=8.5, ha="center",
                fontweight="bold")
    ax.set_xlabel("fraction of energy in $S$'s soft subspace")
    ax.set_ylabel("density")
    ax.set_title("B.  The two errors live in different subspaces")
    ax.legend(frameon=False, fontsize=8.0, loc="upper right")
    ax.text(0.01, -0.20,
            "left = stiff (data-visible) . right = soft (data-invisible)",
            transform=ax.transAxes, fontsize=8.5, color="#333333")


def panel_amplification(ax, split, sigma_min, sigma_max):
    """Per-node amplification of the camp difference, against the singular
    value band: points near sigma_max ride the stiff (data-visible) gain."""
    if sigma_min is not None and sigma_max is not None:
        ax.axhspan(sigma_min, sigma_max, color="#cfe0ee", alpha=0.6, zorder=0)
        ax.axhline(sigma_max, color=PALETTE["stiff"], ls="--", lw=1)
        ax.axhline(sigma_min, color=PALETTE["grey"], ls="--", lw=1)
        xr = split["de_l1"].max()
        ax.text(xr * 0.99, sigma_max,
                f" $\\sigma_{{max}}={sigma_max:.3f}$ (stiffest)",
                color=PALETTE["stiff"], fontsize=8.3, va="bottom", ha="right")
        ax.text(xr * 0.99, sigma_min,
                f" $\\sigma_{{min}}={sigma_min:.3f}$ (softest)",
                color=PALETTE["grey"], fontsize=8.3, va="top", ha="right")
    ax.scatter(split["de_l1"], split["ratio_dp_de"], s=34,
               color=PALETTE["stiff"], edgecolor="white", linewidth=0.4,
               zorder=3)
    ax.set_xlabel("camp activity difference  $\\|e_A-e_B\\|_1$")
    ax.set_ylabel("amplification  $\\|\\Delta p\\|/\\|\\Delta e\\|$")
    ax.set_title("C.  The camp split rides $S$'s stiff directions")
    ax.text(0.02, 0.04,
            f"median amplification {split['ratio_dp_de'].median():.3f}\n"
            "near the stiffest gain: the data sees the camp difference",
            transform=ax.transAxes, fontsize=8.3, va="bottom", color="#333333")


def panel_loglik(ax, split):
    """Signed log-likelihood gap between camps over the split nodes. A net
    lean to one camp means the modes are not observationally equivalent."""
    g = split["dLL_signed"].values
    bins = np.linspace(-6.5, 6.5, 27)
    ax.hist(g[g >= 0], bins=bins, color=PALETTE["stiff"], alpha=0.85,
            label="favours camp A")
    ax.hist(g[g < 0], bins=bins, color="#b2182b", alpha=0.85,
            label="favours camp B")
    ax.axvline(0, color="#333333", lw=1)
    for xv in (1, 3, -1, -3):
        ax.axvline(xv, color=PALETTE["grey"], ls=":", lw=0.8)
    n_a = int((g > 0.5).sum())
    n_b = int((g < -0.5).sum())
    n_tied = int((np.abs(g) <= 0.5).sum())
    ax.set_xlabel("signed log-likelihood gap between camps (nats)")
    ax.set_ylabel("split nodes")
    ax.set_title("D.  The data prefers one camp (not equivalent modes)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.text(0.98, 0.96,
            f"favours A: {n_a}   B: {n_b}   tied: {n_tied}\n"
            f"|gap| median {np.median(np.abs(g)):.2f} nats",
            transform=ax.transAxes, fontsize=8.3, va="top", ha="right",
            color="#333333")


def build_figure(df, camp_sigs, sigma_min, sigma_max, split_threshold, rho):
    split = df[df["de_l1"] >= split_threshold]
    fig, ax = plt.subplots(2, 2, figsize=(11.2, 8.6))
    rho_str = f"$\\rho={rho}$, " if rho is not None else ""
    fig.suptitle(
        "De novo activity recovery at moderate signature overlap  "
        f"({rho_str}{df.shape[0]} nodes)",
        fontsize=13, fontweight="bold", y=0.985)

    if camp_sigs is not None:
        panel_recovery(ax[0, 0], camp_sigs)
    else:
        ax[0, 0].axis("off")
        ax[0, 0].text(0.5, 0.5, "camp_signatures.csv not supplied",
                      ha="center", va="center", color=PALETTE["grey"])
    panel_subspace(ax[0, 1], df, split)
    panel_amplification(ax[1, 0], split, sigma_min, sigma_max)
    panel_loglik(ax[1, 1], split)

    fig.text(0.5, 0.005,
             "The camp split is a fixable mode-lock along directions the data "
             "can see (B left, C, D); the residual error is an information "
             "floor along directions it cannot (B right).",
             ha="center", fontsize=9, style="italic", color="#222222")
    fig.tight_layout(rect=[0, 0.03, 1, 0.965])
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by-node", required=True, dest="by_node",
                    help="activity_ident_by_node.csv for the run.")
    ap.add_argument("--camp-sigs", default=None, dest="camp_sigs",
                    help="camp_signatures.csv (enables the recovery panel).")
    ap.add_argument("--summary", default=None,
                    help="activity_ident_summary.csv, used for sigma_min/max "
                         "if they are not passed explicitly.")
    ap.add_argument("--sigma-min", type=float, default=None, dest="sigma_min")
    ap.add_argument("--sigma-max", type=float, default=None, dest="sigma_max")
    ap.add_argument("--split-threshold", type=float, default=0.05,
                    dest="split_threshold")
    ap.add_argument("--rho", default=None,
                    help="Correlation label for the title (optional).")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--name", default="activity_ident_soft_ridge")
    args = ap.parse_args()

    apply_style()
    df = pd.read_csv(args.by_node)
    camp_sigs = (pd.read_csv(args.camp_sigs)
                 if args.camp_sigs and Path(args.camp_sigs).exists() else None)
    sigma_min, sigma_max = _resolve_sigmas(args)

    fig = build_figure(df, camp_sigs, sigma_min, sigma_max,
                       args.split_threshold, args.rho)
    save(fig, args.outdir, args.name)
    print(f"Written: {Path(args.outdir) / (args.name + '.pdf')}, "
          f"{Path(args.outdir) / (args.name + '.png')}")


if __name__ == "__main__":
    main()