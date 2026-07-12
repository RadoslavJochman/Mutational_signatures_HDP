"""
plot_camp_path.py

Figures for the mixing-geometry section, from the CSVs written by
diagnose_camp_path.py. Two figures:

  fig_camp_path_profile     log-densities along the path, with the
                            likelihood derivative on a twin axis and the straight
                            sampler-coordinate line shown dashed as the logit-geometry
                            reference.

  fig_camp_path_curvature   top: the second directional derivative of logp along the
                            path against the stiff orthogonal walls,
                            the canyon

Usage
    python scripts/plot_camp_path.py --indir ../results/<run>/camp_path \\
        --outdir ../results/<run>/camp_path
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
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    indir, outdir = Path(a.indir), Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(indir / "camp_path_profile.csv")

    import matplotlib
    matplotlib.use("Agg")
    apply_style()
    import matplotlib.pyplot as plt
    P = PALETTE

    def _shift(col):
        return df[col] - df[col].iloc[0]

    # Figure 1
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(df.t, _shift("loglik"), color=P["accent"], lw=2.2, label="log-likelihood")
    ax.plot(df.t, _shift("logp"), color=P["soft"], lw=2.2, label="log-posterior")
    ax.axvline(0, color=P["grey"], ls=":"); ax.axvline(1, color=P["grey"], ls=":")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("interpolation  t   (0 = cluster A, 1 = cluster B, label-aligned)")
    ax.set_ylabel("log-density, shifted to cluster A = 0")
    ax.grid(color=P["grey"], lw=0.6, alpha=0.3); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower center")
    fig.tight_layout(); save(fig, outdir, "fig_camp_path_profile"); plt.close(fig)

    # Figure 2
    fig, axc = plt.subplots(figsize=(7.0, 4.2))
    axc.plot(df.t, df.stiff_eig, color=P["stiff"], lw=2, label="stiffest direction")
    axc.set_ylabel("curvature of logp\n(stiffest direction)", color=P["stiff"])
    axc.tick_params(axis="y", labelcolor=P["stiff"])
    axc.set_xlabel("interpolation  t   (0 = cluster A, 1 = cluster B)")
    axco = axc.twinx()
    axco.plot(df.t, df.on_path_curv, color=P["soft"], lw=2.2, label="along the path (soft)")
    axco.set_ylabel("curvature of logp\n(along path, soft)", color=P["soft"])
    axco.tick_params(axis="y", labelcolor=P["soft"])
    lines = axc.get_lines()[:1] + axco.get_lines()
    axc.legend(lines, [ln.get_label() for ln in lines], frameon=False, fontsize=9,
               loc="lower right")
    axc.grid(color=P["grey"], lw=0.6, alpha=0.3); axc.set_axisbelow(True)
    fig.tight_layout(); save(fig, outdir, "fig_camp_path_curvature"); plt.close(fig)

    print(f"wrote fig_camp_path_profile and fig_camp_path_curvature to {outdir}")


if __name__ == "__main__":
    main()