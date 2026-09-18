"""
plot_switch_recovery.py

Reliability diagram and precision-recall curve for the switch model's
per-node P(active), from switch_recovery.py's outputs.

Left: reliability diagram from switch_calibration.csv (the chain = "mean"
rows), predicted P(active) per bin against the observed active frequency,
marker area proportional to the bin count, with the diagonal as the
perfectly calibrated reference.
Right: precision-recall curve of the chain-mean P(active) against
true_active from switch_nodes.csv, overall (stiff) and per signature (soft),
with the positive prevalence as the no-skill baseline (grey). A signature
whose truth has one class has no curve and is skipped.

Inputs
    --nodes          switch_nodes.csv
    --calibration    switch_calibration.csv
    --outdir         where switch_recovery.pdf / .png are written

Usage
    python scripts/plot_switch_recovery.py --nodes .../switch_nodes.csv \\
        --calibration .../switch_calibration.csv --outdir .../switch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plotting.figure_style import PALETTE, apply_style, save  # noqa: E402


def plot(nodes: pd.DataFrame, cal: pd.DataFrame, outdir):
    apply_style()
    fig, (ax_cal, ax_pr) = plt.subplots(1, 2, figsize=(9.0, 4.0))

    cal_mean = cal[cal["chain"].astype(str) == "mean"]
    if cal_mean.empty:
        cal_mean = cal
    cal_mean = cal_mean[cal_mean["n"] > 0]
    ax_cal.plot([0, 1], [0, 1], ls="--", lw=1, color=PALETTE["grey"], label="perfect")
    sizes = 20 + 200 * cal_mean["n"] / max(cal_mean["n"].max(), 1)
    ax_cal.scatter(
        cal_mean["mean_pred"],
        cal_mean["obs_freq"],
        s=sizes,
        color=PALETTE["stiff"],
        zorder=3,
        label="chain-mean P(active), area = bin count",
    )
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.set_xlabel("predicted P(active)")
    ax_cal.set_ylabel("observed active frequency")
    ax_cal.set_title("Reliability")
    ax_cal.legend(frameon=False, fontsize=8, loc="upper left")

    y_all = nodes["true_active"].to_numpy().astype(int)
    p_all = nodes["p_active_mean"].to_numpy().astype(float)
    if y_all.min() != y_all.max():
        prec, rec, _ = precision_recall_curve(y_all, p_all)
        ax_pr.plot(rec, prec, color=PALETTE["stiff"], lw=2, label="overall")
        ax_pr.axhline(
            y_all.mean(), ls="--", lw=1, color=PALETTE["grey"], label="prevalence"
        )
    for sig, sub in nodes.groupby("signature"):
        y = sub["true_active"].to_numpy().astype(int)
        if y.min() == y.max():
            continue
        prec, rec, _ = precision_recall_curve(y, sub["p_active_mean"].to_numpy())
        ax_pr.plot(rec, prec, color=PALETTE["soft"], lw=1, alpha=0.7, label=str(sig))
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.02)
    ax_pr.set_xlabel("recall")
    ax_pr.set_ylabel("precision")
    ax_pr.set_title("Precision-recall, P(active)")
    ax_pr.legend(frameon=False, fontsize=8, loc="lower left")

    fig.tight_layout()
    save(fig, outdir, "switch_recovery")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    plot(pd.read_csv(a.nodes), pd.read_csv(a.calibration), a.outdir)


if __name__ == "__main__":
    main()
