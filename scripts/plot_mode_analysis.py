"""
plot_mode_analysis.py

Plotting half of the de novo mode characterisation. Reads the tables written by
diagnose_modes.py and renders the figures

Figures (written to --outdir)
    fig_recovery_heatmap   chains x signatures, cell = cosine to truth, columns
                           ordered by exposure, rows grouped by camp. Abundant
                           signatures give uniform columns; low-exposure ones
                           split, which is the mode structure made visible.
    fig_mode_separation    per-chain cosine to truth on the two most
                           spectrum-divergent signatures, coloured by camp. This
                           is in signature-shape space, so it is not distorted by
                           the softmax Jacobian the way activity coordinates are.
    fig_spread_vs_exposure across-chain signature spread against true exposure.
    fig_split_signature    the most divergent signature's spectrum under each
                           camp against truth.

Inputs (from diagnose_modes.py --outdir)
    mode_summary.csv, mode_camps.csv, mode_recovery_matrix.csv, mode_spectra.csv

Usage
    python scripts/plot_mode_analysis.py --indir ../results/<run>/modes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.analysis import DISTRIBUTION_METRICS  # noqa: E402
from src.plotting.figure_style import PALETTE, apply_style, save  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="diagnose_modes.py output dir")
    ap.add_argument("--outdir", default=None, help="default: --indir")
    ap.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="heatmap colour floor; default is the data minimum",
    )
    ap.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="heatmap colour ceiling; default is the data maximum",
    )
    ap.add_argument(
        "--signatures",
        type=int,
        nargs="+",
        default=None,
        help="signature indices to show in the spectrum, activity and "
        "chain-cosine figures; default is the split signature and "
        "the most abundant one as a control",
    )
    ap.add_argument(
        "--metric",
        default=None,
        help="override metric label/orientation; default inherits from "
        "mode_summary.csv (cosine for older runs)",
    )
    ap.add_argument(
        "--show-cosine",
        action="store_true",
        help="also annotate the spectrum panels with cosine",
    )
    args = ap.parse_args()
    indir = Path(args.indir)
    outdir = Path(args.outdir) if args.outdir else indir

    summary = pd.read_csv(indir / "mode_summary.csv")
    mat = pd.read_csv(indir / "mode_recovery_matrix.csv")
    spectra = pd.read_csv(indir / "mode_spectra.csv")
    metric = args.metric or (
        summary["metric"].iloc[0] if "metric" in summary.columns else "cosine"
    )
    metric_fn, higher_is_better = DISTRIBUTION_METRICS[metric]
    metric_label = {
        "hellinger": "Hellinger to truth",
        "tv": "total variation to truth",
        "jensen_shannon": "Jensen-Shannon to truth",
        "cosine": "cosine to truth",
    }.get(metric, f"{metric} to truth")
    metric_short = {
        "hellinger": "Hellinger",
        "tv": "TV",
        "cosine": "cosine",
        "jensen_shannon": "JS",
    }.get(metric, metric)

    K = summary.shape[0]
    sig_cols = [f"sig_{k}" for k in range(K)]
    camp = mat["camp"].values
    M = mat[sig_cols].values  # (chains, K)
    n_chains = M.shape[0]
    exposure = summary.set_index("signature")["exposure"].reindex(range(K)).values
    _bcol = "between_camp" if "between_camp" in summary.columns else "between_camp_cos"
    between = summary.set_index("signature")[_bcol].reindex(range(K)).values
    spread = summary.set_index("signature")["spread"].reindex(range(K)).values
    act_spread = (
        summary.set_index("signature")["act_spread"].reindex(range(K)).values
        if "act_spread" in summary.columns
        else None
    )
    kdiv = int(summary.loc[summary["divergent"], "signature"].iloc[0])
    kctrl = int(exposure.argmax())  # well-determined control
    if kctrl == kdiv:
        kctrl = int(np.argsort(exposure)[-2])

    if args.signatures is None:
        ks = [kdiv, kctrl]  # split, then control
        suffix = ""
    else:
        bad = [k for k in args.signatures if not 0 <= k < K]
        if bad:
            raise SystemExit(f"signature index out of range 0..{K - 1}: {bad}")
        ks = list(args.signatures)
        suffix = "_" + "_".join(map(str, ks))

    import matplotlib

    matplotlib.use("Agg")
    apply_style()
    import matplotlib.pyplot as plt

    P = PALETTE

    order = np.argsort(exposure)  # low -> high
    A = [c for c in range(n_chains) if camp[c] == "A"]
    B = [c for c in range(n_chains) if camp[c] == "B"]
    chain_order = A + B

    # Fig 1: recovery heatmap
    fig, ax = plt.subplots(figsize=(0.7 * K + 2.4, 0.42 * n_chains + 1.8))
    H = M[np.ix_(chain_order, order)]  # (chains, K)
    vmin = args.vmin if args.vmin is not None else float(np.nanmin(M))
    vmax = args.vmax if args.vmax is not None else float(np.nanmax(M))
    cmap = "YlGnBu" if higher_is_better else "YlGnBu_r"  # dark = good either way
    im = ax.imshow(H, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    hmid = 0.5 * (vmin + vmax)
    for i in range(n_chains):
        for j in range(K):
            v = H[i, j]
            dark = (v > hmid) if higher_is_better else (v < hmid)
            ax.text(
                j,
                i,
                f"{v:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if dark else "black",
            )
    ax.set_xticks(range(K))
    ax.set_xticklabels(
        [f"{order[j]}\n{exposure[order[j]]:.1f}" for j in range(K)], fontsize=8
    )
    ax.set_yticks(range(n_chains))
    ax.set_yticklabels([f"ch{c} ({camp[c]})" for c in chain_order], fontsize=8)
    ax.set_xlabel("signature (index above, true exposure below), ordered by exposure")
    ax.set_title(f"Signature recovery per chain ({metric_label})")
    if 0 < len(A) < n_chains:
        ax.axhline(len(A) - 0.5, color="black", lw=1.2)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=metric_label)
    fig.tight_layout()
    save(fig, outdir, "fig_recovery_heatmap")
    plt.close(fig)

    def _tag(k):
        if k == kdiv:
            return "split"
        if k == kctrl and args.signatures is None:
            return "control"
        return ""

    # Fig 2: mode separation strip, one panel per chosen signature. Chains sit in
    # two camp bands; on a split signature the bands separate along the cosine axis,
    # on a well-determined one they sit together near one.
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(
        1, len(ks), figsize=(4.9 * len(ks), 3.0), squeeze=False, sharey=True
    )
    for ax, k in zip(axes[0], ks):
        for cid, idx, col, base in [
            ("A", A, P["stiff"], 1.0),
            ("B", B, P["soft"], 0.0),
        ]:
            if idx:
                y = base + rng.uniform(-0.12, 0.12, size=len(idx))
                ax.scatter(
                    M[idx, k],
                    y,
                    s=80,
                    color=col,
                    edgecolor="black",
                    lw=0.5,
                    label=f"cluster {cid}",
                    zorder=3,
                )
                for c, yc in zip(idx, y):
                    ax.annotate(
                        str(c), (M[c, k], yc), fontsize=7, ha="center", va="center"
                    )
        ax.set_yticks([0.0, 1.0])
        ax.set_yticklabels(["cluster B", "cluster A"])
        ax.set_ylim(-0.5, 1.5)
        vlo, vhi = float(M[:, k].min()), float(M[:, k].max())
        if higher_is_better:
            ax.set_xlim(min(0.95, vlo) - 0.01, 1.005)
        else:
            pad = 0.05 * (vhi + 1e-6)
            ax.set_xlim(-pad, vhi + pad)
        ax.set_xlabel(f"{metric_label}, signature {k} (exposure {exposure[k]:.1f})")
        t = _tag(k)
        ax.set_title(f"Signature {k} ({t})" if t else f"Signature {k}")
    axes[0][0].legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    save(fig, outdir, f"fig_mode_separation{suffix}")
    plt.close(fig)

    # Fig 3: across-chain spread vs exposure, signatures and (when available)
    # activities. Both measure how much the chains' answers disagree, in contrast
    # to r-hat which here is dominated by slow overall mixing.
    two = act_spread is not None
    fig, axes = plt.subplots(
        1, 2 if two else 1, figsize=(9.6 if two else 5.4, 4.0), squeeze=False
    )

    def _spread_panel(ax, vals, ylabel, color):
        ax.scatter(
            exposure, vals, s=60, color=color, edgecolor="black", lw=0.5, zorder=3
        )
        for k in range(K):
            ax.annotate(
                str(k),
                (exposure[k], vals[k]),
                fontsize=8,
                xytext=(4, 3),
                textcoords="offset points",
            )
        ax.set_xscale("log")
        ax.set_xlabel("true signature exposure (log scale)")
        ax.set_ylabel(ylabel)
        ax.grid(color=P["grey"], lw=0.6, alpha=0.35)
        ax.set_axisbelow(True)

    _spread_panel(
        axes[0][0],
        spread,
        f"across-chain spread of signature {metric_short.lower()}",
        P["accent"],
    )
    axes[0][0].set_title("Signatures")
    if two:
        _spread_panel(
            axes[0][1],
            act_spread,
            "relative across-chain spread of activity",
            P["soft"],
        )
        axes[0][1].set_title("Activities")
    else:
        axes[0][0].set_title("Disagreement concentrates at low exposure")
    fig.tight_layout()
    save(fig, outdir, "fig_spread_vs_exposure")
    plt.close(fig)

    def _cos(a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def _head(k):
        t = _tag(k)
        inside = f"{t}, " if t else ""
        return (
            f"Signature {k} ({inside}exposure {exposure[k]:.1f}), "
            f"between-cluster {metric_short.lower()} {between[k]:.3f}"
        )

    def _lab(name, vec, truth):
        d = metric_fn(np.asarray(vec, float), np.asarray(truth, float))
        out = f"{name} ({metric_short.lower()} {d:.3f}"
        if args.show_cosine and metric != "cosine":
            out += f", cos {_cos(vec, truth):.3f}"
        return out + ")"

    def _spectrum_panel(ax, k):
        sp = spectra[spectra["signature"] == k].sort_values("channel")
        ch = sp["channel"].values
        ax.bar(
            ch,
            sp["truth"].values,
            width=0.85,
            color=P["grey"],
            alpha=0.5,
            label="truth",
            zorder=2,
        )
        ax.scatter(
            ch,
            sp["campA"].values,
            s=12,
            color=P["stiff"],
            zorder=4,
            label=_lab("cluster A", sp["campA"], sp["truth"]),
        )
        ax.scatter(
            ch,
            sp["campB"].values,
            s=12,
            color=P["soft"],
            zorder=3,
            label=_lab("cluster B", sp["campB"], sp["truth"]),
        )
        ax.set_xlabel("mutation channel")
        ax.set_ylabel("probability")
        ax.set_xlim(-1, len(ch))
        ax.set_title(_head(k))
        ax.legend(frameon=False, fontsize=8)

    # Fig 4: spectrum, one panel per chosen signature
    fig, axes = plt.subplots(len(ks), 1, figsize=(9.0, 3.1 * len(ks)), squeeze=False)
    for ax, k in zip(axes[:, 0], ks):
        _spectrum_panel(ax, k)
    fig.tight_layout()
    save(fig, outdir, f"fig_split_signature{suffix}")
    plt.close(fig)
    n_fig = 4

    # Fig 5 per-node activity, one panel per chosen signature.
    # Off-diagonal points mean the camps place different activity on that signature.
    act_path = indir / "mode_activity.csv"
    if act_path.exists():
        act = pd.read_csv(act_path)
        wide = "signature" not in act.columns
        if wide:
            ks_act = [kdiv]
            if args.signatures is not None:
                print(
                    "note: mode_activity.csv is the legacy single-signature form; "
                    "re-run diagnose_modes.py for activity on other signatures"
                )
        else:
            ks_act = ks
        fig, axes = plt.subplots(
            1, len(ks_act), figsize=(4.6 * len(ks_act), 4.4), squeeze=False
        )
        for ax, k in zip(axes[0], ks_act):
            d = act if wide else act[act["signature"] == k]
            lim = float(max(d["campA"].max(), d["campB"].max())) * 1.08 + 1e-6
            ax.plot([0, lim], [0, lim], color=P["grey"], lw=1, ls="--", zorder=1)
            ax.scatter(
                d["campA"],
                d["campB"],
                s=42,
                color=P["accent"],
                edgecolor="black",
                lw=0.4,
                alpha=0.85,
                zorder=3,
            )
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_aspect("equal")
            ax.set_xlabel(f"cluster A activity, signature {k}")
            ax.set_ylabel(f"cluster B activity, signature {k}")
            t = _tag(k)
            lead = f"Signature {k} ({t})" if t else f"Signature {k}"
            ax.set_title(
                f"{lead}: total A {d['campA'].sum():.1f}, B {d['campB'].sum():.1f}"
            )
        fig.tight_layout()
        save(fig, outdir, f"fig_activity_split{suffix}")
        plt.close(fig)
        n_fig += 1

    # Fig 6/7: between-chain distance matrices, ordered by camp. A split
    # signature shows two diagonal blocks, a well-determined one a single uniform
    # block. The signature matrix is in the chosen metric; the activity matrix is
    # Bray-Curtis, scale-sensitive so it reflects the level split that defines camps.
    def _chain_fig(csv_path, fig_name, good_low, cbar_label):
        if not csv_path.exists():
            return 0
        ccdf = pd.read_csv(csv_path)
        long = "signature" in ccdf.columns
        vcol = "value" if "value" in ccdf.columns else "cosine"
        ks_cc = ks if long else [kdiv]
        if not long and args.signatures is not None:
            print(
                f"note: {csv_path.name} is the legacy single-signature form; "
                "re-run diagnose_modes.py for other signatures"
            )

        def _ccmat(k):
            if long:
                d = ccdf[ccdf["signature"] == k]
                m = d.pivot(index="chain_i", columns="chain_j", values=vcol).values
            else:
                m = pd.read_csv(csv_path, index_col=0).values
            return m[np.ix_(chain_order, chain_order)]

        mats = {k: _ccmat(k) for k in ks_cc}
        if good_low:
            vmn, vmx, cmap = 0.0, max(float(m.max()) for m in mats.values()), "YlGnBu_r"
        else:
            vmn, vmx, cmap = min(float(m.min()) for m in mats.values()), 1.0, "YlGnBu"
        mid = 0.5 * (vmn + vmx)
        fig, axes = plt.subplots(
            1,
            len(ks_cc),
            figsize=((0.5 * n_chains + 1.8) * len(ks_cc), 0.5 * n_chains + 1.8),
            squeeze=False,
        )
        im = None
        for ax, k in zip(axes[0], ks_cc):
            ccm = mats[k]
            im = ax.imshow(ccm, cmap=cmap, vmin=vmn, vmax=vmx)
            ax.set_xticks(range(n_chains))
            ax.set_xticklabels([f"ch{c}" for c in chain_order], rotation=90, fontsize=7)
            ax.set_yticks(range(n_chains))
            ax.set_yticklabels([f"ch{c} ({camp[c]})" for c in chain_order], fontsize=7)
            for i in range(n_chains):
                for j in range(n_chains):
                    dark = (ccm[i, j] < mid) if good_low else (ccm[i, j] > mid)
                    ax.text(
                        j,
                        i,
                        f"{ccm[i, j]:.3f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if dark else "black",
                    )
            if 0 < len(A) < n_chains:
                ax.axhline(len(A) - 0.5, color="black", lw=1.4)
                ax.axvline(len(A) - 0.5, color="black", lw=1.4)
            t = _tag(k)
            lead = f"Signature {k} ({t})" if t else f"Signature {k}"
            ax.set_title(lead)
        fig.colorbar(im, ax=list(axes[0]), fraction=0.046, pad=0.04, label=cbar_label)
        save(fig, outdir, f"{fig_name}{suffix}")
        plt.close(fig)
        return 1

    sig_csv = indir / "mode_chain_signature.csv"
    if not sig_csv.exists():
        sig_csv = indir / "mode_chain_cosine.csv"  # legacy name
    n_fig += _chain_fig(
        sig_csv,
        "fig_chain_signature",
        not higher_is_better,
        f"between-chain {metric_short.lower()}",
    )
    n_fig += _chain_fig(
        indir / "mode_chain_activity.csv",
        "fig_chain_activity",
        True,
        "between-chain activity distance (Bray-Curtis)",
    )

    # Compact figure for the report: the split signature only, the signature-space
    # matrix beside the activity-space one, so the two sit side by side.
    def _draw_chain(fig, ax, csv_path, k, good_low, title, vmax=None):
        ccdf = pd.read_csv(csv_path)
        vcol = "value" if "value" in ccdf.columns else "cosine"
        m = (
            ccdf[ccdf["signature"] == k]
            .pivot(index="chain_i", columns="chain_j", values=vcol)
            .values[np.ix_(chain_order, chain_order)]
        )
        if good_low:
            vmn, vmx = 0.0, (vmax if vmax is not None else float(m.max()))
        else:
            vmn, vmx = float(m.min()), 1.0
        mid = 0.5 * (vmn + vmx)
        cmap = "YlGnBu_r" if good_low else "YlGnBu"
        im = ax.imshow(m, cmap=cmap, vmin=vmn, vmax=vmx)
        ax.set_xticks(range(n_chains))
        ax.set_yticks(range(n_chains))
        ax.set_xticklabels([f"ch{c}" for c in chain_order], rotation=90, fontsize=7)
        ax.set_yticklabels([f"ch{c} ({camp[c]})" for c in chain_order], fontsize=7)
        for i in range(n_chains):
            for j in range(n_chains):
                dark = (m[i, j] < mid) if good_low else (m[i, j] > mid)
                ax.text(
                    j,
                    i,
                    f"{m[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if dark else "black",
                )
        if 0 < len(A) < n_chains:
            ax.axhline(len(A) - 0.5, color="black", lw=1.4)
            ax.axvline(len(A) - 0.5, color="black", lw=1.4)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    act_csv = indir / "mode_chain_activity.csv"
    if sig_csv.exists() and act_csv.exists():
        rows = ks  # split, then control

        def _colmax(csv_path):
            d = pd.read_csv(csv_path)
            vc = "value" if "value" in d.columns else "cosine"
            return max(float(d[d.signature == k][vc].max()) for k in rows)

        sig_vmax = _colmax(sig_csv) if not higher_is_better else None
        act_vmax = _colmax(act_csv)
        fig, axes = plt.subplots(
            len(rows),
            2,
            figsize=(2 * (0.5 * n_chains + 1.9), len(rows) * (0.5 * n_chains + 1.5)),
            squeeze=False,
        )
        for r, k in enumerate(rows):
            _draw_chain(
                fig,
                axes[r][0],
                sig_csv,
                k,
                not higher_is_better,
                f"signature {k}, signature",
                vmax=sig_vmax,
            )
            _draw_chain(
                fig,
                axes[r][1],
                act_csv,
                k,
                True,
                f"signature {k}, activity",
                vmax=act_vmax,
            )
        fig.tight_layout()
        save(fig, outdir, f"fig_chain_split{suffix}")
        plt.close(fig)
        n_fig += 1

    print(f"wrote {n_fig} figures to {outdir}")


if __name__ == "__main__":
    main()
