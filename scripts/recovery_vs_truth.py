"""
recovery_vs_truth.py

Align a posterior trace to the true labelling and measure recovery as cosine
similarity to truth -- for the inferred signatures (shape space) and for the
per-node activities (composition space) -- then plot both.

Alignment
    For a de novo run (the trace carries a `signatures` variable) the signature
    labels are arbitrary per chain, so each chain is aligned to the true
    labelling by Hungarian matching on signature cosine (chain_perms_to_true);
    the same per-chain permutation is applied to the activity components. For a
    fixed-signature run (no `signatures` variable, or --true-signatures omitted)
    there is no label switching: component k already is true signature k, the
    permutation is the identity, and only the activities are scored.

Why per chain rather than the pooled mean
    On a multimodal (camp-split) run the pooled posterior mean blends chains
    that sit in different labellings and is not a valid estimate. This script
    aligns and scores each chain separately, then reports the across-chain
    spread (best / mean / worst), so camp disagreement is visible rather than
    averaged away. On a clean unimodal run best ~ mean and the spread is tight.

Inputs
    --trace            trace_raw.nc or trace_aligned.nc (either works; each
                       chain is re-aligned to truth independently)
    --true-activities  true_activities.csv  (rows = nodes, cols = signatures)
    --newick           newick_string.nwk    (maps activity rows to node labels)
    --true-signatures  fixed_signatures.csv (optional; omit for fixed-sig runs)

Outputs (written to --outdir)
    recovery_signatures.csv   per signature: true exposure, signature cosine to
                              truth (per chain + best/mean/worst), and the
                              activity-usage-profile cosine (per chain + agg)
    recovery_activities.csv   per node: best/mean/worst-chain activity cosine
    recovery_summary.csv      dataset-level medians/means
    recovery_vs_truth.png     signature panel (if applicable) + activity panel

Interpretation hint
    Read the across-chain spread, not just the mean. A signature (or node) with
    a high best but a low worst is recovered by some chains and not others --
    the camp signature. A uniformly low cosine is a genuine recovery failure.
    On fixed-signature runs only the activity panel is drawn.

Usage
    python scripts/recovery_vs_truth.py \\
        --trace ../results/<run>/trace_raw.nc \\
        --true-activities ../data/<run>/true_activities.csv \\
        --newick ../data/<run>/newick_string.nwk \\
        --true-signatures ../data/<run>/fixed_signatures.csv \\
        --outdir ../results/<run>/recovery
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.analysis import (across_chain, build_forest, chain_perms_to_true,
                                   exposure_errors, node_distances, nodes_by_depth,
                                   signature_distances, usage_distances)


def _aligned_activities(post, perms, newick, truth):
    """Per-chain, per-node activities aligned to the true labelling, matched to
    the rows of `truth`. Returns (labels, A) with A shape (n_nodes, n_chains, K)
    read straight from the e_level_* deterministics; no model is built."""
    dr = nodes_by_depth(build_forest(newick))            # (depth, pos) -> label
    e_vars = sorted([v for v in post.data_vars if v.startswith("e_level")])
    n_chains = post.sizes["chain"]
    labels, rows = [], []
    for ev in e_vars:
        depth = int(ev.split("_")[-1])
        cmean = post[ev].values.mean(axis=1)             # (chains, n_rows, K)
        for r in range(cmean.shape[1]):
            label = dr.get((depth, r))
            if label is None or label not in truth.index:
                continue
            labels.append(label)
            rows.append(np.stack([cmean[c, r][perms[c]] for c in range(n_chains)]))
    return labels, np.stack(rows)                        # (n_nodes, chains, K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--true-activities", required=True)
    ap.add_argument("--newick", required=True)
    ap.add_argument("--true-signatures", default=None,
                    help="omit for fixed-signature runs (no label switching)")
    ap.add_argument("--metrics", nargs="+", default=["tv", "hellinger", "cosine"],
                    help="distribution distances to report (tv, hellinger, js, "
                         "cosine); shared with nmf_baseline.py")
    ap.add_argument("--outdir", default="recovery")
    a = ap.parse_args()

    post = az.from_netcdf(a.trace).posterior
    n_chains = post.sizes["chain"]
    truth = pd.read_csv(a.true_activities, index_col=0)
    K = truth.shape[1]

    do_sigs = (a.true_signatures is not None) and ("signatures" in post.data_vars)
    if a.true_signatures is not None and "signatures" not in post.data_vars:
        print("note: --true-signatures given but the trace has no `signatures` "
              "variable (fixed-signature run); scoring activities only.")

    metrics = a.metrics
    if do_sigs:
        true_S = pd.read_csv(a.true_signatures, index_col=0).values
        S = post["signatures"].values                    # (chains, draws, K, C)
        perms = chain_perms_to_true(S, true_S)
        Smean = S.mean(axis=1)                            # (chains, K, C)
    else:
        perms = [np.arange(K) for _ in range(n_chains)]

    labels, A = _aligned_activities(post, perms, Path(a.newick).read_text().strip(),
                                    truth)  # (nodes, chains, K)
    true_acts = truth.loc[labels].values                 # (n_nodes, K)
    n_nodes = len(labels)

    # score every chain with the shared metric layer, then aggregate over chains
    # with each metric's own direction.
    sig_pc = ([signature_distances(Smean[c][perms[c]], true_S, metrics)
               for c in range(n_chains)] if do_sigs else None)
    use_pc = [usage_distances(A[:, c, :], true_acts, metrics) for c in range(n_chains)]
    nod_pc = [node_distances(A[:, c, :], true_acts, metrics) for c in range(n_chains)]
    expo_pc = np.stack([exposure_errors(A[:, c, :], true_acts)
                        for c in range(n_chains)], axis=1)        # (K, chains)

    def _stack(per_chain):                               # {m: (d, n_chains)}
        return {m: np.stack([pc[m] for pc in per_chain], axis=1) for m in metrics}

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    exposure = true_acts.sum(axis=0)
    use_st = _stack(use_pc)
    nod_st = _stack(nod_pc)
    sig_st = _stack(sig_pc) if do_sigs else None

    # per-signature table
    sig_tbl = {"signature": np.arange(K), "exposure": exposure,
               "rel_exposure_err_mean": expo_pc.mean(1),
               "rel_exposure_err_worst": expo_pc.max(1)}
    for m in metrics:
        if do_sigs:
            b, mn, w = across_chain(sig_st[m], m)
            sig_tbl[f"sig_{m}_best"] = b; sig_tbl[f"sig_{m}_mean"] = mn
            sig_tbl[f"sig_{m}_worst"] = w
        b, mn, w = across_chain(use_st[m], m)
        sig_tbl[f"use_{m}_best"] = b; sig_tbl[f"use_{m}_mean"] = mn
        sig_tbl[f"use_{m}_worst"] = w
    sig_df = pd.DataFrame(sig_tbl).sort_values("exposure").reset_index(drop=True)
    sig_df.to_csv(out / "recovery_signatures.csv", index=False)

    # per-node table
    node_tbl = {"node": labels, "exposure_node": true_acts.sum(1)}
    for m in metrics:
        b, mn, w = across_chain(nod_st[m], m)
        node_tbl[f"act_{m}_best"] = b; node_tbl[f"act_{m}_mean"] = mn
        node_tbl[f"act_{m}_worst"] = w
    node_df = pd.DataFrame(node_tbl)
    node_df.to_csv(out / "recovery_activities.csv", index=False)

    # summary
    summary = {"n_chains": n_chains, "K": K, "n_nodes": n_nodes,
               "rel_exposure_err_max": float(expo_pc.mean(1).max())}
    for m in metrics:
        _, mn, _ = across_chain(nod_st[m], m)
        summary[f"act_{m}_median_mean"] = float(np.median(mn))
        if do_sigs:
            _, smn, _ = across_chain(sig_st[m], m)
            summary[f"sig_{m}_mean_mean"] = float(np.mean(smn))
    pd.DataFrame([summary]).to_csv(out / "recovery_summary.csv", index=False)

    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 50)
    print(sig_df.round(3).to_string(index=False))
    print()
    print(pd.Series(summary).round(4).to_string())

    # plot
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        npan = 2 if do_sigs else 1
        fig, axes = plt.subplots(1, npan, figsize=(6.2 * npan, 4.4), squeeze=False)
        col = 0
        pm = "cosine" if "cosine" in metrics else metrics[0]
        higher_better = (pm == "cosine")
        if do_sigs:
            ax = axes[0, col]; col += 1
            order = np.argsort(exposure)
            x = np.arange(K)
            sc = sig_st[pm][order]                     # (K, chains)
            lo, hi, mn = sc.min(1), sc.max(1), sc.mean(1)
            ax.errorbar(x, mn, yerr=[mn - lo, hi - mn], fmt="D", ms=6,
                        color="#1f3b66", ecolor="#6b8ec4", elinewidth=2,
                        capsize=3, zorder=3,
                        label="across-chain mean (whisker = chain range)")
            ax.set_xticks(x); ax.set_xticklabels(order)
            ax.set_xlabel("true signature (ordered by exposure)")
            ax.set_ylabel(f"signature {pm} to truth")
            ax.set_title("Signature recovery")
            ax.legend(frameon=False, fontsize=8,
                      loc="lower right" if higher_better else "upper right")
        ax = axes[0, col]
        nb, nm, nw = across_chain(nod_st[pm], pm)         # per-node best/mean/worst
        lo_all, hi_all = min(nb.min(), nw.min()), max(nb.max(), nw.max())
        pad = 0.02 * (hi_all - lo_all + 1e-9)
        bins = np.linspace(lo_all - pad, hi_all + pad, 30)
        ax.hist(nb, bins=bins, color="#6b8ec4", edgecolor="#1f3b66",
                alpha=0.85, label="best chain")
        if n_chains > 1:
            ax.hist(nw, bins=bins, color="#d39a9a", edgecolor="#9c4f4f",
                    alpha=0.5, label="worst chain")
        ax.axvline(np.median(nb), color="#1f3b66", ls="--", lw=1.2)
        ax.set_xlabel(f"per-node {pm} to true activity")
        ax.set_ylabel("number of nodes")
        ax.set_title("Activity recovery")
        ax.legend(frameon=False, fontsize=8,
                  loc="upper left" if higher_better else "upper right")
        fig.tight_layout(); fig.savefig(out / "recovery_vs_truth.png", dpi=150)
        print(f"\nwrote {out/'recovery_vs_truth.png'}")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()