"""
switch_recovery.py

Score the switch model's per-node on/off recovery against the simulator's
ground truth (switch_model_plan.md section 6.3). This is the evidence for the
calibrated per-edge / per-node switch detection claim.

Per chain, then best / mean / worst across chains, as recovery_vs_truth.py
does: on a camp-split run the pooled mean blends chains in different
labellings, so each chain is scored on its own and the spread is reported.

Label frames
    a_prob_level_* is in the trace's labelling. For a de novo run pass
    --true-signatures: each chain is aligned to the true labelling by
    Hungarian cosine matching of `signatures` (chain_perms_to_true) and the
    same permutation is applied to a_prob_level_*. For a fixed-signature run
    component k already is true signature k. switch_edges.csv (from
    switch_states.py) is in the aligned trace's frame, so its signature index
    is mapped to the true index with chain 0's permutation.

Inputs
    --trace              trace.nc (fixed) or trace_aligned.nc (de novo)
    --true-active-sets   true_active_sets.csv (rows = nodes, cols = signatures, 0/1)
    --true-activities    true_activities.csv (rows = nodes, cols = signatures)
    --newick             newick_string.nwk (maps a_prob rows to node labels)
    --tree-edges         tree_edges.csv (optional; restricts edge scoring to it)
    --switch-edges       switch_edges.csv from switch_states.py (optional)
    --true-signatures    fixed_signatures.csv (de novo only)
    --threshold          decision threshold on P(active), default 0.5

Outputs (written to --outdir)
    switch_nodes.csv         node, signature, p_active per chain and mean,
                             true_active, true_level
    switch_summary.csv       one row for "overall" and one per signature:
                             AUROC, AUPRC, precision, recall, F1, accuracy at
                             the threshold, Brier, ECE (10 equal-width bins),
                             each as best / mean / worst across chains; plus
                             accuracy at the threshold stratified by the true
                             level (0, (0, 0.05], > 0.05). "On at a level the
                             counts cannot see" is an identification limit, not
                             a model error; the stratified numbers separate the
                             two. A signature that is on (or off) everywhere in
                             truth has no AUROC / AUPRC and gets NaN.
    switch_calibration.csv   reliability bins per chain and for the chain-mean
                             probability (chain = "mean")
    switch_edges_scored.csv  when --switch-edges is given: each edge and
                             signature with p_gain / p_loss / p_switch and the
                             true gain / loss / switch event
    switch_edge_summary.csv  when --switch-edges is given: AUROC, AUPRC,
                             Brier, ECE and accuracy for gain, loss and switch,
                             overall and per signature (the edge probabilities
                             are pooled over chains, so no chain spread here)

Usage
    python scripts/switch_recovery.py --trace ../experiments/<name>/results/trace.nc \\
        --true-active-sets ../experiments/<name>/data/true_active_sets.csv \\
        --true-activities ../experiments/<name>/data/true_activities.csv \\
        --newick ../experiments/<name>/data/newick_string.nwk \\
        --switch-edges ../experiments/<name>/results/switch/switch_edges.csv \\
        --outdir ../experiments/<name>/results/switch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.analysis import (
    chain_perms_to_true,
    expected_calibration_error,
    node_variable_rows,
    reliability_bins,
)
from src.analysis.switch_posterior import registry_frame_to_true

N_BINS = 10
LEVEL_BINS = (
    ("level0", 0.0, 0.0),
    ("level_low", 0.0, 0.05),
    ("level_high", 0.05, np.inf),
)
HIGHER_BETTER = {
    "auroc",
    "auprc",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "acc_level0",
    "acc_level_low",
    "acc_level_high",
}
LOWER_BETTER = {"brier", "ece"}


def binary_metrics(p, y, threshold=0.5) -> dict:
    """AUROC, AUPRC, precision / recall / F1 / accuracy at `threshold`,
    Brier and ECE for probabilities `p` against 0/1 truth `y`. AUROC and
    AUPRC are NaN when `y` has one class (no ranking to score); precision,
    recall and F1 are NaN where undefined (sklearn zero_division=nan)."""
    # a_prob_level_* is sum_s q(s) mask(s, k) evaluated in floating point, so
    # it can overshoot 1 by ~1e-13; sklearn rejects probabilities above 1.
    p = np.clip(np.asarray(p, dtype=float).ravel(), 0.0, 1.0)
    y = np.asarray(y).astype(int).ravel()
    out = {k: np.nan for k in ("auroc", "auprc")}
    if p.size and y.min() != y.max():
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    yhat = (p >= threshold).astype(int)
    if p.size:
        prec, rec, f1, _ = precision_recall_fscore_support(
            y, yhat, average="binary", zero_division=np.nan
        )
        out.update(
            precision=float(prec),
            recall=float(rec),
            f1=float(f1),
            accuracy=float((yhat == y).mean()),
            brier=float(brier_score_loss(y, p, pos_label=1)),
            ece=expected_calibration_error(p, y, N_BINS),
        )
    else:
        out.update(
            {
                k: np.nan
                for k in ("precision", "recall", "f1", "accuracy", "brier", "ece")
            }
        )
    return out


def stratified_accuracy(p, y, level, threshold=0.5) -> dict:
    """Accuracy at `threshold` within true-level bins: exactly 0, (0, 0.05],
    and > 0.05. NaN for an empty bin."""
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y).astype(int).ravel()
    level = np.asarray(level, dtype=float).ravel()
    yhat = (p >= threshold).astype(int)
    out = {}
    for name, lo, hi in LEVEL_BINS:
        m = (level == 0.0) if hi == 0.0 else (level > lo) & (level <= hi)
        out[f"acc_{name}"] = float((yhat[m] == y[m]).mean()) if m.any() else np.nan
    return out


def _best_mean_worst(values: np.ndarray, metric: str):
    """Across-chain best / mean / worst of one metric, NaN-skipping."""
    s = pd.Series(values, dtype=float)
    if s.notna().sum() == 0:
        return np.nan, np.nan, np.nan
    hi, lo, mean = s.max(skipna=True), s.min(skipna=True), s.mean(skipna=True)
    return (hi, mean, lo) if metric in HIGHER_BETTER else (lo, mean, hi)


def score_nodes(P, Y, L, sig_names, threshold=0.5):
    """
    P : (n_nodes, n_chains, K) P(active) per chain, true labelling.
    Y : (n_nodes, K) true active set. L : (n_nodes, K) true level.
    Returns the summary DataFrame (rows: overall + one per signature).
    """
    n_chains = P.shape[1]
    groups = [("overall", slice(None))] + [(s, [k]) for k, s in enumerate(sig_names)]
    rows = []
    for name, sel in groups:
        per_chain = []
        for c in range(n_chains):
            p, y, lvl = P[:, c, sel], Y[:, sel], L[:, sel]
            m = binary_metrics(p, y, threshold)
            m.update(stratified_accuracy(p, y, lvl, threshold))
            per_chain.append(m)
        row = {"group": name, "n_pairs": int(np.size(Y[:, sel]))}
        for metric in per_chain[0]:
            b, mn, w = _best_mean_worst(
                np.array([m[metric] for m in per_chain]), metric
            )
            row[f"{metric}_best"] = b
            row[f"{metric}_mean"] = mn
            row[f"{metric}_worst"] = w
        rows.append(row)
    return pd.DataFrame(rows)


def score_edges(edges: pd.DataFrame, sig_names, threshold=0.5):
    """Edge summary: for gain, loss and switch, the binary metrics overall and
    per signature. `edges` needs p_<event>, true_<event> and `signature`."""
    rows = []
    groups = [("overall", None)] + [(s, s) for s in sig_names]
    for event in ("gain", "loss", "switch"):
        for name, sig in groups:
            sub = edges if sig is None else edges[edges["signature"] == sig]
            m = binary_metrics(sub[f"p_{event}"], sub[f"true_{event}"], threshold)
            rows.append({"event": event, "group": name, "n_edges": len(sub), **m})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--true-active-sets", required=True)
    ap.add_argument("--true-activities", required=True)
    ap.add_argument("--newick", required=True)
    ap.add_argument("--tree-edges", default=None)
    ap.add_argument("--switch-edges", default=None)
    ap.add_argument("--true-signatures", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--outdir", default="switch")
    a = ap.parse_args()

    post = az.from_netcdf(a.trace).posterior
    n_chains = post.sizes["chain"]
    true_active = pd.read_csv(a.true_active_sets, index_col=0)
    true_level = pd.read_csv(a.true_activities, index_col=0)
    sig_names = list(true_active.columns)
    K = len(sig_names)

    if a.true_signatures is not None and "signatures" in post.data_vars:
        true_S = pd.read_csv(a.true_signatures, index_col=0).values
        perms = chain_perms_to_true(post["signatures"].values, true_S)
    else:
        perms = np.tile(np.arange(K), (n_chains, 1))

    newick = Path(a.newick).read_text().strip()
    labels, P = node_variable_rows(
        post, "a_prob_level", newick, perms, keep=true_active.index
    )  # (n_nodes, chains, K)
    Y = true_active.loc[labels].values.astype(int)
    L = true_level.loc[labels].values.astype(float)

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    node_rows = []
    for i, label in enumerate(labels):
        for k, s in enumerate(sig_names):
            row = {"node": label, "signature": s}
            for c in range(n_chains):
                row[f"p_active_chain{c}"] = P[i, c, k]
            row["p_active_mean"] = P[i, :, k].mean()
            row["true_active"] = int(Y[i, k])
            row["true_level"] = L[i, k]
            node_rows.append(row)
    pd.DataFrame(node_rows).to_csv(out / "switch_nodes.csv", index=False)

    summary = score_nodes(P, Y, L, sig_names, a.threshold)
    summary.to_csv(out / "switch_summary.csv", index=False)

    cal = []
    for c in range(n_chains):
        tbl = reliability_bins(P[:, c, :], Y, N_BINS)
        tbl.insert(0, "chain", str(c))
        cal.append(tbl)
    tbl = reliability_bins(P.mean(axis=1), Y, N_BINS)
    tbl.insert(0, "chain", "mean")
    cal.append(tbl)
    pd.concat(cal, ignore_index=True).to_csv(
        out / "switch_calibration.csv", index=False
    )

    if a.switch_edges is not None:
        edges = pd.read_csv(a.switch_edges)
        to_true = registry_frame_to_true(np.asarray(perms[0]))
        edges["signature_index"] = edges["signature"].map(to_true)
        edges["signature"] = edges["signature_index"].map(dict(enumerate(sig_names)))
        if a.tree_edges is not None:
            te = pd.read_csv(a.tree_edges)
            allowed = set(zip(te["parent"], te["child"]))
            edges = edges[
                [(p, c) in allowed for p, c in zip(edges["parent"], edges["child"])]
            ]
        known = set(true_active.index)
        edges = edges[edges["parent"].isin(known) & edges["child"].isin(known)].copy()
        ap_ = true_active.loc[edges["parent"], edges["signature"]].to_numpy().diagonal()
        ac_ = true_active.loc[edges["child"], edges["signature"]].to_numpy().diagonal()
        edges["true_gain"] = ((ap_ == 0) & (ac_ == 1)).astype(int)
        edges["true_loss"] = ((ap_ == 1) & (ac_ == 0)).astype(int)
        edges["true_switch"] = (edges["true_gain"] | edges["true_loss"]).astype(int)
        edges.to_csv(out / "switch_edges_scored.csv", index=False)
        score_edges(edges, sig_names, a.threshold).to_csv(
            out / "switch_edge_summary.csv", index=False
        )


if __name__ == "__main__":
    main()
