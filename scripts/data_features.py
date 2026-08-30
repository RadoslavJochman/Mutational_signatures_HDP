"""
data_features.py

Characterise a generated dataset from its ground truth -- the true signatures
and true activities -- independently of any inference. The metrics are the ones
that govern this model's behaviour: how overlapping the signatures are, how the
overlap conditions the activity-identifiability geometry, how much each signature
is actually used, and how concentrated the usage is.

Inputs
    --signatures   fixed_signatures.csv   (rows = signatures k, 96 channel cols)
    --activities   true_activities.csv     (rows = nodes, cols = signatures k)
    --counts       mutation_count_matrix.csv  (optional; rows = nodes, 96 cols)
    --newick       newick_string.nwk          (optional; for tree depth)

Outputs (written to --outdir)
    data_features_summary.csv      one row of dataset-level scalars
    data_features_signatures.csv   one row per signature
    signature_cosine.csv           K x K cosine similarity of the signature shapes
    activity_correlation.csv       K x K Pearson corr of the per-node activity use

What the columns mean
    Signature overlap (shape space)
        mean/median/max_pair_cos : pairwise cosine of the signature rows. High
                                   overlap is what eventually makes S
                                   non-identifiable.
    Identifiability geometry (the functional consequence of overlap)
        sigma_min_sz / sigma_max_sz / cond_sz : singular values of S restricted
            to the sum-zero (activity-difference) subspace. An activity
            difference de (which is sum-zero) changes the predicted spectrum by
            de @ S; sigma_min_sz is the gain of the *softest* such direction --
            the smaller it is, the more an activity change is invisible to the
            data. This is the quantity that drives the soft activity ridge.
        sigma_*_full / cond_full : the same on the full matrix, for reference.
        eff_rank_sigs : participation ratio of the squared singular values.
    Usage (activity space)
        exposure_k       : total use of signature k = sum over nodes of its
                           activity (in node-equivalents).
        n_active_nodes   : nodes where activity_k exceeds --active-threshold.
        eff_num_sigs     : participation ratio of the exposure vector -- the
                           effective number of signatures the dataset uses
                           (near K if balanced, near 1 if collapsed).
        mean_local_div   : mean over nodes of the per-node activity participation
                           ratio -- how many signatures a typical node mixes.
    Signature shape
        eff_channels     : participation ratio of the signature row -- effective
                           number of channels it spreads over (low = peaked).
        entropy_bits     : Shannon entropy of the signature.
    Co-occurrence
        activity_correlation : Pearson corr of activity columns across nodes --
                           which signatures are used together vs exclusively.

Usage
    python scripts/data_features.py \\
        --signatures ../data/<run>/fixed_signatures.csv \\
        --activities ../data/<run>/true_activities.csv \\
        --counts ../data/<run>/mutation_count_matrix.csv \\
        --newick ../data/<run>/newick_string.nwk \\
        --outdir ../data/<run>/features
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.analysis import build_forest, nodes_by_depth  # noqa: E402


def _participation_ratio(p):
    """Effective number of equally-weighted components of a non-negative vector:
    (sum p)^2 / sum p^2. ~len(p) if uniform, ~1 if one component dominates."""
    p = np.asarray(p, float)
    s = p.sum()
    if s <= 0:
        return np.nan
    p = p / s
    return float(1.0 / (np.sum(p**2) + 1e-12))


def _sumzero_singular_values(S):
    """Singular values of S restricted to the sum-zero subspace of activity
    space. An activity difference de (sum-zero) maps to a spectrum change de @ S;
    these are the gains of that map in orthonormal sum-zero coordinates."""
    K = S.shape[0]
    H = np.eye(K) - np.ones((K, K)) / K  # projector onto sum-zero
    Q = np.linalg.svd(H)[0][:, : K - 1]  # K x (K-1) orthonormal, sum-zero
    return np.linalg.svd(Q.T @ S, compute_uv=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signatures", required=True)
    ap.add_argument("--activities", required=True)
    ap.add_argument("--counts", default=None)
    ap.add_argument("--newick", default=None)
    ap.add_argument(
        "--active-threshold",
        type=float,
        default=0.05,
        help="per-node activity above which a signature counts as active at that node",
    )
    ap.add_argument(
        "--rare-exposure-thr",
        type=float,
        default=6.0,
        help="exposure (node-equivalents) below which a signature is flagged rare",
    )
    ap.add_argument("--outdir", default="features")
    a = ap.parse_args()

    S = pd.read_csv(a.signatures, index_col=0).values.astype(float)  # (K, 96)
    A = pd.read_csv(a.activities, index_col=0)  # (nodes, K)
    acts = A.values.astype(float)
    K, C = S.shape
    n_nodes = acts.shape[0]
    if acts.shape[1] != K:
        raise SystemExit(
            f"activities have {acts.shape[1]} columns but there are {K} signatures"
        )

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # --- signature overlap (shape space) ---
    Sn = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    cosM = Sn @ Sn.T
    pd.DataFrame(cosM, index=range(K), columns=range(K)).to_csv(
        out / "signature_cosine.csv"
    )
    off = cosM[np.triu_indices(K, 1)]
    pair = max(combinations(range(K), 2), key=lambda p: cosM[p])
    cos_no_diag = cosM - np.eye(K)  # for nearest-neighbour lookup
    max_cos_other = cos_no_diag.max(axis=1)
    nearest = cos_no_diag.argmax(axis=1)

    # --- identifiability geometry ---
    sv_full = np.linalg.svd(S, compute_uv=False)
    sv_sz = _sumzero_singular_values(S)
    sv2 = sv_full**2
    eff_rank = float((sv2.sum() ** 2) / ((sv2**2).sum() + 1e-12))

    # --- usage (activity space) ---
    exposure = acts.sum(axis=0)  # (K,) node-equivalents
    mean_act = acts.mean(axis=0)
    max_act = acts.max(axis=0)
    n_active = (acts > a.active_threshold).sum(axis=0)
    eff_num_sigs = _participation_ratio(exposure)
    mean_local_div = float(
        np.mean([_participation_ratio(acts[i]) for i in range(n_nodes)])
    )

    # per-node activity co-occurrence
    corrM = np.corrcoef(acts.T) if n_nodes > 1 else np.full((K, K), np.nan)
    pd.DataFrame(corrM, index=range(K), columns=range(K)).to_csv(
        out / "activity_correlation.csv"
    )

    # --- signature shape ---
    Srow = S / (S.sum(axis=1, keepdims=True) + 1e-12)
    eff_channels = np.array([_participation_ratio(Srow[k]) for k in range(K)])
    entropy = -np.array([(Srow[k] * np.log2(Srow[k] + 1e-12)).sum() for k in range(K)])

    per_sig = (
        pd.DataFrame(
            {
                "signature": np.arange(K),
                "exposure": exposure,
                "mean_activity": mean_act,
                "max_activity": max_act,
                "n_active_nodes": n_active,
                "frac_active_nodes": n_active / n_nodes,
                "eff_channels": eff_channels,
                "entropy_bits": entropy,
                "max_cos_other": max_cos_other,
                "nearest_sig": nearest,
                "rare": exposure < a.rare_exposure_thr,
            }
        )
        .sort_values("exposure")
        .reset_index(drop=True)
    )
    per_sig.to_csv(out / "data_features_signatures.csv", index=False)

    summary = {
        "K": K,
        "n_channels": C,
        "n_nodes": n_nodes,
        "mean_pair_cos": float(off.mean()),
        "median_pair_cos": float(np.median(off)),
        "max_pair_cos": float(off.max()),
        "most_similar_pair": f"{pair[0]}-{pair[1]}",
        "sigma_min_sz": float(sv_sz.min()),
        "sigma_max_sz": float(sv_sz.max()),
        "cond_sz": float(sv_sz.max() / (sv_sz.min() + 1e-12)),
        "sigma_min_full": float(sv_full.min()),
        "sigma_max_full": float(sv_full.max()),
        "cond_full": float(sv_full.max() / (sv_full.min() + 1e-12)),
        "eff_rank_sigs": eff_rank,
        "eff_num_sigs": eff_num_sigs,
        "mean_local_div": mean_local_div,
        "exposure_min": float(exposure.min()),
        "exposure_median": float(np.median(exposure)),
        "exposure_max": float(exposure.max()),
        "n_rare_signatures": int((exposure < a.rare_exposure_thr).sum()),
    }

    if a.counts:
        X = pd.read_csv(a.counts, index_col=0).values.astype(float)
        per_node = X.sum(axis=1)
        summary.update(
            {
                "total_mutations": float(X.sum()),
                "mutations_per_node_median": float(np.median(per_node)),
                "mutations_per_node_min": float(per_node.min()),
                "mutations_per_node_max": float(per_node.max()),
            }
        )

    if a.newick:
        G = build_forest(Path(a.newick).read_text().strip())
        dr = nodes_by_depth(G)  # (depth, pos) -> label
        summary.update(
            {
                "n_tree_nodes": int(G.number_of_nodes()),
                "max_depth": int(max(d for d, _ in dr)) if dr else 0,
            }
        )

    pd.DataFrame([summary]).to_csv(out / "data_features_summary.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print(per_sig.round(3).to_string(index=False))
    print()
    print(pd.Series(summary).to_string())
    print(
        f"\nWritten: {out / 'data_features_summary.csv'}, "
        f"{out / 'data_features_signatures.csv'}, {out / 'signature_cosine.csv'}, "
        f"{out / 'activity_correlation.csv'}"
    )


if __name__ == "__main__":
    main()
