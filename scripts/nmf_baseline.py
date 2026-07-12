"""
nmf_baseline.py

Plain NMF baseline for the de novo comparison. Factorises the observed count
matrix X (N_obs x 96) as W (N_obs x K) times H (K x 96) with scikit-learn NMF,
aligns the recovered signatures to the true labelling by Hungarian matching on
cosine (the same `align` recovery_vs_truth.py uses for the Tree-HDP trace), and
scores recovery with the same cosine metrics so the two methods are directly
comparable on one set of axes.

NMF is a single point estimate, so there is no across-chain spread; each quantity
is one number rather than a best/mean/worst triple. To give NMF a fair chance on a
non-convex objective, several restarts are run and the lowest-reconstruction-error
fit is kept. The default loss is generalised Kullback-Leibler with multiplicative
updates, the Poisson-appropriate choice that classical mutational-signature NMF
uses; pass --frobenius for squared-error instead.

Outputs (to --outdir)
    nmf_recovery_signatures.csv   per signature: true exposure, signature cosine
                                  to truth (sig_cos), usage-profile cosine across
                                  observed nodes (use_cos)
    nmf_recovery_activities.csv   per observed node: activity cosine to truth
    nmf_recovery_summary.csv      n_obs_nodes, K, mean sig_cos, median act_cos,
                                  reconstruction error, restarts, best seed, loss

Usage
    python scripts/nmf_baseline.py \
        --counts ../data/<run>/mutation_count_matrix.csv \
        --true-activities ../data/<run>/true_activities.csv \
        --true-signatures ../data/<run>/fixed_signatures.csv \
        --restarts 10 --outdir ../results/<run>/nmf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.analysis import (align, exposure_errors, node_distances,
                                   signature_distances, usage_distances)


def _orient_counts(counts: pd.DataFrame) -> pd.DataFrame:
    """Return counts as (N_obs x 96) with node labels on the row index."""
    if counts.shape[1] == 96:
        return counts
    if counts.shape[0] == 96:
        return counts.T
    raise ValueError(
        f"count matrix must have a 96-channel axis, got shape {counts.shape}"
    )


def _best_nmf(X: np.ndarray, K: int, restarts: int, loss: str, max_iter: int):
    """Several NMF fits; keep the lowest reconstruction error. The first fit is
    the deterministic nndsvda warm start, the rest are random restarts."""
    best = None
    for r in range(max(1, restarts)):
        if r == 0:
            init = "nndsvda"
            seed = 0
        else:
            init = "random"
            seed = r
        model = NMF(
            n_components=K,
            init=init,
            beta_loss=loss,
            solver="mu" if loss == "kullback-leibler" else "cd",
            max_iter=max_iter,
            random_state=seed,
        )
        W = model.fit_transform(X)
        H = model.components_
        err = float(model.reconstruction_err_)
        if best is None or err < best[0]:
            best = (err, W, H, seed)
    return best  # (err, W, H, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--true-activities", required=True)
    ap.add_argument("--true-signatures", required=True)
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--metrics", nargs="+", default=["tv", "hellinger", "cosine"],
                    help="distribution distances to report (tv, hellinger, js, "
                         "cosine); shared with recovery_vs_truth.py")
    ap.add_argument("--frobenius", action="store_true",
                    help="use squared-error loss instead of the default "
                         "Poisson-appropriate Kullback-Leibler")
    ap.add_argument("--outdir", default="nmf")
    a = ap.parse_args()

    counts = _orient_counts(pd.read_csv(a.counts, index_col=0))
    truth = pd.read_csv(a.true_activities, index_col=0)          # (N_nodes x K)
    true_S = pd.read_csv(a.true_signatures, index_col=0).values  # (K x 96)
    K = true_S.shape[0]
    if truth.shape[1] != K:
        raise ValueError(
            f"true activities have {truth.shape[1]} signatures, "
            f"true signatures have {K}"
        )

    # observed nodes are the count rows that also carry a true activity
    obs = [n for n in counts.index if n in truth.index]
    if not obs:
        raise ValueError(
            "no count-matrix row label matches a true-activity row label; "
            "check that both use the same node labels"
        )
    X = counts.loc[obs].values.astype(float)

    loss = "frobenius" if a.frobenius else "kullback-leibler"
    err, W, H, seed = _best_nmf(X, K, a.restarts, loss, a.max_iter)

    # recovered signatures (row-stochastic) and per-node activity simplex
    Shat = H / (H.sum(axis=1, keepdims=True) + 1e-12)           # (K x 96)
    What = W / (W.sum(axis=1, keepdims=True) + 1e-12)           # (N_obs x K)

    # align recovered signatures to truth; apply the same permutation to usages
    perm, _ = align(Shat, true_S)          # Shat[perm] is in true order
    Shat = Shat[perm]
    What = What[:, perm]

    true_acts = truth.loc[obs].values                          # (N_obs x K)
    exposure = truth.values.sum(axis=0)                        # cohort-wide, all nodes

    metrics = a.metrics
    sig_d = signature_distances(Shat, true_S, metrics)         # {m: (K,)}
    use_d = usage_distances(What, true_acts, metrics)          # {m: (K,)}
    nod_d = node_distances(What, true_acts, metrics)           # {m: (N_obs,)}
    expo_err = exposure_errors(What, true_acts)                # (K,)

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    sig_tbl = {"signature": np.arange(K), "exposure": exposure,
               "rel_exposure_err": expo_err}
    for m in metrics:
        sig_tbl[f"sig_{m}"] = sig_d[m]
        sig_tbl[f"use_{m}"] = use_d[m]
    sig_df = pd.DataFrame(sig_tbl).sort_values("exposure").reset_index(drop=True)
    sig_df.to_csv(out / "nmf_recovery_signatures.csv", index=False)

    node_tbl = {"node": obs, "exposure_node": true_acts.sum(1)}
    for m in metrics:
        node_tbl[f"act_{m}"] = nod_d[m]
    node_df = pd.DataFrame(node_tbl)
    node_df.to_csv(out / "nmf_recovery_activities.csv", index=False)

    summary = {"n_obs_nodes": len(obs), "K": K, "loss": loss,
               "restarts": a.restarts, "best_seed": seed,
               "reconstruction_err": err,
               "rel_exposure_err_max": float(expo_err.max())}
    for m in metrics:
        summary[f"sig_{m}_mean"] = float(np.mean(sig_d[m]))
        summary[f"act_{m}_median"] = float(np.median(nod_d[m]))
    pd.DataFrame([summary]).to_csv(out / "nmf_recovery_summary.csv", index=False)

    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 50)
    print(sig_df.round(3).to_string(index=False))
    print()
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()