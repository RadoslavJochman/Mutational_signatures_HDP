"""
diagnose_modes.py

Analysis half of the de novo mode characterisation. Read a de novo trace, align
every chain to the true signature labelling (so a residual between-chain
disagreement is a genuine mode difference, not a relabelling), split the chains
into camps on their activity fingerprint, and write the tables the mode figures
need.

This overlaps diagnose_camp_sig.py (both align, split camps, and compare
signatures), but writes the per-chain matrix and the camp-mean spectra that the
mode figures need and that the camp-level tables do not carry.

Outputs (written to --outdir)
    mode_summary.csv          per signature: true exposure, signature cosine to
                              truth (best/mean/worst over chains), across-chain
                              spread (best minus worst), between-camp cosine, and
                              a divergent flag for the most spectrum-divergent one.
    mode_camps.csv            chain -> camp membership (A or B).
    mode_recovery_matrix.csv  one row per chain: cosine to truth for each
                              signature (columns sig_0 .. sig_{K-1}) and the camp.
    mode_spectra.csv          long form (signature, channel, truth, campA, campB)
                              of the camp-mean signature spectra against truth.
    mode_activity.csv         long form (signature, node, campA, campB) of the
                              camp-mean per-node activity for every signature.
    mode_chain_signature.csv  long form (signature, chain_i, chain_j, value) of the
                              between-chain signature distance, ordered by camp.
    mode_chain_activity.csv   long form (signature, chain_i, chain_j, value) of the
                              between-chain activity Bray-Curtis, ordered by camp.

Inputs
    --trace            trace_raw.nc        de novo trace (carries `signatures`).
    --true-signatures  fixed_signatures.csv true S (rows = signatures).
    --true-activities  true_activities.csv  rows = nodes, cols = signatures.
    --outdir           output directory.

Usage
    python scripts/diagnose_modes.py \\
        --trace ../results/<run>/trace_raw.nc \\
        --true-signatures ../data/<run>/fixed_signatures.csv \\
        --true-activities ../data/<run>/true_activities.csv \\
        --outdir ../results/<run>/modes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.analysis import (DISTRIBUTION_METRICS, chain_perms_to_true,  # noqa: E402
                                    cosine, detect_camps, per_chain_activity, bray_curtis)


def _aligned(post, true_S):
    """Per-chain posterior-mean signatures and per-component activity, both
    aligned to the true labelling. Returns (Smean_al, pc_al, perms)."""
    S = post["signatures"].values                       # (chain, draw, K, C)
    n_chains = S.shape[0]
    perms = chain_perms_to_true(S, true_S)              # (n_chains, K)
    Smean = S.mean(axis=1)                              # (chain, K, C)
    Smean_al = np.stack([Smean[c][perms[c]] for c in range(n_chains)])
    e_arrays = [post[v].values for v in sorted(post.data_vars)
                if v.startswith("e_level")]             # each (chain,draw,n_d,K)
    pc = per_chain_activity(e_arrays)                   # (chain, K)
    pc_al = np.stack([pc[c][perms[c]] for c in range(n_chains)])
    return Smean_al, pc_al, perms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--true-signatures", required=True)
    ap.add_argument("--true-activities", required=True)
    ap.add_argument("--outdir", default="modes")
    ap.add_argument("--metric", default="hellinger", choices=sorted(DISTRIBUTION_METRICS),
                    help="distance to truth and between camps; default hellinger")
    a = ap.parse_args()
    metric_fn, higher_is_better = DISTRIBUTION_METRICS[a.metric]

    post = az.from_netcdf(a.trace).posterior
    true_S = pd.read_csv(a.true_signatures, index_col=0).values    # (K, C)
    exposure = pd.read_csv(a.true_activities, index_col=0).values.sum(axis=0)
    K, C = true_S.shape
    n_chains = post.sizes["chain"]

    Smean_al, pc_al, perms = _aligned(post, true_S)

    # per-signature distance to truth, per chain, in the chosen metric
    sig_d = np.stack([[metric_fn(Smean_al[c, k], true_S[k]) for c in range(n_chains)]
                      for k in range(K)])               # (K, chains)
    hi, lo = sig_d.max(1), sig_d.min(1)
    best, worst = (hi, lo) if higher_is_better else (lo, hi)
    mean = sig_d.mean(1)
    spread = hi - lo                                    # across-chain disagreement, >= 0

    # relative across-chain spread of the activity each signature receives. Per
    # chain activity is a fraction, so abundant signatures carry large absolute
    # values; normalising by the mean keeps the comparison across exposures fair
    # and matches the logit-frame caution about raw activity magnitudes.
    act_spread = (pc_al.max(0) - pc_al.min(0)) / (pc_al.mean(0) + 1e-12)

    # camps from the activity fingerprint (already aligned to truth)
    camp = detect_camps(pc_al)
    A, B = camp["campA"], camp["campB"]
    SA = Smean_al[A].mean(0) if A else np.zeros_like(true_S)
    SB = Smean_al[B].mean(0) if B else np.zeros_like(true_S)
    between = np.array([metric_fn(SA[k], SB[k]) for k in range(K)])
    between_cos = np.array([cosine(SA[k], SB[k]) for k in range(K)])   # retained
    # spectra differ most where the camps sit furthest apart in the chosen metric
    kdiv = int(np.argmin(between) if higher_is_better else np.argmax(between))

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"signature": np.arange(K), "exposure": exposure,
                  "d_best": best, "d_mean": mean, "d_worst": worst,
                  "spread": spread, "act_spread": act_spread,
                  "between_camp": between, "between_camp_cos": between_cos,
                  "metric": a.metric, "higher_is_better": higher_is_better,
                  "divergent": np.arange(K) == kdiv}
                 ).sort_values("exposure").to_csv(out / "mode_summary.csv", index=False)

    memb = np.array(["A" if c in A else "B" for c in range(n_chains)])
    pd.DataFrame({"chain": np.arange(n_chains), "camp": memb}
                 ).to_csv(out / "mode_camps.csv", index=False)

    mat = pd.DataFrame(sig_d.T, columns=[f"sig_{k}" for k in range(K)])
    mat.insert(0, "camp", memb)
    mat.insert(0, "chain", np.arange(n_chains))
    mat.to_csv(out / "mode_recovery_matrix.csv", index=False)

    spectra = pd.DataFrame({
        "signature": np.repeat(np.arange(K), C),
        "channel": np.tile(np.arange(C), K),
        "truth": true_S.reshape(-1),
        "campA": SA.reshape(-1),
        "campB": SB.reshape(-1),
    })
    spectra.to_csv(out / "mode_spectra.csv", index=False)

    # per-node camp-mean inferred activity for every signature, aligned to truth
    # so component k means the same in every chain. Long form (signature, node,
    # campA, campB) so the plotting step can contrast the split signature with a
    # well-determined control. Because only the product e@S is observed, a camp
    # that shapes a signature differently must place compensating activity on it.
    e_arrays = [post[v].values for v in sorted(post.data_vars)
                if v.startswith("e_level")]             # each (chain,draw,n_d,K)
    blocks = []
    for arr in e_arrays:
        al = np.stack([arr[c][:, :, perms[c]] for c in range(n_chains)])  # aligned
        blocks.append(al.mean(axis=1))                  # (chain, n_d, K): mean over draws
    act = np.concatenate(blocks, axis=1)                # (chain, n_nodes, K)
    n_nodes = act.shape[1]
    campA_act = act[A].mean(0)                          # (n_nodes, K)
    campB_act = act[B].mean(0)
    pd.DataFrame({
        "signature": np.repeat(np.arange(K), n_nodes),
        "node": np.tile(np.arange(n_nodes), K),
        "campA": campA_act.T.reshape(-1),
        "campB": campB_act.T.reshape(-1),
    }).to_csv(out / "mode_activity.csv", index=False)

    # chain-by-chain distance for every signature. Two matrices in long form
    # (signature, chain_i, chain_j, value), ordered by camp so a split signature
    # gives two diagonal blocks and a well-determined one a single uniform block.
    # The signature matrix is the chosen distribution metric between the chains'
    # spectra; the activity matrix is the Bray-Curtis distance between the chains'
    # per-node activity on that signature, scale-sensitive so it sees the level
    # difference that defines the camps, not only the spatial pattern.
    sig_rows, act_rows = [], []
    for k in range(K):
        for i in range(n_chains):
            for j in range(n_chains):
                sig_rows.append((k, i, j, metric_fn(Smean_al[i, k], Smean_al[j, k])))
                act_rows.append((k, i, j, bray_curtis(act[i, :, k], act[j, :, k])))
    pd.DataFrame(sig_rows, columns=["signature", "chain_i", "chain_j", "value"]
                 ).to_csv(out / "mode_chain_signature.csv", index=False)
    pd.DataFrame(act_rows, columns=["signature", "chain_i", "chain_j", "value"]
                 ).to_csv(out / "mode_chain_activity.csv", index=False)

    print(f"camps: A={A} B={B}; most spectrum-divergent signature {kdiv} "
          f"(exposure {exposure[kdiv]:.2f}, between-camp {a.metric} {between[kdiv]:.3f})")
    print(f"wrote 7 CSVs to {out}")


if __name__ == "__main__":
    main()