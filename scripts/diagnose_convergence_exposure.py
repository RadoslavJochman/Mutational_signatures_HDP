"""
diagnose_convergence_exposure.py

Per-signature convergence against true exposure. For each signature it reports a
convergence metric for the signature itself and one for its activity, so the
plotting step can show that both degrade as exposure falls.

Convergence is measured on the label-aligned posterior. Each chain's signatures are
permuted to the true labelling by maximum cosine and the same permutation is applied
to the activities, so component k is the same signature in every chain. Disagreement
between camps then appears as inflated r-hat on the components the chains dispute,
and the rare signatures are the ones that fail to mix. Reported per signature are
the worst-case r-hat (over the 96 channels for the signature, over all nodes for the
activity) and the corresponding bulk effective sample size.

The raw walk variables eta and z are not used here. Their last axis has length K-1,
so a K-permutation does not align them and their r-hat is uninformative. Convergence
is judged on the aligned signatures and activities only.

Inputs
    --trace             ArviZ netCDF with posterior signatures and e_level_* groups
    --true-signatures   CSV, K rows by 96 channels
    --true-activities   CSV, nodes by K (column sums give exposure)
    --outdir

Output
    convergence_exposure.csv : signature, exposure, rhat_sig, rhat_act,
                               ess_sig, ess_act
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.analysis import align  # noqa: E402


def _align_per_draw(S, E, true_S):
    """Align every draw independently to the true labelling and apply the same
    permutation to signatures and activities. A single per-chain permutation
    cannot undo within-chain label switching, which inflates r-hat on every
    component; aligning per draw removes it, so the r-hat that remains is genuine
    disagreement between chains."""
    nc, nd = S.shape[:2]
    S_al = np.empty_like(S)
    E_al = np.empty_like(E)
    for c in range(nc):
        for d in range(nd):
            perm, _ = align(S[c, d], true_S)  # inferred rows in truth order
            S_al[c, d] = S[c, d][perm]
            E_al[c, d] = E[c, d][:, perm]
    return S_al, E_al


def _rhat_ess(arr):
    """arr has shape (chain, draw, *dims); return (rhat, ess) over the dims."""
    dims = ["chain", "draw"] + [f"d{i}" for i in range(arr.ndim - 2)]
    ds = xr.Dataset({"x": xr.DataArray(np.asarray(arr), dims=dims)})
    return az.rhat(ds)["x"].values, az.ess(ds)["x"].values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--true-signatures", required=True)
    ap.add_argument("--true-activities", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    post = az.from_netcdf(args.trace).posterior
    true_S = pd.read_csv(args.true_signatures, index_col=0).values  # (K, C)
    true_A = pd.read_csv(args.true_activities, index_col=0).values  # (nodes, K)
    K = true_S.shape[0]
    exposure = true_A.sum(axis=0)  # (K,)

    S = post["signatures"].values  # (chain,draw,K,C)
    e_arrays = [
        post[v].values for v in sorted(post.data_vars) if v.startswith("e_level")
    ]  # (chain,draw,n_d,K)
    E = np.concatenate(e_arrays, axis=2)  # (chain,draw,nodes,K)
    S_al, E_al = _align_per_draw(S, E, true_S)

    rh_s, es_s = _rhat_ess(S_al)  # (K, C)
    rh_a, es_a = _rhat_ess(E_al)  # (nodes, K)

    pd.DataFrame(
        {
            "signature": np.arange(K),
            "exposure": exposure,
            "rhat_sig": np.nanmax(rh_s, axis=1),  # worst channel
            "rhat_act": np.nanmax(rh_a, axis=0),  # worst node
            "ess_sig": np.nanmin(es_s, axis=1),
            "ess_act": np.nanmin(es_a, axis=0),
        }
    ).to_csv(out / "convergence_exposure.csv", index=False)
    print(f"wrote convergence_exposure.csv to {out}")


if __name__ == "__main__":
    main()
