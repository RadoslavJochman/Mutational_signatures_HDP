"""
Localise a hierarchical funnel between the walk scale and the increments it
multiplies.

Purpose
    A non centered random walk eta_j = eta_parent + sigma z_j couples a single
    scale parameter sigma to the standardised increments z_j. When sigma is
    small the increments are tightly constrained and when it is large they are
    free, so the joint density of (sigma, z) is funnel shaped and a sampler
    cannot pick one step size that works at both the neck and the mouth. This
    script looks for that signature.

Method
    For each draw sigma is a single number and the z components have a spread.
    A funnel makes that spread scale with sigma, so the script reports the
    correlation between log sigma and the per draw standard deviation of each
    z level. It also compares the r_hat of sigma against the r_hat of the z
    and eta blocks it scales, and writes scatter plots of log sigma against
    individual z components.

Inputs
    A posterior trace containing sigma and z_level_* (and optionally
    eta_level_*). The raw, pre alignment trace is preferred because alignment
    does not touch sigma or z.

Outputs (written to the output directory)
    funnel_metrics.csv        per variable: the correlation between log sigma
                              and z spread (for z levels) and the maximum r_hat
    funnel_pairs_sigma_z.png  scatter plots of log sigma against the z
                              components with the worst r_hat

Interpretation hint
    A positive correlation between log sigma and z spread, especially together
    with a healthy sigma r_hat while the z and eta blocks have high r_hat, is
    the funnel signature. In the scatter plots a funnel appears as a wedge that
    widens in z as log sigma rises.

Usage:
    python diagnose_funnel.py --trace path/to/trace_raw.nc \
        [--outdir funnel_diagnosis]
"""

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd


try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--outdir", default="funnel_diagnosis")
    ap.add_argument("--n-pairs", type=int, default=6,
                    help="How many z components to scatter against sigma.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    idata = az.from_netcdf(args.trace)
    post = idata.posterior

    if "sigma" not in post.data_vars:
        raise SystemExit("No 'sigma' in posterior, not a v2-style walk.")

    sigma = post["sigma"].values.reshape(-1)
    log_sigma = np.log(sigma + 1e-12)

    z_vars = sorted([v for v in post.data_vars if v.startswith("z_level")])
    eta_vars = sorted([v for v in post.data_vars if v.startswith("eta_level")])

    rhat = az.rhat(idata)
    rows = [{
        "variable": "sigma",
        "kind": "sigma",
        "funnel_corr": np.nan,
        "rhat_max": float(rhat["sigma"].values),
    }]
    for zv in z_vars:
        arr = post[zv].values
        n = arr.shape[0] * arr.shape[1]
        flat = arr.reshape(n, -1)
        per_draw_spread = flat.std(axis=1)
        c = float(np.corrcoef(log_sigma, per_draw_spread)[0, 1])
        rows.append({
            "variable": zv,
            "kind": "z",
            "funnel_corr": c,
            "rhat_max": float(rhat[zv].max()),
        })
    for ev in eta_vars:
        rows.append({
            "variable": ev,
            "kind": "eta",
            "funnel_corr": np.nan,
            "rhat_max": float(rhat[ev].max()),
        })
    pd.DataFrame(rows).to_csv(outdir / "funnel_metrics.csv", index=False)

    if _HAVE_MPL and z_vars:
        zv = z_vars[0]
        zarr = post[zv].values
        zrh = rhat[zv].values
        n_nodes, Km1 = zrh.shape
        flat_idx = np.argsort(zrh.reshape(-1))[::-1][:args.n_pairs]
        n = len(flat_idx)
        ncol = 3
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax_i, fi in enumerate(flat_idx):
            node_i, comp_i = np.unravel_index(fi, (n_nodes, Km1))
            zvals = zarr[:, :, node_i, comp_i].reshape(-1)
            ax = axes[ax_i]
            ax.scatter(zvals, log_sigma, s=4, alpha=0.3)
            ax.set_xlabel(f"{zv}[{node_i},{comp_i}]")
            ax.set_ylabel("log(sigma)")
            ax.set_title(f"r_hat={zrh[node_i,comp_i]:.2f}", fontsize=9)
        for ax in axes[n:]:
            ax.set_visible(False)
        fig.suptitle("log(sigma) vs z components", fontsize=10)
        fig.tight_layout()
        fig.savefig(outdir / "funnel_pairs_sigma_z.png", dpi=120)
        plt.close(fig)

    msg = f"Written: {outdir / 'funnel_metrics.csv'}"
    if _HAVE_MPL and z_vars:
        msg += f", {outdir / 'funnel_pairs_sigma_z.png'}"
    print(msg)


if __name__ == "__main__":
    main()