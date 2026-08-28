"""
Read the sampler diagnostics stored in a trace to characterise the geometry of
a multimodal posterior.

Purpose
    A multimodal sampling block can arise from two different geometries that
    call for different remedies: a pinched or funnel geometry, where the
    sampler must take tiny steps and may diverge, and well separated modes,
    where each mode samples cleanly but the chains cannot cross between them.
    This script summarises the diagnostics that distinguish them.

Method
    From the sample_stats of a fitted trace it reports the number and rate of
    divergences and where in sampling they fall, the energy diagnostic BFMI
    per chain, the settled step size and tree depth per chain, and the mean
    acceptance rate. It also records a few threshold based flags: a very small
    median step size (below 0.02), a tree depth saturating within one of the
    cap, and a minimum BFMI below 0.3.

Inputs
    A trace with a sample_stats group. The raw, pre alignment trace is
    preferred because alignment leaves sample_stats untouched.

Outputs (written to the output directory)
    sampler_geometry_summary.csv    one row of totals, medians, and the
                                    threshold flags
    sampler_geometry_per_chain.csv  per chain divergences, BFMI, and step size

Interpretation hint
    A non trivial divergence rate, a very small step size, or a saturated tree
    depth point to a pinched or funnel geometry. Few or no divergences with a
    low BFMI point instead to well separated modes the sampler cannot cross.
    Neither set of signs, with an acceptable BFMI, suggests slow mixing rather
    than a pathological geometry.

Usage:
    python diagnose_sampler_geometry.py --trace path/to/trace_raw.nc \
        [--outdir sampler_diagnosis]
"""

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd


def _get(ss, *names):
    """Return the first sample_stats variable that exists, else None."""
    for n in names:
        if n in ss:
            return ss[n]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument(
        "--max-treedepth",
        type=int,
        default=None,
        help="Configured NUTS max_treedepth cap. If omitted, the "
        "saturation flag falls back to the observed max "
        "depth, which can misfire when the run never "
        "approached the cap.",
    )
    ap.add_argument("--outdir", default="sampler_diagnosis")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    idata = az.from_netcdf(args.trace)
    if not hasattr(idata, "sample_stats"):
        raise SystemExit(
            "Trace has no sample_stats group, cannot read sampler "
            "diagnostics.  Re-run inference saving the full InferenceData."
        )
    ss = idata.sample_stats
    n_chains = ss.sizes["chain"]
    n_draws = ss.sizes["draw"]

    div = _get(ss, "diverging")
    total_div = None
    div_frac = np.nan
    div_first_half = np.nan
    per_chain_div = None
    if div is not None:
        d = div.values  # (chain, draw) bool
        total_div = int(d.sum())
        per_chain_div = d.sum(axis=1).astype(int)
        div_frac = total_div / (n_chains * n_draws)
        if total_div > 0:
            draw_idx = np.where(d.any(axis=0))[0]
            div_first_half = int((draw_idx < n_draws / 2).sum())

    try:
        bfmi = np.asarray(az.bfmi(idata), dtype=float).ravel()
    except Exception:
        bfmi = None

    step = _get(ss, "step_size")
    per_chain_step = None
    step_median = np.nan
    if step is not None:
        sv = step.values
        per_chain_step = np.atleast_1d(sv.mean(axis=1) if sv.ndim > 1 else sv)
        step_median = float(np.median(per_chain_step))

    tdepth = _get(ss, "tree_depth", "treedepth")
    td_mean = np.nan
    td_max = np.nan
    if tdepth is not None:
        td = tdepth.values
        td_mean = float(td.mean())
        td_max = int(td.max())

    acc = _get(ss, "acceptance_rate", "mean_tree_accept", "accept")
    acc_mean = float(acc.values.mean()) if acc is not None else np.nan

    flag_tiny_step = bool(step_median < 0.02) if not np.isnan(step_median) else False
    # Saturation = mean depth within one of the cap. Prefer the configured
    # cap; fall back to the observed max
    td_cap = args.max_treedepth if args.max_treedepth is not None else td_max
    flag_saturated_depth = (
        bool(td_mean >= td_cap - 1.0) if not np.isnan(td_mean) else False
    )
    flag_low_bfmi = bool(np.min(bfmi) < 0.3) if bfmi is not None else False

    summary = {
        "n_chains": n_chains,
        "n_draws": n_draws,
        "total_divergences": total_div,
        "divergence_fraction": float(div_frac),
        "divergences_first_half": div_first_half,
        "bfmi_min": float(np.min(bfmi)) if bfmi is not None else np.nan,
        "bfmi_mean": float(np.mean(bfmi)) if bfmi is not None else np.nan,
        "step_size_median": step_median,
        "tree_depth_mean": td_mean,
        "tree_depth_max": td_max,
        "acceptance_mean": acc_mean,
        "flag_tiny_step": flag_tiny_step,
        "flag_saturated_depth": flag_saturated_depth,
        "flag_low_bfmi": flag_low_bfmi,
    }
    pd.DataFrame([summary]).to_csv(outdir / "sampler_geometry_summary.csv", index=False)

    per_chain = pd.DataFrame({"chain": list(range(n_chains))})
    if per_chain_div is not None:
        per_chain["divergences"] = per_chain_div
    if bfmi is not None and len(bfmi) == n_chains:
        per_chain["bfmi"] = bfmi
    if per_chain_step is not None and len(per_chain_step) == n_chains:
        per_chain["step_size"] = per_chain_step
    per_chain.to_csv(outdir / "sampler_geometry_per_chain.csv", index=False)

    print(
        f"Written: {outdir / 'sampler_geometry_summary.csv'}, "
        f"{outdir / 'sampler_geometry_per_chain.csv'}"
    )


if __name__ == "__main__":
    main()
