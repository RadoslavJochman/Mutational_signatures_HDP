"""
Summarise the learned random walk scale sigma across a set of runs and
translate it into interpretable units.

Purpose
    In a non centered logistic normal walk eta_j = eta_parent + sigma z_j with
    e_j = softmax([eta_j, 0]), the single scale sigma is the per branch step
    standard deviation in logit space, shared across branches, depths, and the
    K minus 1 free components. This script collects sigma posterior summaries
    across several traces (for example one per level of a swept parameter) and
    reports two transforms that put it in interpretable terms.

Transforms
    odds_per_branch = exp(sigma)
        the multiplicative change in a component activity to reference odds
        across one branch. A value near 1 means children resemble parents
        (tight); a large value means activities move a lot per branch (loose).
    deep_std = sqrt(sigma_0^2 + max_depth * sigma^2)
        the marginal prior standard deviation of a logit at the deepest node,
        since random walk variance grows linearly with depth. A large value
        means deep nodes are diffuse a priori.

Inputs
    One or more traces given as label=path tokens, each containing the sigma
    variable, plus the fixed root spread sigma_0.

Outputs
    sigma_summary.csv  one row per trace with the sigma posterior mean, sd, and
                       3 and 97 percentiles, ess and r_hat, the maximum depth,
                       deep_std, and odds_per_branch

Interpretation hint
    Compare sigma and its transforms across the runs to see whether the walk
    tightens or loosens as the swept parameter changes. A tight, well pinned
    sigma means the coupling scale is already small; a loose sigma means there
    is room to tighten or structure the coupling.

Usage:
    python inspect_sigma.py \
        --traces 0.1=PATH 0.3=PATH 0.5=PATH 0.7=PATH 0.9=PATH \
        [--sigma-0 1.0] [--var-name sigma] [--out sigma_summary.csv]
"""

import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--traces",
        nargs="+",
        required=True,
        help="CORR=PATH tokens, e.g. 0.3=results/run/trace_aligned.nc",
    )
    ap.add_argument("--sigma-0", type=float, default=1.0)
    ap.add_argument("--var-name", default="sigma")
    ap.add_argument("--out", default="sigma_summary.csv")
    args = ap.parse_args()

    import arviz as az

    rows = []
    for tok in args.traces:
        if "=" not in tok:
            raise SystemExit(f"bad --traces token '{tok}', expected CORR=PATH")
        corr, path = tok.split("=", 1)
        try:
            idata = az.from_netcdf(path)
        except Exception as exc:
            print(f"[{corr}] could not load: {exc}")
            continue
        post = idata.posterior
        if args.var_name not in post.data_vars:
            print(
                f"[{corr}] no '{args.var_name}' in posterior "
                f"(have: {list(post.data_vars)[:8]} ...)"
            )
            continue
        s = np.asarray(post[args.var_name].values).ravel()
        mean, sd = float(s.mean()), float(s.std())
        lo, hi = (float(v) for v in np.percentile(s, [3, 97]))
        try:
            ess = az.ess(idata, var_names=[args.var_name])[args.var_name].values.item()
            rhat = az.rhat(idata, var_names=[args.var_name])[
                args.var_name
            ].values.item()
        except Exception:
            ess, rhat = float("nan"), float("nan")
        depths = sorted(
            int(v.split("_")[-1]) for v in post.data_vars if v.startswith("e_level")
        )
        D = max(depths) if depths else 0
        deep_std = float(np.sqrt(args.sigma_0**2 + D * mean**2))
        rows.append(
            (corr, mean, sd, lo, hi, ess, rhat, D, deep_std, float(np.exp(mean)))
        )

    if not rows:
        print("No sigma summaries produced.")
        return

    df = pd.DataFrame(
        rows,
        columns=[
            "label",
            "sigma_mean",
            "sigma_sd",
            "sigma_p3",
            "sigma_p97",
            "ess",
            "rhat",
            "max_depth",
            "deep_std",
            "odds_per_branch",
        ],
    )
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"\nWritten: {args.out}")


if __name__ == "__main__":
    main()
