"""
scaling_metrics.py

One-row convergence and accuracy summary for a single fixed-signature inference
run. Run it once per tree count and append to a shared CSV; the collected rows
feed the trees-vs-fit table in the report.

Convergence
    max_rhat : worst split-Rhat over the activity variables (e_level_*) and sigma.
    min_ess  : smallest bulk ESS over the same variables.
    eta-level and z-level variables are not matched and so are excluded, since
    they are uninformative by construction in the non-centred walk.

Accuracy (when --true-activities and --newick are given)
    Activities are aligned to the truth
    nodes_by_depth(build_forest(newick)), matched to true_activities by label.
    Signatures are fixed, so there is no label switching and the posterior mean
    is taken over chains and draws. Reports per-node cosine (mean, median) and
    per-node L1 (mean, median); simplex L1 runs 0..2.

Inputs
    --trace            inference_data.nc     ArviZ InferenceData (netCDF).
    --true-activities  true_activities.csv   rows = nodes, cols = signatures.
    --newick           newick_string.nwk     maps e_level rows to node labels.
    --activity-var     activity variable prefix in the posterior (default e_level).
    --n-trees          integer label for this run.
    --out              results CSV to create or append to.

Usage
    python scripts/scaling_metrics.py \\
        --trace ../results/<run>/inference_data.nc \\
        --true-activities ../data/<run>/true_activities.csv \\
        --newick ../data/<run>/newick_string.nwk \\
        --n-trees 5 --out ../results/scaling_results.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.analysis import build_forest, cosine, nodes_by_depth


def _level_vars(post, prefix):
    """Per-level variables <prefix>_<d> in numeric (depth) order."""
    vs = [
        v
        for v in map(str, post.data_vars)
        if re.fullmatch(re.escape(prefix) + r"_?\d+", v)
    ]
    vs.sort(key=lambda v: int(re.findall(r"\d+", v)[-1]))
    return vs


def _convergence_vars(post, activity_var, conv_vars):
    """Variables to judge convergence on: the activity variables and any
    'sigma'. eta-level and z-level are not matched and so are excluded."""
    if conv_vars:
        return list(conv_vars)
    keep = _level_vars(post, activity_var)
    keep += [str(v) for v in post.data_vars if "sigma" in str(v)]
    if not keep:
        raise SystemExit(
            f"no convergence variables matched '{activity_var}_<d>' or 'sigma'; "
            f"available: {list(map(str, post.data_vars))}"
        )
    return keep


def _aligned_activities(post, prefix, newick, truth):
    """Posterior-mean per-node activities aligned to truth. e_level_<depth> row r is the node at
    (depth, r) from nodes_by_depth(forest), matched to truth by label."""
    dr = nodes_by_depth(build_forest(newick))  # (depth, pos) -> label
    labels, A = [], []
    for ev in _level_vars(post, prefix):
        depth = int(re.findall(r"\d+", ev)[-1])
        cmean = np.atleast_2d(
            post[ev].mean(dim=("chain", "draw")).values
        )  # (n_rows, K)
        for r in range(cmean.shape[0]):
            label = dr.get((depth, r))
            if label is None or label not in truth.index:
                continue
            labels.append(label)
            A.append(cmean[r])
    if not labels:
        raise SystemExit(
            "no activity rows mapped to a truth node; check "
            "--newick and the e_level node ordering"
        )
    return labels, np.asarray(A)  # (n_nodes, K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--true-activities", default=None)
    ap.add_argument("--newick", default=None)
    ap.add_argument("--activity-var", default="e_level")
    ap.add_argument(
        "--conv-vars",
        nargs="*",
        default=None,
        help="override the convergence variables; default is the "
        "activity variable(s) and any 'sigma'",
    )
    ap.add_argument("--n-trees", type=int, required=True)
    ap.add_argument("--out", default="scaling_results.csv")
    a = ap.parse_args()

    idata = az.from_netcdf(a.trace)
    post = idata.posterior

    diag = az.summary(
        idata,
        var_names=_convergence_vars(post, a.activity_var, a.conv_vars),
        kind="diagnostics",
    )
    row = {
        "n_trees": a.n_trees,
        "max_rhat": float(diag["r_hat"].max()),
        "min_ess": float(diag["ess_bulk"].min()),
    }

    if a.true_activities and a.newick:
        truth = pd.read_csv(a.true_activities, index_col=0)
        newick = Path(a.newick).read_text().strip()
        labels, A = _aligned_activities(post, a.activity_var, newick, truth)
        true_acts = truth.loc[labels].values.astype(float)
        l1 = np.abs(A - true_acts).sum(axis=1)
        cos = np.array([cosine(A[n], true_acts[n]) for n in range(len(labels))])
        row.update(
            {
                "n_nodes": len(labels),
                "mean_cos": float(cos.mean()),
                "median_cos": float(np.median(cos)),
                "mean_L1": float(l1.mean()),
                "median_L1": float(np.median(l1)),
            }
        )
    elif a.true_activities or a.newick:
        raise SystemExit("accuracy needs both --true-activities and --newick")

    out = Path(a.out)
    df = pd.DataFrame([row])
    if out.exists():
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df = df.drop_duplicates("n_trees", keep="last").sort_values("n_trees")
    df.to_csv(out, index=False)
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
