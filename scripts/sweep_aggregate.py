"""
Assemble a signature correlation sweep into one master table.

Purpose
    Collect, for each run in a sweep over a single varied parameter (here the
    pairwise correlation of the true signatures), a fixed set of geometry,
    recovery, and convergence metrics into one table so the runs can be
    compared side by side.

Method
    For each (correlation, run directory) the metrics are recomputed from the
    canonical input files in that directory rather than parsed from loose
    diagnostic text, so the table is reproducible. Any metric whose source
    file is missing degrades to NaN, so a partial sweep still produces a
    table.

Columns
    Signature geometry (from the true signature matrix):
        smallest_sv        smallest singular value of the true signatures on
                           the simplex tangent (sum zero) subspace
        cond_number        largest_sv / smallest_sv
        mean_pair_cos      mean pairwise cosine of the true signatures
        eff_rank           participation ratio of the singular values
    Recovery and camps (from the aligned trace and true signatures):
        sig_recovery_best  best camp mean cosine to truth, after aligning
                           inferred signatures to true order
        sig_recovery_worst worse camp mean cosine to truth
        camps_agree        whether the chains agree on S (between camp cosine
                           at or above the divergence threshold)
        eff_num_sigs       participation ratio of the dataset mean activity
                           across components (near K if all signatures are
                           used, near 1 if activity collapses onto one)
        mean_local_div     mean per node activity participation ratio
        ess_median         median ess_bulk over the e_level parameters
        rhat_max           maximum r_hat over the e_level parameters
    Activity versus truth (optional, needs true activities and the tree):
        L1_best_median     median over nodes of best chain activity L1 to truth
        spread             mean across chain spread of activity means

    The written CSV also carries auxiliary columns not shown in the printed
    table: n_sig_divergent, camp_sep, n_some_close, n_nodes, eff_rank, and
    subdir.

Interpretation hint
    The metrics together separate three regimes. High sig_recovery_best with
    camps_agree true means the signatures are recovered and any failure is
    confined to the activities. High sig_recovery_best with camps_agree false
    means the chains reconstruct different signature matrices. Low
    sig_recovery_best with eff_num_sigs near 1 means the model has collapsed
    onto a single effective signature. Note that ess_median and rhat_max can
    look best in the collapsed case: good convergence does not imply correct
    recovery, so read them alongside sig_recovery_best and eff_num_sigs rather
    than on their own.

Usage:
    python sweep_aggregate.py \
        --results-root results \
        --data-root    data \
        --runs 0.1=<subdir> 0.3=<subdir> 0.5=<subdir> 0.7=<subdir> 0.9=<subdir> \
        [--true-sigs-name fixed_signatures.csv] \
        [--trace-name trace_aligned.nc] \
        [--true-acts-name true_activities.csv] \
        [--newick-name newick_string.nwk] \
        [--divergence-threshold 0.99] \
        [--out sweep_master.csv]

The subdir is the per run subdirectory name shared under both roots: the
trace is read from results/<subdir>/ and the signatures, tree, and
activities from data/<subdir>/, falling back to data/ for inputs shared
across all runs.
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.analysis import (
    build_forest,
    chain_perms_to_true,
    cosine as _cos,
    detect_camps,
    nodes_by_depth,
    per_chain_activity,
)

def signature_geometry(true_S):
    """Conditioning metrics of the (K, C) true signature matrix."""
    sv = np.linalg.svd(true_S, compute_uv=False)
    sv2 = sv ** 2
    eff_rank = float((sv2.sum() ** 2) / ((sv2 ** 2).sum() + 1e-12))
    Mn = true_S / (np.linalg.norm(true_S, axis=1, keepdims=True) + 1e-12)
    cosM = Mn @ Mn.T
    K = true_S.shape[0]
    mean_pair = float(cosM[np.triu_indices(K, 1)].mean())
    return {
        "smallest_sv": float(sv.min()),
        "cond_number": float(sv.max() / (sv.min() + 1e-12)),
        "mean_pair_cos": mean_pair,
        "eff_rank": eff_rank,
    }


def participation_ratio(p):
    """1 / sum(p_i^2) for a non-negative vector p (normalised internally).

    Equals the number of equally-weighted components; collapses to ~1 when
    one component dominates, ~len(p) when uniform.
    """
    p = np.asarray(p, float)
    s = p.sum()
    if s <= 0:
        return np.nan
    p = p / s
    return float(1.0 / (np.sum(p ** 2) + 1e-12))


def effective_num_signatures(mean_act_per_component):
    """Dataset-level: participation ratio of the mean activity vector."""
    return participation_ratio(mean_act_per_component)


def camp_recovery(S_by_chain, true_S, campA, campB, divergence_threshold):
    """Best/worst camp mean cosine-to-truth and whether camps agree on S."""
    K = true_S.shape[0]
    SA = S_by_chain[campA].mean(axis=0)
    SB = S_by_chain[campB].mean(axis=0)
    between = np.array([_cos(SA[k], SB[k]) for k in range(K)])
    tA = np.array([_cos(SA[k], true_S[k]) for k in range(K)]).mean()
    tB = np.array([_cos(SB[k], true_S[k]) for k in range(K)]).mean()
    n_diff = int(np.sum(between < divergence_threshold))
    return {
        "sig_recovery_best": float(max(tA, tB)),
        "sig_recovery_worst": float(min(tA, tB)),
        "camps_agree": bool(n_diff == 0),
        "n_sig_divergent": n_diff,
    }

def _activity_arrays_and_signatures(post):
    e_vars = sorted([v for v in post.data_vars if v.startswith("e_level")])
    e_arrays = [post[ev].values for ev in e_vars]
    S = post["signatures"].values
    return e_vars, e_arrays, S


def _across_chain_spread(e_arrays):
    """Mean over nodes of the across-chain L1 spread of activity means."""
    spreads = []
    for a in e_arrays:
        cm = a.mean(axis=1)
        for node in range(cm.shape[1]):
            m = cm[:, node, :]
            spreads.append(np.abs(m - m.mean(0, keepdims=True)).sum(1).mean())
    return float(np.mean(spreads)) if spreads else np.nan


def _mean_local_diversity(e_arrays):
    """Average per-node participation ratio of the per-node mean activity."""
    prs = []
    for a in e_arrays:
        cm = a.mean(axis=(0, 1))
        for node in range(cm.shape[0]):
            prs.append(participation_ratio(cm[node]))
    return float(np.nanmean(prs)) if prs else np.nan


def trace_metrics(trace_path, true_S, divergence_threshold):
    """Recovery, camps, effective signatures, spread, ess from the trace."""
    import arviz as az
    idata = az.from_netcdf(str(trace_path))
    post = idata.posterior
    e_vars, e_arrays, S = _activity_arrays_and_signatures(post)
    n_chains = S.shape[0]

    out = {}
    out["eff_num_sigs"] = effective_num_signatures(
        np.vstack([a.mean(axis=(0, 1)) for a in e_arrays]).sum(0))
    out["mean_local_div"] = _mean_local_diversity(e_arrays)
    out["spread"] = _across_chain_spread(e_arrays)

    try:
        ess = az.ess(idata, var_names=e_vars, method="bulk")
        out["ess_median"] = float(np.nanmedian(
            np.concatenate([np.ravel(ess[v].values) for v in e_vars])))
        rhat = az.rhat(idata, var_names=e_vars)
        out["rhat_max"] = float(np.nanmax(
            np.concatenate([np.ravel(rhat[v].values) for v in e_vars])))
    except Exception as exc:
        out["ess_median"] = np.nan
        out["rhat_max"] = np.nan
        print(f"    (ess/rhat skipped: {exc})")

    if true_S is not None:
        perms = chain_perms_to_true(S, true_S)
        S_al = np.stack([S[c][:, perms[c], :] for c in range(n_chains)])
        S_by_chain = S_al.mean(axis=1)
        e_al = [np.stack([a[c][:, :, perms[c]] for c in range(n_chains)])
                for a in e_arrays]
        chain_act = per_chain_activity(e_al)
        _camps = detect_camps(chain_act)
        campA, campB, kstar, sep = (_camps["campA"], _camps["campB"],
                                    _camps["kstar"], _camps["separation"])
        out.update(camp_recovery(S_by_chain, true_S, campA, campB,
                                 divergence_threshold))
        out["camp_sep"] = sep
    return out


def activity_L1(trace_path, true_S, true_acts_path, newick_path):
    """Best chain activity L1 to truth."""
    import arviz as az

    idata = az.from_netcdf(str(trace_path))
    post = idata.posterior
    truth = pd.read_csv(true_acts_path, index_col=0)
    newick = Path(newick_path).read_text().strip()
    S = post["signatures"].values
    n_chains, _, K, _ = S.shape
    perms = chain_perms_to_true(S, true_S)
    dr = nodes_by_depth(build_forest(newick))
    e_vars = sorted([v for v in post.data_vars if v.startswith("e_level")])

    best_L1 = []
    n_close = 0
    n_total = 0
    for ev in e_vars:
        depth = int(ev.split("_")[-1])
        arr = post[ev].values
        for node_row in range(arr.shape[2]):
            label = dr.get((depth, node_row))
            if label is None or label not in truth.index:
                continue
            true_e = truth.loc[label].values
            l1s = []
            for c in range(n_chains):
                m = arr[c, :, node_row, :].mean(axis=0)[perms[c]]
                l1s.append(np.abs(m - true_e).sum())
            bl = min(l1s)
            best_L1.append(bl)
            n_total += 1
            if bl < 0.1:
                n_close += 1
    return {
        "L1_best_median": float(np.median(best_L1)) if best_L1 else np.nan,
        "n_some_close": n_close,
        "n_nodes": n_total,
    }

def _resolve(data_dir, data_root, fname):
    """Look for fname in the per-correlation data subdir first, then fall
    back to the data root (for inputs shared across correlations, e.g. a
    common newick or true_activities)."""
    p = data_dir / fname
    if p.exists():
        return p
    p2 = data_root / fname
    if p2.exists():
        return p2
    return p


def process_run(corr, subdir, results_root, data_root, names,
                divergence_threshold):
    results_dir = results_root / subdir
    data_dir = data_root / subdir
    row = {"correlation": corr, "subdir": subdir}

    sig_path = _resolve(data_dir, data_root, names["true_sigs"])
    true_S = None
    if sig_path.exists():
        true_S = pd.read_csv(sig_path, index_col=0).values
        row.update(signature_geometry(true_S))
    else:
        print(f"    (no {names['true_sigs']} under data: geometry skipped)")

    trace_path = results_dir / names["trace"]
    if trace_path.exists():
        try:
            row.update(trace_metrics(trace_path, true_S, divergence_threshold))
        except Exception as exc:
            print(f"    (trace metrics failed: {exc})")
    else:
        print(f"    (no {names['trace']} under results: "
              f"recovery/ess/eff_sigs skipped)")

    acts_path = _resolve(data_dir, data_root, names["true_acts"])
    nwk_path = _resolve(data_dir, data_root, names["newick"])
    if trace_path.exists() and acts_path.exists() and nwk_path.exists() \
            and true_S is not None:
        try:
            row.update(activity_L1(trace_path, true_S, acts_path, nwk_path))
        except Exception as exc:
            print(f"    (activity L1 failed -- phylox/newick? {exc})")
    else:
        print(f"    (true_activities/newick absent under data: "
              f"activity L1 skipped)")

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True,
                    help="Root holding per-correlation result subdirs "
                         "(traces, analysis outputs).")
    ap.add_argument("--data-root", required=True,
                    help="Root holding per-correlation data subdirs "
                         "(signatures, newick, true activities).")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="CORR=SUBDIR tokens, where SUBDIR is the shared "
                         "subdirectory name under BOTH roots, e.g. "
                         "0.3=config_denovo_easy_more_chains_corr_03")
    ap.add_argument("--true-sigs-name", default="fixed_signatures.csv")
    ap.add_argument("--trace-name", default="trace_aligned.nc")
    ap.add_argument("--true-acts-name", default="true_activities.csv")
    ap.add_argument("--newick-name", default="newick_string.nwk")
    ap.add_argument("--divergence-threshold", type=float, default=0.99)
    ap.add_argument("--out", default="sweep_master.csv")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    data_root = Path(args.data_root)
    names = {"true_sigs": args.true_sigs_name, "trace": args.trace_name,
             "true_acts": args.true_acts_name, "newick": args.newick_name}

    rows = []
    for tok in args.runs:
        if "=" not in tok:
            raise SystemExit(f"bad --runs token '{tok}', expected CORR=SUBDIR")
        corr_s, subdir = tok.split("=", 1)
        corr = float(corr_s)
        print(f"[corr={corr}] results/{subdir}  +  data/{subdir}")
        rows.append(process_run(corr, subdir, results_root, data_root,
                                names, args.divergence_threshold))

    df = pd.DataFrame(rows).sort_values("correlation").reset_index(drop=True)

    # ordered, rounded view for the table
    cols = ["correlation", "smallest_sv", "cond_number", "mean_pair_cos",
            "sig_recovery_best", "sig_recovery_worst", "camps_agree",
            "eff_num_sigs", "mean_local_div", "ess_median", "rhat_max",
            "L1_best_median", "spread"]
    cols = [c for c in cols if c in df.columns]
    view = df[cols].copy()
    for c in view.columns:
        if view[c].dtype.kind == "f":
            view[c] = view[c].round(3)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print(view.to_string(index=False))

    df.to_csv(args.out, index=False)
    print(f"\nWritten: {args.out}")


if __name__ == "__main__":
    main()