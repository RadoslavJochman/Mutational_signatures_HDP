"""
Test whether a between chain split in the activity posterior comes from the
chains inferring different signatures.

Purpose
    When chains mode lock into groups (camps) that disagree about the
    activities, the disagreement can have two sources: the chains may have
    inferred genuinely different signature matrices, or they may share the
    same signatures and differ only in the activity posterior. This script
    separates the two by defining the camps from the activities and then
    testing, as an independent question, whether those camps inferred
    different signatures. The test is non circular because the camps are
    defined by one quantity (activities) and tested on another (signatures).

Method
    Chains are partitioned into two camps from their activity fingerprints
    (or a user supplied split). For each signature index k the cosine between
    the two camps mean signature is compared against the within camp cosine;
    a between camp cosine well below the within camp value means the camps
    disagree on that signature. If the true signatures are supplied, each
    camp mean signature is also scored against truth, and a per camp best
    match is computed to separate a mislabelled signature (best match high but
    aligned slot low) from one a camp genuinely failed to recover (best match
    also low).

Alignment assumption
    Expects an aligned trace, in which each chain signatures and e_level_*
    have been permuted to a common labelling, so component index k refers to
    the same signature in every chain. Scoring against truth uses an inferred
    to true alignment so that index k is comparable to true signature k.

Inputs
    An aligned trace, optionally the true signature matrix, and optionally an
    explicit camp membership override.

Outputs (written to the output directory)
    camp_signatures.csv  per signature between camp cosine, within camp
                         cosine, activity gap, and (if truth is given) each
                         camp cosine to truth and best match
    camp_summary.csv     one row of scalars: camp membership, the culprit
                         activity component, camp separation, the number of
                         divergent signatures, whether the camps agree, and
                         (if truth is given) each camp mean cosine to truth
                         and counts of mislabelled and unrecovered signatures

Interpretation hint
    A between camp cosine much smaller than the within camp cosine on a given
    signature means the camps reconstruct that signature differently. If the
    camps agree on the signatures, the split is in the activity posterior. If
    they disagree but one camp is near truth, a correct mode exists and the
    failure is mixing or initialisation rather than non-identifiability. If
    they disagree and neither camp is near truth, the signatures themselves
    are non-identifiable.

Usage:
    python diagnose_camp_sig.py \
        --trace path/to/trace_aligned.nc \
        [--true-sigs path/to/fixed_signatures.csv] \
        [--camps 2,3,6] [--outdir camp_signatures]
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.analysis import (
    align,
    chain_perms_to_true,
    cosine,
    detect_camps,
    per_chain_activity,
)

def camp_mean_signatures(S_by_chain, camp):
    """Mean signature matrix over the chains in a camp.

    S_by_chain : (n_chains, K, C) per-chain posterior-mean signatures.
    Returns    : (K, C).
    """
    return S_by_chain[camp].mean(axis=0)


def per_signature_cosine(SA, SB):
    """Per-signature cosine between two (K, C) signature matrices."""
    K = SA.shape[0]
    return np.array([cosine(SA[k], SB[k]) for k in range(K)])


def within_camp_cosine(S_by_chain, camp):
    """Per-signature mean pairwise cosine WITHIN a camp (the noise floor).

    Returns (K,) of NaN if the camp has fewer than two chains.
    """
    K = S_by_chain.shape[1]
    if len(camp) < 2:
        return np.full(K, np.nan)
    out = np.zeros(K)
    for k in range(K):
        cs = [cosine(S_by_chain[i, k], S_by_chain[j, k])
              for i, j in combinations(camp, 2)]
        out[k] = float(np.mean(cs))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--true-sigs", dest="true_sigs", default=None,
                    help="fixed_signatures.csv (rows=k). If given, reports "
                         "each camp's cosine-to-truth for divergent sigs.")
    ap.add_argument("--camps", default=None,
                    help="Override auto-detection, e.g. '2,3,6'. The "
                         "remaining chains form the other camp.")
    ap.add_argument("--divergence-threshold", type=float, default=0.99,
                    help="Per-sig between-camp cosine below this counts as "
                         "the camps having a DIFFERENT signature.")
    ap.add_argument("--outdir", default="camp_signatures")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    idata = az.from_netcdf(args.trace)
    post = idata.posterior
    n_chains = post.sizes["chain"]
    if "signatures" not in post.data_vars:
        raise SystemExit("No 'signatures' in posterior.")
    S = post["signatures"].values
    K = S.shape[2]
    e_vars = sorted([v for v in post.data_vars if v.startswith("e_level")])
    e_arrays = [post[ev].values for ev in e_vars]


    true_S = None
    if args.true_sigs:
        true_S = pd.read_csv(args.true_sigs, index_col=0).values
        perms = chain_perms_to_true(S, true_S)
        S = np.stack([S[c][:, perms[c], :] for c in range(n_chains)])
        e_arrays = [np.stack([a[c][:, :, perms[c]] for c in range(n_chains)])
                    for a in e_arrays]
    S_by_chain = S.mean(axis=1)
    chain_act = per_chain_activity(e_arrays)

    if args.camps is not None:
        campA = sorted(int(x) for x in args.camps.split(","))
        campB = sorted(set(range(n_chains)) - set(campA))
        diff = np.abs(chain_act[campA].mean(0) - chain_act[campB].mean(0))
        kstar = int(np.argmax(diff))
        sep = np.nan
        src = "user-specified"
    else:
        d = detect_camps(chain_act)
        campA, campB, kstar, sep = (d["campA"], d["campB"],
                                    d["kstar"], d["separation"])
        src = "auto-detected"

    SA = camp_mean_signatures(S_by_chain, campA)
    SB = camp_mean_signatures(S_by_chain, campB)
    between = per_signature_cosine(SA, SB)
    withinA = within_camp_cosine(S_by_chain, campA)
    withinB = within_camp_cosine(S_by_chain, campB)
    within = np.nanmean(np.vstack([withinA, withinB]), axis=0)

    act_gap = np.abs(chain_act[campA].mean(0) - chain_act[campB].mean(0))

    truth_cos_A = truth_cos_B = None
    bestmatch_A = bestmatch_B = None
    if true_S is not None:
        truth_cos_A = np.array([cosine(SA[k], true_S[k]) for k in range(K)])
        truth_cos_B = np.array([cosine(SB[k], true_S[k]) for k in range(K)])

        bestmatch_A = align(SA, true_S)[1].max(axis=1)
        bestmatch_B = align(SB, true_S)[1].max(axis=1)

    pd.DataFrame({
        "signature": np.arange(K),
        "between_camp_cos": between,
        "within_camp_cos": within,
        "activity_gap": act_gap,
        **({"campA_cos_truth": truth_cos_A, "campB_cos_truth": truth_cos_B,
            "campA_best_match": bestmatch_A, "campB_best_match": bestmatch_B}
           if truth_cos_A is not None else {}),
    }).to_csv(outdir / "camp_signatures.csv", index=False)

    n_diff = int(np.sum(between < args.divergence_threshold))
    summary = {
        "camp_source": src,
        "campA": "|".join(map(str, campA)),
        "campB": "|".join(map(str, campB)),
        "n_chains": n_chains,
        "K": K,
        "culprit_component": int(kstar),
        "culprit_between_cos": float(between[kstar]),
        "camp_separation": float(sep),
        "n_signatures_divergent": n_diff,
        "camps_agree": bool(n_diff == 0),
        "divergence_threshold": float(args.divergence_threshold),
    }
    if truth_cos_A is not None:
        recov = np.minimum(truth_cos_A, truth_cos_B)
        worst_is_B = truth_cos_A.mean() >= truth_cos_B.mean()
        worst_bm = bestmatch_B if worst_is_B else bestmatch_A
        worst_col = truth_cos_B if worst_is_B else truth_cos_A
        summary.update({
            "campA_mean_cos_truth": float(truth_cos_A.mean()),
            "campB_mean_cos_truth": float(truth_cos_B.mean()),
            "best_camp_mean_cos_truth": float(max(truth_cos_A.mean(),
                                                  truth_cos_B.mean())),
            "worst_camp_mean_cos_truth": float(min(truth_cos_A.mean(),
                                                   truth_cos_B.mean())),
            "n_signatures_poor_recovery": int(np.sum(recov < 0.9)),
            "worse_camp_n_mislabelled": int(np.sum((worst_bm > 0.9)
                                                   & (worst_col < 0.9))),
            "worse_camp_n_unrecovered": int(np.sum(worst_bm < 0.9)),
        })
    pd.DataFrame([summary]).to_csv(outdir / "camp_summary.csv", index=False)

    print(f"Written: {outdir/'camp_signatures.csv'}, "
          f"{outdir/'camp_summary.csv'}")


if __name__ == "__main__":
    main()