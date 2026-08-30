"""
camp_direction.py

Decompose the camp-difference direction -- the soft axis of the canyon found by
gradient_barrier.py -- across signature components, in the space that actually
matters for the sampler.

Two fixes over the first cut:
  1. Decomposes the difference in THREE spaces, not just activity:
       activity  Delta_e   = e_B - e_A                  (downstream, Jacobian-distorted)
       logit     Delta_eta = invsm(e_B) - invsm(e_A)    (the sampler's coordinates)
       signature Delta_S   = S_B - S_A                  (signature-shape move)
     The logit decomposition is the relevant one for the canyon: softmax's Jacobian
     de/deta ~= e shrinks low-activity (rare) components in activity space, so raw
     ||Delta_e||^2 is dominated by high-mass well-exposed signatures regardless of
     which direction is soft. ||Delta_eta||^2 is in the space NUTS integrates.
  2. Uses a single chain per camp (mean over its draws) instead of averaging across
     a camp's chains. Raw-trace chains are NOT label-aligned and split_camps can
     group relabelled chains, so cross-chain averaging blends labellings. Within one
     chain the labelling is consistent, so the per-draw mean is clean and denoised.

If --true-signatures is given, components are labelled by their matched true
signature (and exposure), so the rare signature is named explicitly.

Runs in seconds: no compile_logp, no per-draw transform .eval().

Usage
    python scripts/camp_direction.py \
        --trace ../results/<random-start run>/trace_raw.nc \
        --counts ../data/<run>/mutation_count_matrix.csv \
        --newick ../data/<run>/newick_string.nwk --num-signatures 10 \
        --camp-a 0 1 2 5 6 --camp-b 3 4 7 \
        --true-signatures ../data/<run>/fixed_signatures.csv \
        --true-activities ../data/<run>/true_activities.csv \
        --draw-thin 50 \
        --outdir ../results/<run>/camp_direction
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.analysis import (
    activities_draw,
    align,
    build_model,  # noqa: E402
    inv_softmax_last_zero,
    split_camps,
)


def _chain_mean_activities(post, obj, chain, thin):
    """Per-node full-K activities for ONE chain, averaged over its draws (one
    chain => one labelling, so the mean is clean and denoised)."""
    ndraw = post.sizes["draw"]
    order, acc, n = None, None, 0
    for dr in range(0, ndraw, thin):
        o, A = activities_draw(post, obj, chain, dr)
        if order is None:
            order, acc = o, np.zeros_like(A)
        acc += A
        n += 1
    return order, acc / max(n, 1)


def _shares(M):
    """column-wise (per-component) share of squared Frobenius energy of M."""
    e = (np.asarray(M) ** 2).sum(axis=0)
    tot = e.sum()
    return e / max(tot, 1e-300), np.sqrt(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--counts", required=True)
    ap.add_argument("--newick", required=True)
    ap.add_argument("--num-signatures", type=int, required=True)
    ap.add_argument("--camp-a", type=int, nargs="*", default=None)
    ap.add_argument("--camp-b", type=int, nargs="*", default=None)
    ap.add_argument("--true-signatures", default=None)
    ap.add_argument("--true-activities", default=None)
    ap.add_argument("--draw-thin", type=int, default=50)
    ap.add_argument("--outdir", default="camp_direction")
    a = ap.parse_args()

    obj, counts = build_model(a.newick, a.counts, a.num_signatures)
    idata = az.from_netcdf(a.trace)
    post = idata.posterior
    ca, cb = (a.camp_a, a.camp_b) if (a.camp_a and a.camp_b) else split_camps(post, obj)
    rep_a, rep_b = ca[0], cb[0]
    print(f"camp A {ca} (rep chain {rep_a})\ncamp B {cb} (rep chain {rep_b})")

    _, eA = _chain_mean_activities(post, obj, rep_a, a.draw_thin)
    _, eB = _chain_mean_activities(post, obj, rep_b, a.draw_thin)
    S_A = post["signatures"].sel(chain=rep_a).mean("draw").values
    S_B = post["signatures"].sel(chain=rep_b).mean("draw").values

    P, _ = align(S_B, S_A)  # reindex camp B into camp A's labelling
    eB = eB[:, P]
    S_B = S_B[P, :]

    K = eA.shape[1]
    sh_act, l2_act = _shares(eB - eA)  # K
    etaA = inv_softmax_last_zero(eA)  # (N, K-1)
    etaB = inv_softmax_last_zero(eB)
    sh_eta, l2_eta = _shares(etaB - etaA)  # K-1 (anchor dropped)
    sh_sig, l2_sig = _shares((S_B - S_A).T)  # per component over 96 chan

    true_label = [None] * K
    exposure = [None] * K
    if a.true_signatures:
        trueS = pd.read_csv(a.true_signatures, index_col=0).values
        Pa, _ = align(S_A, trueS)
        for k in range(min(len(Pa), K)):
            true_label[k] = int(Pa[k])
        if a.true_activities:
            exp = pd.read_csv(a.true_activities, index_col=0).values.sum(axis=0)
            for k in range(min(len(Pa), K)):
                exposure[k] = float(exp[Pa[k]])

    rows = []
    for k in range(K):
        rows.append(
            {
                "component_A_frame": k,
                "true_signature": true_label[k],
                "true_exposure": exposure[k],
                "share_logit": float(sh_eta[k])
                if k < len(sh_eta)
                else np.nan,  # anchor -> NaN
                "share_activity": float(sh_act[k]),
                "share_signature": float(sh_sig[k]),
            }
        )
    df = (
        pd.DataFrame(rows)
        .sort_values("share_logit", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "camp_direction.csv", index=False)

    print(
        "\nsorted by LOGIT share (the sampler's space; anchor component has no logit):"
    )
    print(df.to_string(index=False))
    top = df.iloc[0]
    print(
        f"\ntop logit-space component: A-frame {int(top.component_A_frame)}"
        + (
            f", true signature {int(top.true_signature)} (exposure {top.true_exposure:.2f})"
            if top.true_signature is not None
            else ""
        )
        + f"  -> {top.share_logit * 100:.1f}% of the camp-difference logit move"
    )


if __name__ == "__main__":
    main()
