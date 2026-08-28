"""
Characterise the activity posterior when chains split into modes (camps).

Purpose
    When the chains agree on the signatures but disagree on the activities,
    this script asks whether the disagreement reflects a genuine information
    limit (the data cannot tell the modes apart) or a sampling failure (the
    data could tell them apart but the sampler is stuck). It also decomposes
    the remaining error of the best chain into the parts the data can and
    cannot constrain.

Method
    All geometry is done on the sum zero subspace of R^K, because activities
    lie on the simplex and any activity difference sums to zero. The softness
    of a direction d is the gain with which it appears in the predicted
    spectrum, the norm of d times S divided by the norm of d, whose extremes
    are the singular values of S on that subspace (the stiffest direction has
    the largest singular value, the softest the smallest). The script reports,
    per node and in aggregate:
      1. Observational equivalence of the modes. The activity difference
         between camps is pushed through the signatures to the difference in
         predicted spectra and measured three ways: the total variation and
         Jensen Shannon distance between the predicted spectra, the spectral
         amplification (norm of the spectrum difference over the norm of the
         activity difference) relative to the singular values, and, when
         observed counts are supplied, the multinomial log likelihood gap the
         data assigns to one mode over the other.
      2. Stiff versus soft error decomposition. The best chain activity error
         (inferred minus truth) is split into the well constrained (stiff) and
         poorly constrained (soft) directions, reported as the fraction of
         error energy in the soft subspace.
      3. Directionality of the camp split. Using the signed log likelihood gap
         and each camp distance to truth, whether one camp wins systematically
         or the camps trade wins node by node.
      4. Whether the truth region is the best mode, via a noise free KL of each
         camp predicted spectrum to the true spectrum and the best single
         chain KL.

Spectrum basis
    By default the predicted spectra use the shared true signatures, which
    isolates the activity question while the camps agree on S. With
    --per-camp-sigs the spectra use each camp own posterior mean signatures,
    the honest per mode comparison once the camps disagree on S. The soft and
    stiff geometry is always referenced to the true signatures.

Inputs
    An aligned trace and the true signatures are required. The true activities
    enable the error decomposition and the truth comparisons; observed counts
    (with the newick tree) enable the data level log likelihood gap.

Outputs (written to the output directory)
    activity_ident_by_node.csv  per node geometry, spectrum distances, log
                                likelihood gaps, error decomposition, and per
                                camp distances and KL to truth
    activity_ident_summary.csv  one row of the aggregate medians and counts
                                across split nodes for each of the four parts

Interpretation hint
    A small predicted spectrum distance with the mode difference concentrated
    in the soft subspace and little log likelihood gap means the modes are
    close to observationally equivalent, an information limit set by signature
    overlap. A non trivial log likelihood gap, or directionality that
    systematically favours one camp, means the data can resolve the split and
    the multimodality is a sampling failure. A best chain error concentrated
    in the soft subspace means the model recovers the part the data
    constrains and errs only where it cannot. A noise free KL favouring a camp
    whose signatures are far from truth points to signature non-identifiability
    rather than an activity side problem.

Usage:
    python diagnose_activity_identifiability.py \
        --trace results/<run>/trace_aligned.nc \
        --true-sigs data/<run>/fixed_signatures.csv \
        [--true-acts data/<run>/true_activities.csv] \
        [--counts data/<run>/node_counts.csv] \
        [--newick data/<run>/newick_string.nwk] \
        [--split-threshold 0.05] [--tv-equiv 0.02] [--per-camp-sigs] \
        [--outdir activity_ident]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.analysis import (
    build_forest,
    chain_perms_to_true,
    cosine,
    detect_camps,
    nodes_by_depth,
)


def sumzero_singular(S):
    """Singular structure of S restricted to the sum-zero (simplex-tangent)
    subspace of activity space.

    S : (K, C).  Returns
        sigmas : (K-1,) descending singular values of S on the sum-zero
                 subspace -- sigmas[-1] is the SOFTEST activity direction.
        U      : (K, K-1) columns = the corresponding directions in R^K
                 (each sum-zero, unit norm), ordered stiff -> soft.
    """
    K = S.shape[0]
    P = np.eye(K) - np.ones((K, K)) / K
    UB, sB, _ = np.linalg.svd(P)
    B = UB[
        :, : K - 1
    ]  # orthonormal basis of the sum-zero subspace (P has K-1 unit eigenvalues)
    M = (
        B.T @ S
    )  # S in that basis; singular values of M are the activity-direction gains
    W, sig, _ = np.linalg.svd(M, full_matrices=False)
    U = B @ W
    return sig, U


def spectral_diff(e_A, e_B, S):
    """Difference in predicted spectra: (e_A - e_B) . S  (length C)."""
    return (e_A - e_B) @ S


def tv_distance(p, q):
    """Total-variation distance between two distributions (0..1)."""
    return 0.5 * float(np.abs(p - q).sum())


def js_divergence(p, q, eps=1e-12):
    """Jensen-Shannon divergence (nats)."""
    p = np.clip(p, eps, None)
    p = p / p.sum()
    q = np.clip(q, eps, None)
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def kl_div(p, q, eps=1e-12):
    """KL(p || q) in nats. With p = true spectrum, KL(p_true || p_camp) is the
    noise-free (overfitting-free) population discrimination: how far a camp's
    predicted spectrum is from the TRUE data distribution. Expected per-node
    log-likelihood gap vs truth is -N * KL."""
    p = np.clip(p, eps, None)
    p = p / p.sum()
    q = np.clip(q, eps, None)
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def decompose_energy(vec, U):
    """Energy of a sum-zero vector along each column of U (stiff->soft).

    Returns coeffs (K-1,) and per-direction energy coeffs**2.
    """
    coeffs = U.T @ vec
    return coeffs, coeffs**2


def soft_energy_fraction(vec, U):
    """Fraction of ||vec||^2 lying in the SOFT (bottom-half) directions of U
    (U ordered stiff->soft).  Isotropic baseline ~0.5."""
    _, energy = decompose_energy(vec, U)
    if energy.sum() <= 0:
        return np.nan
    half = (
        len(energy) // 2
    )  # bottom half = softest dirs; isotropic baseline ~0.5 (exact only for even K-1)
    return float(energy[half:].sum() / energy.sum())


def loglik_gap(counts, p_A, p_B, eps=1e-12):
    """Data log-likelihood gap sum_c counts_c * log(p_A/p_B) (nats).
    Positive favours A; |gap| is the evidence separating the modes."""
    pA = np.clip(p_A, eps, None)
    pB = np.clip(p_B, eps, None)
    return float(np.sum(counts * (np.log(pA) - np.log(pB))))


def directionality_summary(dLL_signed, dL1, eps_d=0.5, eps_t=0.02):
    """Is the camp preference systematic? Inputs are per-(split-)node arrays
    (NaNs allowed): dLL_signed (+ve => data favours camp A) and dL1 = L1_B-L1_A
    (+ve => camp A closer to truth). Returns a dict of counts/consistency/
    agreement; consistency and agreement are NaN when undefined."""
    out = {}
    g = np.asarray(dLL_signed, float)
    g = g[~np.isnan(g)]
    if g.size:
        nA, nB = int((g > eps_d).sum()), int((g < -eps_d).sum())
        dec = nA + nB
        out.update(
            data_favA=nA,
            data_favB=nB,
            data_tied=int((np.abs(g) <= eps_d).sum()),
            net_dLL=float(g.sum()),
            consistency=(max(nA, nB) / dec) if dec else np.nan,
        )
    dl = np.asarray(dL1, float)
    dl = dl[~np.isnan(dl)]
    if dl.size:
        out.update(
            truth_favA=int((dl > eps_t).sum()),
            truth_favB=int((dl < -eps_t).sum()),
            truth_tied=int((np.abs(dl) <= eps_t).sum()),
        )
    gg, dd = np.asarray(dLL_signed, float), np.asarray(dL1, float)
    m = (~np.isnan(gg)) & (~np.isnan(dd)) & (np.abs(gg) > eps_d) & (np.abs(dd) > eps_t)
    out["n_both"] = int(m.sum())
    out["agreement"] = (
        float((np.sign(gg[m]) == np.sign(dd[m])).mean()) if m.any() else np.nan
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--true-sigs", required=True, dest="true_sigs")
    ap.add_argument("--true-acts", default=None, dest="true_acts")
    ap.add_argument("--counts", default=None)
    ap.add_argument("--newick", default=None)
    ap.add_argument("--split-threshold", type=float, default=0.05)
    ap.add_argument("--tv-equiv", type=float, default=0.02)
    ap.add_argument(
        "--per-camp-sigs",
        action="store_true",
        help="Use each camp's own posterior-mean signatures for "
        "the predicted spectra (honest per-mode comparison "
        "when camps disagree on S; use at high overlap). "
        "Default uses the shared true signatures "
        "(activity-isolated comparison).",
    )
    ap.add_argument("--outdir", default="activity_ident")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    import arviz as az

    idata = az.from_netcdf(args.trace)
    post = idata.posterior
    true_S = pd.read_csv(args.true_sigs, index_col=0).values
    K = true_S.shape[0]
    S = post["signatures"].values
    n_chains = S.shape[0]

    # align to truth; camps from per-chain activity
    perms = chain_perms_to_true(S, true_S)
    e_vars = sorted([v for v in post.data_vars if v.startswith("e_level")])
    e_al = [
        np.stack([post[ev].values[c][:, :, perms[c]] for c in range(n_chains)])
        for ev in e_vars
    ]
    chain_act = np.zeros((n_chains, K))
    tot = 0
    for a in e_al:
        chain_act += a.mean(axis=1).sum(axis=1)
        tot += a.shape[2]
    chain_act /= max(tot, 1)
    _camps = detect_camps(chain_act)
    campA, campB, kstar = _camps["campA"], _camps["campB"], _camps["kstar"]

    S_by_chain = np.stack(
        [S[c][:, perms[c], :].mean(axis=0) for c in range(n_chains)]
    )  # (n_chains, K, C)
    S_A = S_by_chain[campA].mean(axis=0)
    S_B = S_by_chain[campB].mean(axis=0)
    btw_cos = np.array([cosine(S_A[k], S_B[k]) for k in range(K)])

    cos_A_true = float(np.mean([cosine(S_A[k], true_S[k]) for k in range(K)]))
    cos_B_true = float(np.mean([cosine(S_B[k], true_S[k]) for k in range(K)]))

    sigmas, U = sumzero_singular(true_S)
    sig_min, sig_max = float(sigmas[-1]), float(sigmas[0])

    dr = None
    if args.newick and (args.true_acts or args.counts):
        try:
            dr = nodes_by_depth(build_forest(Path(args.newick).read_text().strip()))
        except Exception as exc:
            print(f"(node map failed -- phylox/newick? {exc})")
    truth = pd.read_csv(args.true_acts, index_col=0) if args.true_acts else None
    counts = pd.read_csv(args.counts, index_col=0) if args.counts else None

    rows = []
    for ev, a in zip(e_vars, e_al):
        depth = int(ev.split("_")[-1])
        meanA = a[campA].mean(axis=(0, 1))  # (nodes, K)
        meanB = a[campB].mean(axis=(0, 1))
        best = a.mean(axis=1)  # (chain, nodes, K)
        for node in range(a.shape[2]):
            eA, eB = meanA[node], meanB[node]
            de = eA - eB
            de_l1 = float(np.abs(de).sum())
            label = dr.get((depth, node)) if dr is not None else None

            dp = spectral_diff(eA, eB, true_S)
            if args.per_camp_sigs:
                pA = np.mean([best[c, node] @ S_by_chain[c] for c in campA], axis=0)
                pB = np.mean([best[c, node] @ S_by_chain[c] for c in campB], axis=0)
            else:
                pA = eA @ true_S
                pB = eB @ true_S
            pA = pA / pA.sum()
            pB = pB / pB.sum()
            rec = {
                "node": label if label is not None else f"{depth}:{node}",
                "depth": depth,
                "de_l1": de_l1,
                "dp_l2": float(np.linalg.norm(dp)),
                "tv": tv_distance(pA, pB),
                "js": js_divergence(pA, pB),
                "ratio_dp_de": (
                    float(np.linalg.norm(dp) / (np.linalg.norm(de) + 1e-12))
                ),
                "de_soft_frac": soft_energy_fraction(de, U),
            }

            te = (
                truth.loc[label].values
                if (
                    truth is not None
                    and label in (truth.index if truth is not None else [])
                )
                else None
            )
            p_true = None
            if te is not None:
                p_true = te @ true_S
                p_true = p_true / p_true.sum()

            if counts is not None and label in (
                counts.index if counts is not None else []
            ):
                x = counts.loc[label].values.astype(float)
                g = loglik_gap(x, pA, pB)
                rec["N_node"] = float(x.sum())
                rec["dLL_signed"] = g
                rec["dLL"] = abs(g)

                if p_true is not None:
                    llp = lambda p: float(np.sum(x * np.log(np.clip(p, 1e-12, None))))
                    lt = llp(p_true)
                    rec["dll_A_vs_true"] = llp(pA) - lt
                    rec["dll_B_vs_true"] = llp(pB) - lt

            if p_true is not None:
                rec["kl_true_A"] = kl_div(p_true, pA)
                rec["kl_true_B"] = kl_div(p_true, pB)
                kls = [
                    kl_div(p_true, best[c, node] @ S_by_chain[c])
                    for c in range(n_chains)
                ]
                rec["kl_true_bestchain"] = float(min(kls))

            if te is not None:
                l1s = [np.abs(best[c, node] - te).sum() for c in range(n_chains)]
                cbest = int(np.argmin(l1s))
                err = best[cbest, node] - te
                rec["err_l1"] = float(np.abs(err).sum())
                rec["err_soft_frac"] = soft_energy_fraction(err, U)
                rec["L1_A"] = float(np.abs(eA - te).sum())
                rec["L1_B"] = float(np.abs(eB - te).sum())
                rec["dL1_AminusB"] = rec["L1_B"] - rec["L1_A"]
            rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "activity_ident_by_node.csv", index=False)

    split = df[df["de_l1"] >= args.split_threshold]

    summary = {
        "n_chains": n_chains,
        "K": K,
        "n_nodes": len(df),
        "n_split_nodes": len(split),
        "split_threshold": float(args.split_threshold),
        "campA": "|".join(map(str, campA)),
        "campB": "|".join(map(str, campB)),
        "culprit_component": int(kstar),
        "between_camp_cos_min": float(btw_cos.min()),
        "between_camp_cos_mean": float(btw_cos.mean()),
        "camps_share_S": bool(btw_cos.min() >= 0.99),
        "campA_cos_truth": cos_A_true,
        "campB_cos_truth": cos_B_true,
        "spectrum_basis": "per_camp" if args.per_camp_sigs else "shared_true",
        "sigma_max": sig_max,
        "sigma_min": sig_min,
    }

    if len(split):
        summary.update(
            {
                "tv_median": float(split["tv"].median()),
                "tv_max": float(split["tv"].max()),
                "js_median": float(split["js"].median()),
                "ratio_dp_de_median": float(split["ratio_dp_de"].median()),
                "de_soft_frac_median": float(split["de_soft_frac"].median()),
            }
        )
        if "dLL" in split.columns:
            summary.update(
                {
                    "dLL_median": float(split["dLL"].median()),
                    "dLL_max": float(split["dLL"].max()),
                    "N_node_median": float(split["N_node"].median()),
                }
            )

    if "err_soft_frac" in df.columns:
        e = df.dropna(subset=["err_soft_frac"])
        summary.update(
            {
                "err_l1_median": float(e["err_l1"].median()),
                "err_soft_frac_median": float(e["err_soft_frac"].median()),
            }
        )

    have_data = "dLL_signed" in split.columns
    have_truth = "dL1_AminusB" in split.columns
    if len(split) and (have_data or have_truth):
        eps_d, eps_t = 0.5, 0.02  # indifference bands (nats / activity L1)
        D = directionality_summary(
            split["dLL_signed"].values if have_data else np.full(len(split), np.nan),
            split["dL1_AminusB"].values if have_truth else np.full(len(split), np.nan),
            eps_d,
            eps_t,
        )
        for key in (
            "data_favA",
            "data_favB",
            "data_tied",
            "net_dLL",
            "consistency",
            "truth_favA",
            "truth_favB",
            "truth_tied",
            "n_both",
            "agreement",
        ):
            if key in D:
                summary[f"dir_{key}"] = D[key]

    if len(split) and {"kl_true_A", "kl_true_B"} <= set(split.columns):
        if {"dll_A_vs_true", "dll_B_vs_true"} <= set(split.columns):
            summary["dll_A_vs_true_median"] = float(
                split["dll_A_vs_true"].dropna().median()
            )
            summary["dll_B_vs_true_median"] = float(
                split["dll_B_vs_true"].dropna().median()
            )
            summary["dll_best_vs_true_median"] = float(
                split[["dll_A_vs_true", "dll_B_vs_true"]].max(axis=1).dropna().median()
            )
        bothkl = split.dropna(subset=["kl_true_A", "kl_true_B"])
        summary["kl_true_A_median"] = float(split["kl_true_A"].dropna().median())
        summary["kl_true_B_median"] = float(split["kl_true_B"].dropna().median())
        summary["frac_nodes_B_closer"] = (
            float((bothkl["kl_true_B"] < bothkl["kl_true_A"]).mean())
            if len(bothkl)
            else np.nan
        )
        if "kl_true_bestchain" in split.columns:
            summary["kl_true_bestchain_median"] = float(
                split["kl_true_bestchain"].dropna().median()
            )

    pd.DataFrame([summary]).to_csv(outdir / "activity_ident_summary.csv", index=False)
    print(
        f"Written: {outdir / 'activity_ident_by_node.csv'}, "
        f"{outdir / 'activity_ident_summary.csv'}"
    )


if __name__ == "__main__":
    main()
