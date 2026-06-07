"""
Test whether non-orthogonality of the signatures drives multimodality in the
activity posterior.

Purpose
    When two signatures are similar, the data pins their combined contribution
    to the predicted spectrum but is relatively indifferent to how that
    contribution is split between them. That soft trade off can appear in the
    activity posterior as anti correlation between the two components and an
    inflated variance of their difference relative to their sum. This script
    measures the effect and asks whether it scales with signature similarity
    and whether it tracks which nodes are hard to sample.

Method
    For each pair of activity components it computes, within each chain and
    then aggregated over chains, the posterior correlation between the two
    components and a trade off ratio Var(e_a minus e_b) / Var(e_a plus e_b).
    Because activities lie on the simplex, any two components are somewhat
    anti correlated by construction, so a pair is judged relative to the
    distribution over all K(K minus 1)/2 pairs rather than against zero.
    Correlations are taken within chain; pooling across chains in different
    modes would manufacture correlation reflecting mode structure rather than
    a local likelihood ridge. The per pair trade off metrics are then
    regressed on the pair signature cosine, and a per node anti correlation
    load is linked to that node across chain spread.

Inputs
    An aligned trace, the true activities, the true signature matrix, and the
    newick tree. The focus pair defaults to the most similar true signature
    pair and can be overridden.

Outputs (written to the output directory)
    pair_tradeoff_by_node.csv  per node trade off loads and across chain spread
    pair_summary.csv           per pair cosine, median correlation, median
                               trade off ratio, and node count
    pair_metrics.csv           one row of aggregate statistics: rank and linear
                               correlations of cosine against the trade off
                               metrics, variance explained, and rank
                               correlations of the trade off loads against
                               node spread

Interpretation hint
    A negative cosine versus correlation rank correlation together with a
    positive cosine versus trade off ratio rank correlation means similar
    signatures trade off more, so non-orthogonality contributes to the
    multimodality, and the variance explained says how much. If neither is
    significant the trade offs are generic to the simplex rather than driven
    by signature similarity. A positive link between the trade off load and
    node spread means this geometry tracks which nodes are hard. A soft trade
    off on a full rank signature matrix is a sampling ridge, not genuine
    non-identifiability.

Usage:
    python diagnose_pair_tradeoff.py \
        --trace path/to/trace_aligned.nc \
        --truth path/to/true_activities.csv \
        --true-sigs path/to/fixed_signatures.csv \
        --newick path/to/newick_string.nwk \
        [--pair A B] [--active-threshold 0.05] [--outdir pair_tradeoff]
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.analysis import (
    build_forest,
    chain_perms_to_true,
    nodes_by_depth,
)

def _pair_stats(e, a, b):
    """
    e : (draw, K) within-chain activity draws in TRUE coordinates.
    Returns (corr_ab, tradeoff_ratio, active_mass) or None if degenerate.
        corr_ab        : Pearson corr of e_a, e_b across draws.
        tradeoff_ratio : Var(e_a - e_b) / Var(e_a + e_b).  >1 => the split
                         is looser than the sum (ridge signature).
        active_mass    : mean(e_a) + mean(e_b) at this node.
    """
    ea, eb = e[:, a], e[:, b]
    active = float(ea.mean() + eb.mean())
    va, vb = ea.var(), eb.var()
    if va < 1e-12 or vb < 1e-12:
        return None                              # a stuck/degenerate chain
    corr = float(np.corrcoef(ea, eb)[0, 1])
    diff_v = (ea - eb).var()
    sum_v = (ea + eb).var()
    ratio = float(diff_v / sum_v) if sum_v > 1e-12 else np.nan
    return corr, ratio, active


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--truth", required=True,
                    help="true_activities.csv (rows=nodes, cols=k0..K-1).")
    ap.add_argument("--true-sigs", required=True, dest="true_sigs",
                    help="fixed_signatures.csv (rows=k, cols=96).")
    ap.add_argument("--newick", required=True)
    ap.add_argument("--pair", type=int, nargs=2, default=None,
                    help="Override focus pair (true indices). Default: the "
                         "most cosine-similar true signature pair.")
    ap.add_argument("--active-threshold", type=float, default=0.05,
                    help="Min mean(e_a)+mean(e_b) for a node to count: the "
                         "pair must actually be present to trade off.")
    ap.add_argument("--outdir", default="pair_tradeoff")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    idata = az.from_netcdf(args.trace)
    post = idata.posterior
    truth = pd.read_csv(args.truth, index_col=0)
    true_S = pd.read_csv(args.true_sigs, index_col=0).values
    newick = Path(args.newick).read_text().strip()
    K = true_S.shape[0]
    n_chains = post.sizes["chain"]

    Sn = true_S / (np.linalg.norm(true_S, axis=1, keepdims=True) + 1e-12)
    cosS = Sn @ Sn.T
    pair_cos = {(a, b): float(cosS[a, b]) for a, b in combinations(range(K), 2)}
    if args.pair is not None:
        focus = tuple(sorted(args.pair))
    else:
        focus = max(pair_cos, key=pair_cos.get)
    fa, fb = focus

    dr_to_node = nodes_by_depth(build_forest(newick))
    perms = chain_perms_to_true(post["signatures"].values, true_S)
    e_vars = sorted([v for v in post.data_vars if v.startswith("e_level")])

    all_pairs = list(combinations(range(K), 2))
    pair_nodecorr = {p: [] for p in all_pairs}
    pair_noderatio = {p: [] for p in all_pairs}
    node_rows = []

    for ev in e_vars:
        depth = int(ev.split("_")[-1])
        arr = post[ev].values
        for node_row in range(arr.shape[2]):
            label = dr_to_node.get((depth, node_row))
            if label is None or label not in truth.index:
                continue

            e_by_chain = [arr[c, :, node_row, :][:, perms[c]]
                          for c in range(n_chains)]

            chain_means = np.stack([e.mean(axis=0) for e in e_by_chain])
            spread = float(np.abs(
                chain_means - chain_means.mean(0, keepdims=True)
            ).sum(1).mean())

            node_pc = {}
            for p in all_pairs:
                a, b = p
                cs, rs = [], []
                for e in e_by_chain:
                    st = _pair_stats(e, a, b)
                    if st is None:
                        continue
                    corr, ratio, active = st
                    if active < args.active_threshold:
                        continue
                    cs.append(corr)
                    rs.append(ratio)
                if cs:
                    node_pc[p] = (float(np.mean(cs)), float(np.mean(rs)))
                    pair_nodecorr[p].append(node_pc[p][0])
                    pair_noderatio[p].append(node_pc[p][1])

            if not node_pc:
                continue

            anti = {p: max(0.0, -node_pc[p][0]) for p in node_pc}
            plain_load = float(np.mean(list(anti.values())))
            geom_load = float(np.mean([pair_cos[p] * anti[p] for p in anti]))
            node_rows.append({
                "node": label,
                "focus_corr": node_pc.get(focus, (np.nan,))[0],
                "plain_load": plain_load,
                "geom_load": geom_load,
                "n_active_pairs": len(node_pc),
                "chain_mean_spread": spread,
            })

    node_df = pd.DataFrame(node_rows)
    node_df.to_csv(outdir / "pair_tradeoff_by_node.csv", index=False)

    summ = []
    for p in all_pairs:
        if not pair_nodecorr[p]:
            continue
        summ.append({
            "pair": f"{p[0]}-{p[1]}",
            "cosine": pair_cos[p],
            "median_corr": float(np.median(pair_nodecorr[p])),
            "median_tradeoff_ratio": float(np.median(pair_noderatio[p])),
            "n_nodes": len(pair_nodecorr[p]),
        })
    summ_df = pd.DataFrame(summ).sort_values("cosine", ascending=False)
    summ_df.to_csv(outdir / "pair_summary.csv", index=False)

    cos_arr = summ_df["cosine"].values
    corr_arr = summ_df["median_corr"].values
    ratio_arr = summ_df["median_tradeoff_ratio"].values
    sp_corr = spearmanr(cos_arr, corr_arr)
    sp_ratio = spearmanr(cos_arr, ratio_arr)
    pe_corr = pearsonr(cos_arr, corr_arr)
    pe_ratio = pearsonr(cos_arr, ratio_arr)
    r2_corr = float(pe_corr[0] ** 2)
    n_pairs = len(summ_df)

    top_cos = summ_df.iloc[0]
    rank_anti = int(summ_df["median_corr"].rank(ascending=True)[
        summ_df["pair"] == top_cos["pair"]].iloc[0])
    most_anti = summ_df.sort_values("median_corr").iloc[0]

    def _link(col):
        if len(node_df) <= 5:
            return np.nan, np.nan
        return spearmanr(node_df[col], node_df["chain_mean_spread"])
    plain_rho, plain_p = _link("plain_load")
    geom_rho, geom_p = _link("geom_load")
    flink_rho, flink_p = np.nan, np.nan
    fd = node_df.dropna(subset=["focus_corr"])
    if len(fd) > 5:
        flink_rho, flink_p = spearmanr(-fd["focus_corr"],
                                       fd["chain_mean_spread"])

    metrics = {
        "n_chains": n_chains,
        "n_nodes_used": len(node_df),
        "n_pairs": n_pairs,
        "focus_pair": f"{fa}-{fb}",
        "focus_pair_cosine": float(pair_cos[focus]),
        "active_threshold": float(args.active_threshold),
        "spearman_cosine_vs_corr": float(sp_corr[0]),
        "spearman_cosine_vs_corr_p": float(sp_corr[1]),
        "pearson_cosine_vs_corr": float(pe_corr[0]),
        "spearman_cosine_vs_ratio": float(sp_ratio[0]),
        "spearman_cosine_vs_ratio_p": float(sp_ratio[1]),
        "pearson_cosine_vs_ratio": float(pe_ratio[0]),
        "r2_corr_on_cosine": float(r2_corr),
        "most_similar_pair": str(top_cos["pair"]),
        "most_similar_pair_anticorr_rank": int(rank_anti),
        "most_tradedoff_pair": str(most_anti["pair"]),
        "most_tradedoff_pair_corr": float(most_anti["median_corr"]),
        "spearman_plainload_vs_spread": float(plain_rho),
        "spearman_plainload_vs_spread_p": float(plain_p),
        "spearman_geomload_vs_spread": float(geom_rho),
        "spearman_geomload_vs_spread_p": float(geom_p),
        "spearman_focusanticorr_vs_spread": float(flink_rho),
        "spearman_focusanticorr_vs_spread_p": float(flink_p),
    }
    pd.DataFrame([metrics]).to_csv(outdir / "pair_metrics.csv", index=False)

    print(f"Written: {outdir/'pair_metrics.csv'}, "
          f"{outdir/'pair_summary.csv'}, {outdir/'pair_tradeoff_by_node.csv'}")


if __name__ == "__main__":
    main()