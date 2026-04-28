"""
evaluation.py

Pure-analysis functions: no plotting, no I/O.

These functions consume a fitted _BaseTreeHDP model (or raw arrays) and
return DataFrames or dicts that downstream plotting and reporting code
can consume.

Functions
---------
evaluate_inference
    Per-node MAE and cosine similarity between true and posterior-mean
    activity vectors.

get_node_depths
    {node_id: depth} for every node in a composed graph.

compute_depth_stats
    ESS, r_hat, mean and sd for every node-signature pair, annotated
    with tree depth.

compute_alpha_correlations
    Pearson correlation between shared_alpha samples and each
    node-signature activity component.

summarise_depth_stats
    Print a human-readable summary of the depth-stat DataFrame.

summarise_alpha_correlations
    Print a human-readable summary of the alpha-correlation DataFrame.

load_model_fixedSigModel
    Reconstruct a FixedSigHDP model from config + saved trace.

load_model_unknownSigModel
    Reconstruct an UnknownSigHDP model from config + saved trace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import arviz as az
import networkx as nx
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


if TYPE_CHECKING:
    from src.models.hdp_inference import _BaseTreeHDP
from src.models.hdp_inference import FixedSigHDP, UnknownSigHDP

def load_model_fixedSigModel(cfg: dict, trace_path: str) -> FixedSigHDP:
    """
    Reconstruct the FixedSigHDP object from data files and attach the trace.

    Parameters
    ----------
    cfg : dict
    trace_path : str

    Returns
    -------
    FixedSigHDP
    """
    data_cfg = cfg["inference"]["data"]

    with open(data_cfg["newick_string"]) as f:
        newick_string = f.read().strip()

    count_matrix = pd.read_csv(data_cfg["count_matrix"], index_col=0)
    signatures = pd.read_csv(data_cfg["fixed_signatures"], index_col=0).values
    inf_cfg = cfg.get("inference", {})

    model = FixedSigHDP(
        newick_string=newick_string,
        data_matrix=count_matrix,
        fixed_signatures=signatures,
        priors = inf_cfg["priors"],
    )
    model.trace = az.from_netcdf(trace_path)
    return model

def load_model_unknownSigModel(cfg: dict, trace_path: str) -> UnknownSigHDP:
    """
    Reconstruct the UnknownSigHDP object from data files and attach the trace.

    Only the count matrix, Newick string, K_max, and priors are needed —
    no signature file, because signatures were inferred and are stored
    inside the trace itself.

    Parameters
    ----------
    cfg : dict
        Loaded YAML config for the unknown-signature experiment.
    trace_path : str
        Path to the saved ``trace.nc`` ArviZ file.

    Returns
    -------
    UnknownSigHDP
    """
    inf_cfg  = cfg["inference"]
    data_cfg = inf_cfg["data"]

    with open(data_cfg["newick_string"]) as f:
        newick_string = f.read().strip()

    count_matrix = pd.read_csv(data_cfg["count_matrix"], index_col=0)

    model = UnknownSigHDP(
        newick_string=newick_string,
        data_matrix=count_matrix,
        K_max=int(inf_cfg["K_max"]),
        priors=inf_cfg["priors"],
    )
    model.trace = az.from_netcdf(trace_path)
    return model

def evaluate_inference(
    model: "_BaseTreeHDP",
    true_activities: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Compare posterior-mean activity vectors to ground-truth activities.

    For every node that appears in both the model's node_index_map and
    `true_activities`, compute:
    - **MAE**: mean absolute error between posterior mean and true e.
    - **cosine_similarity**: cosine similarity between the two vectors.

    Parameters
    ----------
    model : _BaseTreeHDP
        A fitted inference model with a populated `trace`.
    true_activities : dict
        Mapping {node_label: true_e_vector (shape K,)}.

    Returns
    -------
    pd.DataFrame
        Index: node labels.  Columns: ``mae``, ``cosine_similarity``.
    """
    records = []
    for node_id, (var_name, row_idx) in model.node_index_map.items():
        label = model.graph.nodes[node_id].get("label", str(node_id))
        if label not in true_activities:
            continue

        posterior_mean = model.get_posterior_mean(node_id)   # (K,)
        true_e = true_activities[label]

        mae = float(np.abs(posterior_mean - true_e).mean())
        cosine_sim = float(
            np.dot(posterior_mean, true_e)
            / (np.linalg.norm(posterior_mean) * np.linalg.norm(true_e) + 1e-12)
        )
        records.append({"node": label, "mae": mae, "cosine_similarity": cosine_sim})

    return pd.DataFrame(records).set_index("node")

def get_node_depths(model: "_BaseTreeHDP") -> Dict[str, int]:
    """
    Compute the depth of every node in the composed graph via BFS.

    Parameters
    ----------
    model : _BaseTreeHDP

    Returns
    -------
    dict
        {node_id: depth}
    """
    depths: Dict[str, int] = {}
    roots = [n for n, d in model.graph.in_degree() if d == 0]
    for root in roots:
        for node, depth in nx.single_source_shortest_path_length(
            model.graph, root
        ).items():
            if node not in depths:
                depths[node] = depth
    return depths

def compute_depth_stats(model: "_BaseTreeHDP") -> pd.DataFrame:
    """
    Collect ESS, r_hat, mean and sd for every node-signature pair from the
    ArviZ trace summary, annotated with node depth.

    Parameters
    ----------
    model : _BaseTreeHDP
        Fitted model with a populated trace.

    Returns
    -------
    pd.DataFrame
        Columns: node, depth, signature, mean, sd, ess_bulk, ess_tail, r_hat.
    """
    node_depths = get_node_depths(model)
    records = []
    summary_cache: Dict[str, pd.DataFrame] = {}

    for node_id, (var_name, row_idx) in model.node_index_map.items():
        label = model.graph.nodes[node_id].get("label", str(node_id))
        depth = node_depths.get(node_id)

        if var_name not in summary_cache:
            summary_cache[var_name] = az.summary(model.trace, var_names=[var_name])
        summary = summary_cache[var_name]

        if row_idx is None:
            rows = summary
        else:
            prefix = f"{var_name}[{row_idx},"
            rows = summary[summary.index.str.startswith(prefix)]

        for k, (_, row) in enumerate(rows.iterrows()):
            records.append(
                {
                    "node": label,
                    "depth": depth,
                    "signature": k,
                    "mean": row["mean"],
                    "sd": row["sd"],
                    "ess_bulk": row["ess_bulk"],
                    "ess_tail": row["ess_tail"],
                    "r_hat": row["r_hat"],
                }
            )

    return pd.DataFrame(records)

def summarise_depth_stats(depth_df: pd.DataFrame) -> None:
    """Print a concise summary of ESS and r_hat statistics by depth."""
    print("\n--- ESS / R_HAT BY DEPTH ---")
    print("\nMedian per depth:")
    summary = depth_df.groupby("depth")[["ess_bulk", "ess_tail", "r_hat"]].median()
    print(summary.to_string())

    total = len(depth_df)
    n_bad_rhat = (depth_df["r_hat"] > 1.01).sum()
    n_low_ess = (depth_df["ess_bulk"] < 100).sum()
    print(f"\nParameters with r_hat > 1.01: {n_bad_rhat}/{total} ({100*n_bad_rhat/total:.1f}%)")
    print(f"Parameters with ESS  < 100:   {n_low_ess}/{total} ({100*n_low_ess/total:.1f}%)")

def compute_alpha_correlations(model: "_BaseTreeHDP") -> pd.DataFrame:
    """
    Compute the Pearson correlation between shared_alpha posterior samples
    and every node-signature activity across all chains and draws.

    High correlations indicate non-identifiability between alpha and the
    activity components.

    Parameters
    ----------
    model : _BaseTreeHDP
        Fitted model with a populated trace.

    Returns
    -------
    pd.DataFrame
        Columns: node, signature, correlation_with_alpha, mean_activity.
    """
    alpha_samples = model.trace.posterior["shared_alpha"].values.flatten()
    records = []

    for node_id, _ in model.node_index_map.items():
        label = model.graph.nodes[node_id].get("label", str(node_id))
        posterior = model.get_node_activity_posterior(node_id)          # (chains, draws, K)
        posterior_flat = posterior.reshape(-1, posterior.shape[-1])  # (N, K)
        for k in range(posterior_flat.shape[1]):
            corr = float(np.corrcoef(alpha_samples, posterior_flat[:, k])[0, 1])
            records.append(
                {
                    "node": label,
                    "signature": k,
                    "correlation_with_alpha": corr,
                    "mean_activity": float(posterior_flat[:, k].mean()),
                }
            )

    return pd.DataFrame(records)

def summarise_alpha_correlations(corr_df: pd.DataFrame) -> None:
    """Print a concise summary of alpha-activity correlations."""
    high = corr_df[corr_df["correlation_with_alpha"].abs() > 0.3]
    total = len(corr_df)

    print("\n--- ALPHA CORRELATION DIAGNOSTIC ---")
    print(f"Total node-signature pairs:      {total}")
    print(f"Pairs with corr |r| > 0.3 with alpha: {len(high)} ({100*len(high)/total:.1f}%)")
    print(f"Mean |correlation|:              {corr_df['correlation_with_alpha'].abs().mean():.3f}")
    print(f"Max  |correlation|:              {corr_df['correlation_with_alpha'].abs().max():.3f}")

    print("\nTop 10 most correlated pairs:")
    top10 = corr_df.reindex(
        corr_df["correlation_with_alpha"].abs().nlargest(10).index
    )[["node", "signature", "correlation_with_alpha", "mean_activity"]]
    print(top10.to_string(index=False))

    corr_df = corr_df.copy()
    corr_df["is_near_zero"] = corr_df["mean_activity"] < 0.01
    print("\nMean |correlation| by activity level:")
    print(
        corr_df.groupby("is_near_zero")["correlation_with_alpha"]
        .apply(lambda x: x.abs().mean())
        .rename({True: "near-zero (<0.01)", False: "active (>=0.01)"})
        .to_string()
    )

def compare_alpha(
    model: "_BaseTreeHDP",
    true_alpha: float,
) -> pd.DataFrame:
    """
    Compare the posterior of shared_alpha to the true value used in simulation.

    Returns a one-row DataFrame with posterior mean, sd, and the 94% HDI,
    alongside the true value.

    Parameters
    ----------
    model : _BaseTreeHDP
        Fitted model with a populated trace.
    true_alpha : float
        The alpha value used when generating the data (from
        ``ground_truth_params.json``, key ``"alpha"``).

    Returns
    -------
    pd.DataFrame
        Columns: true_alpha, posterior_mean, posterior_sd, hdi_3%, hdi_97%.
    """
    samples = model.trace.posterior["shared_alpha"].values.flatten()
    hdi = az.hdi(model.trace, var_names=["shared_alpha"], hdi_prob=0.97)
    hdi_lo = float(hdi["shared_alpha"].values[0])
    hdi_hi = float(hdi["shared_alpha"].values[1])

    return pd.DataFrame([{
        "true_alpha":    true_alpha,
        "posterior_mean": float(samples.mean()),
        "posterior_sd":   float(samples.std()),
        "hdi_3%":         hdi_lo,
        "hdi_97%":        hdi_hi,
        "true_in_hdi":    hdi_lo <= true_alpha <= hdi_hi,
    }])


def compare_activities(
        model: "_BaseTreeHDP",
        true_activities: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Per-node comparison of posterior-mean activities vs ground truth.

    Intended for **FixedSigHDP**, where the number of inferred components
    equals the number of true components and the ordering is the same.
    For UnknownSigHDP use ``compare_activities_unknown`` instead, which
    handles the permutation ambiguity correctly.

    For each node present in both the model and ``true_activities``, computes
    MAE, cosine similarity, and per-component true/inferred values.

    Parameters
    ----------
    model : _BaseTreeHDP
        Fitted model with a populated trace.
    true_activities : dict
        ``{node_label: true_e_vector}`` with shape ``(K,)``.

    Returns
    -------
    pd.DataFrame
        Index: node label.
        Columns: ``depth``, ``mae``, ``cosine_similarity``,
                 ``true_k0``, ``inferred_k0``, ``true_k1``, ...
    """
    node_depths = get_node_depths(model)
    records = []

    for node_id in model.node_index_map:
        label = model.graph.nodes[node_id].get("label", str(node_id))
        if label not in true_activities:
            continue

        true_e = true_activities[label]  # (K,)
        inferred_e = model.get_posterior_mean(node_id)  # (K,)

        mae = float(np.abs(inferred_e - true_e).mean())
        cos = float(
            np.dot(inferred_e, true_e)
            / (np.linalg.norm(inferred_e) * np.linalg.norm(true_e) + 1e-12)
        )
        row = {
            "node": label,
            "depth": node_depths.get(node_id),
            "mae": mae,
            "cosine_similarity": cos,
        }
        for k in range(len(true_e)):
            row[f"true_k{k}"] = float(true_e[k])
            row[f"inferred_k{k}"] = float(inferred_e[k])

        records.append(row)

    return pd.DataFrame(records).set_index("node")


def compare_activities_unknown(
        model: "_BaseTreeHDP",
        true_activities: Dict[str, np.ndarray],
        assignment: np.ndarray,
) -> pd.DataFrame:
    """
    Per-node comparison of posterior-mean activities vs ground truth for
    **UnknownSigHDP**, where the inferred signature ordering is arbitrary.

    Uses ``assignment`` from ``align_signatures`` to extract the right
    component from each node's inferred activity vector.  Concretely, for
    each true signature index ``k``, this function reads
    ``inferred_e[assignment[k]]`` and compares it to ``true_e[k]``.
    The K_max - K_true unmatched inferred components are ignored entirely.

    Parameters
    ----------
    model : _BaseTreeHDP
        Fitted UnknownSigHDP model with a populated trace.
    true_activities : dict
        ``{node_label: true_e_vector}`` with shape ``(K_true,)``.
        Pass the *original* true activities — no pre-alignment needed.
    assignment : np.ndarray
        Shape ``(K_true,)``. Direct output of ``align_signatures``:
        ``assignment[k]`` is the index in the inferred posterior that
        corresponds to true signature ``k``.

    Returns
    -------
    pd.DataFrame
        Index: node label.
        Columns: ``depth``, ``mae``, ``cosine_similarity``,
                 ``true_k0``, ``inferred_k0``, ``true_k1``, ...
        ``inferred_k{k}`` holds ``inferred_e[assignment[k]]`` — the
        inferred activity for the component matched to true signature k.
    """
    node_depths = get_node_depths(model)
    K_true = len(assignment)
    records = []

    for node_id in model.node_index_map:
        label = model.graph.nodes[node_id].get("label", str(node_id))
        if label not in true_activities:
            continue

        true_e = true_activities[label]  # (K_true,)
        inferred_e = model.get_posterior_mean(node_id)  # (K_max,)

        # Pick out only the matched components, in true-signature order
        matched_inferred = np.array([inferred_e[assignment[k]] for k in range(K_true)])

        mae = float(np.abs(matched_inferred - true_e).mean())
        cos = float(
            np.dot(matched_inferred, true_e)
            / (np.linalg.norm(matched_inferred) * np.linalg.norm(true_e) + 1e-12)
        )
        row = {
            "node": label,
            "depth": node_depths.get(node_id),
            "mae": mae,
            "cosine_similarity": cos,
        }
        for k in range(K_true):
            row[f"true_k{k}"] = float(true_e[k])
            row[f"inferred_k{k}"] = float(matched_inferred[k])

        records.append(row)

    return pd.DataFrame(records).set_index("node")

def align_signatures(
    inferred_sigs: np.ndarray,
    true_sigs: np.ndarray,
) -> tuple:
    """
    Align inferred signatures to true signatures using the Hungarian algorithm
    on pairwise cosine similarities.

    Because UnknownSigHDP can infer a different number of signatures than
    the true K, this function handles the case where K_inferred >= K_true:
    each true signature is matched to exactly one inferred signature, and
    unmatched inferred signatures are marked as ``-1`` in the mapping.

    Parameters
    ----------
    inferred_sigs : np.ndarray
        Shape (K_inferred, 96). Posterior-mean signatures from the model.
    true_sigs : np.ndarray
        Shape (K_true, 96). Ground-truth signatures.

    Returns
    -------
    aligned_inferred : np.ndarray
        Shape (K_true, 96). Rows of ``inferred_sigs`` reordered so that
        ``aligned_inferred[k]`` is the best match for ``true_sigs[k]``.
    assignment : np.ndarray
        Shape (K_true,). ``assignment[k]`` is the index into ``inferred_sigs``
        matched to ``true_sigs[k]``.
    cosine_sim_matrix : np.ndarray
        Shape (K_true, K_inferred). Full pairwise cosine similarity matrix,
        useful for visualisation.
    """
    from scipy.optimize import linear_sum_assignment

    # Pairwise cosine similarities: (K_true, K_inf)
    true_norm = true_sigs / (np.linalg.norm(true_sigs, axis=1, keepdims=True) + 1e-12)
    inf_norm  = inferred_sigs / (np.linalg.norm(inferred_sigs, axis=1, keepdims=True) + 1e-12)
    cosine_sim_matrix = true_norm @ inf_norm.T   # (K_true, K_inf)

    # Hungarian: maximise cosine similarity = minimise negative similarity
    row_ind, col_ind = linear_sum_assignment(-cosine_sim_matrix)

    aligned_inferred = inferred_sigs[col_ind]   # (K_true, 96)
    assignment = col_ind                         # which inferred index matched each true index

    return aligned_inferred, assignment, cosine_sim_matrix


def compare_signatures(
    inferred_sigs: np.ndarray,
    true_sigs: np.ndarray,
) -> pd.DataFrame:
    """
    Align inferred signatures to true ones and compute per-signature
    cosine similarity and MAE.

    Parameters
    ----------
    inferred_sigs : np.ndarray
        Shape (K_inferred, 96). Posterior-mean signatures.
    true_sigs : np.ndarray
        Shape (K_true, 96). Ground-truth signatures.

    Returns
    -------
    pd.DataFrame
        Index: ``True_Sig_k``.
        Columns: ``matched_inferred_idx``, ``cosine_similarity``, ``mae``.
    """
    aligned, assignment, _ = align_signatures(inferred_sigs, true_sigs)

    records = []
    for k, (inf_idx, aligned_sig) in enumerate(zip(assignment, aligned)):
        cos = float(
            np.dot(aligned_sig, true_sigs[k])
            / (np.linalg.norm(aligned_sig) * np.linalg.norm(true_sigs[k]) + 1e-12)
        )
        mae = float(np.abs(aligned_sig - true_sigs[k]).mean())
        records.append({
            "true_signature":       f"True_Sig_{k}",
            "matched_inferred_idx": int(inf_idx),
            "cosine_similarity":    cos,
            "mae":                  mae,
        })

    return pd.DataFrame(records).set_index("true_signature")
