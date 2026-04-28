"""
plots.py

All visualisation functions for the Tree-HDP project.

Every function follows the same convention:
- Accepts data (arrays, DataFrames, or model objects) as its first argument(s).
- Accepts an optional `save_path` (str or Path).  When provided the figure
  is saved and closed; when None the figure is displayed interactively.
- Returns None (side-effect only).

Sections
--------
Signature plots
    plot_signature_bar        – bar chart of a single signature over 96 channels
    plot_node_signatures      – activities + top-N signatures for one node

Data / cohort plots
    plot_patient_counts       – clustered heatmap of (normalised) mutation profiles
    plot_signatures_heatmap   – heatmap of all K signatures

Posterior diagnostics
    plot_depth_stats          – boxplots of ESS and r_hat by tree depth
    plot_zero_vs_active       – median ESS / r_hat for near-zero vs active components
    plot_alpha_correlations   – distribution and scatter of alpha-activity correlations

Recovery
    plot_recovery_distributions  – histograms of cosine similarity and MAE
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from src.models.hdp_simulator import HDP


def _save_or_show(fig: plt.Figure, save_path: Optional[str], dpi: int = 300) -> None:
    """Save the figure if a path is given, otherwise show it."""
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to '{save_path}'")
    else:
        plt.show()

def plot_signature_bar(
    signature: np.ndarray,
    title: str = "Mutational Signature",
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart of a single mutational signature over 96 trinucleotide channels.

    Parameters
    ----------
    signature : np.ndarray
        Shape (96,).
    title : str
    save_path : str, optional
    """
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.bar(range(96), signature, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("96 Trinucleotide Channels")
    ax.set_ylabel("Probability")
    ax.set_xlim(-0.5, 95.5)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_node_signatures(
    activities: np.ndarray,
    signatures: np.ndarray,
    node_label: str = "",
    alpha: Optional[float] = None,
    top_n: int = 3,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualise the stick-breaking activities and the top-N signatures for a node.

    Parameters
    ----------
    activities : np.ndarray
        Shape (K,) – the mut_activities (stick-breaking weights) or posterior mean.
    signatures : np.ndarray
        Shape (K, 96).
    node_label : str
    alpha : float, optional
        Shown in the activities subplot title if provided.
    top_n : int
        Number of top signatures to plot alongside the activities bar.
    save_path : str, optional
    """
    if len(activities) == 0:
        print(f"No activities to plot for node '{node_label}'.")
        return

    fig = plt.figure(figsize=(15, 4))
    ax1 = plt.subplot(1, top_n + 1, 1)
    ax1.bar(range(len(activities)), activities, color="steelblue")
    title = f"{node_label} – Activities"
    if alpha is not None:
        title += f"\n(alpha={alpha:.2f})"
    ax1.set_title(title)
    ax1.set_xlabel("Signature Index")
    ax1.set_ylabel("Weight")

    top_indices = np.argsort(activities)[::-1][:top_n]
    for i, sig_idx in enumerate(top_indices):
        ax = plt.subplot(1, top_n + 1, i + 2)
        ax.bar(range(96), signatures[sig_idx], color="tomato")
        ax.set_title(f"Signature {sig_idx}\n(Weight: {activities[sig_idx]:.3f})")
        ax.set_xlabel("96 Channels")
        if i == 0:
            ax.set_ylabel("Probability")

    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_node_signatures_from_model(
    hdp_model: "HDP",
    node_label: str,
    top_n: int = 3,
    save_path: Optional[str] = None,
) -> None:
    """
    Convenience wrapper: extract activities and signatures from an HDP
    simulator object and call `plot_node_signatures`.

    Parameters
    ----------
    hdp_model : HDP
        A fitted HDP forward simulator.
    node_label : str
    top_n : int
    save_path : str, optional
    """
    dp_model = hdp_model.get_node_data(node_label, "dp_model")
    plot_node_signatures(
        activities=np.array(dp_model.mut_activities),
        signatures=np.array(dp_model.signatures),
        node_label=node_label,
        alpha=dp_model.alpha,
        top_n=top_n,
        save_path=save_path,
    )

def plot_patient_counts(
    count_matrix: pd.DataFrame,
    title: str = "Relative Mutation Profiles",
    save_path: Optional[str] = None,
) -> None:
    """
    Clustered heatmap of normalised (row-wise) mutation profiles.

    Parameters
    ----------
    count_matrix : pd.DataFrame
        Shape (N_nodes, 96).  Raw counts; rows are normalised internally.
    title : str
    save_path : str, optional
    """
    sns.set_theme(style="white")
    normalised = count_matrix.div(count_matrix.sum(axis=1), axis=0)

    g = sns.clustermap(
        normalised,
        cmap="viridis",
        figsize=(20, 12),
        row_cluster=True,
        col_cluster=False,
        cbar_kws={"label": "Mutation Fraction"},
        rasterized=True,
        linewidths=0,
    )
    g.figure.suptitle(title, fontsize=16, y=1.02)
    g.ax_heatmap.set_xlabel("96 Trinucleotide Mutation Channels", fontsize=12)
    g.ax_heatmap.set_ylabel("Nodes (Clustered)", fontsize=12)
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=8
    )

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        g.figure.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(g.figure)
        print(f"Saved plot to '{save_path}'")
    else:
        plt.show()


def plot_signatures_heatmap(
    signatures: np.ndarray,
    title: str = "Mutational Signatures",
    save_path: Optional[str] = None,
) -> None:
    """
    Heatmap of all K mutational signatures over 96 channels.

    Parameters
    ----------
    signatures : np.ndarray
        Shape (K, 96).
    title : str
    save_path : str, optional
    """
    sns.set_theme(style="white")
    sig_df = pd.DataFrame(
        signatures,
        index=[f"Signature_{i}" for i in range(len(signatures))],
        columns=[f"Channel_{i}" for i in range(96)],
    )

    fig, ax = plt.subplots(figsize=(20, 4))
    sns.heatmap(
        sig_df,
        cmap="magma",
        ax=ax,
        cbar_kws={"label": "Probability"},
        linewidths=0.5,
        linecolor="black",
    )
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("96 Trinucleotide Mutation Channels", fontsize=12)
    ax.set_ylabel("Signatures", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    fig.tight_layout()
    _save_or_show(fig, save_path)

def plot_depth_stats(depth_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Boxplots of ESS (bulk + tail), r_hat, and sd grouped by node depth.

    Parameters
    ----------
    depth_df : pd.DataFrame
        Output of `evaluation.compute_depth_stats`.
    save_path : str, optional
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ["ess_bulk", "ess_tail", "r_hat", "sd"]
    thresholds = {
        "ess_bulk": (100, "orange", "ESS=100"),
        "ess_tail": (100, "orange", "ESS=100"),
        "r_hat": (1.01, "red", "r_hat=1.01"),
    }

    for ax, metric in zip(axes.flatten(), metrics):
        depth_df.boxplot(column=metric, by="depth", ax=ax)
        ax.set_title(f"{metric} by node depth")
        ax.set_xlabel("Depth")
        ax.set_ylabel(metric)
        if metric in thresholds:
            val, color, label = thresholds[metric]
            ax.axhline(val, color=color, linestyle="--", label=label)
            ax.legend()

    fig.suptitle("Sampling quality by tree depth")
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_zero_vs_active(
    depth_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """
    Median ESS and r_hat by depth, split into near-zero vs active components.

    Parameters
    ----------
    depth_df : pd.DataFrame
        Output of `evaluation.compute_depth_stats`.
    save_path : str, optional
    """
    df = depth_df.copy()
    df["is_near_zero"] = df["mean"] < 0.01

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ["ess_bulk", "r_hat"]):
        for is_zero, group in df.groupby("is_near_zero"):
            lbl = "near-zero (<0.01)" if is_zero else "active (≥0.01)"
            group.groupby("depth")[metric].median().plot(
                ax=ax, label=lbl, marker="o"
            )
        ax.set_title(f"Median {metric}: zero vs active components")
        ax.set_xlabel("Depth")
        ax.legend()
        if metric == "r_hat":
            ax.axhline(1.01, color="red", linestyle="--")
        if metric == "ess_bulk":
            ax.axhline(100, color="orange", linestyle="--")

    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_alpha_correlations(
    corr_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """
    Two-panel plot: distribution of alpha-activity correlations, and
    correlation vs mean activity scatter.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Output of `evaluation.compute_alpha_correlations`.
    save_path : str, optional
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(corr_df["correlation_with_alpha"], bins=50, edgecolor="black")
    axes[0].axvline(0.3, color="red", linestyle="--", label="|r|=0.3")
    axes[0].axvline(-0.3, color="red", linestyle="--")
    axes[0].set_title("Distribution of correlations: alpha vs node activities")
    axes[0].set_xlabel("Pearson correlation with shared_alpha")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    sc = axes[1].scatter(
        corr_df["mean_activity"],
        corr_df["correlation_with_alpha"],
        alpha=0.3,
        s=10,
        c=corr_df["mean_activity"],
        cmap="viridis",
    )
    axes[1].axhline(0.3, color="red", linestyle="--", label="|r|=0.3")
    axes[1].axhline(-0.3, color="red", linestyle="--")
    axes[1].set_title("Correlation with alpha vs mean activity")
    axes[1].set_xlabel("Mean posterior activity")
    axes[1].set_ylabel("Correlation with shared_alpha")
    axes[1].legend()
    plt.colorbar(sc, ax=axes[1], label="Mean activity")

    fig.tight_layout()
    _save_or_show(fig, save_path)

def plot_recovery_distributions(
    eval_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """
    Histograms of cosine similarity and MAE from inference evaluation.

    Parameters
    ----------
    eval_df : pd.DataFrame
        Output of `evaluation.evaluate_inference`.  Must contain columns
        ``cosine_similarity`` and ``mae``.
    save_path : str, optional
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    eval_df["cosine_similarity"].hist(bins=30, ax=axes[0], edgecolor="black")
    axes[0].set_title("Cosine Similarity: True vs Inferred Activities")
    axes[0].set_xlabel("Cosine Similarity")
    axes[0].set_ylabel("Count")

    eval_df["mae"].hist(bins=30, ax=axes[1], edgecolor="black")
    axes[1].set_title("MAE: True vs Inferred Activities")
    axes[1].set_xlabel("MAE")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_alpha_recovery(
    alpha_df: pd.DataFrame,
    posterior_samples: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Two-panel plot showing the full posterior of shared_alpha alongside
    the true value.

    Left panel: posterior density with true value marked.
    Right panel: posterior mean ± HDI as an interval, with true value.

    Parameters
    ----------
    alpha_df : pd.DataFrame
        Output of ``evaluation.compare_alpha``.
    posterior_samples : np.ndarray
        Flat array of all MCMC samples for shared_alpha
        (chains × draws,).
    save_path : str, optional
    """
    true_alpha    = float(alpha_df["true_alpha"].iloc[0])
    post_mean     = float(alpha_df["posterior_mean"].iloc[0])
    hdi_lo        = float(alpha_df["hdi_3%"].iloc[0])
    hdi_hi        = float(alpha_df["hdi_97%"].iloc[0])
    true_in_hdi   = bool(alpha_df["true_in_hdi"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: full posterior density
    axes[0].hist(posterior_samples, bins=60, density=True,
                 color="steelblue", alpha=0.7, edgecolor="none")
    axes[0].axvline(true_alpha, color="red",    lw=2, label=f"True α = {true_alpha:.2f}")
    axes[0].axvline(post_mean,  color="orange", lw=2, linestyle="--",
                    label=f"Posterior mean = {post_mean:.2f}")
    axes[0].axvspan(hdi_lo, hdi_hi, alpha=0.15, color="steelblue", label="94% HDI")
    axes[0].set_title("Posterior of shared_alpha")
    axes[0].set_xlabel("shared_alpha")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Right: interval plot
    axes[1].errorbar(
        x=[post_mean], y=[0],
        xerr=[[post_mean - hdi_lo], [hdi_hi - post_mean]],
        fmt="o", color="steelblue", capsize=8, markersize=8,
        label=f"Posterior mean ± 94% HDI",
    )
    axes[1].axvline(true_alpha, color="red", lw=2,
                    label=f"True α = {true_alpha:.2f}")
    cover_str = "✓ True value inside HDI" if true_in_hdi else "✗ True value outside HDI"
    axes[1].set_title(f"Alpha recovery\n{cover_str}")
    axes[1].set_xlabel("shared_alpha")
    axes[1].set_yticks([])
    axes[1].legend()

    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_activity_scatter(
    activity_df: pd.DataFrame,
    K: int,
    save_path: Optional[str] = None,
) -> None:
    """
    Scatter plots of true vs inferred activity for each signature component.

    One panel per signature component k.  Each point is one node.
    Perfect recovery lies on the diagonal.

    Parameters
    ----------
    activity_df : pd.DataFrame
        Output of ``evaluation.compare_activities``.
        Must contain columns ``true_k0, inferred_k0, true_k1, …``.
    K : int
        Number of signature components to plot.
    save_path : str, optional
    """
    ncols = min(K, 4)
    nrows = int(np.ceil(K / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows),
                             squeeze=False)

    for k in range(K):
        ax  = axes[k // ncols][k % ncols]
        col_t = f"true_k{k}"
        col_i = f"inferred_k{k}"
        if col_t not in activity_df.columns:
            ax.set_visible(False)
            continue

        ax.scatter(activity_df[col_t], activity_df[col_i],
                   alpha=0.6, s=20, color="steelblue")
        lim = max(activity_df[col_t].max(), activity_df[col_i].max()) * 1.05
        ax.plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
        ax.set_title(f"Signature {k}")
        ax.set_xlabel("True activity")
        ax.set_ylabel("Inferred activity")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)

    # Hide any unused panels
    for k in range(K, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    fig.suptitle("True vs Inferred Activity — per component", y=1.01)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_activity_heatmap(
    activity_df: pd.DataFrame,
    K: int,
    save_path: Optional[str] = None,
) -> None:
    """
    Side-by-side heatmaps of true and inferred activity matrices.

    Rows are nodes, columns are signature components.  Nodes are sorted
    by tree depth (if the ``depth`` column is present).

    Parameters
    ----------
    activity_df : pd.DataFrame
        Output of ``evaluation.compare_activities``.
    K : int
        Number of signature components.
    save_path : str, optional
    """
    sort_col = "depth" if "depth" in activity_df.columns else None
    df_sorted = activity_df.sort_values(sort_col) if sort_col else activity_df

    true_mat = df_sorted[[f"true_k{k}"     for k in range(K) if f"true_k{k}" in df_sorted.columns]].values
    inf_mat  = df_sorted[[f"inferred_k{k}" for k in range(K) if f"inferred_k{k}" in df_sorted.columns]].values
    node_labels = df_sorted.index.tolist()

    K_actual = true_mat.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(node_labels) * 0.25)),
                             sharey=True)

    vmax = max(true_mat.max(), inf_mat.max())
    kw = dict(vmin=0, vmax=vmax, cmap="YlOrRd", aspect="auto")

    im0 = axes[0].imshow(true_mat,     **kw)
    im1 = axes[1].imshow(inf_mat,      **kw)

    for ax, title, mat in zip(axes, ["True Activities", "Inferred Activities"],
                               [true_mat, inf_mat]):
        ax.set_title(title)
        ax.set_xlabel("Signature component")
        ax.set_xticks(range(K_actual))
        ax.set_xticklabels([f"Sig {k}" for k in range(K_actual)], rotation=45, ha="right")

    axes[0].set_yticks(range(len(node_labels)))
    axes[0].set_yticklabels(node_labels, fontsize=max(5, 9 - len(node_labels) // 20))
    axes[0].set_ylabel("Node (sorted by depth)")

    plt.colorbar(im1, ax=axes[1], label="Activity")
    fig.suptitle("Activity matrix: True vs Inferred", y=1.01)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_signature_recovery(
    sig_comparison_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart of cosine similarity between each true signature and its
    best-matched inferred signature (after Hungarian alignment).

    Parameters
    ----------
    sig_comparison_df : pd.DataFrame
        Output of ``evaluation.compare_signatures``.
    save_path : str, optional
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Cosine similarity per true signature
    axes[0].bar(range(len(sig_comparison_df)),
                sig_comparison_df["cosine_similarity"],
                color="steelblue", edgecolor="black")
    axes[0].axhline(1.0, color="green", linestyle="--", lw=1, label="Perfect recovery")
    axes[0].axhline(0.9, color="orange", linestyle="--", lw=1, label="cos = 0.9")
    axes[0].set_xticks(range(len(sig_comparison_df)))
    axes[0].set_xticklabels(sig_comparison_df.index, rotation=45, ha="right")
    axes[0].set_title("Cosine similarity: true vs best-matched inferred signature")
    axes[0].set_ylabel("Cosine similarity")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()

    # MAE per true signature
    axes[1].bar(range(len(sig_comparison_df)),
                sig_comparison_df["mae"],
                color="tomato", edgecolor="black")
    axes[1].set_xticks(range(len(sig_comparison_df)))
    axes[1].set_xticklabels(sig_comparison_df.index, rotation=45, ha="right")
    axes[1].set_title("MAE: true vs best-matched inferred signature")
    axes[1].set_ylabel("MAE")

    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_signature_comparison_grid(
    inferred_sigs: np.ndarray,
    true_sigs: np.ndarray,
    assignment: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Side-by-side bar charts of each matched true/inferred signature pair.

    One row per true signature, left column = true, right column = inferred match.

    Parameters
    ----------
    inferred_sigs : np.ndarray
        Shape (K_inferred, 96).
    true_sigs : np.ndarray
        Shape (K_true, 96).
    assignment : np.ndarray
        Shape (K_true,). Output of ``align_signatures``.
    save_path : str, optional
    """
    K_true = len(true_sigs)
    fig, axes = plt.subplots(K_true, 2,
                             figsize=(14, 2.5 * K_true),
                             squeeze=False)

    for k in range(K_true):
        inf_k  = assignment[k]
        true_s = true_sigs[k]
        inf_s  = inferred_sigs[inf_k]
        cos    = float(np.dot(true_s, inf_s) /
                       (np.linalg.norm(true_s) * np.linalg.norm(inf_s) + 1e-12))

        axes[k][0].bar(range(96), true_s, color="steelblue")
        axes[k][0].set_title(f"True Sig {k}", fontsize=9)
        axes[k][0].set_ylim(0, max(true_s.max(), inf_s.max()) * 1.1)

        axes[k][1].bar(range(96), inf_s, color="tomato")
        axes[k][1].set_title(f"Inferred Sig {inf_k}  (cos={cos:.3f})", fontsize=9)
        axes[k][1].set_ylim(0, max(true_s.max(), inf_s.max()) * 1.1)

        for ax in axes[k]:
            ax.set_ylabel("Probability", fontsize=7)
            ax.tick_params(axis="x", labelsize=0)

    axes[-1][0].set_xlabel("96 Trinucleotide Channels")
    axes[-1][1].set_xlabel("96 Trinucleotide Channels")
    fig.suptitle("True vs Best-Matched Inferred Signatures (per pair)", y=1.01)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_signature_cosine_heatmap(
    cosine_sim_matrix: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Heatmap of the full pairwise cosine similarity matrix between true and
    inferred signatures.  Highlights the Hungarian assignment.

    Parameters
    ----------
    cosine_sim_matrix : np.ndarray
        Shape (K_true, K_inferred). Output of ``align_signatures``.
    save_path : str, optional
    """
    from scipy.optimize import linear_sum_assignment
    K_true, K_inf = cosine_sim_matrix.shape
    _, col_ind = linear_sum_assignment(-cosine_sim_matrix)

    fig, ax = plt.subplots(figsize=(max(6, K_inf * 0.6), max(4, K_true * 0.5)))
    im = ax.imshow(cosine_sim_matrix, vmin=0, vmax=1,
                   cmap="YlGnBu", aspect="auto")

    # Mark the optimal assignment
    for k_true, k_inf in enumerate(col_ind):
        ax.add_patch(plt.Rectangle(
            (k_inf - 0.5, k_true - 0.5), 1, 1,
            fill=False, edgecolor="red", lw=2,
        ))

    ax.set_xticks(range(K_inf))
    ax.set_xticklabels([f"Inf {k}" for k in range(K_inf)], rotation=45, ha="right")
    ax.set_yticks(range(K_true))
    ax.set_yticklabels([f"True {k}" for k in range(K_true)])
    ax.set_xlabel("Inferred signature index")
    ax.set_ylabel("True signature index")
    ax.set_title("Cosine similarity: True vs Inferred signatures\n(red boxes = Hungarian assignment)")
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    fig.tight_layout()
    _save_or_show(fig, save_path)

