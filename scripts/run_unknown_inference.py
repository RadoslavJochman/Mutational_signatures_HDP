"""
Run de novo signature discovery inference and align the posterior for label
switching.

Purpose
    Fit a model that infers the per node activities and the signature matrix
    S jointly. When S is latent the posterior carries a label switching
    symmetry: any permutation of the K signatures, together with the matching
    permutation of the activity components, leaves the likelihood unchanged,
    so signature index k has no fixed meaning across draws or chains.

Model selection
    The config key inference.model selects the activity prior: 'denovo'
    (default) for the random-walk model DeNovoHDP, or 'ou' for the
    Ornstein-Uhlenbeck model TreeOUHDP (which also reads
    inference.branch_length_scaling). Both share the signature block and
    likelihood, so the alignment and summary below are identical; the OU model
    only adds the global hyperparameters mu, phi, and theta.

Why alignment is needed
    Convergence statistics (r_hat, ess) computed on the raw trace are not
    meaningful, because they would compare chains that merely ordered the
    same signatures differently. A post hoc alignment step is inserted
    between sampling and summary.

Method
    Alignment is per draw. For each posterior draw the K inferred signatures
    are matched to a reference labelling (chain 0 posterior mean) by maximum
    cosine similarity, solved as a linear assignment problem; that draw's
    signatures and per node activity components (e_level_*) are then permuted
    into the reference order. Per draw alignment is robust to within chain
    switching and produces a within chain switching diagnostic as a
    by product: a chain that never switched has a constant permutation across
    all its draws, whereas a varying permutation means that chain switched.

Outputs (written to the run directory)
    trace_raw.nc          unaligned posterior trace
    trace_aligned.nc      posterior trace after per draw alignment
    switching_table.csv   per chain count of distinct permutations and the
                          fraction of draws differing from the dominant one
    inference_summary.csv arviz summary computed on the aligned trace

Interpretation hint
    A chain with n_distinct_perms = 1 stayed in a single labelling; a value
    above one, or a non zero switched_fraction, indicates within chain
    switching and a multimodal posterior. Judge convergence on the aligned
    variables (signatures, e_level_*) only; eta_level_* and z_level_* are not
    aligned, so their r_hat is not informative. For the OU model, sigma, phi
    and theta are label-invariant scalars and their r_hat / ess are
    meaningful as is; mu shares the signature labelling and is not aligned,
    so treat its per-component summary like eta_level_*.

Usage:
    python scripts/run_unknown_inference.py --config configs/<experiment>.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, make_output_dir
from src.models.hdp_inference import DeNovoHDP
from src.models.ou_inference import TreeOUHDP
from src.analysis.analysis import align

def align_trace(trace, activity_var_prefix: str = "e_level"):
    """
    Per-draw alignment of a DeNovoHDP trace to chain 0's mean labelling.

    The `signatures` variable and the per-node activity variables
    (e_level_*) are permuted along their signature axis so that signature k
    means the same thing in every draw of every chain. eta_level_* and
    z_level_* are not permuted: eta carries K-1 free logits, not K aligned
    activity components.

    Parameters
    ----------
    trace : arviz.InferenceData
        Raw trace from DeNovoHDP.sample().
    activity_var_prefix : str
        Prefix of the per-node activity variables to permute alongside the
        signatures.  e_level_* has its last axis = K; eta_level_* has its
        last axis = K-1 (the pinned-logit anchor), so eta is not permuted
        here, only e_level (the interpretable activity) is.

    Returns
    -------
    aligned : arviz.InferenceData
        A copy of `trace` with `signatures` and e_level_* permuted.
    perms : np.ndarray, shape (chains, draws, K)
        The permutation applied to each draw (for the switching report).
    """
    post = trace.posterior
    S = post["signatures"].values
    n_chains, n_draws, K, C = S.shape

    S_ref = S[0].mean(axis=0)

    perms = np.empty((n_chains, n_draws, K), dtype=int)
    for c in range(n_chains):
        for d in range(n_draws):
            perms[c, d], _ = align(S[c, d], S_ref)

    aligned = trace.copy()

    S_aligned = np.empty_like(S)
    for c in range(n_chains):
        for d in range(n_draws):
            S_aligned[c, d] = S[c, d][perms[c, d]]

    sig_da = aligned.posterior["signatures"].copy(data=S_aligned)
    aligned.posterior["signatures"] = sig_da

    for var in list(post.data_vars):
        if not var.startswith(activity_var_prefix):
            continue
        arr = post[var].values
        if arr.shape[-1] != K:
            continue
        out = np.empty_like(arr)
        for c in range(n_chains):
            for d in range(n_draws):
                out[c, d] = arr[c, d][:, perms[c, d]]
        aligned.posterior[var] = aligned.posterior[var].copy(data=out)

    return aligned, perms


def switching_table(perms: np.ndarray) -> pd.DataFrame:
    """
    Summarise within chain label switching from the per draw permutations.

    For each chain it counts how many distinct permutations were applied
    across draws and the fraction of draws that differ from that chain's most
    common permutation.

    Parameters
    ----------
    perms : np.ndarray, shape (chains, draws, K)

    Returns
    -------
    pandas.DataFrame
        One row per chain with columns chain, n_distinct_perms,
        switched_fraction.
    """
    n_chains, n_draws, K = perms.shape
    rows = []
    for c in range(n_chains):
        _, counts = np.unique(perms[c], axis=0, return_counts=True)
        rows.append(
            {
                "chain": c,
                "n_distinct_perms": int(len(counts)),
                "switched_fraction": float(1.0 - counts.max() / n_draws),
            }
        )
    return pd.DataFrame(rows)

def build_model(inf_cfg: dict, newick_string: str, count_matrix, num_signatures: int):
    """
    Instantiate the inference model named by inf_cfg['model'] (default
    'denovo'). Both variants share the signature block, the softmax anchor,
    and the likelihood, so the alignment and summary that follow are identical.

      'denovo' / 'denovo-v1' / 'rw'  -> DeNovoHDP, random-walk activity prior
      'ou' / 'denovo-v2'             -> TreeOUHDP, Ornstein-Uhlenbeck prior
                                        (reads inf_cfg['branch_length_scaling'],
                                        default False)

    Returns
    -------
    (model, label) : the built model and a short class label for logging.
    """
    name = str(inf_cfg.get("model", "denovo")).lower()
    if name in ("denovo", "denovo-v1", "rw"):
        return DeNovoHDP(
            newick_string=newick_string,
            data_matrix=count_matrix,
            num_signatures=num_signatures,
            priors=inf_cfg["priors"],
        ), "DeNovoHDP"
    if name in ("ou", "denovo-v2", "tree-ou"):
        return TreeOUHDP(
            newick_string=newick_string,
            data_matrix=count_matrix,
            num_signatures=num_signatures,
            priors=inf_cfg["priors"],
            branch_length_scaling=bool(inf_cfg.get("branch_length_scaling", False)),
        ), "TreeOUHDP"
    raise ValueError(
        f"Unknown model '{name}'. Use 'denovo' (random walk) or 'ou' "
        f"(Ornstein-Uhlenbeck)."
    )


def run_denovo(cfg: dict) -> None:
    inf_cfg = cfg["inference"]
    data_cfg = inf_cfg["data"]
    result_path = inf_cfg["results_dir"]

    out_dir = make_output_dir(result_path, cfg.get("experiment_name", "experiment"))
    print(f"Output directory: {out_dir}")

    # Load data
    print("Loading data...")
    count_matrix = pd.read_csv(data_cfg["count_matrix"], index_col=0)
    with open(data_cfg["newick_string"]) as f:
        newick_string = f.read().strip()

    num_signatures = int(inf_cfg["num_signatures"])

    # Build the selected model
    model, model_label = build_model(
        inf_cfg, newick_string, count_matrix, num_signatures
    )
    print(f"\nBuilt {model_label} model (K = {num_signatures}).")

    try:
        pm.model_to_graphviz(model.model).render(
            str(out_dir / "bayesian_network"), format="png"
        )
        print(f"Saved Bayesian network graph in {out_dir}.")
    except Exception:
        print("Graphviz not available, skipping model graph.")

    # Optional shared, data-driven initialisation. With init: nmf every chain
    # starts from the same NMF point (init='adapt_diag', no jitter)
    initvals, init_scheme = None, "auto"
    if str(inf_cfg.get("init", "")).lower() == "nmf":
        if model_label != "DeNovoHDP":
            raise ValueError("init: nmf is only supported for model: denovo "
                             "(random walk); the OU walk backs out differently.")
        from src.models.nmf_init import denovo_nmf_initvals
        initvals = denovo_nmf_initvals(
            model, count_matrix,
            sigma_init=float(inf_cfg.get("init_sigma", 0.6)))
        init_scheme = "adapt_diag"
        print("Using shared NMF initialisation (all chains, init=adapt_diag).")

    # Sample
    print("\nStarting MCMC sampler...")
    trace = model.sample(
        draws=inf_cfg["draws"],
        tune=inf_cfg["tune"],
        chains=inf_cfg["chains"],
        cores=inf_cfg["cores"],
        target_accept=inf_cfg["target_accept"],
        max_treedepth=int(inf_cfg.get("max_treedepth", 10)),
        initvals=initvals,
        init=init_scheme,
    )

    # Persist the RAW trace (pre-alignment) so the alignment is reproducible.
    raw_path = out_dir / "trace_raw.nc"
    try:
        trace.to_netcdf(str(raw_path))
        print(f"Saved raw trace to '{raw_path}'")
    except (ValueError, ImportError) as e:
        zarr_path = out_dir / "trace_raw.zarr"
        trace.to_zarr(str(zarr_path))
        print(f"NetCDF backend unavailable ({e}); saved raw trace to '{zarr_path}'")

    # Post-hoc alignment
    print("\nAligning chains (per-draw, to chain 0's mean labelling)...")
    aligned, perms = align_trace(trace)

    # within-chain switching diagnostic, saved as a table
    switch_df = switching_table(perms)
    switch_path = out_dir / "switching_table.csv"
    switch_df.to_csv(switch_path, index=False)
    print(f"Saved within-chain switching table to '{switch_path}'")

    # Persist the aligned trace
    aligned_path = out_dir / "trace_aligned.nc"
    try:
        aligned.to_netcdf(str(aligned_path))
        print(f"\nSaved aligned trace to '{aligned_path}'")
    except (ValueError, ImportError) as e:
        zarr_path = out_dir / "trace_aligned.zarr"
        aligned.to_zarr(str(zarr_path))
        print(f"NetCDF backend unavailable ({e}); saved aligned trace to '{zarr_path}'")

    # Summary computed on the aligned trace
    # r_hat / ess are only meaningful after alignment.
    summary_df = az.summary(aligned)
    summary_df.to_csv(out_dir / "inference_summary.csv")
    print(f"Saved inference summary (aligned) to '{out_dir / 'inference_summary.csv'}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run de novo signature-discovery Tree-HDP inference."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_denovo(cfg)


if __name__ == "__main__":
    main()