"""
scripts/run_inference.py

Run Tree-HDP inference on a mutation count matrix, driven entirely by a
YAML config. inference.model picks the mode:

    fixed   TreeHDP with known signatures (inference.data.fixed_signatures).
            No label switching: component k already is true signature k.
            Artefact: trace.nc.

    denovo  TreeHDP with signatures latent (inference.num_signatures = K).
            Label switching is real: any permutation of the K signatures,
            with the matching permutation of the activity components,
            leaves the likelihood unchanged, so signature index k has no
            fixed meaning across draws or chains. Convergence statistics
            on the raw trace are therefore meaningless, and a post-hoc
            per-draw alignment to chain 0's mean labelling is inserted
            between sampling and summary.
            Artefacts: trace_raw.nc (pre-alignment), trace_aligned.nc,
            switching_table.csv (per-chain switching diagnostic).

inference.model is required and must be set explicitly to 'fixed' or
'denovo'; there is no fallback. Configs written before this runner
collapse (when the mode was picked by which of the two former scripts,
run_inference.py / run_unknown_inference.py, you ran) need it added.
A presence-based guess (e.g. treating inference.num_signatures as an
implicit 'denovo' signal) was considered and rejected: de novo configs
also carry a data.fixed_signatures path (the generator's ground truth,
kept for scoring), so the two signals can disagree and silently
mis-dispatch -- exactly the failure the explicit key exists to prevent.

Usage
-----
    python scripts/run_inference.py --config configs/<experiment>.yaml
"""

import argparse
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.analysis import align
from src.config import load_config, make_output_dir
from src.models.hdp_inference import TreeHDP


def _resolve_model_name(inf_cfg: dict) -> str:
    """
    Resolve which TreeHDP mode to run: 'fixed' (S known) or 'denovo' (S
    latent). inference.model is required; there is no presence-based
    fallback (see the module docstring for why).

    Raises
    ------
    ValueError
        If inference.model is missing, or not 'fixed'/'denovo'.
    """
    if "model" not in inf_cfg:
        raise ValueError(
            "inference.model is required and was not set; add 'model: fixed' "
            "or 'model: denovo' to the config's inference block."
        )
    model_name = str(inf_cfg["model"]).lower()
    if model_name not in ("fixed", "denovo"):
        raise ValueError(
            f"Unknown inference.model '{model_name}'; use 'fixed' or 'denovo'."
        )
    return model_name


def _validate_inference_config(inf_cfg: dict, model_name: str) -> None:
    """
    Check that the config carries what `model_name` needs, and fail with a
    clear message before any data is loaded or any model is built. Assumes
    model_name is already a valid 'fixed'/'denovo' (see _resolve_model_name).

    Raises
    ------
    ValueError
        If 'fixed' has no inference.data.fixed_signatures path, or if
        'denovo' has no inference.num_signatures.
    """
    if model_name == "fixed":
        if not inf_cfg.get("data", {}).get("fixed_signatures"):
            raise ValueError(
                "inference.model: fixed requires inference.data.fixed_signatures "
                "(path to the known signature matrix)."
            )
    else:
        if inf_cfg.get("num_signatures") is None:
            raise ValueError(
                "inference.model: denovo requires inference.num_signatures (K)."
            )


def align_trace(trace, activity_var_prefix: str = "e_level"):
    """
    Per-draw alignment of a de novo (S latent) TreeHDP trace to chain 0's
    mean labelling.

    The `signatures` variable, the forest-pooled usage level `mu_level`,
    and the per-node activity variables (e_level_*) are permuted along
    their signature axis so that signature k means the same thing in every
    draw of every chain.

    Parameters
    ----------
    trace : arviz.InferenceData
        Raw trace from a de novo TreeHDP.sample().
    activity_var_prefix : str
        Prefix of the per-node activity variables to permute alongside the
        signatures. e_level_* has its last axis = K, the aligned
        interpretable activity. eta_level_*/z_level_*/z_root_* (the raw
        non-centred walk variables in unconstrained space) are not
        permuted: their r_hat is uninformative by construction regardless
        of alignment (see CLAUDE.md), so leaving them unaligned costs
        nothing.

    Returns
    -------
    aligned : arviz.InferenceData
        A copy of `trace` with `signatures`, `mu_level`, and e_level_*
        permuted.
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

    if "mu_level" in post.data_vars:
        mu = post["mu_level"].values  # (chains, draws, K)
        mu_aligned = np.empty_like(mu)
        for c in range(n_chains):
            for d in range(n_draws):
                mu_aligned[c, d] = mu[c, d][perms[c, d]]
        aligned.posterior["mu_level"] = aligned.posterior["mu_level"].copy(
            data=mu_aligned
        )

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
    Summarise within-chain label switching from the per-draw permutations.

    For each chain it counts how many distinct permutations were applied
    across draws and the fraction of draws that differ from that chain's
    most common permutation.

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


def run_inference(cfg: dict, model_name: str | None = None) -> None:
    """
    Run Tree-HDP inference end to end: build TreeHDP for `model_name` (or
    the config's resolved mode when `model_name` is None), sample, and
    write that mode's artefact contract. See the module docstring for the
    two contracts and how the mode is resolved.

    Alignment (align_trace, switching_table) is gated on the built model's
    `S_known`, not on the requested model_name, so the artefact contract
    always matches what was actually built.
    """
    inf_cfg = cfg["inference"]
    data_cfg = inf_cfg["data"]
    if model_name is None:
        model_name = _resolve_model_name(inf_cfg)
    _validate_inference_config(inf_cfg, model_name)

    out_dir = make_output_dir(
        cfg["experiment_root"], cfg.get("experiment_name", "experiment"), "results"
    )
    print(f"Output directory: {out_dir}")

    # Load data
    print("Loading data...")
    count_matrix = pd.read_csv(data_cfg["count_matrix"], index_col=0)
    with open(data_cfg["newick_string"]) as f:
        newick_string = f.read().strip()

    # Build model
    print("\nBuilding PyMC model...")
    if model_name == "fixed":
        signatures_df = pd.read_csv(data_cfg["fixed_signatures"], index_col=0)
        model = TreeHDP(
            newick_string=newick_string,
            data_matrix=count_matrix,
            fixed_signatures=signatures_df.values,
            priors=inf_cfg["priors"],
        )
    else:
        model = TreeHDP(
            newick_string=newick_string,
            data_matrix=count_matrix,
            num_signatures=int(inf_cfg["num_signatures"]),
            priors=inf_cfg["priors"],
        )
    print(f"Built TreeHDP (S_known={model.S_known}, K={model.K}).")

    # Optionally save the graphviz Bayesian network diagram
    try:
        pm.model_to_graphviz(model.model).render(
            str(out_dir / "bayesian_network"), format="png"
        )
        print(f"Saved Bayesian network graph in {out_dir}.")
    except Exception:
        print("Graphviz not available, skipping model graph.")

    # Sample
    print("\nStarting MCMC sampler...")
    trace = model.sample(
        draws=inf_cfg["draws"],
        tune=inf_cfg["tune"],
        chains=inf_cfg["chains"],
        cores=inf_cfg["cores"],
        target_accept=inf_cfg["target_accept"],
        max_treedepth=int(inf_cfg.get("max_treedepth", 10)),
    )

    if model.S_known:
        # Fixed signatures: no label switching, one trace, one summary.
        trace_path = out_dir / "trace.nc"
        trace.to_netcdf(str(trace_path))
        print(f"Saved trace to '{trace_path}'")
        summary_df = az.summary(trace)
    else:
        # De novo: persist the RAW trace (pre-alignment) so the alignment
        # is reproducible, then align and persist that too.
        raw_path = out_dir / "trace_raw.nc"
        try:
            trace.to_netcdf(str(raw_path))
            print(f"Saved raw trace to '{raw_path}'")
        except (ValueError, ImportError) as e:
            zarr_path = out_dir / "trace_raw.zarr"
            trace.to_zarr(str(zarr_path))
            print(f"NetCDF backend unavailable ({e}); saved raw trace to '{zarr_path}'")

        print("\nAligning chains (per-draw, to chain 0's mean labelling)...")
        aligned, perms = align_trace(trace)

        switch_df = switching_table(perms)
        switch_path = out_dir / "switching_table.csv"
        switch_df.to_csv(switch_path, index=False)
        print(f"Saved within-chain switching table to '{switch_path}'")

        aligned_path = out_dir / "trace_aligned.nc"
        try:
            aligned.to_netcdf(str(aligned_path))
            print(f"\nSaved aligned trace to '{aligned_path}'")
        except (ValueError, ImportError) as e:
            zarr_path = out_dir / "trace_aligned.zarr"
            aligned.to_zarr(str(zarr_path))
            print(f"NetCDF unavailable ({e}); saved aligned trace to '{zarr_path}'")

        # Summary computed on the aligned trace: r_hat/ess are only
        # meaningful post-alignment.
        summary_df = az.summary(aligned)

    summary_df.to_csv(out_dir / "inference_summary.csv")
    print(f"Saved inference summary to '{out_dir / 'inference_summary.csv'}'")


def run_fixed_sig(cfg: dict) -> None:
    """Run inference with S known, forcing model_name='fixed' regardless of
    cfg. Thin wrapper kept for direct/programmatic callers (e.g. tests)."""
    run_inference(cfg, model_name="fixed")


def run_denovo(cfg: dict) -> None:
    """Run inference with S latent, forcing model_name='denovo' regardless
    of cfg. Thin wrapper kept for direct/programmatic callers (e.g. tests)."""
    run_inference(cfg, model_name="denovo")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tree-HDP inference.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_inference(cfg)


if __name__ == "__main__":
    main()
