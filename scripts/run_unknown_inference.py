"""
scripts/run_unknown_sig_inference.py

Run Bayesian inference with UnknownSigHDP: jointly infers both mutational
signatures and per-node activities from a mutation count matrix.

You only need to specify K_max (an upper bound on the number of signatures)
in the YAML config.  Signatures not needed by the data will have their
activity components shrink toward zero, so the effective number is learned
from the data.

Results written to results/<experiment_name>_<YYYY-MM-DD>/:
    trace.nc                – full ArviZ InferenceData (MCMC samples)
    inference_summary.csv   – ArviZ summary (mean, sd, ESS, r_hat)
    inferred_signatures.csv – posterior-mean signatures, shape (K_max, 96)
    bayesian_network.png    – graphviz model graph (if graphviz is installed)

Usage
-----
    python scripts/run_unknown_sig_inference.py \
        --config configs/unknown_sig_experiment.yaml

Expected YAML structure
-----------------------
    experiment_name: unknown_sig_experiment

    inference:
      results_dir: results/

      data:
        count_matrix:  data/mutation_count_matrix.csv
        newick_string: data/ground_truth_trees.nwk

      K_max: 15

      priors:
        alpha_prior:
          distribution: LogNormal
          mu: 2.5
          sigma: 1.0
        eta: 0.1          # Dirichlet concentration for signature prior

      draws: 2000
      tune: 1000
      chains: 4
      cores: 4
      target_accept: 0.95
"""

import argparse
import sys
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, make_output_dir
from src.models.hdp_inference import UnknownSigHDP
from src.plotting.plots import plot_signatures_heatmap


def run_unknown_sig(cfg: dict) -> None:
    inf_cfg  = cfg["inference"]
    data_cfg = inf_cfg["data"]

    out_dir = make_output_dir(
        inf_cfg["results_dir"],
        cfg.get("experiment_name", "experiment"),
    )
    print(f"Output directory: {out_dir}")

    print("Loading data...")
    count_matrix = pd.read_csv(data_cfg["count_matrix"], index_col=0)

    with open(data_cfg["newick_string"]) as f:
        newick_string = f.read().strip()

    K_max = int(inf_cfg["K_max"])
    print(f"Count matrix:    {count_matrix.shape}")
    print(f"K_max:           {K_max}")

    print("\nBuilding PyMC model (UnknownSigHDP)...")
    model = UnknownSigHDP(
        newick_string=newick_string,
        data_matrix=count_matrix,
        K_max=K_max,
        priors=inf_cfg["priors"],
    )
    print(f"Nodes in model:  {len(model.node_index_map)}")

    # Optional: save the graphviz Bayesian network diagram
    try:
        pm.model_to_graphviz(model.model).render(
            str(out_dir / "bayesian_network"), format="png"
        )
        print(f"Saved Bayesian network graph to '{out_dir}'.")
    except Exception:
        print("Graphviz not available, skipping model graph.")

    print("\nStarting MCMC sampler...")
    trace = model.sample(
        draws=inf_cfg["draws"],
        tune=inf_cfg["tune"],
        chains=inf_cfg["chains"],
        cores=inf_cfg["cores"],
        target_accept=inf_cfg["target_accept"],
    )

    trace_path = out_dir / "trace.nc"
    az.to_netcdf(trace, str(trace_path))
    print(f"Saved trace to '{trace_path}'")

    summary_df = az.summary(trace)
    summary_df.to_csv(out_dir / "inference_summary.csv")
    print(f"Saved inference summary to '{out_dir / 'inference_summary.csv'}'")

    posterior_sigs = model.get_posterior_signatures(mean=True)
    sig_df = pd.DataFrame(
        posterior_sigs,
        index=[f"Signature_{k}" for k in range(K_max)],
        columns=count_matrix.columns,
    )
    sig_path = out_dir / "inferred_signatures.csv"
    sig_df.to_csv(sig_path)
    print(f"Saved inferred signatures to '{sig_path}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run UnknownSigHDP inference (joint signature + activity learning)."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_unknown_sig(cfg)


if __name__ == "__main__":
    main()
