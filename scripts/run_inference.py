"""
scripts/run_inference.py

Run Bayesian inference on a mutation count matrix using either
FixedSigHDP or FullSigHDP, driven entirely by a YAML config.

Results (trace, summary CSV, plots) are written to a dated experiment
directory under results/.

Usage
-----
    python scripts/run_inference.py --config configs/fixed_sig_experiment.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, make_output_dir
from src.models.hdp_inference import FixedSigHDP
from src.analysis.evaluation import evaluate_inference
from src.plotting.plots import plot_recovery_distributions

def run_fixed_sig(cfg: dict) -> None:
    inf_cfg = cfg["inference"]
    data_cfg = inf_cfg["data"]
    result_path= inf_cfg["results_dir"]

    out_dir = make_output_dir(result_path, cfg.get("experiment_name", "experiment"))
    print(f"Output directory: {out_dir}")

    #Load data
    print("Loading data...")
    count_matrix = pd.read_csv(data_cfg["count_matrix"], index_col=0)
    signatures_df = pd.read_csv(data_cfg["fixed_signatures"], index_col=0)
    fixed_signatures = signatures_df.values

    with open(data_cfg["newick_string"]) as f:
        newick_string = f.read().strip()

    # Build model
    print("\nBuilding PyMC model...")
    model = FixedSigHDP(
        newick_string=newick_string,
        data_matrix=count_matrix,
        fixed_signatures=fixed_signatures,
        priors = inf_cfg["priors"],
    )

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
    )

    # Persist trace
    trace_path = out_dir / "trace.nc"
    az.to_netcdf(trace, str(trace_path))
    print(f"Saved trace to '{trace_path}'")

    # Summary statistics
    summary_df = az.summary(trace)
    summary_df.to_csv(out_dir / "inference_summary.csv")
    print(f"Saved inference summary to '{out_dir / 'inference_summary.csv'}'")

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tree-HDP inference.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    run_fixed_sig(cfg)


if __name__ == "__main__":
    main()