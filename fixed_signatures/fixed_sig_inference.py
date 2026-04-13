import json
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import os

from fixed_sig_inference_model import Fixed_sig_HDP
from generate_data_fixed_sig import evaluate_inference


def main():
    output_dir = "../results"
    os.makedirs(output_dir, exist_ok=True)

    # --- Load ground truth params for evaluation ---
    with open("../data/ground_truth_params.json", "r") as f:
        ground_truth_params = json.load(f)
    true_alpha = ground_truth_params["alpha"]
    print(f"Ground truth alpha: {true_alpha}")

    # --- Load data ---
    print("Loading count matrix...")
    count_matrix = pd.read_csv("../data/mutation_count_matrix.csv", index_col=0)

    print("Loading fixed signatures...")
    if not os.path.exists("../data/fixed_signatures.csv"):
        raise FileNotFoundError(
            "Fixed signatures not found. Run generate_data_fixed_sig.py first."
        )
    signatures_df = pd.read_csv("../data/fixed_signatures.csv", index_col=0)
    fixed_signature_matrix = signatures_df.values
    print(f"Loaded {fixed_signature_matrix.shape[0]} signatures.")

    print("Loading tree topology...")
    with open("../data/ground_truth_trees.nwk", "r") as f:
        newick_string = f.read().strip()

    # --- Load ground truth activities for evaluation ---
    true_activities_df = pd.read_csv("../data/true_activities.csv", index_col=0)
    true_activities = {row: true_activities_df.loc[row].values
                       for row in true_activities_df.index}

    # --- Build model ---
    print("\nBuilding PyMC Tree-HDP model...")
    inference_model = Fixed_sig_HDP(
        newick_string=newick_string,
        data_matrix=count_matrix,
        fixed_signatures=fixed_signature_matrix
    )

    try:
        pm.model_to_graphviz(inference_model.model).render(
            os.path.join(output_dir, "bayesian_network"), format="png"
        )
        print("Saved Bayesian network graph.")
    except Exception:
        print("Graphviz not available, skipping model visualization.")

    # --- Sample ---
    print("\nStarting MCMC sampler...")
    trace = inference_model.sample(
        draws=2000,
        tune=1000,
        chains=4,
        cores=4,
        target_accept=0.99
    )

    # --- Persist trace ---
    trace_path = os.path.join(output_dir, "trace.nc")
    az.to_netcdf(trace, trace_path)
    print(f"Saved trace to {trace_path}")

    # --- Summary statistics ---
    summary_df = az.summary(trace)
    summary_df.to_csv(os.path.join(output_dir, "inference_summary.csv"))

    # --- Global parameter recovery ---
    print("\n--- GLOBAL PARAMETER RECOVERY ---")
    alpha_summary = az.summary(trace, var_names=["shared_alpha"])
    print(alpha_summary)
    inferred_alpha_mean = trace.posterior["shared_alpha"].values.mean()
    print(f"\nTrue alpha:     {true_alpha}")
    print(f"Inferred alpha: {inferred_alpha_mean:.3f}  "
          f"({'good' if abs(inferred_alpha_mean - true_alpha) / true_alpha < 0.1 else 'poor'} recovery)")

    print("\nGlobal cohort baseline (e_0):")
    print(az.summary(trace, var_names=["e_0"]))

    # --- Trace plots ---
    az.plot_trace(trace, var_names=["shared_alpha", "e_0"], combined=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trace_plot_global_params.png"), dpi=300)
    plt.close()

    # --- Node-level activity recovery ---
    print("\nEvaluating node-level activity recovery...")
    eval_df = evaluate_inference(inference_model, true_activities)
    eval_df.to_csv(os.path.join(output_dir, "activity_recovery.csv"))

    print(eval_df.describe())
    print(f"\nMean cosine similarity: {eval_df['cosine_similarity'].mean():.3f}")
    print(f"Mean MAE:               {eval_df['mae'].mean():.4f}")

    # --- Plot recovery distributions ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    eval_df["cosine_similarity"].hist(bins=30, ax=axes[0])
    axes[0].set_title("Cosine Similarity: True vs Inferred Activities")
    axes[0].set_xlabel("Cosine Similarity")
    axes[0].set_ylabel("Count")

    eval_df["mae"].hist(bins=30, ax=axes[1])
    axes[1].set_title("MAE: True vs Inferred Activities")
    axes[1].set_xlabel("MAE")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "activity_recovery.png"), dpi=300)
    plt.close()

    print(f"\nPipeline complete. All results saved to '{output_dir}/'.")


if __name__ == "__main__":
    main()