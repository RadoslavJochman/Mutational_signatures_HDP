"""
scripts/generate_data.py

Generate synthetic mutational data from a YAML config and save all
artefacts to the data/ directory.

Supports two modes, selected by the ``inference.model`` key in the config:
  - ``fixed_sig``  (default): uses Forward_HDP_Generator with fixed signatures
  - ``full_hdp``            : uses the stick-breaking HDP simulator

Usage
-----
    python scripts/generate_data.py --config configs/fixed_sig_experiment.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fractions import Fraction

# Make src/ importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, make_output_dir
from src.models.hdp_simulator import (
    Forward_HDP_Generator,
    HDP,
    generate_random_phylogenetic_forest,
)
from src.models.dirichlet_process import DirichletPrior
from src.plotting.plots import (
    plot_signatures_heatmap,
    plot_patient_counts,
    plot_node_signatures,
)

def run_fixed_sig(cfg: dict) -> None:
    sim = cfg["simulation"]
    result_path = sim["results_dir"]
    seed = sim["seed"]
    rng = np.random.default_rng(seed)

    # Creates experiment dir
    output_dir = make_output_dir(result_path, cfg.get("experiment_name", "experiment"))
    print(f"Output directory: {output_dir}")

    # Ground truth signatures
    K = sim["num_signatures"]
    signature_alpha = float(Fraction(sim["signature_alpha"]))
    signatures = rng.dirichlet(np.ones(96) * signature_alpha, size=K)
    sig_df = pd.DataFrame(
        signatures,
        index=[f"Signature_{i}" for i in range(K)],
        columns=[f"Channel_{i}" for i in range(96)],
    )
    sig_path = os.path.join(output_dir,"fixed_signatures.csv")
    sig_df.to_csv(sig_path)
    print(f"Saved {K} signatures to '{sig_path}'")

    # Generate random forest
    forest_cfg = sim["forest"]
    newick_string = generate_random_phylogenetic_forest(
        num_trees=forest_cfg["num_trees"],
        min_leaves=forest_cfg["min_leaves"],
        max_leaves=forest_cfg["max_leaves"],
        min_branch_length=forest_cfg["min_branch_length"],
        max_branch_length=forest_cfg["max_branch_length"],
        rng=rng,
    )
    trees_path = os.path.join(output_dir,"newick_string.nwk")
    with open(trees_path, "w") as f:
        f.write(newick_string)
    print(f"Saved tree topology to '{trees_path}'")

    # Forward simulation
    alpha = sim["alpha"]
    generator_seed = int(rng.integers(0, 2**31))
    generator = Forward_HDP_Generator(
        newick_string=newick_string,
        alpha=alpha,
        fixed_signatures=signatures,
        seed=generator_seed,
    )

    count_matrix = generator.get_mutation_count_matrix()
    true_activities = generator.get_true_activities()

    # Persist data
    count_m_path = os.path.join(output_dir,"mutation_count_matrix.csv")
    count_matrix.to_csv(count_m_path)
    print(f"Saved count matrix  {count_matrix.shape} to '{count_m_path}'")

    activities_path = os.path.join(output_dir,"true_activities.csv")
    activities_df = pd.DataFrame(true_activities).T
    activities_df.index.name = "node"
    activities_df.columns = [f"Signature_{i}" for i in range(K)]
    activities_df.to_csv(activities_path)
    print(f"Saved true activities to '{activities_path}'")

    params = {
        "seed": seed,
        "generator_seed": generator_seed,
        "alpha": alpha,
        "num_signatures": K,
        "signature_alpha": signature_alpha,
        **{f"forest_{k}": v for k, v in forest_cfg.items()},
    }
    params_path = os.path.join(output_dir,"ground_truth_params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved ground truth params to '{params_path}'")

    # Plots
    plot_dir = os.path.join(output_dir,"plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_signatures_heatmap(signatures, save_path=os.path.join(plot_dir,"heatmap_true_signatures.pdf"))
    plot_patient_counts(count_matrix, save_path=os.path.join(plot_dir,"heatmap_mutation_counts.pdf"))

    example_prefixes = [f"T{i}_" for i in range(1, 4)]
    for prefix in example_prefixes:
        label = next((l for l in true_activities if l.startswith(prefix)), None)
        if label:
            plot_node_signatures(
                activities=true_activities[label],
                signatures=signatures,
                node_label=label,
                top_n=3,
                alpha=alpha,
                save_path=os.path.join(plot_dir,f"true_signatures_activ_{label}.png"),
            )

    print("\nData generation complete.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic mutational data.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML experiment config.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    run_fixed_sig(cfg)


if __name__ == "__main__":
    main()