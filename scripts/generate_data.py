"""
scripts/generate_data.py

Generate synthetic tree-structured mutational data from a YAML config and
save all artefacts to the results directory.

This drives `TreeSwitchDriftGenerator`, the switch-plus-drift forward
simulator (simulator_spec.md). `cfg["simulation"]` is passed straight to the
generator: it already matches the plain-dict schema the generator expects
(simulator_spec.md section 3), apart from the config-layer keys handled here
(`seed`, `make_plots`, `n_seeds`), which the generator itself ignores.

Writes to `<experiment_root>/<experiment_name>/data/` (and `.../plots/` when
`make_plots` is set). `experiment_root` and `experiment_name` are top-level
config keys, shared with run_inference.py, so both scripts always agree on
which experiment directory they are writing into.

Usage
-----
    python scripts/generate_data.py --config configs/experiment_config.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, make_output_dir
from src.models.hdp_simulator import TreeSwitchDriftGenerator


def run_generation(cfg: dict) -> None:
    sim = cfg["simulation"]
    exp_root = cfg["experiment_root"]
    exp_name = cfg.get("experiment_name", "experiment")

    output_dir = make_output_dir(exp_root, exp_name, "data")
    print(f"Output directory: {output_dir}")

    generator = TreeSwitchDriftGenerator(sim)
    K = generator.K
    print(
        f"Generated {generator.cfg.forest.n_trees} trees, K={K} signatures "
        f"({', '.join(generator.signature_names)})"
    )

    signatures = generator.get_true_signatures()
    sig_path = os.path.join(output_dir, "fixed_signatures.csv")
    signatures.to_csv(sig_path)
    print(f"Saved {K} signatures to '{sig_path}'")

    true_activities = generator.get_true_activities()
    true_activities.index.name = "node"
    activities_path = os.path.join(output_dir, "true_activities.csv")
    true_activities.to_csv(activities_path)
    print(f"Saved true activities {true_activities.shape} to '{activities_path}'")

    true_active_sets = generator.get_true_active_sets()
    true_active_sets.index.name = "node"
    active_sets_path = os.path.join(output_dir, "true_active_sets.csv")
    true_active_sets.to_csv(active_sets_path)
    print(f"Saved true active sets {true_active_sets.shape} to '{active_sets_path}'")

    edges_df = generator.get_tree_edges()
    edges_path = os.path.join(output_dir, "tree_edges.csv")
    edges_df.to_csv(edges_path, index=False)
    print(f"Saved {len(edges_df)} tree edges to '{edges_path}'")

    count_matrix = generator.get_mutation_count_matrix()
    count_m_path = os.path.join(output_dir, "mutation_count_matrix.csv")
    count_matrix.to_csv(count_m_path)
    print(f"Saved count matrix {count_matrix.shape} to '{count_m_path}'")

    newick_string = generator.get_newick_forest()
    trees_path = os.path.join(output_dir, "newick_string.nwk")
    with open(trees_path, "w") as f:
        f.write(newick_string)
    print(f"Saved tree topology to '{trees_path}'")

    params = generator.get_ground_truth_params()
    params_path = os.path.join(output_dir, "ground_truth_params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved ground truth params to '{params_path}'")

    if sim.get("make_plots", True):
        try:
            from src.plotting.plots import (
                plot_node_signatures,
                plot_patient_counts,
                plot_signatures_heatmap,
            )

            plot_dir = make_output_dir(exp_root, exp_name, "plots")
            plot_signatures_heatmap(
                signatures.to_numpy(),
                save_path=os.path.join(plot_dir, "heatmap_true_signatures.pdf"),
            )
            plot_patient_counts(
                count_matrix,
                save_path=os.path.join(plot_dir, "heatmap_mutation_counts.pdf"),
            )
            for prefix in [f"T{i}_" for i in range(1, 4)]:
                label = next(
                    (lbl for lbl in true_activities.index if lbl.startswith(prefix)),
                    None,
                )
                if label:
                    plot_node_signatures(
                        activities=true_activities.loc[label].to_numpy(),
                        signatures=signatures.to_numpy(),
                        node_label=label,
                        top_n=min(3, K),
                        save_path=os.path.join(
                            plot_dir, f"true_signatures_activ_{label}.png"
                        ),
                    )
            print(f"Saved plots to '{plot_dir}'")
        except ImportError as e:
            print(f"Skipping plots (plotting module unavailable): {e}")

    print("\nData generation complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic tree-structured mutational data."
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML experiment config."
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_generation(cfg)


if __name__ == "__main__":
    main()
