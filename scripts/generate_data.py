"""
scripts/generate_data.py

Generate synthetic tree-structured mutational data from a YAML config and
save all artefacts to the results directory.

This drives `TreeSignatureGenerator` -- the Dirichlet-walk approximation of
the tree-HDP.  Signatures may be synthesized or
loaded from a real COSMIC-style CSV.

Usage
-----
    python scripts/generate_data.py --config configs/experiment_config.yaml
"""

import argparse
import json
import os
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, make_output_dir
from src.models.hdp_simulator import (
    TreeSignatureGenerator,
    generate_random_forest,
)


def _as_float(value) -> float:
    """Accept plain numbers or fraction strings like '1/96' from YAML."""
    if isinstance(value, str):
        return float(Fraction(value))
    return float(value)


def _load_signature_csv(path: str) -> np.ndarray:
    """Load a (K, 96) signature matrix from CSV (signatures as rows)."""
    df = pd.read_csv(path, index_col=0)
    arr = df.to_numpy(dtype=float)
    if arr.shape[1] != 96:
        raise ValueError(
            f"signature file '{path}' must have 96 channel columns, "
            f"got {arr.shape[1]}"
        )
    return arr


def run_generation(cfg: dict) -> None:
    sim = cfg["simulation"]
    rng = np.random.default_rng(sim["seed"])

    output_dir = make_output_dir(
        sim["results_dir"], cfg.get("experiment_name", "experiment")
    )
    print(f"Output directory: {output_dir}")

    sig_cfg = sim["signatures"]
    source = sig_cfg.get("source", "synthesize")

    if source == "load":
        signatures = _load_signature_csv(sig_cfg["path"])
        K = signatures.shape[0]
        gen_kwargs = {"signatures": signatures}
        print(f"Loaded {K} signatures from '{sig_cfg['path']}'")
    elif source == "synthesize":
        K = sig_cfg["num_signatures"]
        gen_kwargs = {
            "n_signatures": K,
            "signature_correlation": _as_float(
                sig_cfg.get("correlation", 0.0)
            ),
        }
        signatures = None
        print(f"Synthesizing {K} signatures "
              f"(correlation={gen_kwargs['signature_correlation']})")
    else:
        raise ValueError(
            f"signatures.source must be 'synthesize' or 'load', got '{source}'"
        )

    forest_cfg = sim["forest"]
    newick_string = generate_random_forest(
        num_trees=forest_cfg["num_trees"],
        min_leaves=forest_cfg["min_leaves"],
        max_leaves=forest_cfg["max_leaves"],
        rng=rng,
        min_branch_length=forest_cfg["min_branch_length"],
        max_branch_length=forest_cfg["max_branch_length"],
    )
    trees_path = os.path.join(output_dir, "newick_string.nwk")
    with open(trees_path, "w") as f:
        f.write(newick_string)
    print(f"Saved tree topology to '{trees_path}'")

    generator_seed = int(rng.integers(0, 2 ** 31))
    generator = TreeSignatureGenerator(
        newick_forest=newick_string,
        alpha=_as_float(sim["alpha"]),
        alpha_0=_as_float(sim.get("alpha_0", 1.0)),
        lam=_as_float(sim["lam"]),
        nb_dispersion=(
            None if sim.get("nb_dispersion") in (None, "none")
            else _as_float(sim["nb_dispersion"])
        ),
        activity_sparsity=_as_float(sim.get("activity_sparsity", 0.0)),
        signature_dropout=_as_float(sim.get("signature_dropout", 0.0)),
        seed=generator_seed,
        **gen_kwargs,
    )

    summary = generator.summary()
    print("Generated:", summary)

    used_signatures = generator.get_true_signatures()
    K = used_signatures.shape[0]
    sig_df = pd.DataFrame(
        used_signatures,
        index=[f"Signature_{i}" for i in range(K)],
        columns=[f"Channel_{i}" for i in range(96)],
    )
    sig_path = os.path.join(output_dir, "fixed_signatures.csv")
    sig_df.to_csv(sig_path)
    print(f"Saved {K} signatures to '{sig_path}'")

    count_matrix = generator.get_mutation_count_matrix()
    count_m_path = os.path.join(output_dir, "mutation_count_matrix.csv")
    count_matrix.to_csv(count_m_path)
    print(f"Saved count matrix {count_matrix.shape} to '{count_m_path}'")

    true_activities = generator.get_true_activities()
    activities_df = pd.DataFrame(true_activities).T
    activities_df.index.name = "node"
    activities_df.columns = [f"Signature_{i}" for i in range(K)]
    activities_path = os.path.join(output_dir, "true_activities.csv")
    activities_df.to_csv(activities_path)
    print(f"Saved true activities {activities_df.shape} to '{activities_path}'")

    edges_df = generator.get_tree_edges()
    edges_path = os.path.join(output_dir, "tree_edges.csv")
    edges_df.to_csv(edges_path, index=False)
    print(f"Saved {len(edges_df)} tree edges to '{edges_path}'")

    # Ground-truth params and dataset summary.
    params = {
        "seed": sim["seed"],
        "generator_seed": generator_seed,
        "alpha": _as_float(sim["alpha"]),
        "alpha_0": _as_float(sim.get("alpha_0", 1.0)),
        "lam": _as_float(sim["lam"]),
        "nb_dispersion": sim.get("nb_dispersion"),
        "num_signatures": K,
        "true_K": generator.true_K,
        "active_signature_indices":
            generator.get_active_signature_indices().tolist(),
        "activity_sparsity": _as_float(sim.get("activity_sparsity", 0.0)),
        "signature_dropout": _as_float(sim.get("signature_dropout", 0.0)),
        "signature_source": source,
        "signature_correlation":
            gen_kwargs.get("signature_correlation"),
        **{f"forest_{k}": v for k, v in forest_cfg.items()},
        "dataset_summary": summary,
    }
    params_path = os.path.join(output_dir, "ground_truth_params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved ground truth params to '{params_path}'")

    if cfg.get("simulation", {}).get("make_plots", True):
        try:
            from src.plotting.plots import (
                plot_signatures_heatmap,
                plot_patient_counts,
                plot_node_signatures,
            )
            plot_dir = os.path.join(output_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            plot_signatures_heatmap(
                used_signatures,
                save_path=os.path.join(plot_dir, "heatmap_true_signatures.pdf"),
            )
            plot_patient_counts(
                count_matrix,
                save_path=os.path.join(plot_dir, "heatmap_mutation_counts.pdf"),
            )
            for prefix in [f"T{i}_" for i in range(1, 4)]:
                label = next(
                    (l for l in true_activities if l.startswith(prefix)), None
                )
                if label:
                    plot_node_signatures(
                        activities=true_activities[label],
                        signatures=used_signatures,
                        node_label=label,
                        top_n=3,
                        alpha=_as_float(sim["alpha"]),
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