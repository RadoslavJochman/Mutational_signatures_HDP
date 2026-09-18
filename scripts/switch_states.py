"""
switch_states.py

Post-hoc on/off state samples for a switch-model trace
(switch_model_plan.md section 6.2). Runs forward-filter backward-sample per
posterior draw (src/analysis/switch_posterior.py) and writes

    state_samples.npz   states: int8 (chains, draws_used, N, K); draw_idx;
                        node_labels (depth-major, the N axis); signature_index
    switch_edges.csv    tumour, parent, child, signature, p_gain, p_loss,
                        p_switch  (pooled over chains and draws; `signature`
                        is the index along the trace's signature axis)

Inputs
    --trace             trace.nc (fixed) or trace_aligned.nc (de novo; the
                        aligned trace is self-consistent per draw, the raw
                        one is not)
    --newick, --counts  the forest and count matrix the model was fit to
    --fixed-signatures  fixed mode only; omitted means S is read from the
                        trace's `signatures`
    --branch-length-source  newick | unit, as in inference.switching
    --thin              use every thin-th draw
    --seed              RNG seed for the state draws

The node marginal from these samples agrees with the mean of
a_prob_level_* to Monte Carlo error; tests/test_smoke_switch.py checks it.
Outputs are files only; nothing is interpreted on stdout.

Usage
    python scripts/switch_states.py --trace ../experiments/<name>/results/trace.nc \\
        --newick ../experiments/<name>/data/newick_string.nwk \\
        --counts ../experiments/<name>/data/mutation_count_matrix.csv \\
        --fixed-signatures ../experiments/<name>/data/fixed_signatures.csv \\
        --outdir ../experiments/<name>/results/switch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.switch_posterior import (
    compile_pruning,
    depth_arrays_from_files,
    edge_probabilities,
    edge_table,
    node_labels,
    sample_states,
)


def run(
    trace_path,
    newick_path,
    counts_path,
    outdir,
    fixed_signatures_path=None,
    branch_length_source="newick",
    thin=1,
    seed=0,
    always_on=(),
    tree_coupled=True,
):
    post = az.from_netcdf(trace_path).posterior
    counts = pd.read_csv(counts_path, index_col=0)
    newick = Path(newick_path).read_text().strip()

    always_on_idx = ()
    if fixed_signatures_path is not None:
        S_df = pd.read_csv(fixed_signatures_path, index_col=0)
        S = S_df.values
        K = S.shape[0]
        names = list(S_df.index)
        missing = [s for s in always_on if s not in names]
        if missing:
            raise SystemExit(f"--always-on names {missing} not in {names}")
        always_on_idx = tuple(sorted(names.index(s) for s in always_on))
        model, depth, nbd_list = depth_arrays_from_files(
            newick,
            counts,
            fixed_signatures=S,
            branch_length_source=branch_length_source,
        )
    else:
        if always_on:
            raise SystemExit("--always-on needs --fixed-signatures (fixed mode only)")
        S = None
        K = post["signatures"].shape[-2]
        model, depth, nbd_list = depth_arrays_from_files(
            newick, counts, num_signatures=K, branch_length_source=branch_length_source
        )

    compiled = compile_pruning(
        K, counts.shape[1], depth, always_on=always_on_idx, tree_coupled=tree_coupled
    )
    states, draw_idx = sample_states(
        post,
        depth,
        K,
        fixed_signatures=S,
        compiled=compiled,
        thin=thin,
        rng=np.random.default_rng(seed),
        always_on=always_on_idx,
    )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    labels = node_labels(model, nbd_list)
    np.savez_compressed(
        outdir / "state_samples.npz",
        states=states,
        draw_idx=draw_idx,
        node_labels=np.array(labels),
        signature_index=np.arange(K),
    )
    edges = edge_probabilities(states, edge_table(model, nbd_list, depth))
    edges.to_csv(outdir / "switch_edges.csv", index=False)
    return outdir / "state_samples.npz", outdir / "switch_edges.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--trace", required=True)
    ap.add_argument("--newick", required=True)
    ap.add_argument("--counts", required=True)
    ap.add_argument("--fixed-signatures", default=None)
    ap.add_argument(
        "--branch-length-source", default="newick", choices=["newick", "unit"]
    )
    ap.add_argument("--thin", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--always-on",
        nargs="*",
        default=[],
        help="signature names the model forced on (inference.switching.always_on); "
        "fixed mode only, must match the fitted model",
    )
    ap.add_argument(
        "--tree-coupled",
        choices=["true", "false"],
        default="true",
        help="inference.switching.tree_coupled of the fitted model; 'false' is the "
        "tree-free ablation (transitions are the root prior)",
    )
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    for p in run(
        a.trace,
        a.newick,
        a.counts,
        a.outdir,
        fixed_signatures_path=a.fixed_signatures,
        branch_length_source=a.branch_length_source,
        thin=a.thin,
        seed=a.seed,
        always_on=tuple(a.always_on),
        tree_coupled=a.tree_coupled == "true",
    ):
        print(p)


if __name__ == "__main__":
    main()
