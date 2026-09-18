"""End-to-end smoke test for the switch model: generate then infer, on the
tiny switching config (experiments/smoke_switch/config.yaml), in both fixed
and de novo mode. Mirrors tests/test_smoke.py's structure and redirection
into a temp working directory; see that file for the plain (switching off)
pipeline.

This is the manual fast gate for switch_model_plan.md item 3 (Checkpoint 2):
it asserts the pipeline runs in both modes, writes the expected files, and
produces a trace with the switch model's variables (lambda_on, lambda_off,
pi_root, a_prob_level_*, e_level_*) at the right shapes and finite. It does
not assert exact numbers, and does not yet call switch_states.py /
switch_recovery.py, which are section 6/item 5 of switch_model_plan.md and
not built yet.
"""

import copy
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "experiments" / "smoke_switch" / "config.yaml"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import generate_data  # noqa: E402
import run_inference  # noqa: E402

from src.analysis.analysis import build_forest, nodes_by_depth  # noqa: E402
from src.config import load_config  # noqa: E402


def _check_switch_trace(idata, depths, n_at_depth, K, chains, draws):
    post = idata.posterior
    assert post.sizes["chain"] == chains
    assert post.sizes["draw"] == draws

    for name in ("lambda_on", "lambda_off", "pi_root"):
        assert name in post.data_vars, f"missing {name}"
        assert post[name].shape[-1] == K
        assert np.isfinite(post[name].values).all()

    assert "observations" not in post.data_vars

    for d in depths:
        a_var, e_var = f"a_prob_level_{d}", f"e_level_{d}"
        assert a_var in post.data_vars, f"missing {a_var}"
        assert e_var in post.data_vars, f"missing {e_var}"
        assert post[a_var].shape[-2:] == (n_at_depth[d], K)
        assert post[e_var].shape[-2:] == (n_at_depth[d], K)

        a_prob = post[a_var].values
        assert np.isfinite(a_prob).all()
        assert ((a_prob >= -1e-9) & (a_prob <= 1 + 1e-9)).all()

        e_level = post[e_var].values
        assert np.isfinite(e_level).all()
        np.testing.assert_allclose(e_level.sum(axis=-1), 1.0, atol=1e-6)


def test_generate_infer_switch_both_modes(tmp_workdir):
    cfg = load_config(CONFIG_PATH)
    name = cfg["experiment_name"]

    exp_dir = tmp_workdir / "experiments" / name
    data_out = exp_dir / "data"

    cfg["experiment_root"] = str(tmp_workdir / "experiments")
    cfg["simulation"]["repertoire"]["path"] = str(
        REPO_ROOT / "COSMIC_sig" / "cosmic_signatures.csv"
    )
    cfg["inference"]["data"] = {
        "count_matrix": str(data_out / "mutation_count_matrix.csv"),
        "newick_string": str(data_out / "newick_string.nwk"),
        "tree_edges": str(data_out / "tree_edges.csv"),
        "fixed_signatures": str(data_out / "fixed_signatures.csv"),
        "true_activities": str(data_out / "true_activities.csv"),
        "ground_truth_params": str(data_out / "ground_truth_params.json"),
    }

    generate_data.run_generation(cfg)
    for fname in [
        "mutation_count_matrix.csv",
        "newick_string.nwk",
        "fixed_signatures.csv",
        "true_activities.csv",
    ]:
        assert (data_out / fname).exists(), f"generate did not write {fname}"

    newick = (data_out / "newick_string.nwk").read_text().strip()
    true_signatures = pd.read_csv(data_out / "fixed_signatures.csv", index_col=0)
    K = true_signatures.shape[0]
    assert K == 3

    dr = nodes_by_depth(build_forest(newick))
    depths = sorted({d for d, _ in dr})
    n_at_depth = {d: sum(1 for dd, _ in dr if dd == d) for d in depths}

    chains = cfg["inference"]["chains"]
    draws = cfg["inference"]["draws"]

    # Fixed mode: separate experiment_name so its results dir does not
    # collide with de novo's (both read the same already-generated data).
    cfg_fixed = copy.deepcopy(cfg)
    cfg_fixed["experiment_name"] = f"{name}_fixed"
    run_inference.run_fixed_sig(cfg_fixed)
    results_fixed = tmp_workdir / "experiments" / f"{name}_fixed" / "results"
    assert (results_fixed / "trace.nc").exists()
    assert (results_fixed / "inference_summary.csv").exists()
    idata_fixed = az.from_netcdf(results_fixed / "trace.nc")
    _check_switch_trace(idata_fixed, depths, n_at_depth, K, chains, draws)

    # De novo mode.
    cfg_denovo = copy.deepcopy(cfg)
    cfg_denovo["experiment_name"] = f"{name}_denovo"
    run_inference.run_denovo(cfg_denovo)
    results_denovo = tmp_workdir / "experiments" / f"{name}_denovo" / "results"
    assert (results_denovo / "trace_raw.nc").exists()
    assert (results_denovo / "trace_aligned.nc").exists()
    assert (results_denovo / "switching_table.csv").exists()
    assert (results_denovo / "inference_summary.csv").exists()
    perms = np.load(results_denovo / "perms.npy")
    assert perms.shape == (chains, draws, K)
    idata_denovo = az.from_netcdf(results_denovo / "trace_aligned.nc")
    _check_switch_trace(idata_denovo, depths, n_at_depth, K, chains, draws)
    assert "signatures" in idata_denovo.posterior.data_vars
    assert idata_denovo.posterior["signatures"].shape[-2:] == (K, 96)
