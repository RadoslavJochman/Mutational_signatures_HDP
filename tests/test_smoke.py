"""End-to-end smoke test: generate then infer then score on a tiny
fixed-signature config (configs/config_smoke.yaml). This is the manual fast
gate described in CLAUDE.md's Testing section -- it asserts the pipeline
runs, writes the expected files, and produces a trace with the expected
variables and finite recovery scores. It does not assert exact numbers.
"""

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "config_smoke.yaml"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import generate_data  # noqa: E402
import recovery_vs_truth  # noqa: E402
import run_inference  # noqa: E402

from src.analysis.analysis import build_forest, nodes_by_depth  # noqa: E402
from src.config import load_config  # noqa: E402


def test_generate_infer_score(tmp_workdir, monkeypatch):
    cfg = load_config(CONFIG_PATH)
    name = cfg["experiment_name"]

    # Redirect data/results into tmp_workdir so the run never touches the
    # repo's data/ or results/; everything else in the config is untouched.
    data_dir = tmp_workdir / "data"
    results_dir = tmp_workdir / "results"
    data_out = data_dir / name
    results_out = results_dir / name

    cfg["simulation"]["results_dir"] = str(data_dir)
    cfg["inference"]["results_dir"] = str(results_dir)
    cfg["inference"]["data"] = {
        "count_matrix": str(data_out / "mutation_count_matrix.csv"),
        "newick_string": str(data_out / "newick_string.nwk"),
        "tree_edges": str(data_out / "tree_edges.csv"),
        "fixed_signatures": str(data_out / "fixed_signatures.csv"),
        "true_activities": str(data_out / "true_activities.csv"),
        "ground_truth_params": str(data_out / "ground_truth_params.json"),
    }

    # ---- generate -----------------------------------------------------
    generate_data.run_generation(cfg)

    for fname in [
        "mutation_count_matrix.csv",
        "newick_string.nwk",
        "tree_edges.csv",
        "fixed_signatures.csv",
        "true_activities.csv",
        "ground_truth_params.json",
    ]:
        assert (data_out / fname).exists(), f"generate did not write {fname}"

    newick = (data_out / "newick_string.nwk").read_text().strip()
    true_signatures = pd.read_csv(data_out / "fixed_signatures.csv", index_col=0)
    true_activities = pd.read_csv(data_out / "true_activities.csv", index_col=0)
    K = true_signatures.shape[0]
    assert true_signatures.shape[1] == 96
    assert true_activities.shape[1] == K
    assert K == cfg["simulation"]["signatures"]["num_signatures"]

    # ---- infer ----------------------------------------------------------
    run_inference.run_fixed_sig(cfg)

    trace_path = results_out / "trace.nc"
    assert trace_path.exists(), "infer did not write trace.nc"
    assert (results_out / "inference_summary.csv").exists(), (
        "infer did not write inference_summary.csv"
    )

    idata = az.from_netcdf(trace_path)
    post = idata.posterior
    assert post.sizes["chain"] == cfg["inference"]["chains"]
    assert post.sizes["draw"] == cfg["inference"]["draws"]

    # Expected variables: sigma and mu_level, plus e_level_<d>/eta_level_<d>
    # for every depth actually present in the generated forest (full
    # K-dimensional ILR walk, no pinned coordinate). Depth 0 draws
    # z_root_0; every other depth draws z_level_<d>.
    dr = nodes_by_depth(build_forest(newick))
    depths = sorted({d for d, _ in dr})
    n_at_depth = {d: sum(1 for dd, _ in dr if dd == d) for d in depths}
    assert depths[0] == 0

    assert "sigma" in post.data_vars
    assert np.isfinite(post["sigma"].values).all()
    assert "mu_level" in post.data_vars
    assert post["mu_level"].shape[-1] == K
    assert np.isfinite(post["mu_level"].values).all()
    for d in depths:
        e_var, eta_var = f"e_level_{d}", f"eta_level_{d}"
        assert e_var in post.data_vars, f"missing {e_var}"
        assert eta_var in post.data_vars, f"missing {eta_var}"
        assert post[e_var].shape[-2:] == (n_at_depth[d], K)
        assert post[eta_var].shape[-2:] == (n_at_depth[d], K)
        assert np.isfinite(post[e_var].values).all()
        if d == 0:
            z_var = f"z_root_{d}"
        else:
            z_var = f"z_level_{d}"
        assert z_var in post.data_vars, f"missing {z_var}"
        assert post[z_var].shape[-2:] == (n_at_depth[d], K)

    # ---- score ------------------------------------------------------------
    recovery_out = results_out / "recovery"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recovery_vs_truth.py",
            "--trace",
            str(trace_path),
            "--true-activities",
            str(data_out / "true_activities.csv"),
            "--newick",
            str(data_out / "newick_string.nwk"),
            "--metrics",
            "cosine",
            "tv",
            "--outdir",
            str(recovery_out),
        ],
    )
    recovery_vs_truth.main()

    for fname in [
        "recovery_signatures.csv",
        "recovery_activities.csv",
        "recovery_summary.csv",
    ]:
        assert (recovery_out / fname).exists(), f"score did not write {fname}"

    activities = pd.read_csv(recovery_out / "recovery_activities.csv")
    assert len(activities) == true_activities.shape[0]

    cos = activities["act_cosine_mean"].to_numpy()
    # total_variation is defined as half the L1 distance (src/analysis/analysis.py),
    # so doubling it recovers the simplex L1, which runs 0..2.
    l1 = 2.0 * activities["act_tv_mean"].to_numpy()

    assert np.isfinite(cos).all()
    assert np.isfinite(l1).all()
    assert ((cos >= -1e-9) & (cos <= 1 + 1e-9)).all()
    assert ((l1 >= -1e-9) & (l1 <= 2 + 1e-9)).all()

    summary = pd.read_csv(recovery_out / "recovery_summary.csv").iloc[0]
    assert np.isfinite(summary["act_cosine_median_mean"])
    assert np.isfinite(summary["act_tv_median_mean"])
