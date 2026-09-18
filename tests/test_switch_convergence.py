"""Slow statistical gate for the switch model (switch_model_plan.md section 4.3).

An easy hand-built forest: one tree of 11 nodes, long branches (2 to 3 per
edge), high burden (3000 mutations per node, multinomial), three catalogue
signatures with SBS36 exactly off in the whole subtree under B and clearly on
(level 0.25) everywhere else. Fixed mode, switching on, 2 chains x 500 draws
with a fixed sampler seed.

Asserts on convergence (split-Rhat, bulk ESS over e_level_*, sigma, lambda_*,
pi_root), on the absence of divergences, and on node-level on/off accuracy at
0.5. Tolerances are generous; this is a gate against regressions in the model
or the sampler geometry, not a benchmark. Run with `pytest -m slow`.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scaling_metrics import _convergence_vars, convergence_row  # noqa: E402

from src.analysis.analysis import node_variable_rows  # noqa: E402
from src.models.hdp_inference import TreeHDP  # noqa: E402

CATALOGUE = REPO_ROOT / "COSMIC_sig" / "cosmic_signatures.csv"
SIGNATURES = ["SBS1", "SBS5", "SBS36"]
NEWICK = "(((H:2.0,I:2.5)D:2.0,E:3.0)B:2.0,((J:2.0,K:2.0)F:2.5,G:3.0)C:2.0)A:0.0;"
NODES = list("ABCDEFGHIJK")
OFF_SUBTREE = {"B", "D", "E", "H", "I"}  # SBS36 exactly off here
BURDEN = 3000
ON_ACTIVITY = np.array([0.40, 0.35, 0.25])
OFF_ACTIVITY = np.array([0.55, 0.45, 0.0])

PRIORS = {
    "sigma_prior": "LogNorm",
    "sigma_prior_parm": {"mu": 0.0, "sigma": 1.0},
    "sigma_0": 1.0,
    "sigma_mu": 2.0,
}
SWITCHING = {
    "enabled": True,
    "branch_length_source": "newick",
    "lambda_on_prior": "LogNorm",
    "lambda_on_prior_parm": {"mu": -1.2, "sigma": 1.0},
    "lambda_off_prior": "LogNorm",
    "lambda_off_prior_parm": {"mu": -1.2, "sigma": 1.0},
    "pi_root_prior": "Beta",
    "pi_root_prior_parm": {"alpha": 1, "beta": 1},
}


def _easy_dataset(seed=20240607):
    rng = np.random.default_rng(seed)
    S = pd.read_csv(CATALOGUE, index_col=0).loc[SIGNATURES].values
    rows, truth = {}, {}
    for node in NODES:
        base = OFF_ACTIVITY if node in OFF_SUBTREE else ON_ACTIVITY
        jitter = rng.normal(scale=0.03, size=3) * (base > 0)
        e = np.clip(base + jitter, 0.0, None)
        e /= e.sum()
        rows[node] = rng.multinomial(BURDEN, e @ S)
        truth[node] = (base > 0).astype(int)
    counts = pd.DataFrame.from_dict(rows, orient="index")
    truth = pd.DataFrame.from_dict(truth, orient="index", columns=SIGNATURES)
    return S, counts, truth


@pytest.mark.slow
def test_switch_model_converges_and_recovers_on_easy_forest():
    S, counts, truth = _easy_dataset()
    model = TreeHDP(
        NEWICK,
        counts,
        priors=PRIORS,
        fixed_signatures=S,
        switching=SWITCHING,
        signature_names=SIGNATURES,
    )
    idata = model.sample(
        draws=500,
        tune=500,
        chains=2,
        cores=2,
        target_accept=0.9,
        random_seed=20240607,
    )
    post = idata.posterior

    divergences = int(idata.sample_stats["diverging"].values.sum())
    assert divergences == 0, f"{divergences} divergent transitions"

    names = _convergence_vars(post, "e_level", None)
    assert {"sigma", "lambda_on", "lambda_off", "pi_root"} <= set(names)
    row = convergence_row(idata, names)
    assert row["max_rhat"] < 1.05, row
    assert row["min_ess"] > 100, row

    K = len(SIGNATURES)
    perms = np.tile(np.arange(K), (post.sizes["chain"], 1))
    labels, P = node_variable_rows(post, "a_prob_level", NEWICK, perms)
    p_active = P.mean(axis=1)  # (n_nodes, K) over chains
    y = truth.loc[labels].values
    accuracy = ((p_active >= 0.5).astype(int) == y).mean()
    assert accuracy >= 0.9, f"node-level accuracy {accuracy:.3f}"
    # the clock signatures are on everywhere and must be called on
    assert (p_active[:, :2] >= 0.5).all()
