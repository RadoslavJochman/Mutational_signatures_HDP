"""Tests for scripts/run_inference.py's registry-driven de novo alignment.

A fabricated two-chain trace: chain 1 is chain 0 with a known permutation
applied along the signature axis of every registered variable. After
align_trace every registered variable must be identical across the two
chains, an unregistered variable must be untouched, and the returned
permutations must undo the known one (switch_model_plan.md section 5).
"""

import sys
from pathlib import Path

import arviz as az
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_inference import align_trace, switching_table  # noqa: E402

K, C, N, DRAWS = 3, 5, 2, 6
PERM = np.array([2, 0, 1])
REGISTRY = {
    "signatures": 0,
    "mu_level": -1,
    "z_root_0": -1,
    "eta_level_0": -1,
    "e_level_0": -1,
    "lambda_on": -1,
    "lambda_off": -1,
    "pi_root": -1,
    "a_prob_level_0": -1,
}


def _fabricated_trace(seed=0):
    rng = np.random.default_rng(seed)
    # well separated signature rows so the Hungarian match is unambiguous
    base = np.eye(K, C) + 0.05 * rng.random((K, C))
    chain0 = {
        "signatures": np.stack(
            [
                (base + 0.01 * rng.random((K, C))) / (base.sum(1, keepdims=True))
                for _ in range(DRAWS)
            ]
        ),
        "mu_level": rng.normal(size=(DRAWS, K)),
        "z_root_0": rng.normal(size=(DRAWS, N, K)),
        "eta_level_0": rng.normal(size=(DRAWS, N, K)),
        "e_level_0": rng.dirichlet(np.ones(K), size=(DRAWS, N)),
        "lambda_on": rng.lognormal(size=(DRAWS, K)),
        "lambda_off": rng.lognormal(size=(DRAWS, K)),
        "pi_root": rng.uniform(size=(DRAWS, K)),
        "a_prob_level_0": rng.uniform(size=(DRAWS, N, K)),
        "sigma": rng.lognormal(size=(DRAWS,)),  # no signature axis
    }
    posterior = {}
    for name, arr0 in chain0.items():
        if name in REGISTRY:
            axis = REGISTRY[name]
            arr1 = np.take(arr0, PERM, axis=axis + 1 if axis >= 0 else axis)
        else:
            arr1 = arr0
        posterior[name] = np.stack([arr0, arr1])  # (chains=2, draws, ...)
    return az.from_dict(posterior=posterior)


def test_align_trace_undoes_known_permutation_on_every_registered_variable():
    trace = _fabricated_trace()
    aligned, perms = align_trace(trace, REGISTRY)

    assert perms.shape == (2, DRAWS, K)
    np.testing.assert_array_equal(perms[0], np.tile(np.arange(K), (DRAWS, 1)))
    # chain 1 draws were chain 0 draws indexed by PERM, so the aligning
    # permutation is PERM's inverse, the same for every draw
    np.testing.assert_array_equal(perms[1], np.tile(np.argsort(PERM), (DRAWS, 1)))

    post = aligned.posterior
    for name in REGISTRY:
        np.testing.assert_allclose(
            post[name].values[1], post[name].values[0], err_msg=name
        )
    # unregistered variable untouched
    np.testing.assert_array_equal(post["sigma"].values, trace.posterior["sigma"].values)
    # and the raw trace was not modified in place
    assert not np.allclose(
        trace.posterior["e_level_0"].values[1], trace.posterior["e_level_0"].values[0]
    )


def test_align_trace_skips_registered_variables_absent_from_trace():
    trace = _fabricated_trace()
    registry = dict(REGISTRY, not_in_trace=-1)
    aligned, _ = align_trace(trace, registry)
    assert "not_in_trace" not in aligned.posterior.data_vars


def test_switching_table_counts_distinct_perms():
    perms = np.zeros((2, 4, K), dtype=int)
    perms[:] = np.arange(K)
    perms[1, 3] = PERM
    table = switching_table(perms)
    assert list(table["n_distinct_perms"]) == [1, 2]
    assert table["switched_fraction"].iloc[1] == 0.25
