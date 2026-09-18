"""
Tests for the `switching` argument on `src.models.hdp_inference.TreeHDP`.

Section references are to `switch_model_plan.md`. `test_bridge_*` is the
model-level bridge test from section 3.4: with everything forced on
(lambda -> 0, pi -> 1), the switch model's marginal likelihood must equal
the plain multinomial's. It evaluates `switch_pruning.prune` directly on
the depth arrays and eta TreeHDP itself builds (the plan's "or evaluate the
pruning function directly" option), which also exercises the
model-integration wiring (node ordering, branch-length normalisation,
counts) -- switch_pruning's own correctness is already covered by
`tests/test_switch_pruning.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytest
from scipy.stats import multinomial

from src.models.hdp_inference import TreeHDP

# A tiny forest: A -> (B -> (C, D), E). E has no data (unobserved leaf); A
# is both a root and observed, so both cases are exercised in one forest.
NEWICK = "((C:0.5,D:0.7)B:0.3,E:1.0)A:0.0;"
NEWICK_NO_LENGTHS = "((C,D)B,E)A;"

PRIORS = {
    "sigma_prior": "LogNorm",
    "sigma_prior_parm": {"mu": 0.0, "sigma": 1.0},
    "sigma_0": 1.0,
    "sigma_mu": 2.0,
    "beta": 0.5,
}


def _switching(**overrides):
    cfg = {
        "enabled": True,
        "lambda_on_prior": "LogNorm",
        "lambda_on_prior_parm": {"mu": -1.2, "sigma": 1.0},
        "lambda_off_prior": "LogNorm",
        "lambda_off_prior_parm": {"mu": -1.2, "sigma": 1.0},
        "pi_root_prior": "Beta",
        "pi_root_prior_parm": {"alpha": 1, "beta": 1},
    }
    cfg.update(overrides)
    return cfg


def _toy_inputs(K=3, C=6, seed=0, newick=NEWICK):
    rng = np.random.default_rng(seed)
    S = rng.dirichlet(np.ones(C), size=K)
    data = pd.DataFrame(rng.integers(1, 10, size=(4, C)), index=["A", "B", "C", "D"])
    return newick, data, K, C, S


def _build_fixed(switching=None, newick=NEWICK, K=3):
    newick, data, K, C, S = _toy_inputs(K=K, newick=newick)
    return TreeHDP(newick, data, priors=PRIORS, fixed_signatures=S, switching=switching)


def _build_denovo(switching=None, K=3):
    newick, data, K, C, S = _toy_inputs(K=K)
    return TreeHDP(newick, data, priors=PRIORS, num_signatures=K, switching=switching)


# ---------------------------------------------------------------------------
# Disabled (default) path must build exactly today's model.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("switching", [None, {"enabled": False}])
def test_switching_disabled_matches_original_model(switching):
    model = _build_fixed(switching=switching)
    var_names = {v.name for v in model.model.free_RVs}
    assert var_names == {"sigma", "mu_level", "z_root_0", "z_level_1", "z_level_2"}
    assert "observations" in {v.name for v in model.model.observed_RVs}
    assert model.model.potentials == []
    det_names = {v.name for v in model.model.deterministics}
    assert "a_prob_level_0" not in det_names

    eta0, e0 = pm.draw(
        [model.model["eta_level_0"], model.model["e_level_0"]], random_seed=0
    )
    softmax_eta0 = np.exp(eta0 - eta0.max(axis=-1, keepdims=True))
    softmax_eta0 /= softmax_eta0.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(e0, softmax_eta0, atol=1e-10)


# ---------------------------------------------------------------------------
# Enabled path: variable names, shapes, no plain multinomial.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build", [_build_fixed, _build_denovo])
def test_switching_enabled_variable_names_and_shapes(build):
    model = build(switching=_switching())
    free_names = {v.name for v in model.model.free_RVs}
    assert {"lambda_on", "lambda_off", "pi_root"} <= free_names
    assert "observations" not in {v.name for v in model.model.observed_RVs}
    assert {v.name for v in model.model.potentials} == {"switch_loglik"}

    K = model.K
    nodes_by_depth = model._get_nodes_by_depth()
    max_depth = max(nodes_by_depth)
    n_by_depth = {d: len(nodes_by_depth[d]) for d in range(max_depth + 1)}

    draw_vars = [
        model.model["lambda_on"],
        model.model["lambda_off"],
        model.model["pi_root"],
    ]
    for d in range(max_depth + 1):
        draw_vars += [model.model[f"a_prob_level_{d}"], model.model[f"e_level_{d}"]]
    drawn = pm.draw(draw_vars, random_seed=1)
    lam_on, lam_off, pi_root = drawn[0], drawn[1], drawn[2]
    assert lam_on.shape == (K,)
    assert lam_off.shape == (K,)
    assert pi_root.shape == (K,)

    rest = drawn[3:]
    for d in range(max_depth + 1):
        a_prob, e_level = rest[2 * d], rest[2 * d + 1]
        assert a_prob.shape == (n_by_depth[d], K)
        assert e_level.shape == (n_by_depth[d], K)
        assert np.all((a_prob >= 0) & (a_prob <= 1))
        np.testing.assert_allclose(e_level.sum(axis=-1), 1.0, atol=1e-8)


def test_signature_axis_registry_disabled():
    model = _build_fixed(switching=None)
    reg = model.signature_axis_vars()
    assert reg == {
        "mu_level": -1,
        "z_root_0": -1,
        "z_level_1": -1,
        "z_level_2": -1,
        "eta_level_0": -1,
        "eta_level_1": -1,
        "eta_level_2": -1,
        "e_level_0": -1,
        "e_level_1": -1,
        "e_level_2": -1,
    }
    assert set(reg) <= set(model.model.named_vars)


def test_signature_axis_registry_denovo_switching():
    model = _build_denovo(switching=_switching())
    reg = model.signature_axis_vars()
    assert reg["signatures"] == 0
    for name in ("lambda_on", "lambda_off", "pi_root"):
        assert reg[name] == -1
    for d in range(3):
        assert reg[f"a_prob_level_{d}"] == -1
        assert reg[f"e_level_{d}"] == -1
    assert set(reg) <= set(model.model.named_vars)
    # every registered axis-(-1) variable really has K on its last axis
    drawn = pm.draw(
        [model.model[n] for n, ax in reg.items() if ax == -1], random_seed=0
    )
    assert all(arr.shape[-1] == model.K for arr in drawn)


# ---------------------------------------------------------------------------
# logp / dlogp finite in both backends (section 3.4).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [None, "JAX"])
@pytest.mark.parametrize("build", [_build_fixed, _build_denovo])
def test_logp_dlogp_finite(build, mode):
    model = build(switching=_switching())
    logp_fn = model.model.compile_logp(mode=mode)
    dlogp_fn = model.model.compile_dlogp(mode=mode)
    ip = model.model.initial_point()

    assert np.isfinite(logp_fn(ip))
    assert np.all(np.isfinite(dlogp_fn(ip)))

    rng = np.random.default_rng(2)
    for _ in range(20):
        point = {k: v + rng.normal(scale=0.3, size=np.shape(v)) for k, v in ip.items()}
        assert np.isfinite(logp_fn(point))
        assert np.all(np.isfinite(dlogp_fn(point)))


# ---------------------------------------------------------------------------
# Bridge test (section 3.4): everything forced on reduces to plain multinomial.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_bridge_reduces_to_plain_multinomial(seed):
    model = _build_fixed(switching=_switching())
    K = model.K

    nodes_by_depth = model._get_nodes_by_depth()
    max_depth = max(nodes_by_depth)
    nodes_by_depth_list = [nodes_by_depth[d] for d in range(max_depth + 1)]
    depth_arrays = model._build_switch_depth_arrays(nodes_by_depth_list)

    eta_vars = [model.model[f"eta_level_{d}"] for d in range(max_depth + 1)]
    eta_vals = pm.draw(eta_vars, random_seed=seed)
    if max_depth == 0:
        eta_vals = [eta_vals]

    import pytensor.tensor as pt

    from src.models.switch_pruning import prune

    # The exact limit, not a nearby point: at pi = 1 - 1e-9 an off state
    # still carries prior mass e^-20.7, and for an extreme prior draw of eta
    # switching a badly fitting signature off can gain far more likelihood
    # than that, so the pruned logZ legitimately exceeds the all-on
    # multinomial (seen for 2 of 5 seeds). Only pi = 1, lambda = 0 makes
    # every off state impossible; prune evaluates it exactly (log_pi is
    # written to be nan-free there, see its docstring). backward is not
    # exercised here: its leave-one-out step is what needs lambda > 0.
    eta_inputs = [pt.as_tensor_variable(e) for e in eta_vals]
    S_t = pt.as_tensor_variable(model.fixed_signatures)
    lambda_on = pt.as_tensor_variable(np.zeros(K))
    lambda_off = pt.as_tensor_variable(np.zeros(K))
    pi_root = pt.as_tensor_variable(np.ones(K))

    *_, logZ, _ = prune(
        eta_inputs, S_t, lambda_on, lambda_off, pi_root, depth_arrays, K
    )
    got_logZ = float(logZ.eval())

    # Independent reference: softmax(eta) @ S, scored as a plain multinomial
    # via scipy (not PyMC), summed over the observed nodes only.
    want_logp = 0.0
    for d, current_nodes in enumerate(nodes_by_depth_list):
        eta_d = eta_vals[d]
        softmax_eta = np.exp(eta_d - eta_d.max(axis=-1, keepdims=True))
        softmax_eta /= softmax_eta.sum(axis=-1, keepdims=True)
        theta_d = softmax_eta @ model.fixed_signatures
        for i, node in enumerate(current_nodes):
            label = model.graph.nodes[node].get("label", str(node))
            if label not in model.data_matrix.index:
                continue
            counts = model.data_matrix.loc[label].values.astype(int)
            if counts.sum() == 0:
                continue
            want_logp += multinomial.logpmf(counts, n=counts.sum(), p=theta_d[i])

    assert np.isfinite(got_logZ)
    assert got_logZ == pytest.approx(want_logp, rel=1e-8)


# ---------------------------------------------------------------------------
# Validation errors.
# ---------------------------------------------------------------------------


def test_bad_branch_length_source_raises():
    with pytest.raises(ValueError, match="branch_length_source"):
        _build_fixed(switching=_switching(branch_length_source="weekly"))


def test_always_on_not_implemented():
    with pytest.raises(NotImplementedError, match="always_on"):
        _build_fixed(switching=_switching(always_on=["SBS1"]))


def test_tree_coupled_false_not_implemented():
    with pytest.raises(NotImplementedError, match="tree_coupled"):
        _build_fixed(switching=_switching(tree_coupled=False))


def test_state_space_cap_raises():
    with pytest.raises(ValueError, match="cap"):
        _build_fixed(switching=_switching(), K=13)


def test_missing_edge_length_raises_under_newick_source():
    with pytest.raises(ValueError, match="length"):
        _build_fixed(switching=_switching(), newick=NEWICK_NO_LENGTHS)


def test_unit_branch_length_source_ignores_missing_lengths():
    model = _build_fixed(
        switching=_switching(branch_length_source="unit"), newick=NEWICK_NO_LENGTHS
    )
    nodes_by_depth = model._get_nodes_by_depth()
    nodes_by_depth_list = [nodes_by_depth[d] for d in range(max(nodes_by_depth) + 1)]
    depth_arrays = model._build_switch_depth_arrays(nodes_by_depth_list)
    for length_d in depth_arrays.length[1:]:
        np.testing.assert_allclose(length_d, 1.0)
