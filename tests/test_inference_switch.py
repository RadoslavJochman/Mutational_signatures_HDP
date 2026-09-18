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


def _build_fixed(switching=None, newick=NEWICK, K=3, priors=PRIORS):
    newick, data, K, C, S = _toy_inputs(K=K, newick=newick)
    return TreeHDP(newick, data, priors=priors, fixed_signatures=S, switching=switching)


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
# always_on (section 7.1).
# ---------------------------------------------------------------------------

SIG_NAMES = ["SBS1", "SBS5", "SBS36"]


def _build_fixed_named(switching, K=3):
    newick, data, K, C, S = _toy_inputs(K=K)
    return TreeHDP(
        newick,
        data,
        priors=PRIORS,
        fixed_signatures=S,
        switching=switching,
        signature_names=SIG_NAMES,
    )


def test_always_on_forces_a_prob_to_one_and_builds():
    model = _build_fixed_named(_switching(always_on=["SBS1"]))
    assert model.switching["always_on_idx"] == (0,)
    nodes_by_depth = model._get_nodes_by_depth()
    draw_vars = [
        model.model[f"a_prob_level_{d}"] for d in range(max(nodes_by_depth) + 1)
    ]
    drawn = pm.draw(draw_vars, random_seed=0)
    for a_prob in drawn:
        assert (a_prob[:, 0] == 1.0).all()
        assert ((a_prob[:, 1:] >= 0) & (a_prob[:, 1:] <= 1)).all()
    logp = model.model.compile_logp()(model.model.initial_point())
    assert np.isfinite(logp)


def test_always_on_validation_errors():
    with pytest.raises(ValueError, match="fixed signatures"):
        _build_denovo(switching=_switching(always_on=["SBS1"]))
    with pytest.raises(ValueError, match="signature_names"):
        _build_fixed(switching=_switching(always_on=["SBS1"]))
    with pytest.raises(ValueError, match="not in the fixed signature index"):
        _build_fixed_named(_switching(always_on=["SBS99"]))


def test_state_space_cap_counts_free_signatures_only():
    # K = 13 exceeds the cap; forcing one signature on brings it to 2**12.
    newick, data, K, C, S = _toy_inputs(K=13)
    names = [f"S{k}" for k in range(13)]
    with pytest.raises(ValueError, match="cap"):
        TreeHDP(
            newick,
            data,
            priors=PRIORS,
            fixed_signatures=S,
            switching=_switching(),
            signature_names=names,
        )
    model = TreeHDP(
        newick,
        data,
        priors=PRIORS,
        fixed_signatures=S,
        switching=_switching(always_on=["S0"]),
        signature_names=names,
    )
    assert model.switching["always_on_idx"] == (0,)


# ---------------------------------------------------------------------------
# walk_branch_length_scaling (section 7.3).
# ---------------------------------------------------------------------------

NEWICK_UNIT = "((C:1,D:1)B:1,E:1)A:0.0;"  # every edge 1, so every l_e = 1
SCALED = dict(PRIORS, walk_branch_length_scaling=True)


def _logp_pairs(model_a, model_b, n=20, seed=0):
    """Model logp of both models at the same n random points around the
    initial point (both share the same free variables)."""
    fa, fb = model_a.model.compile_logp(), model_b.model.compile_logp()
    ip = model_a.model.initial_point()
    assert set(ip) == set(model_b.model.initial_point())
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n):
        point = {k: v + rng.normal(scale=0.3, size=np.shape(v)) for k, v in ip.items()}
        pairs.append((float(fa(point)), float(fb(point))))
    return np.array(pairs)


@pytest.mark.parametrize("with_switching", [False, True])
def test_walk_scaling_with_unit_lengths_equals_unscaled(with_switching):
    sw = _switching() if with_switching else None
    scaled = _build_fixed(switching=sw, newick=NEWICK_UNIT, priors=SCALED)
    plain = _build_fixed(switching=sw, newick=NEWICK_UNIT)
    assert scaled.l_median == 1.0
    pairs = _logp_pairs(scaled, plain)
    assert np.all(np.isfinite(pairs))
    np.testing.assert_allclose(pairs[:, 0], pairs[:, 1], rtol=1e-10)


@pytest.mark.parametrize("with_switching", [False, True])
def test_walk_scaling_with_unequal_lengths_differs(with_switching):
    sw = _switching() if with_switching else None
    scaled = _build_fixed(switching=sw, newick=NEWICK, priors=SCALED)
    plain = _build_fixed(switching=sw, newick=NEWICK)
    assert scaled.l_median == pytest.approx(0.6)  # median of 0.3, 0.5, 0.7, 1.0
    pairs = _logp_pairs(scaled, plain)
    assert np.all(np.isfinite(pairs))
    assert np.abs(pairs[:, 0] - pairs[:, 1]).max() > 1e-6


def test_walk_scaling_length_source_without_switching():
    unit = _build_fixed(
        newick=NEWICK_NO_LENGTHS, priors=dict(SCALED, branch_length_source="unit")
    )
    plain = _build_fixed(newick=NEWICK_NO_LENGTHS)
    pairs = _logp_pairs(unit, plain)
    np.testing.assert_allclose(pairs[:, 0], pairs[:, 1], rtol=1e-10)
    with pytest.raises(ValueError, match="length"):
        _build_fixed(newick=NEWICK_NO_LENGTHS, priors=SCALED)


def test_walk_scaling_default_off_is_unchanged():
    assert "walk_branch_length_scaling" not in PRIORS
    plain = _build_fixed(newick=NEWICK)
    explicit_off = _build_fixed(
        newick=NEWICK, priors=dict(PRIORS, walk_branch_length_scaling=False)
    )
    pairs = _logp_pairs(plain, explicit_off, n=5)
    np.testing.assert_allclose(pairs[:, 0], pairs[:, 1], rtol=1e-12)


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


@pytest.mark.parametrize("build", [_build_fixed, _build_denovo])
def test_tree_coupled_false_builds_and_is_finite(build):
    model = build(switching=_switching(tree_coupled=False))
    assert model.switching["tree_coupled"] is False
    assert {v.name for v in model.model.potentials} == {"switch_loglik"}
    ip = model.model.initial_point()
    assert np.isfinite(model.model.compile_logp()(ip))
    assert np.all(np.isfinite(model.model.compile_dlogp()(ip)))
    coupled = build(switching=_switching())
    assert coupled.switching["tree_coupled"] is True


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
