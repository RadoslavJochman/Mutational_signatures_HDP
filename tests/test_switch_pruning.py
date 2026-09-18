"""
Tests for `src/models/switch_pruning.py`.

Section references are to `switch_model_plan.md`. `test_brute_force_oracle` is
the correctness anchor (plan section 4.1): it enumerates every joint on/off
assignment over a small hand-built forest independently of the module under
test, and checks `logZ`, the node marginals and `e_level` against it. Every
other test here checks one component in isolation (section 4.2).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from scipy.special import gammaln, logsumexp

from src.models.switch_pruning import (
    DepthArrays,
    backward,
    contract_child_to_parent,
    contract_parent_to_child,
    emission_loglik,
    log_transition,
    masked_softmax,
    prune,
    state_grid,
)

# ---------------------------------------------------------------------------
# Shared fixtures: a small two-tree forest.
#
# Tree A: r0 alone (depth 0, no children).
# Tree B: r1 -> c1 -> c2 (depths 0, 1, 2).
# 4 nodes total, matching the plan's own example ((2**K)**N <= 8**4 = 4096
# for the K=3 case brute-forced below).
# ---------------------------------------------------------------------------


def _make_forest(K: int, C: int, rng: np.random.Generator, unobserved=()):
    """Build DepthArrays plus random eta/S/lambda/pi for the shared forest.

    `unobserved` lists (depth, index) pairs whose counts are zeroed and
    `observed` set to False.
    """
    counts0 = rng.integers(0, 6, size=(2, C)).astype("float64")
    counts1 = rng.integers(0, 6, size=(1, C)).astype("float64")
    counts2 = rng.integers(0, 6, size=(1, C)).astype("float64")
    counts = [counts0, counts1, counts2]
    observed = [np.array([True, True]), np.array([True]), np.array([True])]
    for d, i in unobserved:
        counts[d][i] = 0.0
        observed[d][i] = False

    parent_pos = [None, np.array([1]), np.array([0])]
    length = [None, np.array([0.7]), np.array([1.3])]
    tree_of_root = np.array([0, 1])
    depth = DepthArrays.build(counts, observed, parent_pos, length, tree_of_root)

    eta_by_depth = [
        rng.normal(size=(2, K)),
        rng.normal(size=(1, K)),
        rng.normal(size=(1, K)),
    ]
    S = rng.dirichlet(np.ones(C), size=K)
    lambda_on = rng.uniform(0.1, 0.6, size=K)
    lambda_off = rng.uniform(0.1, 0.6, size=K)
    pi = rng.uniform(0.2, 0.8, size=K)
    return depth, eta_by_depth, S, lambda_on, lambda_off, pi


def _compile(K, C, depth, mode=None):
    """Compile a (logZ, *a_prob_by_depth, *e_level_by_depth) function."""
    n_by_depth = [c.shape[0] for c in depth.counts]
    eta_inputs = [
        pt.tensor(f"eta{d}", dtype="float64", shape=(n, K))
        for d, n in enumerate(n_by_depth)
    ]
    S_t = pt.tensor("S", dtype="float64", shape=(K, C))
    lam_on_t = pt.tensor("lambda_on", dtype="float64", shape=(K,))
    lam_off_t = pt.tensor("lambda_off", dtype="float64", shape=(K,))
    pi_t = pt.tensor("pi", dtype="float64", shape=(K,))

    log_beta, log_msg, logT_bd, logpi_vec, logZ, logZ_per_root = prune(
        eta_inputs, S_t, lam_on_t, lam_off_t, pi_t, depth, K
    )
    _, a_prob, e_level = backward(
        eta_inputs, log_beta, log_msg, logT_bd, logpi_vec, logZ_per_root, depth, K
    )
    inputs = eta_inputs + [S_t, lam_on_t, lam_off_t, pi_t]
    outputs = [logZ, *a_prob, *e_level]
    kwargs = {} if mode is None else {"mode": mode}
    return pytensor.function(inputs, outputs, **kwargs)


def _run(f, eta_by_depth, S, lambda_on, lambda_off, pi):
    out = f(*eta_by_depth, S, lambda_on, lambda_off, pi)
    n_depths = len(eta_by_depth)
    logZ = out[0]
    a_prob = out[1 : 1 + n_depths]
    e_level = out[1 + n_depths : 1 + 2 * n_depths]
    return logZ, a_prob, e_level


# ---------------------------------------------------------------------------
# Independent brute-force oracle (plain NumPy, no switch_pruning internals).
# ---------------------------------------------------------------------------


def _np_masked_softmax_row(eta_row, masks):
    w = masks * np.exp(eta_row[None, :] - eta_row.max())
    den = w.sum(axis=1)
    den_safe = np.where(den > 0, den, 1.0)
    return w / den_safe[:, None]  # (2^K, K)


def _np_log_emission_row(eta_row, S, counts_row, observed_row, masks):
    e_all = _np_masked_softmax_row(eta_row, masks)  # (2^K, K)
    theta = np.clip(e_all @ S, 1e-300, 1.0)  # (2^K, C)
    if not observed_row:
        return np.zeros(masks.shape[0])
    ll = (counts_row[None, :] * np.log(theta)).sum(axis=1)
    m_total = counts_row.sum()
    const = gammaln(m_total + 1) - gammaln(counts_row + 1).sum()
    ll = ll + const
    ll[0] = -np.inf
    return ll


def _np_log_transition_matrix(lambda_on, lambda_off, length, masks):
    """Full (2**K, 2**K) log transition matrix for one edge (brute force only)."""
    K = masks.shape[1]
    logT_k = np.zeros((K, 2, 2))
    logT_k[:, 0, 0] = -lambda_on * length
    logT_k[:, 1, 1] = -lambda_off * length
    logT_k[:, 0, 1] = np.log1p(-np.exp(-lambda_on * length))
    logT_k[:, 1, 0] = np.log1p(-np.exp(-lambda_off * length))
    two_k = masks.shape[0]
    out = np.zeros((two_k, two_k))
    for sp in range(two_k):
        for sc in range(two_k):
            total = 0.0
            for k in range(K):
                total += logT_k[k, int(masks[sp, k]), int(masks[sc, k])]
            out[sp, sc] = total
    return out


def _np_log_pi(pi, masks):
    return (masks * np.log(pi)[None, :] + (1 - masks) * np.log1p(-pi)[None, :]).sum(
        axis=1
    )


def brute_force_oracle(
    K, depth: DepthArrays, eta_by_depth, S, lambda_on, lambda_off, pi
):
    """Enumerate every joint state assignment over the whole forest.

    Returns (logZ, a_prob_by_depth, e_level_by_depth), directly comparable to
    `_run`'s outputs.
    """
    masks = state_grid(K)
    n_by_depth = [c.shape[0] for c in depth.counts]
    node_list = [(d, i) for d, n in enumerate(n_by_depth) for i in range(n)]
    N = len(node_list)

    log_em = {
        (d, i): _np_log_emission_row(
            eta_by_depth[d][i], S, depth.counts[d][i], depth.observed[d][i], masks
        )
        for d, i in node_list
    }
    logT = {
        (d, i): _np_log_transition_matrix(
            lambda_on, lambda_off, depth.length[d][i], masks
        )
        for d in range(1, len(n_by_depth))
        for i in range(n_by_depth[d])
    }
    logpi_vec = _np_log_pi(pi, masks)

    two_k = masks.shape[0]
    joint = np.zeros([two_k] * N)
    for combo in itertools.product(range(two_k), repeat=N):
        state_of = dict(zip(node_list, combo))
        logp = 0.0
        for d, i in node_list:
            s = state_of[(d, i)]
            if d == 0:
                logp += logpi_vec[s]
            else:
                parent_i = int(depth.parent_pos[d][i])
                s_parent = state_of[(d - 1, parent_i)]
                logp += logT[(d, i)][s_parent, s]
            logp += log_em[(d, i)][s]
        joint[combo] = logp
    logZ = logsumexp(joint)

    a_prob_by_depth = [np.zeros((n, K)) for n in n_by_depth]
    e_level_by_depth = [np.zeros((n, K)) for n in n_by_depth]
    for pos, (d, i) in enumerate(node_list):
        other_axes = tuple(a for a in range(N) if a != pos)
        log_marginal = logsumexp(joint, axis=other_axes) - logZ  # (2^K,)
        q = np.exp(log_marginal)
        a_prob_by_depth[d][i] = q @ masks
        e_all = _np_masked_softmax_row(eta_by_depth[d][i], masks)
        q_no_off = q.copy()
        q_no_off[0] = 0.0
        e_level_by_depth[d][i] = (q_no_off[:, None] * e_all).sum(
            axis=0
        ) / q_no_off.sum()

    return logZ, a_prob_by_depth, e_level_by_depth


@pytest.mark.parametrize("K", [1, 2, 3])
def test_brute_force_oracle_random(K):
    """Baseline case: random eta/S/lambda/pi/counts, one node unobserved."""
    rng = np.random.default_rng(100 + K)
    C = 6
    depth, eta_by_depth, S, lambda_on, lambda_off, pi = _make_forest(
        K, C, rng, unobserved=[(1, 0)]
    )
    f = _compile(K, C, depth)
    logZ, a_prob, e_level = _run(f, eta_by_depth, S, lambda_on, lambda_off, pi)
    true_logZ, true_a_prob, true_e_level = brute_force_oracle(
        K, depth, eta_by_depth, S, lambda_on, lambda_off, pi
    )

    assert logZ == pytest.approx(true_logZ, rel=1e-8)
    for got, want in zip(a_prob, true_a_prob):
        np.testing.assert_allclose(got, want, atol=1e-8)
    for got, want in zip(e_level, true_e_level):
        np.testing.assert_allclose(got, want, atol=1e-8)
        np.testing.assert_allclose(got.sum(axis=-1), 1.0, atol=1e-8)


def test_brute_force_oracle_small_pi_all_off_matters():
    """Small pi puts real prior mass on all-off; the observed-node exclusion
    (log P(x | a = 0) = -inf) must change the answer, not just be inert."""
    K = 3
    C = 6
    rng = np.random.default_rng(7)
    depth, eta_by_depth, S, lambda_on, lambda_off, _ = _make_forest(K, C, rng)
    pi = np.full(K, 0.05)

    f = _compile(K, C, depth)
    logZ, a_prob, e_level = _run(f, eta_by_depth, S, lambda_on, lambda_off, pi)
    true_logZ, true_a_prob, true_e_level = brute_force_oracle(
        K, depth, eta_by_depth, S, lambda_on, lambda_off, pi
    )

    assert logZ == pytest.approx(true_logZ, rel=1e-8)
    for got, want in zip(a_prob, true_a_prob):
        np.testing.assert_allclose(got, want, atol=1e-8)
    for got, want in zip(e_level, true_e_level):
        np.testing.assert_allclose(got, want, atol=1e-8)

    # Sanity: with such a small pi, some node's all-off state carries real
    # prior mass, so a_prob must differ meaningfully from 1 somewhere.
    assert min(a.min() for a in a_prob) < 0.95


def test_brute_force_oracle_reduces_to_plain_multinomial():
    """pi -> 1, lambda -> 0: only the all-on state keeps essentially all the
    mass, so e_level must be close to softmax(eta) (the switch model with
    everything forced on is the current model, cf. the bridge test in
    section 3.4).

    lambda is tiny but not exactly zero: at lambda = 0 the transition matrix
    becomes a hard identity (no mixing across states at all), which can make
    the downward pass's leave-one-out subtraction hit -inf - (-inf) = NaN at
    an all-off entry that a real (lambda > 0) run never reaches (the
    LogNormal prior on lambda has support (0, inf), never exactly 0). This
    is exactly why section 3.4 writes "lambda -> 0", not "= 0".
    """
    K = 3
    C = 6
    rng = np.random.default_rng(11)
    depth, eta_by_depth, S, _, _, _ = _make_forest(K, C, rng)
    lambda_on = np.full(K, 1e-8)
    lambda_off = np.full(K, 1e-8)
    pi = np.full(K, 1 - 1e-9)

    f = _compile(K, C, depth)
    logZ, a_prob, e_level = _run(f, eta_by_depth, S, lambda_on, lambda_off, pi)
    true_logZ, true_a_prob, true_e_level = brute_force_oracle(
        K, depth, eta_by_depth, S, lambda_on, lambda_off, pi
    )

    assert logZ == pytest.approx(true_logZ, rel=1e-8)
    for got, want in zip(a_prob, true_a_prob):
        np.testing.assert_allclose(got, want, atol=1e-6)
        # near 1: pi = 1 - 1e-9 leaves a small residual off-mass that the
        # likelihood ratio can amplify, so this is a loose sanity check, not
        # the tight oracle comparison above.
        assert got.min() > 0.99
    for got, want, eta_d in zip(e_level, true_e_level, eta_by_depth):
        np.testing.assert_allclose(got, want, atol=1e-6)
        softmax_eta = np.exp(eta_d - eta_d.max(axis=-1, keepdims=True))
        softmax_eta /= softmax_eta.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(got, softmax_eta, atol=1e-3)


# ---------------------------------------------------------------------------
# Component tests (section 4.2).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("K", [1, 2, 3, 4])
def test_state_grid_convention(K):
    masks = state_grid(K)
    assert masks.shape == (2**K, K)
    reshaped = masks.reshape((2,) * K + (K,))
    for idx in itertools.product([0, 1], repeat=K):
        np.testing.assert_array_equal(reshaped[idx], np.array(idx, dtype="float64"))


@pytest.mark.parametrize("K", [1, 2, 3, 4])
def test_contraction_matches_kronecker(K):
    rng = np.random.default_rng(K)
    n = 3
    masks = state_grid(K)
    length = rng.uniform(0.2, 1.5, size=n)
    lambda_on = rng.uniform(0.1, 0.6, size=K)
    lambda_off = rng.uniform(0.1, 0.6, size=K)
    log_beta_child_val = np.log(rng.dirichlet(np.ones(2**K), size=n))

    logT_t = log_transition(
        pt.as_tensor_variable(lambda_on),
        pt.as_tensor_variable(lambda_off),
        pt.as_tensor_variable(length),
    )
    log_beta_child_t = pt.tensor("lbc", dtype="float64", shape=(n, 2**K))
    out_t = contract_child_to_parent(log_beta_child_t, logT_t, K)
    f = pytensor.function([log_beta_child_t], out_t)
    got = f(log_beta_child_val)

    want = np.stack(
        [
            logsumexp(
                _np_log_transition_matrix(lambda_on, lambda_off, length[node], masks)
                + log_beta_child_val[node][None, :],
                axis=1,
            )
            for node in range(n)
        ]
    )
    np.testing.assert_allclose(got, want, atol=1e-8)


@pytest.mark.parametrize("K", [1, 2, 3])
def test_identity_transition_is_identity(K):
    """lambda -> 0 gives an identity transition: the child's belief passes
    through the parent axis unchanged."""
    n = 2
    rng = np.random.default_rng(K + 50)
    length = rng.uniform(0.2, 1.5, size=n)
    lambda_on = np.zeros(K)
    lambda_off = np.zeros(K)
    log_beta_child_val = np.log(rng.dirichlet(np.ones(2**K), size=n))

    logT_t = log_transition(
        pt.as_tensor_variable(lambda_on),
        pt.as_tensor_variable(lambda_off),
        pt.as_tensor_variable(length),
    )
    log_beta_child_t = pt.tensor("lbc", dtype="float64", shape=(n, 2**K))
    out_t = contract_child_to_parent(log_beta_child_t, logT_t, K)
    f = pytensor.function([log_beta_child_t], out_t)
    got = f(log_beta_child_val)
    np.testing.assert_allclose(got, log_beta_child_val, atol=1e-10)


def test_contract_parent_to_child_transposes():
    K, n = 2, 2
    rng = np.random.default_rng(3)
    lambda_on = rng.uniform(0.1, 0.6, size=K)
    lambda_off = rng.uniform(0.1, 0.6, size=K)
    length = rng.uniform(0.2, 1.0, size=n)
    logT_t = log_transition(
        pt.as_tensor_variable(lambda_on),
        pt.as_tensor_variable(lambda_off),
        pt.as_tensor_variable(length),
    )
    x_t = pt.tensor("x", dtype="float64", shape=(n, 2**K))
    up = contract_child_to_parent(x_t, logT_t, K)
    down = contract_parent_to_child(x_t, logT_t.transpose(0, 1, 3, 2), K)
    f = pytensor.function([x_t], [up, down])
    x_val = np.log(rng.dirichlet(np.ones(2**K), size=n))
    got_up, got_down = f(x_val)
    np.testing.assert_allclose(got_up, got_down, atol=1e-10)


def test_masked_softmax_emission_properties():
    K, n = 3, 4
    rng = np.random.default_rng(4)
    masks = state_grid(K)
    eta_val = rng.normal(size=(n, K))
    eta_t = pt.tensor("eta", dtype="float64", shape=(n, K))
    e_all_t = masked_softmax(eta_t, masks)
    f = pytensor.function([eta_t], e_all_t)
    e_all = f(eta_val)

    np.testing.assert_allclose(e_all[:, 0, :], 0.0, atol=1e-12)
    row_sums = e_all[:, 1:, :].sum(axis=-1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    for i in range(n):
        want = _np_masked_softmax_row(eta_val[i], masks)
        np.testing.assert_allclose(e_all[i], want, atol=1e-10)


def test_emission_loglik_unobserved_is_zero():
    K, C, n = 2, 4, 3
    rng = np.random.default_rng(5)
    masks = state_grid(K)
    eta_val = rng.normal(size=(n, K))
    S_val = rng.dirichlet(np.ones(C), size=K)
    counts_val = rng.integers(0, 5, size=(n, C)).astype("float64")
    observed_val = np.array([True, False, True])

    eta_t = pt.tensor("eta", dtype="float64", shape=(n, K))
    S_t = pt.tensor("S", dtype="float64", shape=(K, C))
    counts_t = pt.tensor("counts", dtype="float64", shape=(n, C))
    observed_t = pt.tensor("observed", dtype="bool", shape=(n,))
    e_all_t = masked_softmax(eta_t, masks)
    log_em_t = emission_loglik(e_all_t, S_t, counts_t, observed_t)
    f = pytensor.function([eta_t, S_t, counts_t, observed_t], log_em_t)
    log_em = f(eta_val, S_val, counts_val, observed_val)

    np.testing.assert_allclose(log_em[1], 0.0, atol=1e-12)
    assert log_em[0, 0] == -np.inf
    assert log_em[2, 0] == -np.inf
    assert np.all(np.isfinite(log_em[0, 1:]))
    assert np.all(np.isfinite(log_em[2, 1:]))


def test_log1mexp_path():
    K = 4
    rng = np.random.default_rng(6)
    lam = rng.uniform(0.05, 2.0, size=K)
    length = rng.uniform(0.1, 3.0, size=1)
    lam_t = pt.as_tensor_variable(lam)
    length_t = pt.as_tensor_variable(length)
    logT_t = log_transition(lam_t, lam_t, length_t)
    f = pytensor.function([], logT_t)
    logT = f()
    p_gain = np.exp(logT[0, :, 0, 1])
    p_stay_off = np.exp(logT[0, :, 0, 0])
    np.testing.assert_allclose(p_gain + p_stay_off, 1.0, atol=1e-12)

    # finite at a tiny hazard, where naive 1 - exp(-x) would underflow to 0
    tiny = pt.as_tensor_variable(np.array([1e-12]))
    logT_tiny = log_transition(lam_t, lam_t, tiny)
    f_tiny = pytensor.function([], logT_tiny)
    out = f_tiny()
    assert np.all(np.isfinite(out))
    assert out[0, 0, 0, 1] < 0  # log of a tiny but nonzero probability


def test_prune_and_backward_jax_matches_default_backend():
    """JAX compilation of prune and backward on a tiny input; the sampler is
    numpyro, so a graph that only works in C is a failure (section 4.2)."""
    K, C = 2, 4
    rng = np.random.default_rng(8)
    depth, eta_by_depth, S, lambda_on, lambda_off, pi = _make_forest(K, C, rng)

    f_default = _compile(K, C, depth, mode=None)
    f_jax = _compile(K, C, depth, mode="JAX")

    out_default = f_default(*eta_by_depth, S, lambda_on, lambda_off, pi)
    out_jax = f_jax(*eta_by_depth, S, lambda_on, lambda_off, pi)

    for got, want in zip(out_jax, out_default):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=1e-8)
