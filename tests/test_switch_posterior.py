"""Tests for src/analysis/switch_posterior.py (switch_model_plan.md 6.1).

The correctness anchor reuses tests/test_switch_pruning.py's brute-force
oracle: on the branching forest, forward-filter backward-sample draws from
the compiled pruning must reproduce the exact node marginals and the exact
per-edge gain / loss probabilities to Monte Carlo error.
"""

import arviz as az
import numpy as np
import pandas as pd
import pytest

from src.analysis.switch_posterior import (
    compile_pruning,
    depth_arrays_from_files,
    edge_probabilities,
    edge_table,
    ffbs_draw,
    node_labels,
    node_marginals,
    registry_frame_to_true,
    sample_states,
)
from src.models.switch_pruning import state_grid
from tests.test_switch_pruning import _make_branching_forest, brute_force_oracle


def _oracle_setup(K=2, C=5, seed=3):
    rng = np.random.default_rng(seed)
    depth, eta, S, lam_on, lam_off, _ = _make_branching_forest(
        K, C, rng, unobserved=[(2, 1)]
    )
    pi = np.full(K, 0.3)
    truth = brute_force_oracle(K, depth, eta, S, lam_on, lam_off, pi, with_edges=True)
    return depth, eta, S, lam_on, lam_off, pi, truth


def test_ffbs_marginals_and_edge_events_match_oracle():
    K, C = 2, 5
    depth, eta, S, lam_on, lam_off, pi, truth = _oracle_setup(K, C)
    _, true_a_prob, _, true_edges = truth
    compiled = compile_pruning(K, C, depth)
    log_beta, _, logT, logpi = compiled(eta, S, lam_on, lam_off, pi)

    rng = np.random.default_rng(0)
    masks = state_grid(K).astype(np.int8)
    n_samples = 6000
    draws = np.stack(
        [
            masks[ffbs_draw(log_beta, logT, logpi, depth, state_grid(K), rng)]
            for _ in range(n_samples)
        ]
    )
    marg = draws.mean(axis=0)  # (N, K)

    want = np.concatenate(true_a_prob)
    np.testing.assert_allclose(marg, want, atol=0.03)

    # edge events: P(parent off, child on) and P(parent on, child off)
    offsets = np.cumsum([0] + [c.shape[0] for c in depth.counts])
    for d in range(1, depth.max_depth + 1):
        for i in range(depth.counts[d].shape[0]):
            p_idx = offsets[d - 1] + depth.parent_pos[d][i]
            c_idx = offsets[d] + i
            ap, ac = draws[:, p_idx, :].astype(bool), draws[:, c_idx, :].astype(bool)
            gain = (~ap & ac).mean(axis=0)
            loss = (ap & ~ac).mean(axis=0)
            np.testing.assert_allclose(gain, true_edges["gain"][(d, i)], atol=0.03)
            np.testing.assert_allclose(loss, true_edges["loss"][(d, i)], atol=0.03)


def test_sample_states_shapes_thin_and_marginals():
    K, C = 2, 5
    depth, eta, S, lam_on, lam_off, pi, truth = _oracle_setup(K, C)
    _, true_a_prob, _, _ = truth
    chains, draws = 1, 3000
    post = {
        f"eta_level_{d}": np.broadcast_to(e, (chains, draws) + e.shape).copy()
        for d, e in enumerate(eta)
    }
    post["lambda_on"] = np.broadcast_to(lam_on, (chains, draws, K)).copy()
    post["lambda_off"] = np.broadcast_to(lam_off, (chains, draws, K)).copy()
    post["pi_root"] = np.broadcast_to(pi, (chains, draws, K)).copy()
    idata = az.from_dict(posterior=post)

    states, draw_idx = sample_states(
        idata.posterior,
        depth,
        K,
        fixed_signatures=S,
        thin=3,
        rng=np.random.default_rng(1),
    )
    N = sum(c.shape[0] for c in depth.counts)
    assert states.dtype == np.int8
    assert states.shape == (chains, 1000, N, K)
    assert np.array_equal(draw_idx, np.arange(0, draws, 3))
    assert set(np.unique(states)) <= {0, 1}
    np.testing.assert_allclose(
        node_marginals(states), np.concatenate(true_a_prob), atol=0.04
    )


def test_edge_table_and_probabilities_from_model():
    newick = "((C:0.5,D:0.7)B:0.3,E:1.0)A:0.0;"
    rng = np.random.default_rng(0)
    K, C = 2, 6
    S = rng.dirichlet(np.ones(C), size=K)
    counts = pd.DataFrame(rng.integers(1, 10, size=(4, C)), index=["A", "B", "C", "D"])
    model, depth, nbd_list = depth_arrays_from_files(newick, counts, fixed_signatures=S)
    labels = node_labels(model, nbd_list)
    assert set(labels) == {"A", "B", "C", "D", "E"}
    edges = edge_table(model, nbd_list, depth)
    assert set(zip(edges["parent"], edges["child"])) == {
        ("A", "B"),
        ("A", "E"),
        ("B", "C"),
        ("B", "D"),
    }
    assert (edges["tumour"] == 0).all()

    N = len(labels)
    states = np.zeros((2, 4, N, K), dtype=np.int8)
    idx = {lab: i for i, lab in enumerate(labels)}
    states[:, :, idx["A"], 0] = 0
    states[:, :, idx["B"], 0] = 1  # gain on A -> B, signature 0, in every sample
    states[:, :, idx["C"], 0] = 1  # and kept on below B, so B -> C, B -> D
    states[:, :, idx["D"], 0] = 1  # carry no event for either signature
    states[:, :, idx["A"], 1] = 1
    states[:, :, idx["E"], 1] = 0  # loss on A -> E, signature 1
    probs = edge_probabilities(states, edges)
    assert list(probs.columns) == [
        "tumour",
        "parent",
        "child",
        "signature",
        "p_gain",
        "p_loss",
        "p_switch",
    ]
    ab0 = probs[
        (probs.parent == "A") & (probs.child == "B") & (probs.signature == 0)
    ].iloc[0]
    assert ab0.p_gain == 1.0 and ab0.p_loss == 0.0 and ab0.p_switch == 1.0
    ae1 = probs[
        (probs.parent == "A") & (probs.child == "E") & (probs.signature == 1)
    ].iloc[0]
    assert ae1.p_loss == 1.0 and ae1.p_gain == 0.0
    bc = probs[(probs.parent == "B") & (probs.child == "C")]
    assert (bc.p_switch == 0.0).all()


def test_registry_frame_to_true_inverts_perm():
    perm0 = np.array([2, 0, 1])  # true slot i holds trace row perm0[i]
    mapping = registry_frame_to_true(perm0)
    for i in range(3):
        assert mapping[int(perm0[i])] == i


@pytest.mark.parametrize("K", [1, 3])
def test_compile_pruning_matches_direct_eval(K):
    """The compiled function returns the same arrays as evaluating prune's
    graph directly (one implementation, two ways of running it)."""
    import pytensor.tensor as pt

    from src.models.switch_pruning import prune

    rng = np.random.default_rng(K)
    C = 4
    depth, eta, S, lam_on, lam_off, pi = _make_branching_forest(K, C, rng)
    compiled = compile_pruning(K, C, depth)
    lb, lm, lt, lp = compiled(eta, S, lam_on, lam_off, pi)

    ref = prune(
        [pt.as_tensor_variable(e) for e in eta],
        pt.as_tensor_variable(S),
        pt.as_tensor_variable(lam_on),
        pt.as_tensor_variable(lam_off),
        pt.as_tensor_variable(pi),
        depth,
        K,
    )
    for d in range(depth.max_depth + 1):
        np.testing.assert_allclose(lb[d], ref[0][d].eval())
        if d > 0:
            np.testing.assert_allclose(lm[d], ref[1][d].eval())
            np.testing.assert_allclose(lt[d], ref[2][d].eval())
    assert lm[0] is None and lt[0] is None
    np.testing.assert_allclose(lp, ref[3].eval())
