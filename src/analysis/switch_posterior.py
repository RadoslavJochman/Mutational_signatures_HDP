"""
switch_posterior.py

Post-hoc state samples for the switch model (switch_model_plan.md section 6.1).

`compile_pruning` compiles `switch_pruning.prune` once, for one fixed forest,
into a `pytensor.function` from (eta per depth, S, lambda_on, lambda_off, pi)
to (log_beta per depth, log_msg per depth, logT per depth, log_pi). It is the
same graph `TreeHDP` builds inside the model, with the same static per-depth
sizes (one implementation of the pruning maths); here it is evaluated per
posterior draw in NumPy.

`sample_states` is forward-filter backward-sample per draw: the root state is
drawn from softmax(log_pi + log_beta[0]) and each child from
softmax(logT(s_parent -> .) + log_beta_child), top down. It returns int8
`(chains, draws, N, K)` with nodes in depth-major order (the order
`e_level_<d>` / `a_prob_level_<d>` are concatenated in). From the samples any
joint functional follows: `edge_probabilities` gives per-edge gain, loss and
switch probabilities, and `node_marginals` the per-node activation frequency.
The latter must agree with the mean of `a_prob_level_*` (the
Rao-Blackwellised estimate from `switch_pruning.backward`) to Monte Carlo
error; `tests/test_smoke_switch.py` asserts that cross-check.

In de novo mode this runs on `trace_aligned.nc`, which `run_inference.py`
makes self-consistent per draw by permuting every signature-axis variable
(`TreeHDP.signature_axis_vars`) by the same permutation. Cost is
`O(N 2^K C)` per draw; negligible at K = 5, minutes for thousands of draws
at K = 10 (use `thin`).

The depth arrays come from `TreeHDP._build_switch_depth_arrays`
(`depth_arrays_from_files` builds the model to get them) rather than from a
second builder, so node order, branch-length normalisation and the observed
set are exactly the model's.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt

from src.models.switch_pruning import DepthArrays, prune, state_grid

# Priors are irrelevant to the depth arrays; these only have to build.
_BUILD_PRIORS = {
    "sigma_prior": "LogNorm",
    "sigma_prior_parm": {"mu": 0.0, "sigma": 1.0},
    "sigma_0": 1.0,
    "beta": 0.5,
}
_BUILD_SWITCHING = {
    "enabled": True,
    "lambda_on_prior": "LogNorm",
    "lambda_on_prior_parm": {"mu": -1.2, "sigma": 1.0},
    "lambda_off_prior": "LogNorm",
    "lambda_off_prior_parm": {"mu": -1.2, "sigma": 1.0},
    "pi_root_prior": "Beta",
    "pi_root_prior_parm": {"alpha": 1, "beta": 1},
}


def depth_arrays_from_files(
    newick: str,
    counts: pd.DataFrame,
    fixed_signatures: Optional[np.ndarray] = None,
    num_signatures: Optional[int] = None,
    branch_length_source: str = "newick",
):
    """
    Build a `TreeHDP` (switching enabled) for `newick` and `counts` and
    return `(model, depth_arrays, nodes_by_depth_list)`, the model's own
    depth bookkeeping. Exactly one of `fixed_signatures` / `num_signatures`
    is given, as for `TreeHDP`.
    """
    from src.models.hdp_inference import TreeHDP

    switching = dict(_BUILD_SWITCHING, branch_length_source=branch_length_source)
    model = TreeHDP(
        newick,
        counts,
        priors=_BUILD_PRIORS,
        fixed_signatures=fixed_signatures,
        num_signatures=num_signatures,
        switching=switching,
    )
    nbd = model._get_nodes_by_depth()
    nodes_by_depth_list = [nbd[d] for d in range(max(nbd) + 1)]
    return (
        model,
        model._build_switch_depth_arrays(nodes_by_depth_list),
        nodes_by_depth_list,
    )


def compile_pruning(K: int, C: int, depth: DepthArrays, mode=None):
    """
    Compile `switch_pruning.prune` for one forest.

    Returns a callable `run(eta_by_depth, S, lambda_on, lambda_off, pi)`
    taking NumPy arrays (`eta_by_depth[d]` of shape `(n_d, K)`, `S` of
    `(K, C)`, the rest `(K,)`) and returning
    `(log_beta_by_depth, log_msg_by_depth, logT_by_depth, logpi_vec)` as
    NumPy arrays, with `log_msg_by_depth[0]` and `logT_by_depth[0]` None.
    Every shape is static (Python ints from `depth`), as the pruning graph
    requires.
    """
    n_by_depth = [c.shape[0] for c in depth.counts]
    eta_in = [
        pt.tensor(f"eta{d}", dtype="float64", shape=(n, K))
        for d, n in enumerate(n_by_depth)
    ]
    S_t = pt.tensor("S", dtype="float64", shape=(K, C))
    lam_on_t = pt.tensor("lambda_on", dtype="float64", shape=(K,))
    lam_off_t = pt.tensor("lambda_off", dtype="float64", shape=(K,))
    pi_t = pt.tensor("pi", dtype="float64", shape=(K,))

    log_beta, log_msg, logT, logpi_vec, _, _ = prune(
        eta_in, S_t, lam_on_t, lam_off_t, pi_t, depth, K
    )
    outputs = list(log_beta) + list(log_msg[1:]) + list(logT[1:]) + [logpi_vec]
    kwargs = {} if mode is None else {"mode": mode}
    fn = pytensor.function(eta_in + [S_t, lam_on_t, lam_off_t, pi_t], outputs, **kwargs)
    n_depths = len(n_by_depth)

    def run(eta_by_depth, S, lambda_on, lambda_off, pi):
        out = fn(*eta_by_depth, S, lambda_on, lambda_off, pi)
        out = [np.asarray(o) for o in out]
        lb = out[:n_depths]
        lm = [None] + out[n_depths : 2 * n_depths - 1]
        lt = [None] + out[2 * n_depths - 1 : 3 * n_depths - 2]
        return lb, lm, lt, out[-1]

    return run


def _sample_rows(logits: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One categorical draw per row of unnormalised log-probabilities."""
    m = logits.max(axis=1, keepdims=True)
    m = np.where(np.isinf(m), 0.0, m)
    p = np.exp(logits - m)
    p /= p.sum(axis=1, keepdims=True)
    cdf = np.cumsum(p, axis=1)
    u = rng.random(logits.shape[0])[:, None]
    idx = (u > cdf).sum(axis=1)
    return np.minimum(idx, logits.shape[1] - 1)


def ffbs_draw(
    log_beta: Sequence[np.ndarray],
    logT: Sequence[Optional[np.ndarray]],
    logpi_vec: np.ndarray,
    depth: DepthArrays,
    masks: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    One forward-filter backward-sample of every node's joint state.

    Roots from `softmax(log_pi + log_beta[0])`; then, depth by depth, each
    child from `softmax(logT(s_parent -> .) + log_beta_child)`. Returns the
    flat state index per node, shape `(N,)`, in depth-major order.
    """
    K = masks.shape[1]
    masks_int = masks.astype(int)
    ar = np.arange(K)[None, :]
    states: List[np.ndarray] = [_sample_rows(logpi_vec[None, :] + log_beta[0], rng)]
    for d in range(1, depth.max_depth + 1):
        parent_states = states[d - 1][depth.parent_pos[d]]  # (n_d,)
        sp_bits = masks_int[parent_states]  # (n_d, K)
        lt = logT[d]  # (n_d, K, 2, 2): [i=parent bit, j=child bit]
        rows = np.take_along_axis(lt, sp_bits[:, :, None, None], axis=2)[:, :, 0, :]
        # rows: (n_d, K, 2) = logT[:, k, parent bit, child bit j]
        trans = rows[:, ar, masks_int].sum(axis=-1)  # (n_d, 2^K)
        states.append(_sample_rows(trans + log_beta[d], rng))
    return np.concatenate(states)


def sample_states(
    post,
    depth: DepthArrays,
    K: int,
    fixed_signatures: Optional[np.ndarray] = None,
    compiled=None,
    thin: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward-filter backward-sample one joint state per posterior draw.

    Parameters
    ----------
    post : xarray posterior with `eta_level_<d>`, `lambda_on`, `lambda_off`,
        `pi_root` and, when S is latent, `signatures` (aligned trace in de
        novo mode).
    depth : the model's `DepthArrays`.
    K : number of signatures.
    fixed_signatures : (K, C) array when S is known; None reads
        `post["signatures"]` per draw.
    compiled : output of `compile_pruning` (built here if None).
    thin : use every `thin`-th draw.
    rng : NumPy Generator (default_rng(0) if None).

    Returns
    -------
    states : int8 `(chains, n_draws_used, N, K)` on/off per node and
        signature, nodes in depth-major order.
    draw_idx : the draw indices used, shape `(n_draws_used,)`.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n_chains, n_draws = post.sizes["chain"], post.sizes["draw"]
    draw_idx = np.arange(0, n_draws, max(int(thin), 1))
    n_depths = depth.max_depth + 1
    eta_all = [post[f"eta_level_{d}"].values for d in range(n_depths)]
    lam_on_all = post["lambda_on"].values
    lam_off_all = post["lambda_off"].values
    pi_all = post["pi_root"].values
    S_all = None if fixed_signatures is not None else post["signatures"].values
    C = fixed_signatures.shape[1] if fixed_signatures is not None else S_all.shape[-1]
    if compiled is None:
        compiled = compile_pruning(K, C, depth)
    masks = state_grid(K)
    masks_int = masks.astype(np.int8)
    N = sum(c.shape[0] for c in depth.counts)

    states = np.empty((n_chains, len(draw_idx), N, K), dtype=np.int8)
    for c in range(n_chains):
        for t, d in enumerate(draw_idx):
            eta = [e[c, d] for e in eta_all]
            S = fixed_signatures if S_all is None else S_all[c, d]
            log_beta, _, logT, logpi_vec = compiled(
                eta, S, lam_on_all[c, d], lam_off_all[c, d], pi_all[c, d]
            )
            s = ffbs_draw(log_beta, logT, logpi_vec, depth, masks, rng)
            states[c, t] = masks_int[s]
    return states, draw_idx


def node_marginals(states: np.ndarray) -> np.ndarray:
    """`P(a_jk = 1)` estimated from the samples, shape `(N, K)`."""
    return states.astype(float).mean(axis=(0, 1))


def node_labels(model, nodes_by_depth_list: Sequence[list]) -> List[str]:
    """Node labels in depth-major order, matching `sample_states`'s N axis."""
    return [
        model.graph.nodes[n].get("label", str(n))
        for nodes in nodes_by_depth_list
        for n in nodes
    ]


def edge_table(
    model, nodes_by_depth_list: Sequence[list], depth: DepthArrays
) -> pd.DataFrame:
    """
    Every parent -> child edge as `tumour, parent, child, parent_idx,
    child_idx`, the indices into the depth-major node axis. `tumour` is the
    tree index (depth 0 root order, i.e. Newick record order).
    """
    labels = node_labels(model, nodes_by_depth_list)
    offsets = np.cumsum([0] + [len(n) for n in nodes_by_depth_list])
    rows = []
    for d in range(1, depth.max_depth + 1):
        for i in range(len(nodes_by_depth_list[d])):
            p_idx = int(offsets[d - 1] + depth.parent_pos[d][i])
            c_idx = int(offsets[d] + i)
            rows.append(
                {
                    "tumour": int(depth.tree_id[d][i]),
                    "parent": labels[p_idx],
                    "child": labels[c_idx],
                    "parent_idx": p_idx,
                    "child_idx": c_idx,
                }
            )
    return pd.DataFrame(rows)


def edge_probabilities(states: np.ndarray, edges: pd.DataFrame) -> pd.DataFrame:
    """
    Per-edge, per-signature gain / loss / switch probabilities from the
    samples, pooled over chains and draws.

    `p_gain = P(a_parent = 0, a_child = 1)`, `p_loss = P(a_parent = 1,
    a_child = 0)`, `p_switch = p_gain + p_loss`. Long format:
    `tumour, parent, child, signature, p_gain, p_loss, p_switch`, where
    `signature` is the index along the trace's signature axis.
    """
    a = states.reshape(-1, states.shape[2], states.shape[3]).astype(bool)  # (S, N, K)
    K = a.shape[2]
    rows = []
    for e in edges.itertuples(index=False):
        ap, ac = a[:, e.parent_idx, :], a[:, e.child_idx, :]
        p_gain = (~ap & ac).mean(axis=0)
        p_loss = (ap & ~ac).mean(axis=0)
        for k in range(K):
            rows.append(
                {
                    "tumour": e.tumour,
                    "parent": e.parent,
                    "child": e.child,
                    "signature": k,
                    "p_gain": float(p_gain[k]),
                    "p_loss": float(p_loss[k]),
                    "p_switch": float(p_gain[k] + p_loss[k]),
                }
            )
    return pd.DataFrame(rows)


def registry_frame_to_true(perm0: np.ndarray) -> Dict[int, int]:
    """
    Map a trace-frame signature index to the true index, given `perms[0]`
    from `chain_perms_to_true` on the aligned trace (`S[perm0]` is in true
    order, so true slot i holds trace row perm0[i]).
    """
    inv = np.argsort(perm0)
    return {int(k): int(inv[k]) for k in range(len(perm0))}
