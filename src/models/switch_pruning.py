"""
switch_pruning.py

Exact marginalisation of per-node signature on/off states by Felsenstein
pruning over a tree-structured, factorised two-state Markov chain. Pure
PyTensor graph builders: no PyMC, no config, no Newick/forest parsing. See
`switch_model_plan.md` sections 1 and 2 for the derivation.

State space
-----------
Each node j carries a latent `a_j in {0,1}^K` (never sampled; always
marginalised). Signature k switches independently along each edge as a
two-state chain: gain 0->1 at rate `lambda_on_k`, loss 1->0 at rate
`lambda_off_k`; the root state is `Bernoulli(pi_k)`. Conditional on `a_j`,
node j's emission is a masked-softmax multinomial:

    e_j(a) = a * exp(eta_j) / sum_k a_k exp(eta_jk)
    theta_j(a) = e_j(a) @ S
    x_j | a_j ~ Multinomial(M_j, theta_j(a_j)),  with P(x_j | a_j = 0) = 0
                                                  whenever node j is observed

Nodes are grouped by depth, one list entry per depth, in the same order as
`_get_nodes_by_depth` (`DepthArrays` below holds this bookkeeping). Depth 0
holds the forest's roots; every other depth's nodes each have a parent at
the depth above.

Static shapes
-------------
`n_d` (nodes at depth d), `K`, `2**K` and `C` must all be Python ints, fixed
when the graph is built, not read off a tensor's runtime `.shape`. The
per-signature axis-wise contraction below reshapes tensors to `(n_d, 2, R)`
with `R` a Python int. A dynamic shape anywhere in these reshapes fails to
JIT-trace under `mode="JAX"` (the sampler is numpyro): a `pt.tensordot`
version of the emission step failed this way during the design spike; the
fix was plain matmul (`e_all @ S`) plus tensors declared with static shapes
throughout. See `switch_model_plan.md` section 9.

Numerical safety
-----------------
`-inf` enters the graph only as a `pt.set_subtensor` constant (the all-off
emission at an observed node, see `emission_loglik`), never as the result of
`pt.log` of a possibly-zero quantity, and never as an unselected branch of a
`pt.where` that a gradient could flow through: JAX's `where` propagates NaN
gradients from the branch that was not selected, so every `where` here has
both branches finite.

`pt.logsumexp` is never used: in PyTensor 2.31.7 it is the naive
`log(sum(exp(x)))` (no max-shift), which silently returns `-inf` for any
input more negative than about -745, since `exp(x)` underflows to exactly
0.0 there. Real per-node log-likelihoods reach into the -1000s at a burden
of a few hundred mutations, well past that threshold; a smoke-config run on
real generated data hit this (`switch_loglik = -inf` at a perfectly
ordinary point, traced to this). `_stable_logsumexp` below does the
max-shift by hand and is used everywhere a log-sum-exp reduction is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pytensor.tensor as pt


def _stable_logsumexp(x: pt.TensorVariable, axis: int) -> pt.TensorVariable:
    """Log-sum-exp along one axis, stable at any magnitude.

    Do not use `pt.logsumexp` (see the module docstring). `x_max` is
    replaced by 0 when it is `-inf` (every entry along `axis` is `-inf`),
    so the result is `-inf` in that case rather than `-inf - (-inf) = nan`.
    """
    x_max = pt.max(x, axis=axis, keepdims=True)
    safe_max = pt.where(pt.isinf(x_max), pt.zeros_like(x_max), x_max)
    summed = pt.sum(pt.exp(x - safe_max), axis=axis, keepdims=True)
    result = pt.log(summed) + safe_max
    return pt.squeeze(result, axis=axis)


def state_grid(K: int) -> np.ndarray:
    """Return the `(2**K, K)` 0/1 mask array of every joint on/off state.

    Row `s` has bit `k` equal to `(s >> (K - 1 - k)) & 1`, so flat index `0`
    is all-off, and flat index `s` corresponds to position
    `(s_0, ..., s_{K-1})` in a tensor of shape `(2,) * K` in C order.
    """
    if K == 0:
        return np.zeros((1, 0))  # one state, no free signatures
    s = np.arange(2**K)
    bits = [(s >> (K - 1 - k)) & 1 for k in range(K)]
    return np.stack(bits, axis=1).astype("float64")


def full_masks(K: int, always_on: Sequence[int] = ()) -> Tuple[np.ndarray, List[int]]:
    """State masks over the free signatures only, with `always_on` reinserted.

    The joint state space is `state_grid` over the `K - m` signatures not in
    `always_on`; each always-on signature is a constant 1 column, so it is on
    in every state and never switches. Returns `(masks, free)`: `masks` of
    shape `(2**(K - m), K)` and `free`, the indices of the switching
    signatures in increasing order (the axis order of the state grid). With
    `always_on` empty this is `state_grid(K)` and `free = range(K)`.
    """
    always_on = sorted(set(int(k) for k in always_on))
    for k in always_on:
        if not 0 <= k < K:
            raise ValueError(f"always_on index {k} outside range(K={K})")
    free = [k for k in range(K) if k not in always_on]
    grid = state_grid(len(free))
    masks = np.ones((grid.shape[0], K))
    masks[:, free] = grid
    return masks, free


def masked_softmax(eta: pt.TensorVariable, masks: np.ndarray) -> pt.TensorVariable:
    """Masked-softmax activity under every joint state.

    Parameters
    ----------
    eta : (n, K) tensor.
    masks : (2**K, K) array from `state_grid`.

    Returns
    -------
    e_all : (n, 2**K, K) tensor. Rows sum to one for every state `s != 0`;
        the all-off row (`s = 0`) is exactly zero.
    """
    masks_t = pt.as_tensor_variable(masks)  # (2^K, K)
    eta_row = eta[:, None, :]  # (n, 1, K)
    row_max = pt.max(eta, axis=-1, keepdims=True)[:, None, :]  # (n, 1, 1)
    w = masks_t[None, :, :] * pt.exp(eta_row - row_max)  # (n, 2^K, K)
    den = pt.sum(w, axis=-1)  # (n, 2^K); zero only at the all-off state
    den_safe = pt.where(den > 0, den, 1.0)  # both branches finite constants/den
    return w / den_safe[:, :, None]


def emission_loglik(
    e_all: pt.TensorVariable,
    S: pt.TensorVariable,
    counts: pt.TensorVariable,
    observed: pt.TensorVariable,
    has_all_off: bool = True,
) -> pt.TensorVariable:
    """Multinomial log-emission per node per joint state.

    Parameters
    ----------
    e_all : (n, n_states, K) tensor from `masked_softmax`.
    S : (K, C) tensor, the signature matrix.
    counts : (n, C) tensor, observed mutation counts (any values at
        unobserved nodes; ignored there).
    observed : (n,) bool tensor.
    has_all_off : whether state 0 is the all-off state (true for
        `state_grid`; false when `full_masks` has an always-on signature, so
        no state has every signature off and none is excluded).

    Returns
    -------
    log_em : (n, n_states) tensor. Includes the multinomial normalising
        constant, so this is the true log-emission, not a quantity that is
        only proportional to it. With `has_all_off`, the all-off column
        (`s = 0`) is `-inf` where `observed` is true (a node with counts
        cannot have every signature off) and `0` where `observed` is false
        (an unobserved node's likelihood does not depend on its state).
    """
    theta = (
        e_all @ S
    )  # (n, 2^K, C), plain matmul -- never pt.tensordot (see module docstring)
    log_theta = pt.log(pt.clip(theta, 1e-300, 1.0))
    ll = pt.sum(counts[:, None, :] * log_theta, axis=-1)  # (n, 2^K)
    m_total = pt.sum(counts, axis=-1)  # (n,)
    norm_const = m_total.astype("float64")
    norm_const = pt.gammaln(norm_const + 1) - pt.sum(pt.gammaln(counts + 1), axis=-1)
    ll = ll + norm_const[:, None]
    log_em = pt.where(observed[:, None], ll, pt.zeros_like(ll))
    if not has_all_off:
        return log_em
    all_off_value = pt.where(observed, -np.inf, 0.0)  # constant per node, no grad path
    log_em = pt.set_subtensor(log_em[:, 0], all_off_value)
    return log_em


def log_pi(pi: pt.TensorVariable, masks: np.ndarray) -> pt.TensorVariable:
    """Root-state log-prior over every joint state.

    `log_pi(s) = sum_k [s_k log(pi_k) + (1 - s_k) log(1 - pi_k)]`.

    Parameters
    ----------
    pi : (K,) tensor, per-signature root activation probability.
    masks : (2**K, K) array from `state_grid`.

    Returns
    -------
    (2**K,) tensor.

    Notes
    -----
    Written as a per-signature `where` summed over k rather than the matmul
    `masks @ log(pi) + (1 - masks) @ log1p(-pi)`. The two agree for every
    `pi` in (0, 1), which is all a Beta prior ever samples, but the matmul
    form gives `0 * log(0) = nan` in the all-on row at `pi = 1` exactly,
    while this form gives 0 there and `-inf` for every other state. That
    exact limit (`pi = 1`, `lambda = 0`: every signature forced on) is the
    bridge test of switch_model_plan.md section 3.4, where the pruned
    marginal likelihood must equal the plain multinomial's, so it has to be
    evaluable. Gradients at that boundary are undefined in either form.
    """
    masks_t = pt.as_tensor_variable(masks)  # (2^K, K)
    per_k = pt.where(masks_t > 0, pt.log(pi)[None, :], pt.log1p(-pi)[None, :])
    return pt.sum(per_k, axis=-1)


def log_transition(
    lambda_on: pt.TensorVariable,
    lambda_off: pt.TensorVariable,
    length: pt.TensorVariable,
) -> pt.TensorVariable:
    """Per-node, per-signature log transition matrix along the edge to the parent.

    Parameters
    ----------
    lambda_on, lambda_off : (K,) tensors.
    length : (n,) tensor, normalised branch length to the parent.

    Returns
    -------
    logT : (n, K, 2, 2) tensor. `logT[:, k, i, j] = log P(s_k: i -> j)`,
        `i` the parent's state, `j` the child's state.
    """
    l_col = length[:, None]  # (n, 1)
    neg_on = -lambda_on[None, :] * l_col  # (n, K)
    neg_off = -lambda_off[None, :] * l_col  # (n, K)
    log_p_gain = pt.log1mexp(neg_on)  # (n, K): P(0 -> 1)
    log_p_loss = pt.log1mexp(neg_off)  # (n, K): P(1 -> 0)
    row0 = pt.stack([neg_on, log_p_gain], axis=-1)  # (n, K, 2): i=0, j=[0,1]
    row1 = pt.stack([log_p_loss, neg_off], axis=-1)  # (n, K, 2): i=1, j=[0,1]
    return pt.stack([row0, row1], axis=-2)  # (n, K, 2, 2): axis -2 is i, axis -1 is j


def log_transition_iid(pi: pt.TensorVariable, n: int) -> pt.TensorVariable:
    """The tree-free ablation's transition: every edge draws the child's
    state from the root prior, independent of the parent's state and of the
    branch length, so states are i.i.d. across nodes and the tree carries no
    information about on/off (switch_model_plan.md section 7.2).

    Parameters
    ----------
    pi : (K,) tensor, per-signature activation probability.
    n : number of nodes at this depth (Python int).

    Returns
    -------
    logT : (n, K, 2, 2) tensor with `logT[:, k, i, 1] = log(pi_k)` and
        `logT[:, k, i, 0] = log(1 - pi_k)` for both parent states `i`, the
        same layout as `log_transition` so the contractions are unchanged.
    """
    row = pt.stack([pt.log1p(-pi), pt.log(pi)], axis=-1)  # (K, 2): j = 0, 1
    per_k = pt.stack([row, row], axis=-2)  # (K, 2, 2): i, then j
    return pt.tile(per_k[None, :, :, :], (n, 1, 1, 1))


def contract_child_to_parent(
    log_beta_child: pt.TensorVariable, logT: pt.TensorVariable, K: int
) -> pt.TensorVariable:
    """Sum a child depth's belief into a message indexed by the parent's state.

    For each signature axis `k` in turn, contracts the child's state `j`
    against `logT[:, k, i, j]`, leaving the parent's state `i`. Cost
    `O(K * 2**K)`, never builds the full `4**K` Kronecker transition.

    Parameters
    ----------
    log_beta_child : (n, 2**K) tensor.
    logT : (n, K, 2, 2) tensor from `log_transition`; `logT[:, k, i, j] =
        log P(s_k: i -> j)`, `i` the parent's state, `j` the child's.
    K : number of signatures (Python int; fixes every reshape below).

    Returns
    -------
    (n, 2**K) tensor, axis `s` now indexing the PARENT's joint state.
    """
    n = log_beta_child.type.shape[0]
    if n is None:
        raise ValueError(
            "contract_child_to_parent needs a static shape[0] on log_beta_child "
            "(see the module docstring on static shapes)."
        )
    x = log_beta_child.reshape((n,) + (2,) * K)
    r = 2 ** (K - 1)
    for k in range(K):
        perm = [0, k + 1] + [i for i in range(1, K + 1) if i != k + 1]
        x = x.transpose(perm)
        x2 = x.reshape((n, 2, r))  # child's s_k at axis 1
        lt = logT[:, k, :, :]  # (n, 2(i), 2(j))
        combined = lt[:, :, :, None] + x2[:, None, :, :]  # (n, 2(i), 2(j), R)
        contracted = _stable_logsumexp(combined, axis=2)  # (n, 2(i)=parent, R)
        x = contracted.reshape((n, 2) + (2,) * (K - 1))
        inv_perm = [0] * (K + 1)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        x = x.transpose(inv_perm)
    return x.reshape((n, 2**K))


def contract_parent_to_child(
    log_belief_parent: pt.TensorVariable, logT: pt.TensorVariable, K: int
) -> pt.TensorVariable:
    """Same contraction, the other direction (parent's state contracted away).

    Equivalent to `contract_child_to_parent` with `logT` transposed on its
    last two axes, so the surviving axis is the CHILD's state.
    """
    logT_rev = logT.transpose(0, 1, 3, 2)
    return contract_child_to_parent(log_belief_parent, logT_rev, K)


@dataclass
class DepthArrays:
    """Per-depth NumPy bookkeeping, in `_get_nodes_by_depth` order.

    Attributes
    ----------
    counts : list of (n_d, C) arrays.
    observed : list of (n_d,) bool arrays.
    parent_pos : list, `None` at depth 0; otherwise (n_d,) int, index of
        each node's parent within depth `d - 1`.
    length : list, `None` at depth 0; otherwise (n_d,) float, normalised
        branch length to the parent.
    tree_id : list of (n_d,) int arrays, which forest tree each node
        belongs to (shared with that tree's root at depth 0).
    """

    counts: List[np.ndarray]
    observed: List[np.ndarray]
    parent_pos: List[Optional[np.ndarray]]
    length: List[Optional[np.ndarray]]
    tree_id: List[np.ndarray]

    @property
    def max_depth(self) -> int:
        return len(self.counts) - 1

    @classmethod
    def build(
        cls,
        counts: Sequence[np.ndarray],
        observed: Sequence[np.ndarray],
        parent_pos: Sequence[Optional[np.ndarray]],
        length: Sequence[Optional[np.ndarray]],
        tree_of_root: np.ndarray,
    ) -> "DepthArrays":
        """Construct from per-depth arrays, deriving `tree_id` by depth.

        `tree_of_root` gives depth 0's tree id per root; every deeper
        depth's `tree_id` is chased down through `parent_pos`.
        """
        tree_id = [np.asarray(tree_of_root)]
        for d in range(1, len(counts)):
            tree_id.append(tree_id[d - 1][parent_pos[d]])
        return cls(
            counts=list(counts),
            observed=list(observed),
            parent_pos=list(parent_pos),
            length=list(length),
            tree_id=tree_id,
        )


def prune(
    eta_by_depth: Sequence[pt.TensorVariable],
    S: pt.TensorVariable,
    lambda_on: pt.TensorVariable,
    lambda_off: pt.TensorVariable,
    pi: pt.TensorVariable,
    depth: DepthArrays,
    K: int,
    always_on: Sequence[int] = (),
    tree_coupled: bool = True,
):
    """Upward (Felsenstein pruning) pass over the whole forest.

    Parameters
    ----------
    eta_by_depth : one (n_d, K) tensor per depth.
    S : (K, C) tensor.
    lambda_on, lambda_off, pi : (K,) tensors.
    depth : `DepthArrays`.
    K : number of signatures (Python int).
    always_on : indices of signatures forced on (see `full_masks`). They are
        dropped from the state grid, so the state space is `2**(K - m)`, and
        their `lambda`/`pi` entries are ignored.
    tree_coupled : with False, every edge transition is replaced by the root
        prior (`log_transition_iid`): states are i.i.d. across nodes,
        `lambda_on`/`lambda_off` and the branch lengths are ignored, and the
        tree carries no information about on/off. The tree-free ablation.

    Returns
    -------
    log_beta_by_depth : list of (n_d, n_states) tensors.
    log_msg_by_depth : list, `None` at depth 0; otherwise (n_d, n_states)
        tensors, the message each node sent to its parent.
    logT_by_depth : list, `None` at depth 0; otherwise (n_d, K - m, 2, 2)
        tensors over the free signatures, so `backward` need not recompute
        them.
    logpi_vec : (n_states,) tensor.
    logZ : scalar tensor, the total forest log-likelihood (sum over roots),
        entering the model as `pm.Potential("switch_loglik", logZ)`.
    logZ_per_root : (n_0,) tensor.
    """
    masks, free = full_masks(K, always_on)
    n_free, n_states = len(free), masks.shape[0]
    has_all_off = n_free == K
    if not has_all_off:
        free_idx = np.asarray(free)
        lambda_on, lambda_off, pi = (
            lambda_on[free_idx],
            lambda_off[free_idx],
            pi[free_idx],
        )
    logpi_vec = log_pi(pi, state_grid(n_free))
    max_depth = depth.max_depth
    n_by_depth = [c.shape[0] for c in depth.counts]

    acc = [pt.zeros((n, n_states)) for n in n_by_depth]
    log_beta: List[Optional[pt.TensorVariable]] = [None] * (max_depth + 1)
    log_msg: List[Optional[pt.TensorVariable]] = [None] * (max_depth + 1)
    logT_by_depth: List[Optional[pt.TensorVariable]] = [None] * (max_depth + 1)

    for d in range(max_depth, -1, -1):
        eta_d = pt.specify_shape(eta_by_depth[d], (n_by_depth[d], K))
        e_all = masked_softmax(eta_d, masks)
        counts_d = pt.as_tensor_variable(depth.counts[d].astype("float64"))
        observed_d = pt.as_tensor_variable(depth.observed[d].astype(bool))
        log_beta[d] = (
            emission_loglik(e_all, S, counts_d, observed_d, has_all_off=has_all_off)
            + acc[d]
        )
        if d > 0:
            length_d = pt.as_tensor_variable(depth.length[d].astype("float64"))
            if tree_coupled:
                logT_d = log_transition(lambda_on, lambda_off, length_d)
            else:
                logT_d = log_transition_iid(pi, n_by_depth[d])
            logT_by_depth[d] = logT_d
            log_msg[d] = contract_child_to_parent(log_beta[d], logT_d, n_free)
            acc[d - 1] = pt.inc_subtensor(acc[d - 1][depth.parent_pos[d]], log_msg[d])

    logZ_per_root = _stable_logsumexp(
        logpi_vec[None, :] + log_beta[0], axis=1
    )  # (n_0,)
    logZ = pt.sum(logZ_per_root)
    return log_beta, log_msg, logT_by_depth, logpi_vec, logZ, logZ_per_root


def backward(
    eta_by_depth: Sequence[pt.TensorVariable],
    log_beta_by_depth: Sequence[pt.TensorVariable],
    log_msg_by_depth: Sequence[Optional[pt.TensorVariable]],
    logT_by_depth: Sequence[Optional[pt.TensorVariable]],
    logpi_vec: pt.TensorVariable,
    logZ_per_root: pt.TensorVariable,
    depth: DepthArrays,
    K: int,
    always_on: Sequence[int] = (),
):
    """Downward pass: per-node state posterior and the reported Deterministics.

    Takes the outputs of `prune` (plus `eta_by_depth` again, to rebuild
    `e_all`), with the same `always_on`. Cost is not on the sampler: every
    output here is a Deterministic, evaluated after sampling.

    Returns
    -------
    log_q_by_depth : list of (n_d, n_states) tensors, the normalised log
        joint state posterior per node.
    a_prob_by_depth : list of (n_d, K) tensors, `P(a_jk = 1 | data)`;
        exactly 1 in every `always_on` column.
    e_level_by_depth : list of (n_d, K) tensors, the state-mixed expected
        activity (rows sum to one); this is what `e_level_d` becomes when
        switching is enabled.
    """
    masks, free = full_masks(K, always_on)
    n_free = len(free)
    has_all_off = n_free == K
    masks_t = pt.as_tensor_variable(masks)  # (n_states, K)
    max_depth = depth.max_depth
    n0 = depth.counts[0].shape[0]

    log_alpha: List[Optional[pt.TensorVariable]] = [None] * (max_depth + 1)
    log_alpha[0] = pt.tile(logpi_vec[None, :], (n0, 1))
    for d in range(1, max_depth + 1):
        combined_parent = log_alpha[d - 1] + log_beta_by_depth[d - 1]  # (n_{d-1}, 2^K)
        leave_one_out = (
            combined_parent[depth.parent_pos[d]] - log_msg_by_depth[d]
        )  # (n_d, 2^K)
        log_alpha[d] = contract_parent_to_child(leave_one_out, logT_by_depth[d], n_free)

    log_q_by_depth: List[pt.TensorVariable] = []
    a_prob_by_depth: List[pt.TensorVariable] = []
    e_level_by_depth: List[pt.TensorVariable] = []
    for d in range(max_depth + 1):
        logZ_node = logZ_per_root[depth.tree_id[d]]  # (n_d,)
        log_q_d = log_alpha[d] + log_beta_by_depth[d] - logZ_node[:, None]  # (n_d, 2^K)
        q_d = pt.exp(log_q_d)
        a_prob_d = q_d @ masks_t  # (n_d, K)
        if not has_all_off:
            on_idx = [k for k in range(K) if k not in free]
            a_prob_d = pt.set_subtensor(a_prob_d[:, on_idx], 1.0)

        e_all_d = masked_softmax(eta_by_depth[d], masks)  # (n_d, n_states, K)
        q_no_off = pt.set_subtensor(q_d[:, 0], 0.0) if has_all_off else q_d
        numer = pt.sum(q_no_off[:, :, None] * e_all_d, axis=1)  # (n_d, K)
        denom = pt.sum(q_no_off, axis=1, keepdims=True)  # (n_d, 1)
        e_level_d = numer / denom

        log_q_by_depth.append(log_q_d)
        a_prob_by_depth.append(a_prob_d)
        e_level_by_depth.append(e_level_d)

    return log_q_by_depth, a_prob_by_depth, e_level_by_depth
