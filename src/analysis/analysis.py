"""
Shared helpers for the de novo analysis scripts.

Logit-simplex maps
    softmax_last_zero / inv_softmax_last_zero, the anchored softmax link the
    activity walk uses and its inverse, as numpy counterparts of the model's
    pytensor version.

Distribution distances
    total_variation, hellinger, jensen_shannon, bray_curtis, cosine and
    relative_exposure_error, with the DISTRIBUTION_METRICS registry and the
    builders that map a named metric over signatures, per-signature usage
    profiles or per-node compositions (signature_distances, usage_distances,
    node_distances, exposure_errors), plus across_chain for best/mean/worst.

Signature alignment
    align (Hungarian on cosine) and chain_perms_to_true, which undo the de novo
    label-switching symmetry before anything is compared.

Forest and graph helpers
    node_order, node_depths, node_label, build_forest and nodes_by_depth, for
    walking a fitted model's graph or a bare newick string.

The non-centred walk
    forward_walk / inverse_walk convert between the free walk variables
    (eta_level_*, z_level_*) and per-node activities; activities_mean /
    activities_draw read per-node activities out of a posterior.

Camp detection
    per_chain_activity, detect_camps and split_camps split the chains into the
    two clusters the de novo posterior falls into (named "camps" here for
    historical reasons, the report calls them clusters).

Model construction
    build_model and DEFAULT_PRIORS build a DeNovoHDP from file paths with the
    shared prior configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def softmax_last_zero(eta: np.ndarray) -> np.ndarray:
    """(..., K-1) free logits -> (..., K) simplex point, last component pinned
    to logit 0. numpy counterpart of the model's pytensor `_softmax_last_zero`."""
    eta = np.asarray(eta, dtype=float)
    zeros = np.zeros(eta.shape[:-1] + (1,))
    full = np.concatenate([eta, zeros], axis=-1)
    full = full - full.max(axis=-1, keepdims=True)
    e = np.exp(full)
    return e / e.sum(axis=-1, keepdims=True)


def inv_softmax_last_zero(e: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """(..., K) simplex point -> (..., K-1) logits, anchor = last component:
    eta_k = log(e_k) - log(e_{K-1}). Inverse of `softmax_last_zero` up to the
    floor. Default floor matches the value used in real runs (nmf_init)."""
    e = np.clip(np.asarray(e, dtype=float), floor, None)
    return np.log(e[..., :-1]) - np.log(e[..., -1:])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _row_normalise(X: np.ndarray) -> np.ndarray:
    """Each row of X scaled to unit L2 norm (builds the cosine matrix in align)."""
    X = np.asarray(X, dtype=float)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def _as_dist(x: np.ndarray) -> np.ndarray:
    """Non-negative vector renormalised to sum one, read as a distribution."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, None)
    s = x.sum()
    return x / s if s > 0 else x


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats, summed over the support of p."""
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """Total-variation distance, half the L1: the fraction of probability mass
    in the wrong place. Bounded in [0, 1], linear so it does not saturate."""
    p, q = _as_dist(p), _as_dist(q)
    return 0.5 * float(np.abs(p - q).sum())


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    """Hellinger distance, the chord between the square-roots. Bounded in
    [0, 1], symmetric, a proper metric; sensitive on the small components."""
    p, q = _as_dist(p), _as_dist(q)
    return float(np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2.0))


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon distance, the square-root of the base-2 JS divergence: a
    symmetrised, bounded ([0, 1]) KL that stays finite when supports differ."""
    p, q = _as_dist(p), _as_dist(q)
    m = 0.5 * (p + q)
    return float(np.sqrt(0.5 * (_kl(p, m) + _kl(q, m)) / np.log(2.0)))


def relative_exposure_error(recovered_total: float, true_total: float) -> float:
    """|recovered - true| / true for a signature's total exposure: the level
    error the shape distances and cosine are all scale-invariant to."""
    return float(abs(recovered_total - true_total) / (abs(true_total) + 1e-12))


def bray_curtis(a: np.ndarray, b: np.ndarray) -> float:
    """Bray-Curtis dissimilarity for two non-negative abundance vectors, the L1
    difference over the summed totals. Bounded in [0, 1], symmetric, and unlike
    cosine it is scale-sensitive, so two activity patterns that differ only in
    overall level are still far apart. Reduces to total variation once the
    vectors are normalised, so it sits in the same family as the shape metrics."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.abs(a - b).sum() / (a.sum() + b.sum() + 1e-12))


# name -> (function, higher_is_better) for the table builders and aggregator
DISTRIBUTION_METRICS = {
    "tv": (total_variation, False),
    "hellinger": (hellinger, False),
    "js": (jensen_shannon, False),
    "cosine": (cosine, True),
}


def _apply(metric: str, est_rows: np.ndarray, true_rows: np.ndarray) -> np.ndarray:
    """Map a named metric over matched rows of two (n, d) arrays -> (n,)."""
    fn = DISTRIBUTION_METRICS[metric][0]
    return np.array([fn(est_rows[i], true_rows[i]) for i in range(len(true_rows))])


def signature_distances(
    sig_est: np.ndarray, true_S: np.ndarray, metrics: Sequence[str]
) -> Dict[str, np.ndarray]:
    """Per-signature shape distance for each named metric. Both (K, C), already
    aligned to the same labelling. Returns {metric: (K,)}."""
    return {m: _apply(m, sig_est, true_S) for m in metrics}


def usage_distances(
    act_est: np.ndarray, true_acts: np.ndarray, metrics: Sequence[str]
) -> Dict[str, np.ndarray]:
    """Per-signature usage-profile distance: each signature's activity across
    nodes, est vs true. Both (n_nodes, K), aligned. Returns {metric: (K,)}."""
    return {m: _apply(m, act_est.T, true_acts.T) for m in metrics}


def node_distances(
    act_est: np.ndarray, true_acts: np.ndarray, metrics: Sequence[str]
) -> Dict[str, np.ndarray]:
    """Per-node composition distance: each node's activity vector, est vs true.
    Both (n_nodes, K), aligned. Returns {metric: (n_nodes,)}."""
    return {m: _apply(m, act_est, true_acts) for m in metrics}


def exposure_errors(act_est: np.ndarray, true_acts: np.ndarray) -> np.ndarray:
    """Per-signature relative exposure error from aligned (n_nodes, K) activity
    arrays: column totals compared, recovered vs true. Returns (K,)."""
    rec, tru = act_est.sum(0), true_acts.sum(0)
    return np.array([relative_exposure_error(rec[k], tru[k]) for k in range(len(tru))])


def across_chain(
    values: np.ndarray, metric: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate a (..., n_chains) array over the last axis into best, mean,
    worst, with best/worst set by the metric's direction (cosine high is good,
    distances low is good)."""
    mean = values.mean(-1)
    hi, lo = values.max(-1), values.min(-1)
    higher_is_better = DISTRIBUTION_METRICS[metric][1]
    best, worst = (hi, lo) if higher_is_better else (lo, hi)
    return best, mean, worst


def align(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hungarian alignment of `source` rows to `target` rows on cosine.

    Returns
    -------
    perm : (n_target,) int
        Reorder index: ``source[perm]`` is row-matched to ``target`` (perm[i]
        is the source row assigned to target row i). The forward map
        source_row -> target_slot is ``np.argsort(perm)``.
    cos_matrix : (n_target, n_source)
        Pairwise cosine similarity (target rows x source rows).
    """
    cos_matrix = _row_normalise(target) @ _row_normalise(source).T
    _, perm = linear_sum_assignment(-cos_matrix)
    return perm, cos_matrix


def node_order(model) -> List:
    """Flat list of graph node-ids in BFS-by-depth order -- the order the
    e_level_* / z_level_* arrays are concatenated in."""
    nbd = model._get_nodes_by_depth()
    return [n for d in sorted(nbd) for n in nbd[d]]


def node_depths(model) -> Dict:
    """{node_id: depth} via BFS from every root of the model graph."""
    import networkx as nx

    depths: Dict = {}
    roots = [n for n, d in model.graph.in_degree() if d == 0]
    for r in roots:
        for n, depth in nx.single_source_shortest_path_length(model.graph, r).items():
            depths.setdefault(n, depth)
    return depths


def node_label(model, n) -> str:
    """The stored 'label' of graph node `n`, falling back to str(n)."""
    return model.graph.nodes[n].get("label", str(n))


def _parents(model, nodes) -> list:
    """Parent of each node in `nodes` in the model graph (None for a root)."""
    return [(list(model.graph.predecessors(n)) or [None])[0] for n in nodes]


def build_forest(newick_string):
    """Build a directed forest from a newick string (one tree per record) and
    return it as a networkx DiGraph with nodes relabelled to their labels.

    Note: this parses the newick independently of the model (via phylox); for a
    fitted model prefer reading ``model.graph`` directly. Kept for the no-model
    callers (e.g. sweep_aggregate)."""
    import networkx as nx
    import phylox

    G = nx.DiGraph()
    for s in newick_string.split(";"):
        if not s.strip():
            continue
        t = phylox.DiNetwork.from_newick(s + ";")  # ';' restored for phylox
        mapping = {n: t.nodes[n].get("label", str(n)) for n in t.nodes()}
        G = nx.compose(G, nx.relabel_nodes(t, mapping))
    return G


def nodes_by_depth(G):
    """Map (depth, position) -> node label for a forest graph, where depth is
    distance from a root and position is first-reached order at that depth.
    (Graph-level helper; for a fitted model use `node_order`/`node_depths`.)"""
    import networkx as nx

    roots = [n for n, d in G.in_degree() if d == 0]
    seen, by_depth = {}, {}
    for r in roots:
        for n, depth in nx.single_source_shortest_path_length(G, r).items():
            if n not in seen:
                seen[n] = depth
                by_depth.setdefault(depth, []).append(n)
    dr = {}
    for depth, nodes in by_depth.items():
        for i, n in enumerate(nodes):
            dr[(depth, i)] = n
    return dr


def forward_walk(model, free: Dict[str, np.ndarray], sigma: float) -> Dict:
    """{eta_level_0, z_level_d, ...} (one draw or an initvals dict) -> {node_id:
    e_j} per-node full-K activities. Used for the NMF-init activity, the
    interpolation forward-walk fallback, and the displacement diagnostic."""
    nbd = model._get_nodes_by_depth()
    eta_of, e_of = {}, {}
    for d in sorted(nbd):
        current = nbd[d]
        parents = _parents(model, current)
        if parents[0] is None:
            arr = np.asarray(free[f"eta_level_{d}"])
            for i, n in enumerate(current):
                eta_of[n] = arr[i]
        else:
            z = np.asarray(free[f"z_level_{d}"])
            for i, (n, p) in enumerate(zip(current, parents)):
                eta_of[n] = eta_of[p] + sigma * z[i]
        for n in current:
            e_of[n] = softmax_last_zero(eta_of[n])
    return e_of


def inverse_walk(model, e_by_node: Dict, sigma: float) -> Dict[str, np.ndarray]:
    """{node_id (or label): e_j} -> {eta_level_0, z_level_d, ...} by inverse-
    softmax and backing z out of the non-centred walk (z = (eta_child -
    eta_parent)/sigma). Internal nodes absent from `e_by_node` inherit their
    parent's eta (z = 0). Shared core of nmf_init and the interpolation backout."""
    nbd = model._get_nodes_by_depth()
    eta_of, out = {}, {}
    for d in sorted(nbd):
        current = nbd[d]
        parents = _parents(model, current)
        eta_rows, z_rows = [], []
        for n, p in zip(current, parents):
            lab = node_label(model, n)
            if lab in e_by_node:
                eta = inv_softmax_last_zero(e_by_node[lab])
            elif n in e_by_node:
                eta = inv_softmax_last_zero(e_by_node[n])
            elif p is not None:
                eta = eta_of[p]
            else:
                eta = np.zeros(model.K - 1)
            eta_of[n] = eta
            eta_rows.append(eta)
            if p is not None:
                z_rows.append((eta - eta_of[p]) / sigma)
        if parents[0] is None:
            out[f"eta_level_{d}"] = np.array(eta_rows)
        else:
            out[f"z_level_{d}"] = np.array(z_rows)
    return out


def activities_mean(
    post, model, chains: Optional[Sequence[int]] = None
) -> Tuple[List, np.ndarray]:
    """Posterior-mean per-node activities. Returns (node_order, A) with A shape
    (n_nodes, K) in node_order, averaged over the given chains (all if None)
    and all draws. Reads the e_level_* deterministics."""
    nbd = model._get_nodes_by_depth()
    sel = dict(chain=list(chains)) if chains is not None else {}
    blocks = [
        post[f"e_level_{d}"].isel(**sel).mean(("chain", "draw")).values
        for d in sorted(nbd)
    ]
    return node_order(model), np.concatenate(blocks, axis=0)


def activities_draw(post, model, chain: int, draw: int) -> Tuple[List, np.ndarray]:
    """Per-node activities for a single (chain, draw). Reads e_level_* if
    present, else forward-walks the free RVs. Returns (node_order, A)."""
    nbd = model._get_nodes_by_depth()
    depths = sorted(nbd)
    if all(f"e_level_{d}" in post for d in depths):
        blocks = [
            post[f"e_level_{d}"].isel(chain=chain, draw=draw).values for d in depths
        ]
        return node_order(model), np.concatenate(blocks, axis=0)
    sigma = float(post["sigma"].isel(chain=chain, draw=draw).values)
    free = {
        v: post[v].isel(chain=chain, draw=draw).values
        for v in post.data_vars
        if v.startswith(("eta_level_", "z_level_"))
    }
    e_of = forward_walk(model, free, sigma)
    order = node_order(model)
    return order, np.stack([e_of[n] for n in order])


def per_chain_activity(e_arrays):
    """Mean activity per (chain, component), averaged over draws and nodes.

    e_arrays : list of (n_chains, n_draws, n_nodes, K) arrays.
    Returns  : (n_chains, K).
    """
    n_chains = e_arrays[0].shape[0]
    K = e_arrays[0].shape[3]
    acc = np.zeros((n_chains, K))
    total_nodes = 0
    for arr in e_arrays:
        acc += arr.mean(axis=1).sum(axis=1)  # mean over draws, sum nodes
        total_nodes += arr.shape[2]
    return acc / max(total_nodes, 1)


def detect_camps(chain_act):
    """
    Split chains into two camps along the most between-chain-variable activity
    component (break at the largest gap along that axis).

    chain_act : (n_chains, K).
    Returns a dict with the splitting component kstar, the camp memberships
    campA (low side) and campB (high side), a separation quality (largest gap
    over the median of the other gaps along the splitting axis), and the axis.
    """
    kstar = int(np.argmax(chain_act.var(axis=0)))
    axis = chain_act[:, kstar]
    order = np.argsort(axis)
    gaps = np.diff(axis[order])
    split = int(np.argmax(gaps))
    campA = sorted(order[: split + 1].tolist())
    campB = sorted(order[split + 1 :].tolist())
    if len(gaps) > 1:
        others = np.delete(gaps, split)
        sep = float(gaps[split] / (np.median(others) + 1e-12))
    else:
        sep = np.inf
    return {
        "kstar": kstar,
        "campA": campA,
        "campB": campB,
        "separation": sep,
        "axis": axis,
    }


def split_camps(
    post, model, chains: Optional[Sequence[int]] = None
) -> Tuple[list, list]:
    """Partition chains into two camps by activity fingerprint, via detect_camps
    (gap on the highest-between-chain-variance component). Returns (campA, campB)
    as chain-index lists. This is the canonical camp rule; the KMeans variants
    that used to live in the scripts should call this."""
    nbd = model._get_nodes_by_depth()
    e_arrays = [post[f"e_level_{d}"].values for d in sorted(nbd)]
    pc = per_chain_activity(e_arrays)  # (chain, K), each in its own frame
    S = post["signatures"].values  # (chain, draw, K, C)
    ref = S[0].mean(axis=0)
    pc_al = np.empty_like(pc)
    for c in range(pc.shape[0]):
        perm, _ = align(S[c].mean(axis=0), ref)  # source[perm] ~ ref
        pc_al[c] = pc[c][perm]
    res = detect_camps(pc_al)
    return res["campA"], res["campB"]


def chain_perms_to_true(S, true_S):
    """
    Per-chain permutation mapping the trace signature labelling to the true
    labelling, by Hungarian alignment on cosine.

    S      : (chain, draw, K, C) signatures from the chain-0 aligned trace.
    true_S : (K, C) true signatures.
    Returns: perms (n_chains, K) where S[c].mean(0)[perms[c]] is in true order.
    """
    n_chains, _, K, _ = S.shape
    perms = np.empty((n_chains, K), dtype=int)
    for c in range(n_chains):
        # align(source=trace sigs, target=true): source[perm] is in true order.
        perm, _ = align(S[c].mean(axis=0), true_S)
        perms[c] = perm
    return perms


DEFAULT_PRIORS = {
    "sigma_prior": "LogNorm",
    "sigma_prior_parm": {"mu": 0.0, "sigma": 1.0},
    "sigma_0": 2.0,
    "beta": 0.5,
}


def build_model(
    newick_path: str,
    counts_path: str,
    num_signatures: int,
    priors: Optional[dict] = None,
):
    """Construct a DeNovoHDP from file paths with the shared prior config the
    scripts had all hard-coded. Returns (model, counts). Pass `priors` to
    override DEFAULT_PRIORS."""
    import pandas as pd

    from src.models.hdp_inference import DeNovoHDP

    counts = pd.read_csv(counts_path, index_col=0)
    newick = Path(newick_path).read_text().strip()
    return DeNovoHDP(newick, counts, num_signatures, priors or DEFAULT_PRIORS), counts
