"""
Shared helpers for the de novo analysis scripts.

These functions are used by more than one script in scripts/, so they are
collected here to keep a single definition of each. They cover cosine
similarity, alignment of inferred signatures to the true labelling, splitting
chains into two camps by their activity fingerprint, averaging activity per
chain, and building a phylogenetic forest from a newick string together with a
map from (depth, position) to node label.
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
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def _row_normalise(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

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
    return model.graph.nodes[n].get("label", str(n))

def _parents(model, nodes) -> list:
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
        t = phylox.DiNetwork.from_newick(s + ";")   # ';' restored for phylox
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


def activities_mean(post, model, chains: Optional[Sequence[int]] = None
                    ) -> Tuple[List, np.ndarray]:
    """Posterior-mean per-node activities. Returns (node_order, A) with A shape
    (n_nodes, K) in node_order, averaged over the given chains (all if None)
    and all draws. Reads the e_level_* deterministics."""
    nbd = model._get_nodes_by_depth()
    sel = dict(chain=list(chains)) if chains is not None else {}
    blocks = [post[f"e_level_{d}"].isel(**sel).mean(("chain", "draw")).values
              for d in sorted(nbd)]
    return node_order(model), np.concatenate(blocks, axis=0)


def activities_draw(post, model, chain: int, draw: int) -> Tuple[List, np.ndarray]:
    """Per-node activities for a single (chain, draw). Reads e_level_* if
    present, else forward-walks the free RVs. Returns (node_order, A)."""
    nbd = model._get_nodes_by_depth()
    depths = sorted(nbd)
    if all(f"e_level_{d}" in post for d in depths):
        blocks = [post[f"e_level_{d}"].isel(chain=chain, draw=draw).values for d in depths]
        return node_order(model), np.concatenate(blocks, axis=0)
    sigma = float(post["sigma"].isel(chain=chain, draw=draw).values)
    free = {v: post[v].isel(chain=chain, draw=draw).values
            for v in post.data_vars if v.startswith(("eta_level_", "z_level_"))}
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
        acc += arr.mean(axis=1).sum(axis=1)       # mean over draws, sum nodes
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
    campB = sorted(order[split + 1:].tolist())
    if len(gaps) > 1:
        others = np.delete(gaps, split)
        sep = float(gaps[split] / (np.median(others) + 1e-12))
    else:
        sep = np.inf
    return {"kstar": kstar, "campA": campA, "campB": campB,
            "separation": sep, "axis": axis}

def split_camps(post, model, chains: Optional[Sequence[int]] = None) -> Tuple[list, list]:
    """Partition chains into two camps by activity fingerprint, via detect_camps
    (gap on the highest-between-chain-variance component). Returns (campA, campB)
    as chain-index lists. This is the canonical camp rule; the KMeans variants
    that used to live in the scripts should call this."""
    nbd = model._get_nodes_by_depth()
    e_arrays = [post[f"e_level_{d}"].values for d in sorted(nbd)]
    pc = per_chain_activity(e_arrays)              # (chain, K), each in its own frame
    S = post["signatures"].values                  # (chain, draw, K, C)
    ref = S[0].mean(axis=0)
    pc_al = np.empty_like(pc)
    for c in range(pc.shape[0]):
        perm, _ = align(S[c].mean(axis=0), ref)    # source[perm] ~ ref
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

def build_model(newick_path: str, counts_path: str, num_signatures: int,
                priors: Optional[dict] = None):
    """Construct a DeNovoHDP from file paths with the shared prior config the
    scripts had all hard-coded. Returns (model, counts). Pass `priors` to
    override DEFAULT_PRIORS."""
    import pandas as pd
    from src.models.hdp_inference import DeNovoHDP
    counts = pd.read_csv(counts_path, index_col=0)
    newick = Path(newick_path).read_text().strip()
    return DeNovoHDP(newick, counts, num_signatures, priors or DEFAULT_PRIORS), counts
