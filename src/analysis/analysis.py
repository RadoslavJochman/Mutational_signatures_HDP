"""
Shared helpers for the de novo analysis scripts.

These functions are used by more than one script in scripts/, so they are
collected here to keep a single definition of each. They cover cosine
similarity, alignment of inferred signatures to the true labelling, splitting
chains into two camps by their activity fingerprint, averaging activity per
chain, and building a phylogenetic forest from a newick string together with a
map from (depth, position) to node label.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def cos(a, b):
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) /
                 (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def chain_perms_to_true(S, true_S):
    """
    Per chain permutation mapping the trace signature labelling to the true
    labelling, by Hungarian alignment on cosine.

    The runner aligns draws only to chain 0 labelling, which is an arbitrary
    permutation away from the true order. Comparing a trace slot to true_S[k]
    without this step compares mismatched signatures and yields spuriously low
    cosines.

    S      : (chain, draw, K, C) signatures from the chain 0 aligned trace.
    true_S : (K, C) true signatures.
    Returns: perms (n_chains, K) where perms[c] reorders trace slots into true
             order, so S[c].mean(0)[perms[c]] is in true order.
    """
    n_chains, _, K, _ = S.shape
    tn = true_S / (np.linalg.norm(true_S, axis=1, keepdims=True) + 1e-12)
    perms = np.empty((n_chains, K), dtype=int)
    for c in range(n_chains):
        Sm = S[c].mean(axis=0)
        sn = Sm / (np.linalg.norm(Sm, axis=1, keepdims=True) + 1e-12)
        cos_mat = sn @ tn.T                       # (K_trace, K_true)
        ri, ci = linear_sum_assignment(-cos_mat)
        inv = np.empty(K, dtype=int)
        inv[ci] = ri                              # inv[true] = trace slot
        perms[c] = inv
    return perms


def detect_camps(chain_act):
    """
    Split chains into two camps along the most between chain variable activity
    component.

    chain_act : (n_chains, K).
    Returns a dict with the splitting component kstar, the camp memberships
    campA (low side) and campB (high side), a separation quality (the largest
    gap divided by the median of the other gaps along the splitting axis), and
    the axis values.
    """
    n_chains, K = chain_act.shape
    kstar = int(np.argmax(chain_act.var(axis=0)))
    axis = chain_act[:, kstar]
    order = np.argsort(axis)
    gaps = np.diff(axis[order])
    split = int(np.argmax(gaps))                  # break at the largest gap
    campA = sorted(order[: split + 1].tolist())   # low side
    campB = sorted(order[split + 1:].tolist())    # high side
    if len(gaps) > 1:
        others = np.delete(gaps, split)
        sep = float(gaps[split] / (np.median(others) + 1e-12))
    else:
        sep = np.inf
    return {"kstar": kstar, "campA": campA, "campB": campB,
            "separation": sep, "axis": axis}


def per_chain_activity(e_arrays):
    """
    Mean activity per (chain, component), averaged over draws and nodes.

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


def build_forest(newick_string):
    """
    Build a directed forest from a newick string (one tree per record) and
    return it as a networkx DiGraph with nodes relabelled to their labels.
    """
    import networkx as nx
    import phylox
    G = nx.DiGraph()
    for s in newick_string.split(";"):
        if not s.strip():
            continue
        t = phylox.DiNetwork.from_newick(s + ";")   # ';' restored for phylox; same topology/labels as the model's parse
        mapping = {n: t.nodes[n].get("label", str(n)) for n in t.nodes()}
        G = nx.compose(G, nx.relabel_nodes(t, mapping))
    return G


def nodes_by_depth(G):
    """
    Map (depth, position) to node label for a forest, where depth is the
    distance from a root and position is the order in which nodes at that depth
    are first reached.
    """
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
