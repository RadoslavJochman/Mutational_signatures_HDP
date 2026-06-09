"""
nmf_init.py

Shared, data-driven initial values for the de novo Tree-HDP samplers.

Purpose
    The camp split at moderate overlap is a stiff, data-visible mode-lock: the
    data prefers the truth-closer camp, but chains started from random points
    fall into different basins. Starting every chain from the same informed
    point lets them share the data-preferred basin instead. This builds that
    point from a non-negative matrix factorisation of the count matrix and
    returns it as an initvals dict for pm.sample.

Method
    NMF factorises the observed counts X (N_obs x 96) as W (N_obs x K) times
    H (K x 96). The rows of H, normalised, initialise the signatures; the rows
    of W, normalised, give a per-node activity simplex point that is mapped to
    the model's free variables:
      - signatures   : row-normalised H
      - eta (root)   : inverse softmax of the activity, last component pinned
      - z_level_<d>  : (eta_node - eta_parent) / sigma_init, backing the
                       non-centred walk out of the activity path
    Node order follows the model's own BFS (_get_nodes_by_depth and the graph
    predecessors), so the arrays line up with eta_level_* / z_level_*.
    Unobserved internal nodes inherit their parent's eta (z = 0 there).

    The same point is returned for every chain, so all chains start together.
    Use it with init='adapt_diag' to keep the shared start.

Scope
    Written for the random-walk de novo model (DeNovoHDP): free variables
    signatures, sigma, eta_level_0, z_level_1, ... For the OU model the walk
    backing-out differs (mu, phi, reversion), so this helper targets the
    random walk, which is where the camp split was diagnosed.

Usage
    from src.analysis.nmf_init import denovo_nmf_initvals
    init = denovo_nmf_initvals(model, count_matrix, sigma_init=0.6, seed=0)
    model.sample(..., initvals=init, init="adapt_diag")
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF


def _inv_softmax_last_zero(e: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Map a (K,) simplex point to (K-1,) logits with the last component as
    the pinned zero reference: eta_k = log(e_k) - log(e_{K-1})."""
    e = np.clip(e, floor, None)
    return np.log(e[:-1]) - np.log(e[-1])


def denovo_nmf_initvals(
        model,
        count_matrix: pd.DataFrame,
        sigma_init: float = 0.6,
        seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Shared NMF initial values for a DeNovoHDP-style model.

    Parameters
    ----------
    model : a built _BaseTreeHDP subclass with .graph, .K, ._get_nodes_by_depth
    count_matrix : observed counts, index = node labels, 96 columns
    sigma_init : walk-scale value used to back z out of the activity path
    seed : NMF random_state

    Returns
    -------
    dict keyed by free-variable name, suitable for pm.sample(initvals=...).
    """
    K = model.K
    X = count_matrix.values.astype(float)

    # NMF of the counts into K activity components and K signatures.
    nmf = NMF(n_components=K, init="nndsvda", random_state=seed, max_iter=500)
    W = nmf.fit_transform(X)
    H = nmf.components_

    signatures_init = H + 1e-4
    signatures_init = signatures_init / signatures_init.sum(axis=1, keepdims=True)
    e_obs = W / np.clip(W.sum(axis=1, keepdims=True), 1e-12, None)
    label_to_e = {lab: e_obs[i] for i, lab in enumerate(count_matrix.index)}

    # eta for every graph node, in the model's BFS order, parents first.
    nodes_by_depth = model._get_nodes_by_depth()
    max_depth = max(nodes_by_depth) if nodes_by_depth else 0

    def node_label(n):
        return model.graph.nodes[n].get("label", str(n))

    eta_of = {}
    init: Dict[str, np.ndarray] = {}

    for depth in range(0, max_depth + 1):
        current = nodes_by_depth.get(depth, [])
        if not current:
            continue
        parents = [
            (list(model.graph.predecessors(n)) or [None])[0] for n in current
        ]
        eta_rows, z_rows = [], []
        for n, p in zip(current, parents):
            lab = node_label(n)
            if lab in label_to_e:
                eta = _inv_softmax_last_zero(label_to_e[lab])
            elif p is not None:
                eta = eta_of[p]
            else:
                eta = np.zeros(K - 1)
            eta_of[n] = eta
            eta_rows.append(eta)
            if p is not None:
                z_rows.append((eta - eta_of[p]) / sigma_init)

        if parents[0] is None:
            # root level is a free, centred eta in the random-walk model
            init[f"eta_level_{depth}"] = np.array(eta_rows)
        else:
            init[f"z_level_{depth}"] = np.array(z_rows)

    init["signatures"] = signatures_init
    init["sigma"] = float(sigma_init)
    return init