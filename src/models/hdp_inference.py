"""
hdp_inference.py

PyMC-based Bayesian inference models for the Tree-HDP.

Classes
-------
_BaseTreeHDP
    Abstract base that handles all tree-topology bookkeeping shared by
    every inference variant: parsing the Newick forest, composing
    individual trees into a single directed graph, computing node depths,
    and providing a `get_node_posterior` accessor.

    Concrete subclasses only need to implement `_build_pymc_model`.

FixedSigHDP
    Infers per-node signature *activities* while treating signatures as
    known.

    model_structure: fixed-sig-v2

    The activity walk is a logistic-normal random walk in unconstrained
    space:
        eta_j ~ Normal(eta_parent, sigma^2)   (non-centered)
        e_j   = softmax(eta_j)
    This replaces the v1 Dirichlet walk e_j ~ Dir(alpha * e_parent), whose
    simplex-corner geometry forced an `eps` pseudo-count.  The logistic-
    normal walk samples cleanly with no eps.  The concentration parameter
    alpha is replaced by a Gaussian step scale sigma; the last component
    of eta is pinned to 0 at every node as the softmax identifiability
    anchor.

    The earlier Dirichlet-walk model (fixed-sig-v1) is preserved in the
    git history under the tag `fixed-sig-v1`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import sys
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import phylox
import pymc as pm
import pytensor.tensor as pt
from fractions import Fraction

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import get_prior

class _BaseTreeHDP(ABC):
    """
    Abstract base for all Tree-HDP PyMC inference models.

    Responsibilities
    ----------------
    - Parse one or more Newick trees and compose them into a single DiGraph.
    - Compute depth for every node (BFS from each root).
    - Group nodes by depth for vectorised PyMC variable construction.
    - Provide ``get_node_activity_posterior`` so analysis code can query any
      node without knowing the internal variable naming scheme.
    - Expose a ``sample`` method with a consistent signature.

    Subclass contract
    -----------------
    Implement ``_build_pymc_model`` to populate ``self.model`` and
    ``self.node_index_map``.

    ``self.node_index_map`` must map every internal node ID to a
    (pymc_var_name, row_index_or_None) tuple
    """

    def __init__(self, newick_string: str, data_matrix: pd.DataFrame):
        self.data_matrix = data_matrix

        # node_id -> (pymc_var_name, row_idx_or_None)
        self.node_index_map: Dict[str, Tuple[str, Optional[int]]] = {}

        # Compose all trees in the Newick string into one graph
        self.graph = nx.DiGraph()
        individual_trees = [
            phylox.DiNetwork.from_newick(s)
            for s in newick_string.split(";")
            if s.strip()
        ]
        for tree in individual_trees:
            mapping = {n: tree.nodes[n].get("label", str(n)) for n in tree.nodes()}
            relabeled = nx.relabel_nodes(tree, mapping)
            self.graph = nx.compose(self.graph, relabeled)

        self.model: Optional[pm.Model] = None
        self.trace = None
        self._build_pymc_model()

    @abstractmethod
    def _build_pymc_model(self) -> None:
        """Construct the PyMC model and populate self.node_index_map."""

    def _get_nodes_by_depth(self) -> Dict[int, list]:
        """
        BFS from every root to assign a depth to each node.

        Returns
        -------
        dict
            {depth: [node_id, ...]} sorted by depth.
        """
        roots = [n for n, d in self.graph.in_degree() if d == 0]
        seen: Dict[str, int] = {}
        nodes_by_depth: Dict[int, list] = {}

        for root in roots:
            for node, depth in nx.single_source_shortest_path_length(self.graph, root).items():
                if node not in seen:
                    seen[node] = depth
                    nodes_by_depth.setdefault(depth, []).append(node)

        return nodes_by_depth

    def get_node_activity_posterior(self, node_id: str) -> np.ndarray:
        """
        Return posterior samples for a node's activity vector.

        Parameters
        ----------
        node_id : str
            Internal node ID (key in ``node_index_map``).

        Returns
        -------
        np.ndarray
            Shape (chains, draws, K).

        Raises
        ------
        ValueError
            If the trace has not been computed yet.
        KeyError
            If the node ID is not in the model.
        """
        if self.trace is None:
            raise ValueError("No trace found.  Run `sample()` first.")
        if node_id not in self.node_index_map:
            raise KeyError(
                f"Node '{node_id}' not in model.  "
                f"Available: {list(self.node_index_map.keys())}"
            )
        var_name, row_idx = self.node_index_map[node_id]
        samples = self.trace.posterior[var_name]

        return samples.values[:, :, row_idx, :]  # (chains, draws, K)

    def get_posterior_mean(self, node_id: str) -> np.ndarray:
        """
        Convenience wrapper: return the posterior mean activity vector for a node.

        Returns
        -------
        np.ndarray
            Shape (K,).
        """
        return self.get_node_activity_posterior(node_id).mean(axis=(0, 1))

    def sample(
            self,
            draws: int = 1000,
            tune: int = 1000,
            chains: int = 4,
            cores: int = 4,
            target_accept: float = 0.95,
            max_treedepth: int = 10,
    ):
        """
        Run the NUTS sampler.

        Parameters
        ----------
        draws, tune, chains, cores, target_accept, max_treedepth :
            standard pm.sample / NUTS args.

        Returns
        -------
        arviz.InferenceData
        """
        if self.model is None:
            raise ValueError("Model has not been built yet.")
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                nuts_sampler="numpyro",
                nuts={"max_tree_depth": max_treedepth},
            )
        return self.trace


class FixedSigHDP(_BaseTreeHDP):
    """
    Tree-HDP inference model with fixed signatures and a logistic-normal
    activity walk.

    model_structure: fixed-sig-v2

    Reparameterisation of the v1 Dirichlet-walk model (preserved in git
    under the tag `fixed-sig-v1`).  Instead of the per-node Dirichlet
    walk e_j ~ Dir(alpha * e_parent), activities follow a Gaussian random
    walk in unconstrained space and are mapped to the simplex by softmax:

        sigma     ~ Given by the config parameter 'sigma_prior'
        eta_root  ~ Normal(0, sigma_0^2)        (K-1 free components)
        eta_j     =  eta_parent + sigma * z_j   (non-centered)
        z_j       ~ Normal(0, 1)
        e_j       =  softmax([eta_j, 0])
        x_ji      ~ Multinomial(M_j, e_j @ signatures)

    Why this form
    -------------
    The v1 Dirichlet walk has concentration entries near zero (e_parent is
    itself near the simplex boundary), which places mass at the simplex
    corners -- geometry NUTS samples badly, and the reason v1 needed the
    `eps` pseudo-count.  Here the walk happens in unconstrained R^(K-1)
    with Gaussian increments: no corners, no funnel, no eps.

    Identifiability
    ---------------
    softmax is shift-invariant (softmax(x) == softmax(x + c)).  To make the
    model identified, the last component of eta is pinned to 0 at every
    node -- the standard reference-category anchor.  eta therefore has K-1
    free components per node; e_j is still a full K-vector on the simplex.

    Notes
    -----
    - `sigma` (Gaussian step scale) replaces v1's `shared_alpha`
      (Dirichlet concentration).  Small sigma => children resemble parents;
      large sigma => fast drift.  The two are NOT the same parameter and
      are not directly comparable.
    - The recovered `eta_root` / baseline is not directly comparable to
      v1's Dirichlet `e_0`; the per-node activity vectors e_j (on the
      simplex) are the quantities to compare across models.

    Parameters
    ----------
    newick_string : str
        Semicolon-separated Newick trees.
    data_matrix : pd.DataFrame
        Shape (N_observed, 96).  Index must match node labels.
    fixed_signatures : np.ndarray
        Shape (K, 96).
    priors : dict
        Prior config dict.  Reads:
          - 'sigma_prior' / 'sigma_prior_parm'  : prior on the walk scale
            sigma (e.g. a HalfNormal or LogNormal).
          - 'sigma_0' (optional, default 1.0)   : std of the root baseline
            eta_root ~ Normal(0, sigma_0^2).
    """

    def __init__(
        self,
        newick_string: str,
        data_matrix: pd.DataFrame,
        fixed_signatures: np.ndarray,
        priors: dict,
    ):
        self.fixed_signatures = fixed_signatures
        self.K = fixed_signatures.shape[0]
        self.priors = priors
        super().__init__(newick_string, data_matrix)

    @staticmethod
    def _softmax_last_zero(eta_free: pt.TensorVariable) -> pt.TensorVariable:
        """
        Map an (..., K-1) array of free logits to an (..., K) simplex point,
        with the last component pinned to logit 0 (the identifiability
        anchor).
        """
        # pad a column of zeros for the reference category
        zeros = pt.zeros_like(eta_free[..., :1])
        eta_full = pt.concatenate([eta_free, zeros], axis=-1)
        return pt.special.softmax(eta_full, axis=-1)

    def _build_pymc_model(self) -> None:
        Km1 = self.K - 1                       # free logit dimensions
        sigma_0 = float(Fraction(str(self.priors.get("sigma_0", 1.0))))

        nodes_by_depth = self._get_nodes_by_depth()
        max_depth = max(nodes_by_depth.keys()) if nodes_by_depth else 0

        with pm.Model() as self.model:
            signatures = pt.as_tensor_variable(self.fixed_signatures)

            # Global walk-scale parameter (replaces v1's shared_alpha).
            sigma = get_prior(self.priors, "sigma_prior", dim=1)(name="sigma")

            # node_id -> eta vector (K-1 free logits)
            node_etas: Dict[str, pt.TensorVariable] = {}
            # node_id -> e_j vector (K-simplex), for the likelihood
            node_es: Dict[str, pt.TensorVariable] = {}

            for depth in range(0, max_depth + 1):
                current_nodes = nodes_by_depth.get(depth, [])
                if not current_nodes:
                    continue
                n_cur = len(current_nodes)

                parent_nodes = [
                    list(self.graph.predecessors(n))[0]
                    if list(self.graph.predecessors(n)) else None
                    for n in current_nodes
                ]

                if parent_nodes[0] is None:
                    # Root level: eta_root ~ Normal(0, sigma_0^2), centered.
                    eta_name = f"eta_level_{depth}"
                    eta_level = pm.Normal(
                        eta_name, mu=0.0, sigma=sigma_0,
                        shape=(n_cur, Km1),
                    )
                else:
                    # Non-root: non-centered walk
                    #   eta_j = eta_parent + sigma * z_j,  z_j ~ N(0,1)
                    parent_eta_stack = pt.stack(
                        [node_etas[p] for p in parent_nodes]
                    )
                    z_name = f"z_level_{depth}"
                    z_level = pm.Normal(
                        z_name, mu=0.0, sigma=1.0,
                        shape=(n_cur, Km1),
                    )
                    eta_name = f"eta_level_{depth}"
                    eta_level = pm.Deterministic(
                        eta_name, parent_eta_stack + sigma * z_level
                    )

                # map this level's logits to the simplex
                e_name = f"e_level_{depth}"
                e_level = pm.Deterministic(
                    e_name, self._softmax_last_zero(eta_level)
                )

                for i, node in enumerate(current_nodes):
                    node_etas[node] = eta_level[i]
                    node_es[node] = e_level[i]
                    self.node_index_map[node] = (e_name, i)

            # Likelihood
            observed_es, obs_counts = [], []
            for node in self.graph.nodes():
                label = self.graph.nodes[node].get("label", str(node))
                if label in self.data_matrix.index:
                    counts = self.data_matrix.loc[label].values
                    if counts.sum() > 0:
                        observed_es.append(node_es[node])
                        obs_counts.append(counts)

            if observed_es:
                obs_counts_matrix = np.array(obs_counts, dtype=np.int32)
                n_mutations = obs_counts_matrix.sum(axis=1)
                e_matrix = pt.stack(observed_es)
                expected_probs = pt.dot(e_matrix, signatures)
                pm.Multinomial(
                    "observations",
                    n=n_mutations,
                    p=expected_probs,
                    observed=obs_counts_matrix,
                )