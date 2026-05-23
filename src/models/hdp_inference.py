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
    Parameters recovered: shared_alpha, e_0, e_j for every node.
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
    Tree-HDP inference model with fixed (known) mutational signatures.

    model_structure: fixed-sig-v1

    Infers per-node activity vectors e_j and a shared concentration parameter
    alpha. Signatures are treated as constants.

    Generative model
    --------------------------------------------------
        shared_alpha ~ Given by the config parameter 'alpha_prior'
        e_0          ~ Dir(1_K)                         (global cohort baseline)
        e_j          ~ Dir(shared_alpha * e_parent)     (per node, plain Dirichlet)
        x_ji         ~ Multinomial(M_j, e_j @ signatures)

    Parameters
    ----------
    newick_string : str
        Semicolon-separated Newick trees.
    data_matrix : pd.DataFrame
        Shape (N_observed, 96). Index must match node labels.
    fixed_signatures : np.ndarray
        Shape (K, 96).
    priors : dict
        Prior config dict.
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

    def _build_pymc_model(self) -> None:
        # eps: pseudo-count for the per-node activity prior. Config-driven
        # via priors["eps"].  eps = 0 recovers the plain (v1) form.
        # eps is read for interface uniformity but v1 uses no
        # pseudo-count; eps has no effect here (set eps: 0 in config).
        eps = float(Fraction(str(self.priors.get("eps", 0.0))))

        nodes_by_depth = self._get_nodes_by_depth()
        max_depth = max(nodes_by_depth.keys()) if nodes_by_depth else 0

        with pm.Model() as self.model:
            signatures = pt.as_tensor_variable(self.fixed_signatures)

            # Global parameters
            shared_alpha = get_prior(self.priors, "alpha_prior", dim=1)(name="shared_alpha")

            e_0_values = pm.Dirichlet("e_0", a=np.ones(self.K))

            node_es: Dict[str, pt.TensorVariable] = {}

            for depth in range(0, max_depth + 1):
                current_nodes = nodes_by_depth.get(depth, [])
                if not current_nodes:
                    continue

                parent_nodes = [
                    list(self.graph.predecessors(n))[0] if list(self.graph.predecessors(n)) else None
                    for n in current_nodes
                ]
                if parent_nodes[0] is None:
                    parent_e_stack = pt.stack([e_0_values] * len(current_nodes))
                else:
                    parent_e_stack = pt.stack([node_es[p] for p in parent_nodes])

                a_matrix = shared_alpha * parent_e_stack + eps
                e_name = f"e_level_{depth}"
                e_level_values = pm.Dirichlet(e_name, a=a_matrix,
                                              shape=(len(current_nodes), self.K))

                for i, node in enumerate(current_nodes):
                    node_es[node] = e_level_values[i]
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

