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

FullSigHDP
    Simultaneously infers both signatures and activities.
    Currently implemented for a *single* tree only —
    multi-tree support is planned.  Parameters recovered: signatures,
    alpha_0, alpha_j, e_j for every node.
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
    - Provide `get_node_posterior` so analysis code can query any node
      without knowing the internal variable naming scheme.
    - Expose a `sample` method with a consistent signature.

    Subclass contract
    -----------------
    Implement `_build_pymc_model` to populate `self.model` and
    `self.node_index_map`.

    `self.node_index_map` must map every internal node ID to a
    (pymc_var_name, row_index_or_None) tuple, following the same
    convention as Fixed_sig_HDP: depth-0 nodes get their own scalar
    variable (row_index = None), deeper nodes are packed into a batched
    Dirichlet (row_index = int).
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
            Internal node ID (key in `node_index_map`).

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

        # if it is root we don't have to index, we have variable for each root node
        if row_idx is None:
            return samples.values  # (chains, draws, K)
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
    ):
        """
        Run the NUTS sampler.

        Parameters
        ----------
        draws : int
        tune : int
        chains : int
        cores : int
        target_accept : float

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
            )
        return self.trace

class FixedSigHDP(_BaseTreeHDP):
    """
    Tree-HDP inference model with fixed (known) mutational signatures.

    Infers per-node activity vectors e_j and a shared concentration
    parameter alpha.  Signatures are treated as constants.

    Generative model (mirroring Forward_HDP_Generator exactly)
    ----------------------------------------------------------
        shared_alpha ~ LogNormal(mu, sigma)
        e_0          ~ Dir(shared_alpha / K * 1_K)
        e_root       ~ Dir(shared_alpha * e_0)          for each root
        e_j          ~ Dir(shared_alpha * e_parent(j))  for each non-root
        x_ji         ~ Multinomial(M_j, e_j @ signatures)

    Parameters
    ----------
    newick_string : str
        Semicolon-separated Newick trees.
    data_matrix : pd.DataFrame
        Shape (N_observed, 96).  Index must match node labels.
    fixed_signatures : np.ndarray
        Shape (K, 96).
    priors:
    """

    def __init__(
        self,
        newick_string: str,
        data_matrix: pd.DataFrame,
        fixed_signatures: np.ndarray,
        priors: dict
    ):
        self.fixed_signatures = fixed_signatures
        self.K = fixed_signatures.shape[0]
        self.priors = priors
        super().__init__(newick_string, data_matrix)

    def _build_pymc_model(self) -> None:
        nodes_by_depth = self._get_nodes_by_depth()
        max_depth = max(nodes_by_depth.keys()) if nodes_by_depth else 0

        with pm.Model() as self.model:
            signatures = pt.as_tensor_variable(self.fixed_signatures)

            # Global parameters
            shared_alpha = get_prior(self.priors, "alpha_prior", dim=1)(name="shared_alpha")


            e_0_values = pm.Dirichlet("e_0", a=np.ones(self.K))

            node_es: Dict[str, pt.TensorVariable] = {}

            # Depth 0: one named variable per root
            for root in nodes_by_depth.get(0, []):
                e_root_values = pm.Dirichlet(f"e_{root}", a=(shared_alpha * e_0_values) + 1.01)
                node_es[root] = e_root_values
                self.node_index_map[root] = (f"e_{root}", None)

            # Depth 1+: batched Gamma per depth level
            for depth in range(1, max_depth + 1):
                current_nodes = nodes_by_depth.get(depth, [])
                if not current_nodes:
                    continue

                parent_nodes = [
                    list(self.graph.predecessors(n))[0] for n in current_nodes
                ]
                parent_e_stack = pt.stack([node_es[p] for p in parent_nodes])
                a_matrix = (shared_alpha * parent_e_stack) + 1.01

                e_name = f"e_level_{depth}"
                e_level_values = pm.Dirichlet(e_name, a=a_matrix,
                                              shape=(len(current_nodes), self.K))

                for i, node in enumerate(current_nodes):
                    node_es[node] = e_level_values[i]
                    self.node_index_map[node] = (e_name, i)

            # Likelihood — unchanged
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


class UnknownSigHDP(_BaseTreeHDP):
    """
    Tree-HDP inference model that jointly learns mutational signatures and
    per-node activities.

    The signatures theta_k ~ Dir(eta * 1_96) are inferred from data alongside
    per-node activity vectors e_j.  You only need to specify K_max, an upper
    bound on the number of signatures.  Signatures that are not needed by the
    data will be "switched off" — their corresponding activity components will
    shrink toward zero — so the effective number of signatures is learned
    automatically, as long as K_max is large enough.

    This mirrors FixedSigHDP exactly in the activity part of the model.
    The only addition is the signature layer above it:

        theta_k  ~ Dir(eta * 1_96)              k = 1, …, K_max   (signature prior)
        shared_alpha ~ LogNormal(mu, sigma)                        (concentration prior)
        e_0      ~ Dir(1_K)                                        (global baseline)
        e_root   ~ Dir(shared_alpha * e_0)                         (per root node)
        e_j      ~ Dir(shared_alpha * e_parent)                    (per non-root node)
        x_ji     ~ Multinomial(M_j,  e_j @ theta)                 (observed mutations)

    Parameters
    ----------
    newick_string : str
        Semicolon-separated Newick trees (same format as FixedSigHDP).
    data_matrix : pd.DataFrame
        Shape (N_observed, 96).  Index must match node labels in the tree.
    K_max : int
        Maximum number of signatures.  In practice 2–3× the number you
        expect to be active is a reasonable choice.
    priors : dict
        Same prior config dict used by FixedSigHDP.  The following keys
        are read:
          - ``"alpha_prior"``  (required) — prior on shared_alpha, passed
            to ``get_prior`` exactly as in FixedSigHDP.
          - ``"signatures_eta"`` (optional, default 0.1) — symmetric
            Dirichlet concentration for each signature theta_k.  Values
            below 1 encourage sparse, peaked signatures; values above 1
            push signatures toward uniform.  (Also accepted as ``"eta"``
            for backwards compatibility.)

    Attributes
    ----------
    K_max : int
    signature_var_name : str
        Name of the signature variable in the PyMC model (``"signatures"``).
        Use this when calling ``model.trace.posterior["signatures"]``.

    Notes
    -----
    Posterior access
    ~~~~~~~~~~~~~~~~
    Activities are accessed exactly as in FixedSigHDP::

        model.get_node_activity_posterior(node_id)   # (chains, draws, K_max)
        model.get_posterior_mean(node_id)            # (K_max,)

    The inferred signatures live in the trace under ``"signatures"``::

        sigs = model.trace.posterior["signatures"].mean(("chain", "draw"))
        # shape: (K_max, 96)

    A convenience accessor is provided::

        sigs = model.get_posterior_signatures()      # (K_max, 96) posterior mean
        sigs = model.get_posterior_signatures(mean=False)  # (chains, draws, K_max, 96)

    Identifiability
    ~~~~~~~~~~~~~~~
    Signatures and activities are only jointly identified up to permutation
    of the K_max components.  If you compare runs, align signatures by cosine
    similarity before comparing.  The ``evaluate_inference`` function in
    ``evaluation.py`` handles this correctly because it matches on node
    labels, not on signature indices.
    """

    # Name under which inferred signatures are stored in the PyMC model/trace.
    signature_var_name: str = "signatures"

    def __init__(
            self,
            newick_string: str,
            data_matrix: pd.DataFrame,
            K_max: int,
            priors: dict,
    ):
        self.K_max = K_max
        self.priors = priors
        super().__init__(newick_string, data_matrix)

    def _build_pymc_model(self) -> None:
        nodes_by_depth = self._get_nodes_by_depth()
        max_depth = max(nodes_by_depth.keys()) if nodes_by_depth else 0

        # Concentration for the signature prior.
        eta = float(Fraction(self.priors.get("signatures_eta", 0.1)))

        with pm.Model() as self.model:

            signatures = pm.Dirichlet(
                self.signature_var_name,
                a=np.full(96, eta),  # 1-D: broadcast to every row
                shape=(self.K_max, 96),  # K_max independent 96-dim Dirichlets
            )

            shared_alpha = get_prior(self.priors, "alpha_prior", dim=1)(
                name="shared_alpha"
            )

            e_0 = pm.Dirichlet(
                "e_0",
                a=np.ones(self.K_max),
            )

            node_es: Dict[str, pt.TensorVariable] = {}

            # ── Depth 0: one variable per root node ───────────────────────
            for root in nodes_by_depth.get(0, []):
                e_root = pm.Dirichlet(
                    f"e_{root}",
                    a=(shared_alpha * e_0) + 1.01,
                )
                node_es[root] = e_root
                self.node_index_map[root] = (f"e_{root}", None)

            # ── Depth 1+: batched Dirichlet per depth level ───────────────
            for depth in range(1, max_depth + 1):
                current_nodes = nodes_by_depth.get(depth, [])
                if not current_nodes:
                    continue

                parent_nodes = [
                    list(self.graph.predecessors(n))[0] for n in current_nodes
                ]
                parent_e_stack = pt.stack([node_es[p] for p in parent_nodes])
                a_matrix = shared_alpha * parent_e_stack + 1.01  # (N_current, K_max)

                e_name = f"e_level_{depth}"

                e_level = pm.Dirichlet(
                    e_name,
                    a=a_matrix,
                    shape=(len(current_nodes), self.K_max),
                )

                for i, node in enumerate(current_nodes):
                    node_es[node] = e_level[i]
                    self.node_index_map[node] = (e_name, i)

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
                e_matrix = pt.stack(observed_es)  # (N_obs, K_max)
                expected_probs = pt.dot(e_matrix, signatures)  # (N_obs, 96)
                pm.Multinomial(
                    "observations",
                    n=n_mutations,
                    p=expected_probs,
                    observed=obs_counts_matrix,
                )

    def get_posterior_signatures(self, mean: bool = True) -> np.ndarray:
        """
        Return posterior samples (or their mean) for the inferred signatures.

        Parameters
        ----------
        mean : bool
            If True (default) return the posterior mean over all chains and
            draws, shape (K_max, 96).
            If False return the full posterior array, shape
            (chains, draws, K_max, 96).

        Returns
        -------
        np.ndarray

        Raises
        ------
        ValueError
            If ``sample()`` has not been called yet.
        """
        if self.trace is None:
            raise ValueError("No trace found.  Run `sample()` first.")
        sigs = self.trace.posterior[self.signature_var_name]
        if mean:
            return sigs.mean(("chain", "draw")).values  # (K_max, 96)
        return sigs.values  # (chains, draws, K_max, 96)