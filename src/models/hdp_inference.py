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

TreeHDP
    Infers per-node signature activities under a shared ILR (sum-zero)
    random walk, with the signature matrix S either fixed (known
    signatures) or latent (S inferred jointly, de novo). See the class
    docstring for the model structure.

    This unifies the former FixedSigHDP and DeNovoHDP classes, which
    differed only in how S entered the likelihood and, incidentally, in
    the walk parameterisation (FixedSigHDP used an anchored softmax with
    the last logit pinned to 0; DeNovoHDP used the sum-zero ILR walk with
    a pooled usage level, needed to stop chains splitting into clusters on
    the harder de novo problem). Both now share the ILR walk; the fixed
    case is that walk with S clamped to a constant instead of drawn. The
    earlier anchored-softmax fixed-sig model and the pre-ILR de novo model
    are preserved in the git history under the tags `fixed-sig-v2` and
    `denovo-v1` respectively.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from fractions import Fraction
from pathlib import Path
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import phylox
import pymc as pm
import pytensor.tensor as pt

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
        """Parse the Newick forest into one directed graph and build the model."""
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
            for node, depth in nx.single_source_shortest_path_length(
                self.graph, root
            ).items():
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
        initvals=None,
        init: str = "auto",
    ):
        """
        Run the NUTS sampler.

        Parameters
        ----------
        draws, tune, chains, cores, target_accept, max_treedepth :
            standard pm.sample / NUTS args.
        initvals :
            optional dict (or list of dicts) of starting values. A single dict
            starts every chain from the same point; pair with
            init='adapt_diag' for a shared start without jitter.
        init :
            pm.sample initialisation scheme (default 'auto', unchanged).

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
                initvals=initvals,
                init=init,
            )
        return self.trace


class TreeHDP(_BaseTreeHDP):
    """
    Tree-HDP inference model with a shared ILR (sum-zero) activity walk.
    The signature matrix S is either fixed (known signatures) or latent
    (inferred jointly, de novo); everything else is the same model.

        sigma      ~ prior (config 'sigma_prior')       (walk scale)
        mu_level   ~ ZeroSumNormal(sigma_mu)             (K,) forest-pooled
                                                          usage level
        z_root     ~ ZeroSumNormal(1)                    (n_root, K)
        eta_root   =  mu_level + sigma_0 * z_root        (each tree root,
                                                          non-centered)
        z_j        ~ ZeroSumNormal(1)                    (K,)
        eta_j      =  eta_parent + sigma * z_j           (non-centered)
        e_j        =  softmax(eta_j)                     (full K, no
                                                          pinned coordinate)
        x_ji       ~ Multinomial(M_j, e_j @ S)

    S known (fixed signatures)
        S is a constant, passed in as `fixed_signatures`. Component k
        already is true signature k: there is no label switching, and
        posterior means over chains and draws are directly meaningful.

    S latent (de novo)
        S_k ~ Dir(beta * 1_C) for k = 1..K, inferred jointly with the
        activities. This introduces two non-identifiabilities:

        1. Label switching. Signature index k has no fixed meaning: any
           permutation of the K signatures, with the matching permutation
           of the activity components, leaves the likelihood unchanged.
           Posterior means computed by averaging raw draws across chains
           (or across draws, if a chain switches mid-run) are therefore
           MEANINGLESS.
        2. S / e trade-off. Only the product e_j @ S is observed, so a
           continuum of (S, e) pairs fit nearly equally well. This is
           broken only by the structure of the priors -- the tree walk on
           e and the Dirichlet concentration on S.

        This class does not solve either. Chain alignment and scoring
        against ground-truth signatures are intentionally left to external
        post-processing: align chains to a common labelling first, THEN
        compute posterior means or call the activity accessors.
        `get_posterior_mean` inherited from the base class will silently
        return a permutation-averaged (wrong) result if called on an
        un-aligned trace.

    Parameters
    ----------
    newick_string : str
        Semicolon-separated Newick trees.
    data_matrix : pd.DataFrame
        Shape (N_observed, C).  Index must match node labels.
    priors : dict
        Prior config dict.  Reads:
          - 'sigma_prior' / 'sigma_prior_parm' : prior on the walk scale.
          - 'sigma_0' (optional, default 1.0)  : scale of each root's
            deviation from mu_level (z_root), not an absolute root scale.
          - 'sigma_mu' (optional, default 2.0) : scale of the forest-pooled
            usage level mu_level ~ ZeroSumNormal(sigma_mu).
          - 'beta' (optional, default 0.5)     : Dirichlet concentration
            for the signature prior S_k ~ Dir(beta * 1_C). Read only when
            S is latent.
    fixed_signatures : np.ndarray, optional
        Shape (K, C). Pass to fix S (known-signature setting); K is taken
        from this array. Exactly one of `fixed_signatures` /
        `num_signatures` must be given.
    num_signatures : int, optional
        K, the number of signatures to discover, with S latent
        (de novo setting). Exactly one of `fixed_signatures` /
        `num_signatures` must be given.

    Notes
    -----
    Call with keyword arguments. This replaces the former FixedSigHDP and
    DeNovoHDP classes, whose constructors took different positional
    arguments (see the module docstring); positional calls written against
    either are not compatible with this constructor.
    """

    def __init__(
        self,
        newick_string: str,
        data_matrix: pd.DataFrame,
        priors: dict,
        fixed_signatures: Optional[np.ndarray] = None,
        num_signatures: Optional[int] = None,
    ):
        if (fixed_signatures is None) == (num_signatures is None):
            raise ValueError(
                "TreeHDP needs exactly one of fixed_signatures (S known) "
                "or num_signatures (S latent)."
            )
        if fixed_signatures is not None:
            self.S_known = True
            self.fixed_signatures = np.asarray(fixed_signatures)
            self.K = self.fixed_signatures.shape[0]
        else:
            self.S_known = False
            self.K = int(num_signatures)
        self.priors = priors
        self.n_channels = data_matrix.shape[1]
        super().__init__(newick_string, data_matrix)

    def _build_signature_block(self) -> pt.TensorVariable:
        """
        Return the (K, C) signature tensor: a constant when S is known, or
        a latent pm.Dirichlet, S_k ~ Dir(beta * 1_C), when S is inferred
        jointly (de novo). Must be called inside `self.model`.
        """
        if self.S_known:
            return pt.as_tensor_variable(self.fixed_signatures)
        beta = float(Fraction(str(self.priors.get("beta", 0.5))))
        return pm.Dirichlet(
            "signatures",
            a=beta * np.ones(self.n_channels),
            shape=(self.K, self.n_channels),
        )

    def _build_pymc_model(self) -> None:
        """Build the shared-walk PyMC model (structure in the class docstring)."""
        sigma_0 = float(Fraction(str(self.priors.get("sigma_0", 1.0))))
        sigma_mu = float(Fraction(str(self.priors.get("sigma_mu", 2.0))))

        nodes_by_depth = self._get_nodes_by_depth()
        max_depth = max(nodes_by_depth.keys()) if nodes_by_depth else 0

        with pm.Model() as self.model:
            signatures = self._build_signature_block()

            sigma = get_prior(self.priors, "sigma_prior", dim=1)(name="sigma")

            mu_level = pm.ZeroSumNormal("mu_level", sigma=sigma_mu, shape=(self.K,))
            node_etas: Dict[str, pt.TensorVariable] = {}
            node_es: Dict[str, pt.TensorVariable] = {}

            for depth in range(0, max_depth + 1):
                current_nodes = nodes_by_depth.get(depth, [])
                if not current_nodes:
                    continue
                n_cur = len(current_nodes)

                parent_nodes = [
                    list(self.graph.predecessors(n))[0]
                    if list(self.graph.predecessors(n))
                    else None
                    for n in current_nodes
                ]

                if parent_nodes[0] is None:
                    eta_name = f"eta_level_{depth}"
                    z_root = pm.ZeroSumNormal(
                        f"z_root_{depth}", sigma=1.0, shape=(n_cur, self.K)
                    )
                    eta_level = pm.Deterministic(
                        eta_name, mu_level[None, :] + sigma_0 * z_root
                    )
                else:
                    parent_eta_stack = pt.stack([node_etas[p] for p in parent_nodes])
                    z_name = f"z_level_{depth}"
                    z_level = pm.ZeroSumNormal(z_name, sigma=1.0, shape=(n_cur, self.K))
                    eta_name = f"eta_level_{depth}"
                    eta_level = pm.Deterministic(
                        eta_name, parent_eta_stack + sigma * z_level
                    )

                e_name = f"e_level_{depth}"
                e_level = pm.Deterministic(
                    e_name, pt.special.softmax(eta_level, axis=-1)
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

    def get_signatures_posterior(self) -> np.ndarray:
        """
        Return posterior samples of the signature matrix.

        Returns
        -------
        np.ndarray
            Shape (chains, draws, K, n_channels).

        Raises
        ------
        ValueError
            If S is fixed (there is nothing latent to return) or if no
            trace has been sampled yet.

        Notes
        -----
        When S is latent these samples are subject to label switching
        across chains (and possibly within a chain).  Align chains to a
        common labelling BEFORE averaging -- a raw mean over chains is not
        meaningful.
        """
        if self.S_known:
            raise ValueError(
                "Signatures are fixed on this model; there is no "
                "'signatures' posterior to return."
            )
        if self.trace is None:
            raise ValueError("No trace found.  Run `sample()` first.")
        return self.trace.posterior["signatures"].values
