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
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import phylox
import pymc as pm
import pytensor.tensor as pt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import get_prior
from src.models.switch_pruning import DepthArrays
from src.models.switch_pruning import backward as switch_backward
from src.models.switch_pruning import prune as switch_prune


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
        eta_j      =  eta_parent + sigma * z_j           (non-centered; times
                                                          sqrt(l_e) with
                                                          walk_branch_length_scaling)
        e_j        =  softmax(eta_j)                     (full K, no
                                                          pinned coordinate)
        x_ji       ~ Multinomial(M_j, e_j @ S)

    With `switching` given (see below), `e_j` and the likelihood change:
    each signature carries a marginalised per-node on/off state, exactly
    marginalised by Felsenstein pruning (`src/models/switch_pruning.py`,
    see `switch_model_plan.md`). `eta_j` is unchanged; `e_j` becomes the
    state-mixed Bayes activity and the plain `pm.Multinomial` observation
    is replaced by a `pm.Potential` holding the pruned log-likelihood.
    `switching=None` (the default) builds exactly the model above.

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
          - 'walk_branch_length_scaling' (optional, default False) : scale
            each walk step by the square root of the normalised branch
            length, eta_j = eta_parent + sigma * sqrt(l_e) * z_j, with the
            same l_e = L_e / L_median as the switch block (the simulator's
            branch-length-scaled drift). Works with switching on or off;
            with every l_e = 1 it is the unscaled model.
          - 'branch_length_source' (optional, default 'newick') : where the
            lengths for the scaled walk come from when switching is off
            ('newick' | 'unit'); with switching on, its
            `branch_length_source` is used for both.
    fixed_signatures : np.ndarray, optional
        Shape (K, C). Pass to fix S (known-signature setting); K is taken
        from this array. Exactly one of `fixed_signatures` /
        `num_signatures` must be given.
    num_signatures : int, optional
        K, the number of signatures to discover, with S latent
        (de novo setting). Exactly one of `fixed_signatures` /
        `num_signatures` must be given.
    switching : dict, optional
        `None` (default) or `{"enabled": False}` builds exactly the model
        above -- same variables, same `pm.Multinomial` likelihood. With
        `"enabled": True`, reads:
          - 'branch_length_source' (default 'newick') : 'newick' reads each
            edge's `length` attribute (normalised by the forest median
            branch length) and fails loudly if any edge lacks one; 'unit'
            sets every branch length to 1 (for Newick trees with no lengths,
            e.g. the real-data SNV tree).
          - 'lambda_on_prior' / 'lambda_on_prior_parm',
            'lambda_off_prior' / 'lambda_off_prior_parm' : `get_prior`-style
            prior on the per-signature gain/loss hazard, shape (K,).
          - 'pi_root_prior' / 'pi_root_prior_parm' : `get_prior`-style prior
            on the per-signature root activation probability, shape (K,).
          - 'always_on' (fixed mode only; default []) : signature names
            forced on everywhere. They are dropped from the state grid
            (state space `2**(K - m)`) and `a_prob_level_*` is exactly 1 for
            them. Biologically the clock signatures (SBS1, SBS5). Needs
            `signature_names`.
          - 'tree_coupled' (default True) : with False, every edge
            transition is the root prior, independent of the parent's state
            and the branch length: states are i.i.d. across nodes and the
            tree carries no information about on/off. The walk is unchanged,
            so this isolates what the Markov coupling along the tree adds to
            on/off recovery (the tree-free switching ablation).
        `K - m` above 12 raises `ValueError`.
    signature_names : sequence of str, optional
        Names of the rows of `fixed_signatures`, needed to resolve
        `switching.always_on`.

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
        switching: Optional[dict] = None,
        signature_names: Optional[Sequence[str]] = None,
    ):
        self.signature_names = (
            None if signature_names is None else list(signature_names)
        )
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
        self.switching = self._resolve_switching(switching)
        self.priors = priors
        self.n_channels = data_matrix.shape[1]
        # name -> axis, for every model variable carrying a signature axis;
        # filled in as _build_pymc_model creates them (see signature_axis_vars).
        self.signature_axis: Dict[str, int] = {}
        super().__init__(newick_string, data_matrix)

    def signature_axis_vars(self) -> Dict[str, int]:
        """
        Every model variable with a signature axis, as `name -> axis` (the
        axis within the variable's own shape, not counting chain/draw).

        `signatures` is axis 0 (present only when S is latent); every
        `(..., K)` variable is axis -1: `mu_level`, `z_root_<d>`,
        `z_level_<d>`, `eta_level_<d>`, `e_level_<d>`, and with switching
        enabled `lambda_on`, `lambda_off`, `pi_root`, `a_prob_level_<d>`.

        This is the registry a de novo post-hoc alignment permutes, so that
        signature k means the same thing in every draw of every chain for
        every variable, not just the ones a hard-coded list happened to
        name. Permuting a ZeroSumNormal draw keeps it zero-sum, so aligning
        `z_*` is harmless and makes the aligned trace self-consistent.
        """
        return dict(self.signature_axis)

    def _resolve_switching(self, switching: Optional[dict]) -> Optional[dict]:
        """
        Validate and normalise the `switching` constructor argument.

        Returns `None` when switching is disabled (the default), or the
        validated config dict when enabled. See the class docstring for the
        keys read. Called before `self.K` is otherwise used, so `self.K`
        must already be set (the constructor does this).

        Raises
        ------
        ValueError
            If `branch_length_source` is not 'newick'/'unit'; if `always_on`
            is given with S latent, without `signature_names`, or names a
            signature not in the index; or if `K - len(always_on)` exceeds
            the state-space cap of 12 (`2**12` states).
        """
        if not switching or not switching.get("enabled", False):
            return None
        cfg = dict(switching)
        cfg.setdefault("branch_length_source", "newick")
        if cfg["branch_length_source"] not in ("newick", "unit"):
            raise ValueError(
                "switching.branch_length_source must be 'newick' or 'unit', "
                f"got {cfg['branch_length_source']!r}."
            )
        always_on = list(cfg.get("always_on") or [])
        if always_on:
            if not self.S_known:
                raise ValueError(
                    "switching.always_on is only valid with fixed signatures; de "
                    "novo signatures have no identity to force on."
                )
            if self.signature_names is None:
                raise ValueError(
                    "switching.always_on names signatures, so TreeHDP needs "
                    "signature_names (the fixed signature matrix's index)."
                )
            names = list(self.signature_names)
            missing = [s for s in always_on if s not in names]
            if missing:
                raise ValueError(
                    f"switching.always_on names {missing} are not in the fixed "
                    f"signature index {names}."
                )
            cfg["always_on_idx"] = tuple(sorted(names.index(s) for s in always_on))
        else:
            cfg["always_on_idx"] = ()
        cfg["tree_coupled"] = bool(cfg.get("tree_coupled", True))
        state_space_cap = 12
        n_free = self.K - len(cfg["always_on_idx"])
        if n_free > state_space_cap:
            raise ValueError(
                f"switching state space is 2**(K - len(always_on)) = 2**{n_free} "
                f"states; that exceeds the cap of 2**{state_space_cap}. Use "
                "always_on for the clock signatures or reduce K."
            )
        return cfg

    def _build_signature_block(self) -> pt.TensorVariable:
        """
        Return the (K, C) signature tensor: a constant when S is known, or
        a latent pm.Dirichlet, S_k ~ Dir(beta * 1_C), when S is inferred
        jointly (de novo). Must be called inside `self.model`.
        """
        if self.S_known:
            return pt.as_tensor_variable(self.fixed_signatures)
        beta = float(Fraction(str(self.priors.get("beta", 0.5))))
        self.signature_axis["signatures"] = 0
        return pm.Dirichlet(
            "signatures",
            a=beta * np.ones(self.n_channels),
            shape=(self.K, self.n_channels),
        )

    def _normalised_edge_lengths(self, nodes_by_depth_list: List[list], source: str):
        """
        Per-depth normalised branch lengths `l_e = L_e / L_median`, the one
        length convention shared by the switch block and the branch-length-
        scaled walk. `L_median` is the median over every edge of the forest.

        Returns `(length_by_depth, l_median)`: `length_by_depth[0]` is None,
        `length_by_depth[d]` is a `(n_d,)` array for `d >= 1`. With
        `source == 'unit'` every length is 1 and `l_median = 1`. Also sets
        `self.l_median` (rates in the switch block are per median branch, so
        comparing a fitted `lambda` to the simulator's needs this factor).

        Raises
        ------
        ValueError
            If `source` is not 'newick'/'unit', or is 'newick' and some edge
            has no `length` attribute.
        """
        if source not in ("newick", "unit"):
            raise ValueError(
                f"branch_length_source must be 'newick' or 'unit', got {source!r}."
            )
        if source == "newick":
            lengths = [d.get("length") for _, _, d in self.graph.edges(data=True)]
            if any(length is None for length in lengths):
                raise ValueError(
                    "branch_length_source='newick' requires a 'length' attribute "
                    "on every edge; found one or more edges without one. Use "
                    "branch_length_source='unit' for a Newick forest with no "
                    "branch lengths."
                )
            l_median = float(np.median(lengths)) if lengths else 1.0
        else:
            l_median = 1.0

        length_by_depth: List[Optional[np.ndarray]] = [None]
        for current_nodes in nodes_by_depth_list[1:]:
            length_d = np.ones(len(current_nodes))
            if source == "newick":
                for i, node in enumerate(current_nodes):
                    parent = list(self.graph.predecessors(node))[0]
                    length_d[i] = self.graph.edges[parent, node]["length"] / l_median
            length_by_depth.append(length_d)
        self.l_median = l_median
        return length_by_depth, l_median

    def _build_switch_depth_arrays(
        self, nodes_by_depth_list: List[list]
    ) -> DepthArrays:
        """
        Build the NumPy `DepthArrays` the switch-pruning graph needs, from
        `self.graph`'s edge `length` attributes and `self.data_matrix`.

        Parameters
        ----------
        nodes_by_depth_list : list of list of node id
            One entry per depth, in the same order used to build
            `eta_level_<d>` (so switch_pruning's node order matches the
            walk's).

        Raises
        ------
        ValueError
            If `branch_length_source` is 'newick' and some edge has no
            `length` attribute.
        """
        length_by_depth, _ = self._normalised_edge_lengths(
            nodes_by_depth_list, self.switching["branch_length_source"]
        )

        node_pos: Dict[str, int] = {}
        counts_by_depth, observed_by_depth = [], []
        parent_pos_by_depth = []

        for depth, current_nodes in enumerate(nodes_by_depth_list):
            n_cur = len(current_nodes)
            counts_d = np.zeros((n_cur, self.n_channels))
            observed_d = np.zeros(n_cur, dtype=bool)
            for i, node in enumerate(current_nodes):
                label = self.graph.nodes[node].get("label", str(node))
                if label in self.data_matrix.index:
                    row = self.data_matrix.loc[label].values.astype("float64")
                    if row.sum() > 0:
                        counts_d[i] = row
                        observed_d[i] = True
                node_pos[node] = i
            counts_by_depth.append(counts_d)
            observed_by_depth.append(observed_d)

            if depth == 0:
                parent_pos_by_depth.append(None)
                continue
            parent_pos_d = np.zeros(n_cur, dtype=int)
            for i, node in enumerate(current_nodes):
                parent = list(self.graph.predecessors(node))[0]
                parent_pos_d[i] = node_pos[parent]
            parent_pos_by_depth.append(parent_pos_d)

        tree_of_root = np.arange(len(nodes_by_depth_list[0]))
        return DepthArrays.build(
            counts_by_depth,
            observed_by_depth,
            parent_pos_by_depth,
            length_by_depth,
            tree_of_root,
        )

    def _build_switch_block(
        self,
        signatures: pt.TensorVariable,
        eta_by_depth: List[pt.TensorVariable],
        nodes_by_depth_list: List[list],
        node_es: Dict[str, pt.TensorVariable],
    ) -> None:
        """
        Build the switching likelihood: priors on lambda_on/lambda_off/
        pi_root, the pruned log-likelihood Potential, and the
        a_prob_level_<d>/e_level_<d> Deterministics. Must be called inside
        `self.model`, after the walk block has built `eta_by_depth`.
        """
        depth_arrays = self._build_switch_depth_arrays(nodes_by_depth_list)
        lambda_on = get_prior(self.switching, "lambda_on_prior", dim=self.K)(
            name="lambda_on"
        )
        lambda_off = get_prior(self.switching, "lambda_off_prior", dim=self.K)(
            name="lambda_off"
        )
        pi_root = get_prior(self.switching, "pi_root_prior", dim=self.K)(name="pi_root")
        self.signature_axis.update(lambda_on=-1, lambda_off=-1, pi_root=-1)

        always_on = self.switching["always_on_idx"]
        log_beta, log_msg, logT_by_depth, logpi_vec, logZ, logZ_per_root = switch_prune(
            eta_by_depth,
            signatures,
            lambda_on,
            lambda_off,
            pi_root,
            depth_arrays,
            self.K,
            always_on=always_on,
            tree_coupled=self.switching["tree_coupled"],
        )
        pm.Potential("switch_loglik", logZ)

        _, a_prob_by_depth, e_level_by_depth = switch_backward(
            eta_by_depth,
            log_beta,
            log_msg,
            logT_by_depth,
            logpi_vec,
            logZ_per_root,
            depth_arrays,
            self.K,
            always_on=always_on,
        )
        for depth, current_nodes in enumerate(nodes_by_depth_list):
            pm.Deterministic(f"a_prob_level_{depth}", a_prob_by_depth[depth])
            e_level = pm.Deterministic(f"e_level_{depth}", e_level_by_depth[depth])
            self.signature_axis[f"a_prob_level_{depth}"] = -1
            self.signature_axis[f"e_level_{depth}"] = -1
            for i, node in enumerate(current_nodes):
                node_es[node] = e_level[i]

    def _build_pymc_model(self) -> None:
        """Build the shared-walk PyMC model (structure in the class docstring)."""
        sigma_0 = float(Fraction(str(self.priors.get("sigma_0", 1.0))))
        sigma_mu = float(Fraction(str(self.priors.get("sigma_mu", 2.0))))

        nodes_by_depth = self._get_nodes_by_depth()
        nodes_by_depth_list: List[list] = [
            nodes_by_depth[d] for d in sorted(nodes_by_depth)
        ]
        switching_enabled = self.switching is not None

        # Branch-length-scaled walk (priors.walk_branch_length_scaling):
        # eta_j = eta_parent + sigma * sqrt(l_e) * z_j, with the same
        # l_e = L_e / L_median as the switch block.
        walk_scaling = bool(self.priors.get("walk_branch_length_scaling", False))
        sqrt_l_by_depth: List[Optional[np.ndarray]] = [None] * len(nodes_by_depth_list)
        if walk_scaling:
            source = (
                self.switching["branch_length_source"]
                if switching_enabled
                else self.priors.get("branch_length_source", "newick")
            )
            length_by_depth, _ = self._normalised_edge_lengths(
                nodes_by_depth_list, source
            )
            sqrt_l_by_depth = [None] + [np.sqrt(ln) for ln in length_by_depth[1:]]

        with pm.Model() as self.model:
            signatures = self._build_signature_block()

            sigma = get_prior(self.priors, "sigma_prior", dim=1)(name="sigma")

            mu_level = pm.ZeroSumNormal("mu_level", sigma=sigma_mu, shape=(self.K,))
            self.signature_axis["mu_level"] = -1
            node_etas: Dict[str, pt.TensorVariable] = {}
            node_es: Dict[str, pt.TensorVariable] = {}
            eta_by_depth: List[pt.TensorVariable] = []

            for depth, current_nodes in enumerate(nodes_by_depth_list):
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
                    step = sigma * z_level
                    if walk_scaling:
                        sqrt_l = pt.as_tensor_variable(sqrt_l_by_depth[depth])
                        step = sigma * sqrt_l[:, None] * z_level
                    eta_level = pm.Deterministic(eta_name, parent_eta_stack + step)

                z_var = f"z_root_{depth}" if parent_nodes[0] is None else z_name
                self.signature_axis[z_var] = -1
                self.signature_axis[eta_name] = -1

                for i, node in enumerate(current_nodes):
                    node_etas[node] = eta_level[i]
                    self.node_index_map[node] = (f"e_level_{depth}", i)

                eta_by_depth.append(eta_level)

            if switching_enabled:
                self._build_switch_block(
                    signatures, eta_by_depth, nodes_by_depth_list, node_es
                )
                return

            for depth, current_nodes in enumerate(nodes_by_depth_list):
                e_level = pm.Deterministic(
                    f"e_level_{depth}",
                    pt.special.softmax(eta_by_depth[depth], axis=-1),
                )
                self.signature_axis[f"e_level_{depth}"] = -1
                for i, node in enumerate(current_nodes):
                    node_es[node] = e_level[i]

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
