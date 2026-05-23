"""
tree_signature_generator.py

Forward simulator for tree-structured mutational signature data.

This is the data-generating process for simulation studies of the Tree-HDP
inference model.  It implements a finite-dimensional approximation of the
nested tree-HDP. Instead of a nested Dirichlet process at every node,
activities follow a Dirichlet random walk down the tree.
This keeps the generator simple, neutral with respect to the inference
model, and compatible with both "signatures known" and "signatures unknown"
experiments.

Generative process
-------------------
    e_0  ~ Dir( (alpha_0 / K) * 1_K )      # cohort baseline (shared)
    [zero out a random subset of e_0]
    e_j  ~ Dir( alpha * e_parent )         # activities flow down each tree
                                           #   root's parent is e_0
    M_j  ~ NegBinom(mean=lam_j, ...)       # mutation count per node
    x_j  ~ Multinomial( M_j, e_j @ S )     # observed 96-channel counts

Key modelling choices
---------------------
* Strict inheritance.  The Dirichlet walk is taken only over the active
  support.  A signature that is zero in a parent stays exactly zero in every
  descendant, it can never be resurrected.  With cohort-level zeroing this
  means the active signature set is fixed by e_0 and identical at every node;
  `alpha` then controls only how activities vary within that fixed set.
* Zero-mutation nodes are kept.  Connecting nodes with M_j = 0 remain in the
  tree topology and in the ground-truth activity dict, but are excluded
  from the observed count matrix.
* Optional per-node signature dropout.  If enabled, a signature active in a
  parent may switch fully off in a child, the only mechanism for within
  tree change of support. Off by default.

Signatures
----------
Either supply a fixed (K, 96) matrix (e.g. real COSMIC SBS signatures) via
`signatures=`, or let the generator synthesize K signatures with a tunable
pairwise-correlation target via `n_signatures=` and `signature_correlation=`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import phylox
from phylox.constants import LABEL_ATTR

N_CHANNELS = 96

def synthesize_signatures(
    n_signatures: int,
    correlation: float,
    rng: np.random.Generator,
    n_channels: int = N_CHANNELS,
) -> np.ndarray:
    """
    Generate `n_signatures` synthetic signatures, each a distribution over
    `n_channels` trinucleotide channels, with a tunable degree of pairwise
    overlap.

    Parameters
    ----------
    n_signatures : int
        Number of signatures K to generate.
    correlation : float in [0, 1]
        0.0  -> signatures drawn independently (near-orthogonal, easy to
                distinguish).
        1.0  -> all signatures are perturbations of one shared base profile
                (highly correlated, hard to distinguish -- closer to the
                real difficulty of COSMIC signature extraction).
    rng : np.random.Generator
    n_channels : int

    Returns
    -------
    np.ndarray, shape (n_signatures, n_channels)
        Each row sums to 1.
    """
    if not 0.0 <= correlation <= 1.0:
        raise ValueError("correlation must be in [0, 1]")

    base = rng.dirichlet(np.ones(n_channels))

    sigs = np.empty((n_signatures, n_channels))
    for k in range(n_signatures):
        own = rng.dirichlet(np.ones(n_channels))
        profile = correlation * base + (1.0 - correlation) * own

        # More peaks
        profile = profile ** 1.5

        # Normalization
        sigs[k] = profile / profile.sum()
    return sigs


def _validate_signatures(signatures: np.ndarray) -> np.ndarray:
    """Check shape and that rows are valid probability vectors."""
    signatures = np.asarray(signatures, dtype=float)
    if signatures.ndim != 2:
        raise ValueError("signatures must be a 2-D array of shape (K, 96)")
    if signatures.shape[1] != N_CHANNELS:
        raise ValueError(
            f"signatures must have {N_CHANNELS} columns, got {signatures.shape[1]}"
        )
    row_sums = signatures.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("each signature row must sum to 1")
    if np.any(signatures < 0):
        raise ValueError("signatures must be non-negative")
    return signatures

@dataclass
class _NodeTruth:
    """Ground-truth record for a single node."""
    tumour: int
    label: str
    e_vector: np.ndarray
    num_mutations: int
    is_root: bool
    parent_label: Optional[str]

class TreeSignatureGenerator:
    """
    Forward simulator for tree-structured mutational signature data.

    Parameters
    ----------
    newick_forest : str
        One or more Newick trees, semicolon-separated.  Each tree is an
        independent tumour.  Branch lengths, if present, are used as the
        per-node mutation-count mean (overriding `lam`).
    signatures : np.ndarray, optional
        Fixed (K, 96) signature matrix.  If given, `n_signatures` and
        `signature_correlation` are ignored.
    n_signatures : int, optional
        Number of signatures to synthesize.  Required if `signatures` is None.
    signature_correlation : float, default 0.0
        Pairwise-overlap target for synthesized signatures (see
        `synthesize_signatures`).  Ignored when `signatures` is given.
    alpha : float, default 1.0
        Concentration of the activity random walk e_j ~ Dir(alpha * e_parent).
        High  -> children closely resemble parents (tree structure is strong).
        Low   -> children drift fast (tree structure is weak).
    alpha_0 : float, default 1.0
        Concentration of the cohort baseline e_0 ~ Dir((alpha_0/K) * 1_K).
    lam : float, default 1000.
        Mean mutations per node, used when an edge has no branch length.
    nb_dispersion : float, default 2.0
        Dispersion of the negative-binomial count model.  M_j has mean lam_j
        and variance lam_j * (1 + lam_j / nb_dispersion).  Smaller values ->
        more over-dispersed (more realistic burden spread).  Set to None for
        a plain Poisson.
    activity_sparsity : float, default 0.0
        Fraction of the K signatures forced to exactly zero in the cohort
        baseline e_0.  With strict inheritance these signatures are absent at
        every node.  true_K = K - round(activity_sparsity * K).
    signature_dropout : float, default 0.0
        Per-node probability that a signature active in the parent is switched
        fully off in the child.  The only mechanism for within-tree change of
        support.  Off (0.0) by default.
    seed : int, default 42
        RNG seed.
    """

    def __init__(
        self,
        newick_forest: str,
        signatures: Optional[np.ndarray] = None,
        n_signatures: Optional[int] = None,
        signature_correlation: float = 0.0,
        alpha: float = 1.0,
        alpha_0: float = 1.0,
        lam: float = 1000.0,
        nb_dispersion: Optional[float] = 2.0,
        activity_sparsity: float = 0.0,
        signature_dropout: float = 0.0,
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)
        self.alpha = float(alpha)
        self.alpha_0 = float(alpha_0)
        self.lam = float(lam)
        self.nb_dispersion = nb_dispersion
        self.signature_dropout = float(signature_dropout)

        if not 0.0 <= activity_sparsity < 1.0:
            raise ValueError("activity_sparsity must be in [0, 1)")
        self.activity_sparsity = float(activity_sparsity)

        if signatures is not None:
            self.signatures = _validate_signatures(signatures)
        else:
            if n_signatures is None:
                raise ValueError(
                    "provide either `signatures` or `n_signatures`"
                )
            self.signatures = synthesize_signatures(
                n_signatures, signature_correlation, self.rng
            )
        self.K = self.signatures.shape[0]

        self.e_0, self.active_mask = self._make_cohort_baseline()
        self.true_K = int(self.active_mask.sum())

        self.graphs: List[nx.DiGraph] = []
        for s in newick_forest.split(";"):
            if s.strip():
                self.graphs.append(phylox.DiNetwork.from_newick(s + ";"))

        self._truth: Dict[str, _NodeTruth] = {}
        self._simulate()

    def _make_cohort_baseline(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw e_0 ~ Dir((alpha_0/K) * 1_K), then zero a random subset so that
        true K is an exact integer.
        Returns (e_0, active_mask).
        """
        n_zero = int(round(self.activity_sparsity * self.K))
        n_zero = min(n_zero, self.K - 1)  # always keep >= 1 signature

        active_mask = np.ones(self.K, dtype=bool)
        if n_zero > 0:
            off = self.rng.choice(self.K, size=n_zero, replace=False)
            active_mask[off] = False

        conc = np.full(self.K, self.alpha_0 / self.K)
        e_0 = np.zeros(self.K)
        e_0[active_mask] = self.rng.dirichlet(conc[active_mask])
        return e_0, active_mask

    def _draw_child_activity(self, parent_e: np.ndarray) -> np.ndarray:
        """
        e_child ~ Dir(alpha * parent_e), taken strictly over the parent's
        active support.  Signatures with zero parent mass stay exactly zero.

        With signature_dropout > 0, each currently-active signature may be
        switched off before the draw.
        """
        support = parent_e > 0.0

        if self.signature_dropout > 0.0 and support.sum() > 1:
            keep = support.copy()
            for k in np.where(support)[0]:
                if self.rng.random() < self.signature_dropout:
                    keep[k] = False
            if keep.sum() == 0:                       # never drop everything
                keep[self.rng.choice(np.where(support)[0])] = True
            support = keep

        child = np.zeros_like(parent_e)
        conc = self.alpha * parent_e[support]
        conc = np.maximum(conc, 1e-12)
        child[support] = self.rng.dirichlet(conc)
        return child

    def _draw_count(self, mean: float) -> int:
        """Negative-binomial (or Poisson) draw for the per-node mutation count."""
        if self.nb_dispersion is None:
            return int(self.rng.poisson(mean))

        r = self.nb_dispersion
        rate = self.rng.gamma(shape=r, scale=mean / r)
        return int(self.rng.poisson(rate))

    def _simulate(self) -> None:
        """Traverse every tree top-down, drawing activities and counts."""
        for t_idx, graph in enumerate(self.graphs):
            for node in nx.topological_sort(graph):
                parents = list(graph.predecessors(node))
                label = graph.nodes[node].get(LABEL_ATTR, str(node))

                if not parents:
                    parent_e = self.e_0
                    parent_label = None
                    is_root = True
                    mean = self.lam
                else:
                    p = parents[0]
                    parent_e = graph.nodes[p]["e_vector"]
                    parent_label = graph.nodes[p].get(LABEL_ATTR, str(p))
                    is_root = False
                    edge = graph.get_edge_data(p, node) or {}
                    mean = float(edge.get("length", self.lam))

                e_vec = self._draw_child_activity(parent_e)
                count = self._draw_count(mean)

                graph.nodes[node]["e_vector"] = e_vec
                graph.nodes[node]["num_mutations"] = count

                self._truth[label] = _NodeTruth(
                    tumour=t_idx,
                    label=label,
                    e_vector=e_vec,
                    num_mutations=count,
                    is_root=is_root,
                    parent_label=parent_label,
                )

    def get_mutation_count_matrix(self) -> pd.DataFrame:
        """
        Observed data: an (N x 96) integer count matrix.

        Nodes with zero mutations are excluded from this matrix (they have
        nothing observed) but remain in the topology and ground truth.

        Returns
        -------
        pd.DataFrame
            Index: node labels.  Columns: Channel_0 ... Channel_95.
        """
        labels, rows = [], []
        for graph in self.graphs:
            for node in graph.nodes():
                m = graph.nodes[node]["num_mutations"]
                if m == 0:
                    continue
                e_vec = graph.nodes[node]["e_vector"]
                probs = e_vec @ self.signatures
                probs = probs / probs.sum()
                rows.append(self.rng.multinomial(m, probs))
                labels.append(graph.nodes[node].get(LABEL_ATTR, str(node)))

        cols = [f"Channel_{i}" for i in range(N_CHANNELS)]
        return pd.DataFrame(rows, index=labels, columns=cols)

    def get_tree_edges(self) -> pd.DataFrame:
        """
        Tree topology as a (parent_label, child_label) edge list, including
        edges to zero-mutation nodes.  This is what a tree-aware inference
        model consumes.

        Returns
        -------
        pd.DataFrame with columns ['tumour', 'parent', 'child'].
        """
        records = []
        for label, tr in self._truth.items():
            if tr.parent_label is not None:
                records.append(
                    {"tumour": tr.tumour,
                     "parent": tr.parent_label,
                     "child": label}
                )
        return pd.DataFrame(records, columns=["tumour", "parent", "child"])

    def get_true_activities(self) -> Dict[str, np.ndarray]:
        """
        Ground-truth activity vector (length K) for every node, including
        zero-mutation nodes.

        Returns
        -------
        dict {node_label: e_vector}
        """
        return {lbl: tr.e_vector.copy() for lbl, tr in self._truth.items()}

    def get_true_signatures(self) -> np.ndarray:
        """The (K, 96) signature matrix used to generate the data."""
        return self.signatures.copy()

    def get_active_signature_indices(self) -> np.ndarray:
        """Indices of the signatures with non-zero cohort baseline mass."""
        return np.where(self.active_mask)[0]

    def summary(self) -> Dict[str, object]:
        """Quick description of the generated dataset."""
        counts = [tr.num_mutations for tr in self._truth.values()]
        return {
            "n_tumours": len(self.graphs),
            "n_nodes_total": len(self._truth),
            "n_nodes_observed": int(np.sum(np.array(counts) > 0)),
            "K": self.K,
            "true_K": self.true_K,
            "alpha": self.alpha,
            "mean_mutations": float(np.mean(counts)) if counts else 0.0,
            "min_mutations": int(np.min(counts)) if counts else 0,
            "max_mutations": int(np.max(counts)) if counts else 0,
        }

def generate_random_forest(
    num_trees: int,
    min_leaves: int,
    max_leaves: int,
    rng: np.random.Generator,
    min_branch_length: int = 100,
    max_branch_length: int = 3000,
) -> str:
    """
    Build a random phylogenetic forest and return it as a semicolon-separated
    Newick string.  Each edge gets an independent integer branch length drawn
    uniformly from [min_branch_length, max_branch_length]; that length is the
    per-node mutation-count mean, so burden varies node to node.

    Parameters
    ----------
    num_trees : int
    min_leaves, max_leaves : int
        Inclusive range for leaves per tree.
    rng : np.random.Generator
    min_branch_length, max_branch_length : int
        Inclusive range for per-edge count means.
    """
    from phylox.generators.randomTC import (
        generate_network_random_tree_child_sequence,
    )

    newicks = []
    for t in range(num_trees):
        n_leaves = int(rng.integers(min_leaves, max_leaves + 1))
        tree_seed = int(rng.integers(0, 2 ** 31))
        tree = generate_network_random_tree_child_sequence(
            n_leaves, 0, seed=tree_seed
        )
        for i, node in enumerate(nx.topological_sort(tree)):
            tree.nodes[node][LABEL_ATTR] = f"T{t + 1}_{i + 1}"
        for u, v in tree.edges():
            tree[u][v]["length"] = int(
                rng.integers(min_branch_length, max_branch_length + 1)
            )
        newicks.append(tree.newick())
    return "".join(newicks)