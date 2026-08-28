"""
ou_inference.py

De novo Tree-HDP inference with an Ornstein-Uhlenbeck activity prior.

Class
-----
TreeOUHDP
    Same model as DeNovoHDP (latent signatures, logistic-normal activity
    walk, multinomial likelihood) with one change: the random-walk activity
    prior is replaced by a mean-reverting Ornstein-Uhlenbeck walk that pools
    every node toward a shared global centre mu.

    model_structure: denovo-v2

Motivation
----------
The random walk of denovo-v1 constrains only local parent-child differences;
the common soft mode (all nodes sliding together along the signatures' softest
direction) is anchored only at the root and stays weakly identified, which is
the ridge that splits chains into camps at moderate overlap. Reverting every
node toward a shared centre bounds that mode and de-ridges the soft directions
the likelihood leaves flat, while the data still dominates the stiff ones.

The OU walk on a branch of weight w_j is
    eta_j = mu + phi_j (eta_parent - mu) + s_j z_j,   z_j ~ Normal(0, 1)
with
    phi_j = phi ** w_j = exp(-theta w_j),     theta = -log(phi)
    s_j   = sigma sqrt( (1 - phi_j^2) / (2 theta) )
written non-centred (the sampler sees the standardised z_j, not eta_j).

phi in (0, 1) is the per-unit-weight persistence: phi -> 1 is no reversion
(the denovo-v1 random walk, with s_j -> sigma sqrt(w_j)); smaller phi pools
harder toward mu. The likelihood is flat in the soft directions and will not
demand reversion on its own, so phi needs an informative prior with mass below
1; see the `phi_prior` config entry.

Branch-length scaling
---------------------
The edge weight w_j is 1 for every edge by default (the discrete OU, which
matches the current simulator: activities there drift a constant amount per
edge, independent of branch length, because the edge `length` attribute is the
mutation-count mean, not an activity-drift time). Set `branch_length_scaling`
to read the edge `length` attribute and use w_j = length_j / mean_length, the
continuous-time OU on the tree, for data where branch length is an
activity-drift time. Off by default.

Identifiability
---------------
Unchanged from DeNovoHDP: label switching and the S / e trade-off are not
solved here and are left to external alignment before any cross-chain average.
A raw mean over chains of `signatures`, `eta_level_*` or `e_level_*` is not
meaningful. `mu` and `phi` are global scalars/vectors and are not subject to
label switching on their own, but `mu`'s components share the signature
labelling, so align before interpreting `mu` per component.

Nesting / regression
--------------------
With `mu` fixed to zero and `phi` fixed near 1 the walk reduces to the
denovo-v1 random walk eta_j = eta_parent + sigma z_j; fitting in that setting
should reproduce the denovo-v1 / fixed-sig-v2 behaviour. Use this as the
regression check before trusting any OU result.

Parameters
----------
newick_string : str
    Semicolon-separated Newick trees.
data_matrix : pd.DataFrame
    Shape (N_observed, 96). Index must match node labels.
num_signatures : int
    K, the number of signatures to discover.
priors : dict
    Prior config dict. Reads:
      - 'sigma_prior' / 'sigma_prior_parm' : prior on the diffusion scale.
      - 'sigma_0' (optional, default 1.0)  : root-baseline std.
      - 'beta' (optional, default 0.5)     : signature Dirichlet concentration.
      - 'mu_fixed' (optional, default False): if true, mu is pinned to zero
        (the regression / no-pooling setting); otherwise mu is learned.
      - 'mu_tau' (optional, default 1.0)   : std of the learned mu prior
        mu ~ Normal(0, mu_tau^2).
      - 'phi_fixed' (optional)             : pin the persistence to this value
        in (0, 1); skips the phi prior.
      - 'phi_prior' / 'phi_prior_parm'     : prior on the persistence phi, if
        not fixed. Defaults to Beta(5, 2) (mean 0.71) when neither is given.
branch_length_scaling : bool, default False
    Use per-edge weights from the `length` attribute (continuous-time OU)
    rather than w_j = 1 (discrete OU).
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import get_prior
from src.models.hdp_inference import _BaseTreeHDP


class TreeOUHDP(_BaseTreeHDP):
    """De novo Tree-HDP with an Ornstein-Uhlenbeck (mean-reverting) activity
    prior. See module docstring for the model and the regression check."""

    def __init__(
        self,
        newick_string: str,
        data_matrix: pd.DataFrame,
        num_signatures: int,
        priors: dict,
        branch_length_scaling: bool = False,
    ):
        self.K = int(num_signatures)
        self.priors = priors
        self.n_channels = data_matrix.shape[1]
        self.branch_length_scaling = bool(branch_length_scaling)
        super().__init__(newick_string, data_matrix)

    @staticmethod
    def _softmax_last_zero(eta_free: pt.TensorVariable) -> pt.TensorVariable:
        """Map (..., K-1) free logits to a (..., K) simplex point with the
        last logit pinned to 0, identical to the DeNovoHDP anchor."""
        zeros = pt.zeros_like(eta_free[..., :1])
        eta_full = pt.concatenate([eta_free, zeros], axis=-1)
        return pt.special.softmax(eta_full, axis=-1)

    def _edge_weight(self, parent: str, node: str) -> float:
        """Weight w_j for the branch parent -> node. 1.0 unless branch-length
        scaling is on, in which case the `length` edge attribute normalised by
        the mean edge length, falling back to 1.0 when absent."""
        if not self.branch_length_scaling:
            return 1.0
        edge = self.graph.get_edge_data(parent, node) or {}
        return float(edge.get("length", self._mean_edge_length))

    def _build_pymc_model(self) -> None:
        Km1 = self.K - 1
        sigma_0 = float(Fraction(str(self.priors.get("sigma_0", 1.0))))
        beta = float(Fraction(str(self.priors.get("beta", 0.5))))
        mu_fixed = bool(self.priors.get("mu_fixed", False))
        mu_tau = float(Fraction(str(self.priors.get("mu_tau", 1.0))))
        C = self.n_channels

        # mean edge length, used only to normalise weights when scaling is on
        if self.branch_length_scaling:
            lengths = [
                float(d.get("length", 1.0)) for _, _, d in self.graph.edges(data=True)
            ]
            self._mean_edge_length = float(np.mean(lengths)) if lengths else 1.0
        else:
            self._mean_edge_length = 1.0

        nodes_by_depth = self._get_nodes_by_depth()
        max_depth = max(nodes_by_depth.keys()) if nodes_by_depth else 0

        with pm.Model() as self.model:
            # Signature block, identical to DeNovoHDP.
            signatures = pm.Dirichlet(
                "signatures",
                a=beta * np.ones(C),
                shape=(self.K, C),
            )

            # Diffusion scale (same role and prior as the v1 walk scale).
            sigma = get_prior(self.priors, "sigma_prior", dim=1)(name="sigma")

            # Global pooling centre mu (the shared baseline the tree reverts
            # to). Pinned to zero in the regression setting.
            if mu_fixed:
                mu = pt.zeros(Km1)
            else:
                mu = pm.Normal("mu", mu=0.0, sigma=mu_tau, shape=(Km1,))

            # Per-unit-weight persistence phi in (0, 1); theta = -log(phi).
            if "phi_fixed" in self.priors:
                phi = pt.as_tensor_variable(
                    float(Fraction(str(self.priors["phi_fixed"])))
                )
            elif "phi_prior" in self.priors:
                phi = get_prior(self.priors, "phi_prior", dim=1)(name="phi")
            else:
                phi = pm.Beta("phi", alpha=5.0, beta=2.0)
            # clip keeps theta finite and positive at the phi -> 1 edge
            theta = pm.Deterministic("theta", -pt.log(pt.clip(phi, 1e-6, 1.0 - 1e-7)))

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
                    # Root: eta_root = mu + sigma_0 * z_root (non-centred).
                    z_name = f"z_level_{depth}"
                    z_level = pm.Normal(z_name, mu=0.0, sigma=1.0, shape=(n_cur, Km1))
                    eta_name = f"eta_level_{depth}"
                    eta_level = pm.Deterministic(
                        eta_name, mu[None, :] + sigma_0 * z_level
                    )
                else:
                    # OU step toward mu, non-centred.
                    #   phi_j = phi ** w_j,   s_j = sigma sqrt((1-phi_j^2)/(2 theta))
                    parent_eta_stack = pt.stack([node_etas[p] for p in parent_nodes])
                    w = np.array(
                        [
                            self._edge_weight(p, n)
                            for p, n in zip(parent_nodes, current_nodes)
                        ],
                        dtype=float,
                    )
                    phi_j = phi ** pt.as_tensor_variable(w)  # (n_cur,)
                    inc_var = sigma**2 * (1.0 - phi_j**2) / (2.0 * theta)
                    s_j = pt.sqrt(inc_var)  # (n_cur,)

                    z_name = f"z_level_{depth}"
                    z_level = pm.Normal(z_name, mu=0.0, sigma=1.0, shape=(n_cur, Km1))
                    eta_name = f"eta_level_{depth}"
                    eta_level = pm.Deterministic(
                        eta_name,
                        mu[None, :]
                        + phi_j[:, None] * (parent_eta_stack - mu[None, :])
                        + s_j[:, None] * z_level,
                    )

                e_name = f"e_level_{depth}"
                e_level = pm.Deterministic(e_name, self._softmax_last_zero(eta_level))

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
        """Posterior samples of the discovered signatures, shape
        (chains, draws, K, n_channels). Subject to label switching: align
        before averaging."""
        if self.trace is None:
            raise ValueError("No trace found.  Run `sample()` first.")
        return self.trace.posterior["signatures"].values
