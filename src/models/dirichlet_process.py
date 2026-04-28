"""
dirichlet_process.py

Core probability primitives for the Tree-HDP model.

Classes
-------
Measure
    Abstract base class: anything that can be sampled to yield a
    probability vector over mutation channels.

DirichletPrior
    The global base measure H ~ Dir(alpha_prior * 1_96).
    Sits at the very top of the hierarchy.

DirichletProcess
    A single node's measure G_j ~ DP(alpha_j, G_parent).
    Implements the stick-breaking construction lazily: new atoms are
    only created when a sample falls in the unexplored part of the stick.
"""

import bisect
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from scipy.stats import beta, dirichlet, multinomial, uniform


class Measure(ABC):
    """
    Abstract base class representing a probability measure over mutation channels.

    Both the global prior H and every node-level Dirichlet Process G_j are
    Measures: they expose a single `sample()` method that returns a probability
    vector of length `dimensions` (typically 96 trinucleotide channels).
    """

    seed_generator: np.random.Generator = None

    @abstractmethod
    def sample(self) -> np.ndarray:
        """
        Draw a single sample from this measure.

        Returns
        -------
        np.ndarray
            A probability vector over the mutation channels (sums to 1).
        """


class DirichletPrior(Measure):
    """
    The global base measure H at the top of the hierarchy.

    Samples a completely new mutational signature theta_tilde ~ Dir(alpha_prior * 1_d).
    A low alpha_prior (e.g. 0.1) produces sparse, peaked signatures;
    a high alpha_prior pushes signatures toward the uniform distribution.

    Parameters
    ----------
    dimensions : int
        Number of mutation channels (default 96 for trinucleotide context).
    alpha_prior : float
        Concentration of the symmetric Dirichlet prior (default 0.5).
    seed_generator : np.random.Generator, optional
        Shared RNG.  If None a fresh default_rng() is created.
    """

    def __init__(
        self,
        dimensions: int = 96,
        alpha_prior: float = 0.5,
        seed_generator: np.random.Generator = None,
    ):
        self.dimensions = dimensions
        self.prior_alpha = np.ones(dimensions) * alpha_prior
        self.seed_generator = (
            seed_generator if seed_generator is not None else np.random.default_rng()
        )

    def sample(self) -> np.ndarray:
        """
        Draw a new mutational signature theta_tilde from the Dirichlet prior.

        Returns
        -------
        np.ndarray
            Shape (dimensions,).  Sums to 1.
        """
        return dirichlet.rvs(self.prior_alpha, random_state=self.seed_generator)[0]


class DirichletProcess(Measure):
    """
    A single node's probability measure G_j ~ DP(alpha_j, G_parent).

    Internally implements the stick-breaking construction lazily:
    atoms are only materialised when a sample falls into the
    as-yet-unexplored portion of the stick.

    Parameters
    ----------
    alpha : float
        Concentration parameter.  Large alpha -> G_j stays close to the
        parent measure; small alpha -> G_j concentrates on few atoms.
    base_measure : Measure
        The parent measure (either DirichletPrior or another
        DirichletProcess).  New atoms are drawn from this measure.

    Attributes
    ----------
    mut_activities : list of float
        Stick-breaking weights (e_k) realised so far.
    signatures : list of np.ndarray
        Corresponding atoms (theta_tilde_k) drawn from base_measure.
    remaining_stick : float
        Fraction of the unit stick not yet broken off.
    """

    def __init__(self, alpha: float, base_measure: Measure):
        self.alpha = alpha
        self.base_measure = base_measure
        # Inherit the shared RNG from the parent so the whole tree uses one stream
        self.seed_generator = base_measure.seed_generator

        self.mut_activities: List[float] = []
        self.signatures: List[np.ndarray] = []
        self.remaining_stick: float = 1.0
        self._cumsums: List[float] = [0.0]

    def _break_new_stick_piece(self) -> None:
        """
        Materialise one new atom via the stick-breaking process.

        Draws a Beta(1, alpha) fraction of the remaining stick, appends
        the resulting weight and a fresh signature drawn from base_measure.
        """
        fraction = beta.rvs(1, self.alpha, random_state=self.seed_generator)
        e_k = fraction * self.remaining_stick
        self.remaining_stick -= e_k
        self.mut_activities.append(e_k)
        self._cumsums.append(self._cumsums[-1] + e_k)
        self.signatures.append(self.base_measure.sample())

    def sample(self) -> np.ndarray:
        """
        Draw a mutational signature theta_i from this node's mixture G_j.

        A uniform U(0,1) deviate is compared against the cumulative
        stick weights.  If it falls beyond the already-broken portion,
        new pieces are broken until it is covered.

        Returns
        -------
        np.ndarray
            The selected 96-dimensional probability vector.
        """
        a = uniform.rvs(0, 1, random_state=self.seed_generator)
        while 1.0 - self.remaining_stick <= a:
            self._break_new_stick_piece()
        index = bisect.bisect(self._cumsums, a) - 1
        return self.signatures[index]

    def sample_mutation(self) -> int:
        """
        Convenience wrapper: draw a signature then draw a single trinucleotide
        mutation channel from it.

        Returns
        -------
        int
            An integer in [0, dimensions).
        """
        theta = self.sample()
        return int(np.argmax(multinomial.rvs(1, theta, random_state=self.seed_generator)))