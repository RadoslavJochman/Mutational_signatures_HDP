from abc import ABC, abstractmethod
import bisect
import numpy as np
from typing import List
from scipy.stats import dirichlet
from scipy.stats import beta
from scipy.stats import uniform

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

class Measure(ABC):
    """
    Abstract Base Class representing a probability measure.
    Both the Global Prior (H) and any Dirichlet Process (G) are measures
    that can be sampled from.
    """

    @abstractmethod
    def sample(self) -> np.ndarray:
        """
        Draw a sample from the probability measure.

        Returns:
            np.ndarray: A probability vector over the 96 trinucleotide channels.
        """
        pass


class DirichletPrior(Measure):
    """
    Represents the base measure H at the very top of the hierarchy.
    """

    def __init__(self, dimensions: int = 96):
        """
        Initializes the symmetric prior for the mutational signatures.

        Args:
            dimensions (int): The number of mutation channels (default 96).
        """
        self.dimensions = dimensions
        self.prior_alpha = np.ones(self.dimensions) / self.dimensions

    def sample(self) -> np.ndarray:
        """
        Draws a completely new mutational signature (theta_tilde) from the
        Dirichlet prior.

        Returns:
            np.ndarray: A 'self.dimensions'-dimensional probability vector.
        """
        return dirichlet.rvs(self.prior_alpha, random_state=rng)[0]


class DirichletProcess(Measure):
    """
    Represents a specific node's probability measure, G_j ~ DP(alpha_j, G_s).
    """

    def __init__(self, alpha: float, base_measure: Measure):
        """
        Initializes the Dirichlet Process.

        Args:
            alpha (float): The concentration parameter controlling variance.
            base_measure (Measure): The parent measure to draw signatures from.
                                    Can be DirichletPrior (H) or another DirichletProcess (G_s).
        """
        self.alpha = alpha
        self.base_measure = base_measure

        # State tracking for the stick-breaking process (e ~ Stick(alpha))
        self.mut_activities: List[float] = []  # Stores the lengths of the broken stick pieces (e_i)
        self.signatures: List[np.ndarray] = []  # Stores the signatures (theta_tilde_k) drawn from base_measure
        self.remaining_stick: float = 1.0  # Tracks how much of the stick is left to break
        self._cumsums = [0,]

    def _break_new_stick_piece(self):
        """
        Breaks a random fraction of the `remaining_stick` using a Beta distribution beta(1, 'self.alpha').
        Draws a new signature from `self.base_measure.sample()`.
        Appends the new activity and signature to the state trackers.
        """
        fraction = beta.rvs(1, self.alpha, random_state=rng)
        e_i = fraction * self.remaining_stick
        self.remaining_stick -= e_i
        self.mut_activities.append(e_i)
        self._cumsums.append(self._cumsums[-1] + e_i)
        self.signatures.append(self.base_measure.sample())

    def sample(self) -> np.ndarray:
        """
        Draws a specific mutational signature (theta_i) from this node's mixture distribution G.

        Logic:
        - Generate a random number between 0 and 1.
        - Iterate through `mut_activities`. If the random number falls within an existing activity,
          return the corresponding signature.
        - If it falls in the `remaining_stick`, call `_break_new_stick_piece()` until a new
          signature is selected.

        Returns:
            np.ndarray: The selected 96-dimensional probability vector.
        """
        a = uniform.rvs(0, 1, random_state=rng)
        while 1 - self.remaining_stick <= a:
            self._break_new_stick_piece()
        index = bisect.bisect(self._cumsums, a)-1
        return self.signatures[index]