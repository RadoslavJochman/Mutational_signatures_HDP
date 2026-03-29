from abc import ABC, abstractmethod
import bisect
import numpy as np
from typing import List, Dict, Optional
from scipy.stats import dirichlet, uniform, beta, multinomial
import phylox
import networkx as nx

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


class HDP:
    """
    Manages the tree topology using PhyloX,
    and orchestrates the HDP.
    """

    def __init__(self, newick_string: str, global_prior: Measure, alpha_0: float, alpha_dict: Dict[str, float] = None, default_alpha_j: float = 2.0):
        """
        Args:
            newick_string: The tree topology in Newick format.
            global_prior: The base measure H.
            alpha_0: Concentration parameter for the root G_0.
            alpha_dict: A dictionary mapping specific node names to their unique alpha_j.
            default_alpha_j: The fallback alpha_j if a node is not in alpha_dict.
        """
        self.global_prior = global_prior
        self.alpha_0 = alpha_0
        self.alpha_dict = alpha_dict if alpha_dict is not None else {}
        self.default_alpha_j = default_alpha_j

        # Parse the Newick string
        self.graph = phylox.DiNetwork.from_newick(newick_string)

        # Populate the graph with Dirichlet processes
        self._initialize_dp_models()

    def _initialize_dp_models(self):
        """
        Traverses the DAG in topological order to instantiate the DPs.
        This guarantees parents are fully initialized before children.
        """
        for node in nx.topological_sort(self.graph):
            parents = list(self.graph.predecessors(node))

            if not parents:
                # Root node: G_0 ~ DP(alpha_0, H)
                dp = DirichletProcess(alpha=self.alpha_0, base_measure=self.global_prior)
            else:
                # Child node: G_j ~ DP(alpha_j, G_s)
                parent_dp = self.graph.nodes[parents[0]]['dp_model']
                node_alpha = self.alpha_dict.get(node, self.default_alpha_j)
                dp = DirichletProcess(alpha=node_alpha, base_measure=parent_dp)

            # Store the model and initialize the mutations list
            self.graph.nodes[node]['dp_model'] = dp
            self.graph.nodes[node]['mutations'] = []

            # Extract branch length from edge data to use as num_mutations
            # Assuming the root has 0 mutations, and children inherit branch length from their incoming edge
            if parents:
                edge_data = self.graph.get_edge_data(parents[0], node)
                # Fallback to 0 if length isn't specified
                branch_length = edge_data.get('length', 0.0)
            else:
                branch_length = 0.0

            self.graph.nodes[node]['num_mutations'] = int(branch_length)

    def generate_all_data(self) -> Dict[str, List[int]]:
        """
        Iterates through the graph, samples from the DPs, and generates mutations.
        """
        results = {}
        for node in self.graph.nodes():
            dp_model = self.graph.nodes[node]['dp_model']
            num_mutations = self.graph.nodes[node]['num_mutations']
            mutations_list = self.graph.nodes[node]['mutations']

            for _ in range(num_mutations):
                theta_ji = dp_model.sample()
                x_ji = np.argmax(multinomial.rvs(1, theta_ji, random_state=rng))
                mutations_list.append(int(x_ji))

            results[node] = mutations_list

        return results