"""
hdp_simulator.py

Forward simulation models that generate synthetic mutational data by
sampling down a phylogenetic tree according to the Tree-HDP generative process.

Classes
-------
HDP
    Full generative model using the stick-breaking Dirichlet Process.
    Mutational signatures are drawn on-the-fly from a DirichletPrior.
    Useful for exploring the prior, visualising tree structure, and
    generating synthetic data when signatures are *unknown*.

Forward_HDP_Generator
    Simplified generative model that uses a *fixed* set of known signatures.
    At each node, activities e_j ~ Dir(alpha * e_parent) are sampled, then
    mutations are drawn from the resulting mixture.  This mirrors exactly the
    mathematical assumptions of the Fixed_sig_HDP PyMC inference model,
    making it the correct data-generating process for simulation studies.
"""

from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
import phylox
from networkx.drawing.nx_pydot import graphviz_layout
from phylox.constants import LABEL_ATTR
from phylox.generators.randomTC import generate_network_random_tree_child_sequence
import matplotlib.pyplot as plt

from src.models.dirichlet_process import DirichletPrior, DirichletProcess, Measure


# ---------------------------------------------------------------------------
# HDP – stick-breaking forward simulation (unknown signatures)
# ---------------------------------------------------------------------------

class HDP:
    """
    Tree-HDP forward simulator using the stick-breaking Dirichlet Process.

    Parses a Newick string into a directed tree, attaches a DirichletProcess
    to every node, and generates trinucleotide mutations by sampling down the
    hierarchy.

    Parameters
    ----------
    newick_string : str
        Phylogenetic tree in Newick format.  Branch lengths are interpreted
        as the number of mutations assigned to that branch.
    global_prior : Measure
        The base measure H (typically a DirichletPrior).
    alpha_0 : float
        Concentration parameter for the root-level DP G_0 ~ DP(alpha_0, H).
    alpha_dict : dict, optional
        Mapping {node_label: alpha_j} for nodes that should have a custom
        concentration parameter.  Unlisted nodes use `default_alpha_j`.
    default_alpha_j : float
        Fallback concentration parameter for nodes not in `alpha_dict`.
    """

    def __init__(
        self,
        newick_string: str,
        global_prior: Measure,
        alpha_0: float,
        alpha_dict: Optional[Dict[str, float]] = None,
        default_alpha_j: float = 2.0,
    ):
        self.global_prior = global_prior
        self.alpha_0 = alpha_0
        self.alpha_dict = alpha_dict or {}
        self.default_alpha_j = default_alpha_j
        self.seed_generator = global_prior.seed_generator

        self.graph = phylox.DiNetwork.from_newick(newick_string)
        self._initialize_dp_models()

    def _initialize_dp_models(self) -> None:
        """
        Walk the DAG in topological order (parents before children) and
        attach a DirichletProcess to every node.
        """
        self.G_0 = DirichletProcess(alpha=self.alpha_0, base_measure=self.global_prior)

        for node in nx.topological_sort(self.graph):
            parents = list(self.graph.predecessors(node))
            label = self.graph.nodes[node].get("label", str(node))

            if not parents:
                # Root node: draws from the global G_0
                node_alpha = self.alpha_dict.get(label, self.default_alpha_j)
                dp = DirichletProcess(alpha=node_alpha, base_measure=self.G_0)
                branch_length = 0.0
            else:
                parent_dp = self.graph.nodes[parents[0]]["dp_model"]
                node_alpha = self.alpha_dict.get(label, self.default_alpha_j)
                dp = DirichletProcess(alpha=node_alpha, base_measure=parent_dp)
                edge_data = self.graph.get_edge_data(parents[0], node)
                branch_length = edge_data.get("length", 0.0)

            self.graph.nodes[node]["dp_model"] = dp
            self.graph.nodes[node]["mutations"] = []
            self.graph.nodes[node]["num_mutations"] = int(branch_length)

    def generate_all_data(self) -> Dict[str, List[int]]:
        """
        Sample mutations for every node according to branch lengths.

        Returns
        -------
        dict
            Mapping {node_label: [mutation_channel_index, ...]}.
        """
        results = {}
        for node in self.graph.nodes():
            dp_model = self.graph.nodes[node]["dp_model"]
            num_mutations = self.graph.nodes[node]["num_mutations"]
            mutations_list = self.graph.nodes[node]["mutations"]

            for _ in range(num_mutations):
                mutations_list.append(dp_model.sample_mutation())

            results[self.graph.nodes[node]["label"]] = mutations_list

        return results

    def get_node_data(self, node_label: str, data_type: Optional[str] = None):
        """
        Retrieve stored data for a named node.

        Parameters
        ----------
        node_label : str
        data_type : str, optional
            Key into the node attribute dict (e.g. 'dp_model', 'mutations').
            If None the full attribute dict is returned.
        """
        node_id = self.graph.label_to_node_dict[node_label]
        if data_type is None:
            return self.graph.nodes[node_id]
        return self.graph.nodes[node_id][data_type]

    def get_mutation_count_matrix(self) -> pd.DataFrame:
        """
        Aggregate per-node mutation lists into an (N × 96) count matrix.

        Nodes with zero mutations are excluded.

        Returns
        -------
        pd.DataFrame
            Index: node labels.  Columns: Channel_0 … Channel_95.
        """
        node_labels, count_rows = [], []

        for node in self.graph.nodes():
            label = self.graph.nodes[node].get("label", str(node))
            mutations = self.graph.nodes[node].get("mutations", [])
            if not mutations:
                continue
            count_rows.append(np.bincount(mutations, minlength=96))
            node_labels.append(label)

        columns = [f"Channel_{i}" for i in range(96)]
        return pd.DataFrame(count_rows, index=node_labels, columns=columns)

    def plot_tree(self, save_path: Optional[str] = None) -> None:
        """
        Draw the phylogenetic tree with edge labels showing the first few
        sampled mutation channel indices.

        Parameters
        ----------
        save_path : str, optional
            If provided the figure is saved at 600 dpi to this path.
        """
        n = len(self.graph.nodes())

        safe_mapping = {nd: f"Node_{str(nd).replace('-', 'M')}" for nd in self.graph.nodes()}
        reverse_mapping = {v: k for k, v in safe_mapping.items()}

        layout_graph = nx.DiGraph()
        layout_graph.add_nodes_from(safe_mapping.values())
        for u, v in self.graph.edges():
            layout_graph.add_edge(safe_mapping[u], safe_mapping[v])

        safe_pos = graphviz_layout(layout_graph, prog="dot")
        pos = {reverse_mapping[s]: coords for s, coords in safe_pos.items()}

        node_size = max(400, 150_000 // max(1, n))
        font_size = max(6, node_size // 500)
        fig_w = max(10, n * 0.6)
        fig_h = max(8, n * 0.4)

        plt.figure(figsize=(fig_w, fig_h))
        nx.draw(
            self.graph,
            pos,
            labels=dict(self.graph.nodes(data="label")),
            node_color="lightblue",
            node_size=node_size,
            font_size=font_size,
            font_weight="bold",
            edge_color="gray",
            arrows=True,
            arrowsize=max(10, 25 - n // 2),
        )
        edge_labels = {
            (i, j): self.graph.nodes("mutations")[j][:6]
            for i, j in self.graph.edges()
        }
        nx.draw_networkx_edge_labels(
            self.graph,
            pos,
            edge_labels=edge_labels,
            font_color="red",
            font_size=max(5, font_size - 1),
            font_weight="bold",
        )

        if save_path:
            plt.savefig(save_path, dpi=600)
            print(f"Saved tree plot to '{save_path}'")
        plt.show()

class Forward_HDP_Generator:
    """
    Generative model for the *fixed-signature* Tree-HDP.

    Mirrors the mathematical assumptions of Fixed_sig_HDP exactly so that
    simulation studies can verify posterior recovery.

    The generative process is:
        e_0  ~ Dir(alpha/K * 1_K)            # global cohort baseline
        e_j  ~ Dir(alpha * e_parent)          # per-node activities
        x_ji ~ Categorical(e_j @ signatures)  # observed mutations

    Parameters
    ----------
    newick_string : str
        One or more Newick trees separated by semicolons.  Each tree is
        treated as an independent tumour.
    alpha : float
        Shared concentration parameter used at every node.
    fixed_signatures : np.ndarray
        Shape (K, 96).  The fixed mutational signature matrix.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        newick_string: str,
        alpha: float,
        fixed_signatures: np.ndarray,
        seed: int = 42,
    ):
        self.alpha = alpha
        self.fixed_signatures = fixed_signatures
        self.K = fixed_signatures.shape[0]
        self.rng = np.random.default_rng(seed)

        # Global cohort baseline
        self.e_0 = self.rng.dirichlet(np.ones(self.K) * (self.alpha / self.K))

        # Build one graph per tree in the forest
        self.graphs = [
            phylox.DiNetwork.from_newick(s)
            for s in newick_string.split(";")
            if s.strip()
        ]
        self._initialize_node_activities()

    def _initialize_node_activities(self) -> None:
        """
        Traverse every tree in topological order, drawing activity vectors
        e_j ~ Dir(alpha * e_parent) for each node.
        """
        for graph in self.graphs:
            for node in nx.topological_sort(graph):
                parents = list(graph.predecessors(node))

                parent_e = (
                    self.e_0 if not parents
                    else graph.nodes[parents[0]]["e_vector"]
                )

                a_vector = np.clip(self.alpha * parent_e, 1e-9, None)
                graph.nodes[node]["e_vector"] = self.rng.dirichlet(a_vector)

                if parents:
                    edge_data = graph.get_edge_data(parents[0], node)
                    branch_length = edge_data.get("length", 0.0)
                else:
                    branch_length = 0.0

                graph.nodes[node]["num_mutations"] = int(branch_length)


    def get_mutation_count_matrix(self) -> pd.DataFrame:
        """
        Generate the (N × 96) mutation count matrix.

        Nodes with zero mutations are skipped.

        Returns
        -------
        pd.DataFrame
            Index: node labels.  Columns: Channel_0 … Channel_95.
        """
        node_labels, count_rows = [], []

        for graph in self.graphs:
            for node in graph.nodes():
                num_mut = graph.nodes[node]["num_mutations"]
                # We skip roots which don't have any mutations
                if num_mut <= 0:
                    continue
                e_vector = graph.nodes[node]["e_vector"]
                expected_probs = np.dot(e_vector, self.fixed_signatures)
                counts = self.rng.multinomial(num_mut, expected_probs)
                label = graph.nodes[node].get(LABEL_ATTR, str(node))
                node_labels.append(label)
                count_rows.append(counts)

        columns = [f"Channel_{i}" for i in range(96)]
        return pd.DataFrame(count_rows, index=node_labels, columns=columns)

    def get_true_activities(self) -> Dict[str, np.ndarray]:
        """
        Return the ground-truth activity vectors for all nodes.

        Returns
        -------
        dict
            Mapping {node_label: e_vector (shape K,)}.
        """
        return {
            graph.nodes[node].get(LABEL_ATTR, str(node)): graph.nodes[node]["e_vector"]
            for graph in self.graphs
            for node in graph.nodes()
        }

def generate_random_phylogenetic_forest(
    num_trees: int,
    min_leaves: int,
    max_leaves: int,
    min_branch_length: int,
    max_branch_length: int,
    rng: np.random.Generator,
) -> str:
    """
    Generate a random forest of phylogenetic trees and return them as a
    concatenated Newick string (semicolon-separated).

    Parameters
    ----------
    num_trees : int
    min_leaves, max_leaves : int
        Inclusive range for the number of leaves per tree.
    min_branch_length, max_branch_length : int
        Inclusive range for integer branch lengths (= number of mutations).
    rng : np.random.Generator
        Shared RNG for reproducibility.

    Returns
    -------
    str
        Concatenated Newick strings, one tree per semicolon-delimited entry.
    """
    forest_newicks = []

    for tree_idx in range(num_trees):
        n_leaves = int(rng.integers(min_leaves, max_leaves + 1))
        tree_seed = int(rng.integers(0, 2**31))

        phylo_tree = generate_network_random_tree_child_sequence(
            n_leaves, 0, seed=tree_seed
        )

        for i, node in enumerate(nx.topological_sort(phylo_tree)):
            phylo_tree.nodes[node][LABEL_ATTR] = f"T{tree_idx + 1}_{i + 1}"

        for u, v in phylo_tree.edges():
            phylo_tree[u][v]["length"] = int(
                rng.integers(min_branch_length, max_branch_length + 1)
            )

        forest_newicks.append(phylo_tree.newick())

    return "".join(forest_newicks)