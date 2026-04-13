import numpy as np
import phylox
import networkx as nx
import pymc as pm
import pytensor.tensor as pt
import pandas as pd


class Fixed_sig_HDP:
    def __init__(self, newick_string: str, data_matrix: pd.DataFrame, fixed_signatures: np.ndarray):
        self.data_matrix = data_matrix
        self.fixed_signatures = fixed_signatures
        self.K = fixed_signatures.shape[0]

        # node_id -> (pymc_variable_name, row_index)
        # row_index is None for depth-0 nodes (they get their own variable)
        self.node_index_map = {}

        self.graph = nx.DiGraph()
        individual_trees = [phylox.DiNetwork.from_newick(s) for s in newick_string.split(';') if s.strip()]

        for tree_idx, tree in enumerate(individual_trees):
            mapping = {}
            for n in tree.nodes():
                label = tree.nodes[n].get('label', str(n))
                new_id = f"t{tree_idx}_{label}"
                mapping[n] = new_id
            relabeled = nx.relabel_nodes(tree, mapping)
            for old_n, new_n in mapping.items():
                relabeled.nodes[new_n]['label'] = tree.nodes[old_n].get('label', str(old_n))
            self.graph = nx.compose(self.graph, relabeled)

        self.model = None
        self.trace = None
        self._build_tree_hdp_model()

    def _build_tree_hdp_model(self):
        with pm.Model() as self.model:
            signatures = pt.as_tensor_variable(self.fixed_signatures)
            shared_alpha = pm.LogNormal("shared_alpha", mu=2.5, sigma=1.0)
            e_0 = pm.Dirichlet("e_0", a=np.ones(self.K) * (shared_alpha / self.K))

            roots = [n for n, d in self.graph.in_degree() if d == 0]

            nodes_by_depth = {}
            for root in roots:
                for node, depth in nx.single_source_shortest_path_length(self.graph, root).items():
                    if node not in {n for nodes in nodes_by_depth.values() for n in nodes}:
                        nodes_by_depth.setdefault(depth, []).append(node)

            node_es = {}

            # Depth 0: each root gets its own named variable
            for root in nodes_by_depth.get(0, []):
                node_name = str(root).replace("-", "_").replace(".", "_")
                e_root = pm.Dirichlet(f"e_{node_name}", a=(shared_alpha * e_0))
                node_es[root] = e_root
                self.node_index_map[root] = (f"e_{node_name}", None)

            # Depth 1+: batched by depth, but indexed in node_index_map
            max_depth = max(nodes_by_depth.keys()) if nodes_by_depth else 0
            for depth in range(1, max_depth + 1):
                current_nodes = nodes_by_depth.get(depth, [])
                if not current_nodes:
                    continue

                parent_nodes = [list(self.graph.predecessors(n))[0] for n in current_nodes]
                parent_e_stack = pt.stack([node_es[p] for p in parent_nodes])
                a_matrix = shared_alpha * parent_e_stack

                var_name = f"e_level_{depth}"
                e_level = pm.Dirichlet(var_name, a=a_matrix, shape=(len(current_nodes), self.K))

                for i, node in enumerate(current_nodes):
                    node_es[node] = e_level[i]
                    self.node_index_map[node] = (var_name, i)

            # Likelihood
            observed_es = []
            obs_counts = []

            for node in self.graph.nodes():
                label = self.graph.nodes[node].get('label', str(node))
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
                    observed=obs_counts_matrix
                )

    def get_node_posterior(self, node_id: str) -> np.ndarray:
        """
        Returns posterior samples for a node's activity vector.
        Shape: (chains, draws, K)
        """
        if node_id not in self.node_index_map:
            raise KeyError(f"Node '{node_id}' not found. Available nodes: {list(self.node_index_map.keys())}")

        var_name, row_idx = self.node_index_map[node_id]
        samples = self.trace.posterior[var_name]

        if row_idx is None:
            return samples.values  # (chains, draws, K)
        else:
            return samples.values[:, :, row_idx, :]  # (chains, draws, K)

    def sample(self, draws=1000, tune=1000, chains=4, cores=4, target_accept=0.95):
        if self.model is None:
            raise ValueError("Model has not been built yet.")
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                nuts_sampler="numpyro"
            )
        return self.trace