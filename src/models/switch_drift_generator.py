"""
switch_drift_generator.py

Phase-one implementation of the switch-plus-drift forward simulator described
in simulator_spec.md. This is an interim module: `TreeSwitchDriftGenerator`
and its helpers live here, separate from `hdp_simulator.py`, so the rewrite is
purely additive and reviewable on its own. Phase two moves this module's
contents into `hdp_simulator.py`, deletes the old `TreeSignatureGenerator`,
and removes this module; nothing here is meant to be a permanent second
generator module.

The module takes plain dicts and dataclasses only. It has no dependency on
`src/config.py` or YAML -- parsing a config file into the dicts this module
consumes is `generate_data.py`'s job (phase two).

Generative process (simulator_spec.md sections 2 and 4)
--------------------------------------------------------
Each node j carries a binary active set a_j over the repertoire and an
activity vector e_j on the simplex with off-signatures at exactly zero.

    Switch:  each switch unit's on/off state evolves along the tree as a
             two-state gain/loss process, with flip probability rising with
             branch length (or flat, per `switching.branch_length_scaling`).
    Drift:   among the on-signatures, e_j drifts from e_parent by a
             mean-preserving Dirichlet step whose concentration scales with
             branch length (floored so a long branch cannot collapse the
             draw to a one-hot vector).
    Counts:  x_j ~ DirichletMultinomial(M_j, kappa * e_j @ S), or a plain
             Multinomial as a well-specified control.

One `np.random.Generator`, seeded once, threads through every draw (topology,
switching, drift, burden, counts), so one seed reproduces one forest exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import phylox
from phylox.constants import LABEL_ATTR

N_CHANNELS = 96


# Config dataclasses (the parsed, validated form of the plain-dict config)


@dataclass(frozen=True)
class SwitchUnit:
    """One switch unit: signatures that gain/lose activity together."""

    signatures: Tuple[str, ...]
    root_prob: float
    lambda_on: float
    lambda_off: float


@dataclass(frozen=True)
class SwitchingConfig:
    enabled: bool
    branch_length_scaling: bool
    units: Tuple[SwitchUnit, ...] = ()


@dataclass(frozen=True)
class LevelsConfig:
    enabled: bool
    concentration: float
    branch_length_scaling: bool
    concentration_floor: float
    root_concentration: float
    activation_pseudocount: float


@dataclass(frozen=True)
class BranchLengthConfig:
    distribution: str  # "lognormal" or "empirical"
    params: Dict[str, float] = field(default_factory=dict)
    path: Optional[str] = None


@dataclass(frozen=True)
class ForestConfig:
    n_trees: int
    nodes_per_tree: Tuple[int, int]
    depth: Tuple[int, int]
    max_attempts: int
    branch_lengths: BranchLengthConfig


@dataclass(frozen=True)
class BurdenConfig:
    mean: float
    distribution: str  # "negbin"
    dispersion: float


@dataclass(frozen=True)
class CountsConfig:
    model: str  # "dirichlet_multinomial" or "multinomial"
    kappa: Optional[float]


@dataclass(frozen=True)
class RepertoireConfig:
    source: str
    path: str
    signatures: Tuple[str, ...]


@dataclass(frozen=True)
class GeneratorConfig:
    """The fully parsed, validated generator config (everything but the seed)."""

    repertoire: RepertoireConfig
    switching: SwitchingConfig
    levels: LevelsConfig
    forest: ForestConfig
    burden: BurdenConfig
    counts: CountsConfig


@dataclass
class _NodeRecord:
    """Ground-truth record for a single node."""

    tumour: int
    label: str
    parent_label: Optional[str]
    is_root: bool
    branch_length: Optional[float]
    active_set: np.ndarray  # (K,) bool
    e_vector: np.ndarray  # (K,) float
    burden: int
    counts: np.ndarray  # (96,) int


# Config parsing and validation (simulator_spec.md section 3)


def _require(cfg: Dict[str, Any], key: str, section: str) -> Any:
    if key not in cfg or cfg[key] is None:
        raise ValueError(f"{section} config missing required key '{key}'")
    return cfg[key]


def parse_repertoire_config(cfg: Dict[str, Any]) -> RepertoireConfig:
    source = str(_require(cfg, "source", "repertoire"))
    path = str(_require(cfg, "path", "repertoire"))
    signatures = list(_require(cfg, "signatures", "repertoire"))
    if not signatures:
        raise ValueError("repertoire.signatures must be non-empty")
    if len(set(signatures)) != len(signatures):
        raise ValueError("repertoire.signatures must not contain duplicates")
    return RepertoireConfig(source=source, path=path, signatures=tuple(signatures))


def parse_switching_config(
    cfg: Dict[str, Any], repertoire_signatures: Sequence[str]
) -> SwitchingConfig:
    enabled = bool(_require(cfg, "enabled", "switching"))
    branch_length_scaling = bool(_require(cfg, "branch_length_scaling", "switching"))
    if not enabled:
        # units are optional and ignored when switching is off.
        return SwitchingConfig(
            enabled=False, branch_length_scaling=branch_length_scaling, units=()
        )

    raw_units = cfg.get("units")
    if not raw_units:
        raise ValueError("switching.units is required when switching.enabled is true")

    units: List[SwitchUnit] = []
    for i, u in enumerate(raw_units):
        sigs = tuple(_require(u, "signatures", f"switching.units[{i}]"))
        if not sigs:
            raise ValueError(f"switching.units[{i}].signatures must be non-empty")
        root_prob = float(_require(u, "root_prob", f"switching.units[{i}]"))
        if not 0.0 <= root_prob <= 1.0:
            raise ValueError(
                f"switching.units[{i}].root_prob must be in [0, 1], got {root_prob}"
            )
        lambda_on = float(_require(u, "lambda_on", f"switching.units[{i}]"))
        lambda_off = float(_require(u, "lambda_off", f"switching.units[{i}]"))
        if branch_length_scaling:
            if lambda_on < 0.0 or lambda_off < 0.0:
                raise ValueError(
                    f"switching.units[{i}] lambda_on/lambda_off must be "
                    "non-negative when switching.branch_length_scaling is true"
                )
        else:
            if not (0.0 <= lambda_on <= 1.0) or not (0.0 <= lambda_off <= 1.0):
                raise ValueError(
                    f"switching.units[{i}] lambda_on/lambda_off must be in "
                    "[0, 1] when switching.branch_length_scaling is false"
                )
        units.append(
            SwitchUnit(
                signatures=sigs,
                root_prob=root_prob,
                lambda_on=lambda_on,
                lambda_off=lambda_off,
            )
        )

    all_unit_sigs = [s for u in units for s in u.signatures]
    if len(all_unit_sigs) != len(set(all_unit_sigs)):
        raise ValueError(
            "switching.units signatures overlap: each signature must belong "
            "to exactly one unit"
        )
    repertoire_set = set(repertoire_signatures)
    unit_set = set(all_unit_sigs)
    if unit_set != repertoire_set:
        missing = sorted(repertoire_set - unit_set)
        extra = sorted(unit_set - repertoire_set)
        parts = []
        if missing:
            parts.append(f"missing from any unit: {missing}")
        if extra:
            parts.append(f"in a unit but not in repertoire.signatures: {extra}")
        raise ValueError(
            "switching.units must partition repertoire.signatures exactly; "
            + "; ".join(parts)
        )

    if not any(u.root_prob == 1.0 and u.lambda_off == 0.0 for u in units):
        raise ValueError(
            "switching.units must include at least one always-on unit "
            "(root_prob=1.0, lambda_off=0.0); this is the clock guard that "
            "keeps every node's active set non-empty"
        )

    return SwitchingConfig(
        enabled=True,
        branch_length_scaling=branch_length_scaling,
        units=tuple(units),
    )


def parse_levels_config(cfg: Dict[str, Any]) -> LevelsConfig:
    enabled = bool(_require(cfg, "enabled", "levels"))
    walk = cfg.get("walk", "dirichlet")
    if walk != "dirichlet":
        raise ValueError(f"levels.walk must be 'dirichlet', got {walk!r}")
    concentration = float(_require(cfg, "concentration", "levels"))
    if concentration <= 0.0:
        raise ValueError("levels.concentration must be positive")
    branch_length_scaling = bool(_require(cfg, "branch_length_scaling", "levels"))
    concentration_floor = float(_require(cfg, "concentration_floor", "levels"))
    if concentration_floor <= 0.0:
        raise ValueError("levels.concentration_floor must be positive")
    if branch_length_scaling and concentration_floor > concentration:
        raise ValueError(
            "levels.concentration_floor must be <= levels.concentration when "
            "levels.branch_length_scaling is true"
        )
    root_concentration = float(_require(cfg, "root_concentration", "levels"))
    if root_concentration <= 0.0:
        raise ValueError("levels.root_concentration must be positive")
    activation_pseudocount = float(_require(cfg, "activation_pseudocount", "levels"))
    if activation_pseudocount <= 0.0:
        raise ValueError("levels.activation_pseudocount must be positive")
    return LevelsConfig(
        enabled=enabled,
        concentration=concentration,
        branch_length_scaling=branch_length_scaling,
        concentration_floor=concentration_floor,
        root_concentration=root_concentration,
        activation_pseudocount=activation_pseudocount,
    )


def parse_branch_length_config(cfg: Dict[str, Any]) -> BranchLengthConfig:
    distribution = _require(cfg, "distribution", "forest.branch_lengths")
    if distribution == "lognormal":
        params = _require(cfg, "params", "forest.branch_lengths")
        mu = float(_require(params, "mu", "forest.branch_lengths.params"))
        sigma = float(_require(params, "sigma", "forest.branch_lengths.params"))
        if sigma <= 0.0:
            raise ValueError("forest.branch_lengths.params.sigma must be positive")
        return BranchLengthConfig(
            distribution="lognormal", params={"mu": mu, "sigma": sigma}
        )
    if distribution == "empirical":
        path = str(_require(cfg, "path", "forest.branch_lengths"))
        return BranchLengthConfig(distribution="empirical", params={}, path=path)
    raise ValueError(
        "forest.branch_lengths.distribution must be 'lognormal' or "
        f"'empirical', got {distribution!r}"
    )


def parse_forest_config(cfg: Dict[str, Any]) -> ForestConfig:
    n_trees = int(_require(cfg, "n_trees", "forest"))
    if n_trees <= 0:
        raise ValueError("forest.n_trees must be positive")
    nodes_per_tree = tuple(int(x) for x in _require(cfg, "nodes_per_tree", "forest"))
    if len(nodes_per_tree) != 2 or not (2 <= nodes_per_tree[0] <= nodes_per_tree[1]):
        raise ValueError(
            "forest.nodes_per_tree must be [min, max] with 2 <= min <= max"
        )
    depth = tuple(int(x) for x in _require(cfg, "depth", "forest"))
    if len(depth) != 2 or not (1 <= depth[0] <= depth[1]):
        raise ValueError("forest.depth must be [min, max] with 1 <= min <= max")
    max_attempts = int(_require(cfg, "max_attempts", "forest"))
    if max_attempts <= 0:
        raise ValueError("forest.max_attempts must be positive")
    branch_lengths = parse_branch_length_config(
        _require(cfg, "branch_lengths", "forest")
    )
    return ForestConfig(
        n_trees=n_trees,
        nodes_per_tree=nodes_per_tree,
        depth=depth,
        max_attempts=max_attempts,
        branch_lengths=branch_lengths,
    )


def parse_burden_config(cfg: Dict[str, Any]) -> BurdenConfig:
    mean = float(_require(cfg, "mean", "burden"))
    if mean <= 0.0:
        raise ValueError("burden.mean must be positive")
    distribution = _require(cfg, "distribution", "burden")
    if distribution != "negbin":
        raise ValueError(f"burden.distribution must be 'negbin', got {distribution!r}")
    dispersion = float(_require(cfg, "dispersion", "burden"))
    if dispersion <= 0.0:
        raise ValueError("burden.dispersion must be positive")
    return BurdenConfig(mean=mean, distribution=distribution, dispersion=dispersion)


def parse_counts_config(cfg: Dict[str, Any]) -> CountsConfig:
    model = _require(cfg, "model", "counts")
    if model not in ("dirichlet_multinomial", "multinomial"):
        raise ValueError(
            "counts.model must be 'dirichlet_multinomial' or 'multinomial', "
            f"got {model!r}"
        )
    kappa = cfg.get("kappa")
    if model == "dirichlet_multinomial":
        if kappa is None:
            raise ValueError(
                "counts.kappa is required when counts.model is 'dirichlet_multinomial'"
            )
        kappa = float(kappa)
        if kappa <= 0.0:
            raise ValueError("counts.kappa must be positive")
    elif kappa is not None:
        kappa = float(kappa)
    return CountsConfig(model=model, kappa=kappa)


def parse_generator_config(config: Dict[str, Any]) -> GeneratorConfig:
    """Parse and validate the plain-dict config into a `GeneratorConfig`."""
    repertoire = parse_repertoire_config(_require(config, "repertoire", "simulation"))
    switching = parse_switching_config(
        _require(config, "switching", "simulation"), repertoire.signatures
    )
    levels = parse_levels_config(_require(config, "levels", "simulation"))
    forest = parse_forest_config(_require(config, "forest", "simulation"))
    burden = parse_burden_config(_require(config, "burden", "simulation"))
    counts = parse_counts_config(_require(config, "counts", "simulation"))
    return GeneratorConfig(
        repertoire=repertoire,
        switching=switching,
        levels=levels,
        forest=forest,
        burden=burden,
        counts=counts,
    )


# Repertoire loading


def load_catalogue(path: str, signature_names: Sequence[str]) -> pd.DataFrame:
    """Load and validate the requested signatures from a catalogue CSV.

    The CSV is read with the signature name as the row index and channels as
    columns (the `fixed_signatures.csv` layout). Rows are returned in the
    order `signature_names` asks for, regardless of file order.
    """
    try:
        catalogue = pd.read_csv(path, index_col=0)
    except FileNotFoundError as exc:
        raise ValueError(f"repertoire catalogue not found: '{path}'") from exc

    missing = [s for s in signature_names if s not in catalogue.index]
    if missing:
        raise ValueError(f"signatures {missing} not found in catalogue '{path}'")

    sub = catalogue.loc[list(signature_names)].astype(float)
    if sub.shape[1] != N_CHANNELS:
        raise ValueError(
            f"catalogue '{path}' must have {N_CHANNELS} channel columns, "
            f"got {sub.shape[1]}"
        )
    row_sums = sub.sum(axis=1).to_numpy()
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(f"every signature row in '{path}' must sum to 1")
    if (sub.to_numpy() < 0).any():
        raise ValueError(f"signatures in '{path}' must be non-negative")
    return sub


# Forest sampling (simulator_spec.md section 5)


def _find_root(graph: nx.DiGraph):
    roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    if len(roots) != 1:
        raise AssertionError(f"expected exactly one root, found {len(roots)}")
    return roots[0]


def _tree_depth(graph: nx.DiGraph, root) -> int:
    lengths = nx.single_source_shortest_path_length(graph, root)
    return max(lengths.values())


def _load_empirical_lengths(path: str) -> np.ndarray:
    """Read a single column of positive branch lengths, header optional."""
    values: List[float] = []
    with open(path) as fh:
        for line in fh:
            token = line.strip().split(",")[0].strip()
            if not token:
                continue
            try:
                values.append(float(token))
            except ValueError:
                continue  # a header line
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError(f"no branch lengths found in empirical file '{path}'")
    if np.any(arr <= 0.0):
        raise ValueError(f"empirical branch lengths in '{path}' must all be positive")
    return arr


def _sample_branch_length(
    branch_cfg: BranchLengthConfig,
    rng: np.random.Generator,
    empirical_cache: Dict[str, np.ndarray],
) -> float:
    if branch_cfg.distribution == "lognormal":
        return float(
            rng.lognormal(
                mean=branch_cfg.params["mu"], sigma=branch_cfg.params["sigma"]
            )
        )
    # empirical
    assert branch_cfg.path is not None
    if branch_cfg.path not in empirical_cache:
        empirical_cache[branch_cfg.path] = _load_empirical_lengths(branch_cfg.path)
    pool = empirical_cache[branch_cfg.path]
    return float(rng.choice(pool))


def sample_forest(
    forest_cfg: ForestConfig, rng: np.random.Generator
) -> List[phylox.DiNetwork]:
    """Rejection-sample `forest_cfg.n_trees` trees against nodes_per_tree/depth.

    For each tree, draw a candidate leaf count, build it with phylox's
    tree-child generator (0 reticulations), and accept it only if the
    realised node count and depth both fall in range; otherwise resample, up
    to `forest_cfg.max_attempts` times, after which a RuntimeError names the
    configured ranges and the attempt count reached.
    """
    from phylox.generators.randomTC import generate_network_random_tree_child_sequence

    empirical_cache: Dict[str, np.ndarray] = {}
    min_n, max_n = forest_cfg.nodes_per_tree
    min_d, max_d = forest_cfg.depth
    leaf_high = max(2, max_n)

    trees: List[phylox.DiNetwork] = []
    for t in range(forest_cfg.n_trees):
        accepted = None
        for _attempt in range(1, forest_cfg.max_attempts + 1):
            n_leaves = int(rng.integers(2, leaf_high + 1))
            tree_seed = int(rng.integers(0, 2**31 - 1))
            candidate = generate_network_random_tree_child_sequence(
                n_leaves, 0, seed=tree_seed
            )
            n_nodes = candidate.number_of_nodes()
            root = _find_root(candidate)
            depth = _tree_depth(candidate, root)
            if min_n <= n_nodes <= max_n and min_d <= depth <= max_d:
                accepted = candidate
                break
        if accepted is None:
            raise RuntimeError(
                "could not sample a tree with nodes_per_tree in "
                f"[{min_n}, {max_n}] and depth in [{min_d}, {max_d}] after "
                f"{forest_cfg.max_attempts} attempts; loosen nodes_per_tree "
                "or depth, or raise forest.max_attempts"
            )
        for i, node in enumerate(nx.topological_sort(accepted)):
            accepted.nodes[node][LABEL_ATTR] = f"T{t + 1}_{i + 1}"
        for u, v in accepted.edges():
            accepted[u][v]["length"] = _sample_branch_length(
                forest_cfg.branch_lengths, rng, empirical_cache
            )
        trees.append(accepted)
    return trees


# Switch process (simulator_spec.md sections 2 and 4)


def draw_root_states(
    units: Sequence[SwitchUnit], rng: np.random.Generator
) -> np.ndarray:
    """Draw each unit's root on/off state from its `root_prob`."""
    draws = rng.random(len(units))
    root_probs = np.array([u.root_prob for u in units])
    return draws < root_probs


def evolve_units(
    state: np.ndarray,
    length: float,
    units: Sequence[SwitchUnit],
    switching_cfg: SwitchingConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evolve unit states across one edge of length `length`.

    An off unit gains with probability `1 - exp(-lambda_on * length)` and an
    on unit loses with `1 - exp(-lambda_off * length)` when
    `switching_cfg.branch_length_scaling` is true; otherwise `lambda_on` and
    `lambda_off` are used directly as flat per-edge probabilities.
    """
    draws = rng.random(len(units))
    new_state = state.copy()
    for i, unit in enumerate(units):
        if switching_cfg.branch_length_scaling:
            p_gain = -np.expm1(-unit.lambda_on * length)
            p_loss = -np.expm1(-unit.lambda_off * length)
        else:
            p_gain = unit.lambda_on
            p_loss = unit.lambda_off
        if state[i]:
            if draws[i] < p_loss:
                new_state[i] = False
        else:
            if draws[i] < p_gain:
                new_state[i] = True
    return new_state


def expand_units_to_signatures(
    unit_state: np.ndarray, unit_of_signature: np.ndarray
) -> np.ndarray:
    """Map a per-unit boolean state to a per-signature active-set mask."""
    return unit_state[unit_of_signature]


def _unit_index_for_signatures(
    signature_names: Sequence[str], units: Sequence[SwitchUnit]
) -> np.ndarray:
    index = {name: i for i, name in enumerate(signature_names)}
    unit_of = np.full(len(signature_names), -1, dtype=int)
    for u_idx, unit in enumerate(units):
        for sig in unit.signatures:
            unit_of[index[sig]] = u_idx
    return unit_of


# Drift process (simulator_spec.md section 4)


def effective_concentration(
    levels_cfg: LevelsConfig, length: float, l_median: float
) -> float:
    """The total Dirichlet concentration for a step across an edge.

    `c * L_median / L_e`, floored at `concentration_floor`, when
    `branch_length_scaling` is true; a constant `c` otherwise.
    """
    if levels_cfg.branch_length_scaling:
        return max(
            levels_cfg.concentration * l_median / length, levels_cfg.concentration_floor
        )
    return levels_cfg.concentration


def carry_forward(
    e_parent: np.ndarray, a_child: np.ndarray, activation_pseudocount: float
) -> np.ndarray:
    """Build the renormalised base composition for the child's Dirichlet step.

    Restrict `e_parent` to `a_child`'s support, add `activation_pseudocount`
    to every signature that is newly on (active in the child, zero in the
    parent), and renormalise to sum to 1. This is the same rule regardless of
    whether the edge carries an activation, a deactivation, both, or neither.
    """
    base = np.zeros_like(e_parent)
    base[a_child] = e_parent[a_child]
    newly_on = a_child & (e_parent == 0.0)
    base[newly_on] += activation_pseudocount
    total = base.sum()
    if total <= 0.0:
        raise AssertionError(
            "carry-forward produced a non-positive base; the clock-guard "
            "invariant (a_child always non-empty) must have been violated"
        )
    return base / total


def dirichlet_step(
    base: np.ndarray, conc: float, levels_cfg: LevelsConfig, rng: np.random.Generator
) -> np.ndarray:
    """Draw the child activity: Dirichlet(conc * base) restricted to base's support.

    Mean-preserving: E[e_child] = base. When `levels_cfg.enabled` is false, no
    noise is added and `base` is returned unchanged (levels stay fixed until
    the active set itself changes).
    """
    if not levels_cfg.enabled:
        return base.copy()
    support = base > 0.0
    e = np.zeros_like(base)
    alpha = conc * base[support]
    e[support] = rng.dirichlet(alpha)
    return e


def draw_root_activity(
    a_root: np.ndarray, root_concentration: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw the root activity: a flat Dirichlet(root_concentration) over a_root."""
    e = np.zeros(a_root.shape[0])
    n_active = int(a_root.sum())
    alpha = np.full(n_active, root_concentration)
    e[a_root] = rng.dirichlet(alpha)
    return e


# Burden and counts (simulator_spec.md sections 6 and 7)


def draw_burden(burden_cfg: BurdenConfig, rng: np.random.Generator) -> int:
    """Negative-binomial burden draw, parameterised by mean and dispersion."""
    rate = rng.gamma(
        shape=burden_cfg.dispersion, scale=burden_cfg.mean / burden_cfg.dispersion
    )
    return int(rng.poisson(rate))


def draw_counts(
    burden: int,
    p_channels: np.ndarray,
    counts_cfg: CountsConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw the observed 96-channel counts for one node."""
    if counts_cfg.model == "multinomial":
        return rng.multinomial(burden, p_channels)
    # dirichlet_multinomial
    alpha = np.maximum(counts_cfg.kappa * p_channels, 1e-12)
    theta = rng.dirichlet(alpha)
    return rng.multinomial(burden, theta)


# The generator


class TreeSwitchDriftGenerator:
    """Forward simulator implementing the switch-plus-drift process.

    Parameters
    ----------
    config : dict
        Plain dict matching the schema in simulator_spec.md section 3, minus
        `seed`, `n_seeds`, `results_dir`, and `make_plots` (sweep- and
        config-layer concerns handled by `generate_data.py` in phase two).
        Required top-level keys: `repertoire`, `switching`, `levels`,
        `forest`, `burden`, `counts`.
    seed : int, optional
        RNG seed. If omitted, `config['seed']` is used instead; at least one
        of the two is required.
    """

    def __init__(self, config: Dict[str, Any], seed: Optional[int] = None):
        resolved_seed = config.get("seed") if seed is None else seed
        if resolved_seed is None:
            raise ValueError(
                "a seed is required, either as config['seed'] or the seed argument"
            )
        self.seed = int(resolved_seed)
        self.rng = np.random.default_rng(self.seed)

        self.cfg = parse_generator_config(config)
        self.signature_names: List[str] = list(self.cfg.repertoire.signatures)
        self.K = len(self.signature_names)

        self.S = load_catalogue(self.cfg.repertoire.path, self.signature_names)
        self._s_matrix = self.S.to_numpy()

        if self.cfg.switching.enabled:
            self._unit_of_signature = _unit_index_for_signatures(
                self.signature_names, self.cfg.switching.units
            )
        else:
            self._unit_of_signature = None

        self.forest: List[phylox.DiNetwork] = sample_forest(self.cfg.forest, self.rng)

        all_lengths = [
            graph[u][v]["length"] for graph in self.forest for u, v in graph.edges()
        ]
        if not all_lengths:
            raise ValueError("the sampled forest has no edges")
        self.l_median = float(np.median(all_lengths))

        self._records: List[_NodeRecord] = []
        self._simulate()
        self._count_matrix = self._build_count_matrix()

    # switch helpers

    def _active_set_from_state(self, state: Optional[np.ndarray]) -> np.ndarray:
        if self.cfg.switching.enabled:
            return expand_units_to_signatures(state, self._unit_of_signature)
        return np.ones(self.K, dtype=bool)

    def _root_active_set(self) -> Tuple[Optional[np.ndarray], np.ndarray]:
        if self.cfg.switching.enabled:
            state = draw_root_states(self.cfg.switching.units, self.rng)
        else:
            state = None
        return state, self._active_set_from_state(state)

    def _child_active_set(
        self, parent_state: Optional[np.ndarray], branch_length: float
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        if self.cfg.switching.enabled:
            state = evolve_units(
                parent_state,
                branch_length,
                self.cfg.switching.units,
                self.cfg.switching,
                self.rng,
            )
        else:
            state = None
        return state, self._active_set_from_state(state)

    # simulation

    def _simulate(self) -> None:
        for t_idx, graph in enumerate(self.forest):
            node_state: Dict[Any, Optional[np.ndarray]] = {}
            node_e: Dict[Any, np.ndarray] = {}

            for node in nx.topological_sort(graph):
                parents = list(graph.predecessors(node))
                label = graph.nodes[node][LABEL_ATTR]

                if not parents:
                    parent_label = None
                    branch_length = None
                    state, a = self._root_active_set()
                    assert a.any(), f"clock guard violated at root '{label}'"
                    e = draw_root_activity(
                        a, self.cfg.levels.root_concentration, self.rng
                    )
                else:
                    p = parents[0]
                    parent_label = graph.nodes[p][LABEL_ATTR]
                    branch_length = float(graph.get_edge_data(p, node)["length"])
                    state, a = self._child_active_set(node_state[p], branch_length)
                    assert a.any(), f"clock guard violated at node '{label}'"
                    base = carry_forward(
                        node_e[p], a, self.cfg.levels.activation_pseudocount
                    )
                    conc = effective_concentration(
                        self.cfg.levels, branch_length, self.l_median
                    )
                    e = dirichlet_step(base, conc, self.cfg.levels, self.rng)

                node_state[node] = state
                node_e[node] = e

                burden = draw_burden(self.cfg.burden, self.rng)
                p_channels = e @ self._s_matrix
                p_channels = p_channels / p_channels.sum()
                counts = draw_counts(burden, p_channels, self.cfg.counts, self.rng)

                self._records.append(
                    _NodeRecord(
                        tumour=t_idx,
                        label=label,
                        parent_label=parent_label,
                        is_root=not parents,
                        branch_length=branch_length,
                        active_set=a,
                        e_vector=e,
                        burden=burden,
                        counts=counts,
                    )
                )

    def _build_count_matrix(self) -> pd.DataFrame:
        cols = [f"Channel_{i}" for i in range(N_CHANNELS)]
        labels, rows = [], []
        for r in self._records:
            if r.burden == 0:
                continue
            labels.append(r.label)
            rows.append(r.counts)
        return pd.DataFrame(rows, index=labels, columns=cols)

    # accessors

    def get_true_signatures(self) -> pd.DataFrame:
        """The (K, 96) signature matrix, indexed by real signature name."""
        return self.S.copy()

    def get_true_activities(self) -> pd.DataFrame:
        """Ground-truth activities for every node (nodes x signature name)."""
        rows = {r.label: r.e_vector for r in self._records}
        return pd.DataFrame.from_dict(
            rows, orient="index", columns=self.signature_names
        )

    def get_true_active_sets(self) -> pd.DataFrame:
        """Ground-truth active sets for every node (nodes x signature name)."""
        rows = {r.label: r.active_set.astype(int) for r in self._records}
        return pd.DataFrame.from_dict(
            rows, orient="index", columns=self.signature_names
        )

    def get_mutation_count_matrix(self) -> pd.DataFrame:
        """Observed counts. Drawn once at construction; repeated calls are
        idempotent (see the NQ6 fix note in hdp_simulator.py's history)."""
        return self._count_matrix.copy()

    def get_tree_edges(self) -> pd.DataFrame:
        """Tree topology with branch lengths (parent, child, length)."""
        records = []
        for r in self._records:
            if r.parent_label is not None:
                records.append(
                    {
                        "tumour": r.tumour,
                        "parent": r.parent_label,
                        "child": r.label,
                        "length": r.branch_length,
                    }
                )
        return pd.DataFrame(records, columns=["tumour", "parent", "child", "length"])

    def get_newick_forest(self) -> str:
        """The forest as a semicolon-separated Newick string."""
        return "".join(graph.newick() for graph in self.forest)

    def get_ground_truth_params(self) -> Dict[str, Any]:
        """The resolved generator config plus L_median and seed, nested to
        mirror the config schema (simulator_spec.md section 1)."""

        def unit_to_dict(u: SwitchUnit) -> Dict[str, Any]:
            return {
                "signatures": list(u.signatures),
                "root_prob": u.root_prob,
                "lambda_on": u.lambda_on,
                "lambda_off": u.lambda_off,
            }

        return {
            "seed": self.seed,
            "repertoire": {
                "source": self.cfg.repertoire.source,
                "path": self.cfg.repertoire.path,
                "signatures": list(self.cfg.repertoire.signatures),
            },
            "switching": {
                "enabled": self.cfg.switching.enabled,
                "branch_length_scaling": self.cfg.switching.branch_length_scaling,
                "units": [unit_to_dict(u) for u in self.cfg.switching.units],
            },
            "levels": {
                "enabled": self.cfg.levels.enabled,
                "walk": "dirichlet",
                "concentration": self.cfg.levels.concentration,
                "branch_length_scaling": self.cfg.levels.branch_length_scaling,
                "concentration_floor": self.cfg.levels.concentration_floor,
                "root_concentration": self.cfg.levels.root_concentration,
                "activation_pseudocount": self.cfg.levels.activation_pseudocount,
            },
            "forest": {
                "n_trees": self.cfg.forest.n_trees,
                "nodes_per_tree": list(self.cfg.forest.nodes_per_tree),
                "depth": list(self.cfg.forest.depth),
                "max_attempts": self.cfg.forest.max_attempts,
                "branch_lengths": {
                    "distribution": self.cfg.forest.branch_lengths.distribution,
                    "params": dict(self.cfg.forest.branch_lengths.params),
                    "path": self.cfg.forest.branch_lengths.path,
                },
            },
            "burden": {
                "mean": self.cfg.burden.mean,
                "distribution": self.cfg.burden.distribution,
                "dispersion": self.cfg.burden.dispersion,
            },
            "counts": {
                "model": self.cfg.counts.model,
                "kappa": self.cfg.counts.kappa,
            },
            "l_median": self.l_median,
        }
