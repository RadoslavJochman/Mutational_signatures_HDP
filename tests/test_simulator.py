"""Tests for TreeSwitchDriftGenerator (src/models/switch_drift_generator.py).

Phase-one tests: constructed directly against hand-built config dicts, no
YAML and no generate_data.py (see simulator_spec.md section 14). Only the
import line changes when phase two merges this module into hdp_simulator.py.

Covers the nine invariants from simulator_spec.md section 14: (a) shapes;
(b) byte-for-byte seed determinism; (c) loader validation rules; (d)
renormalise-then-scale on all four edge kinds; (e) the concentration floor at
extreme branch length; (f) the rejection sampler's retry and hard failure;
(g) mean-preservation of the Dirichlet step; (h) activation preserving
continuing signatures' relative proportions; (i) off-signatures exactly zero
and agreeing with the active set.
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.switch_drift_generator import (
    BranchLengthConfig,
    ForestConfig,
    LevelsConfig,
    TreeSwitchDriftGenerator,
    carry_forward,
    dirichlet_step,
    effective_concentration,
    parse_generator_config,
    sample_forest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOGUE_PATH = str(REPO_ROOT / "COSMIC_sig" / "cosmic_signatures.csv")

RNG_SEED = 20240607


def _base_config(**overrides) -> dict:
    """A small, valid config: two clock signatures plus one switching pair.

    All three names (SBS1, SBS5, SBS36) exist in COSMIC_sig/cosmic_signatures.csv.
    """
    config = {
        "repertoire": {
            "source": "cosmic",
            "path": CATALOGUE_PATH,
            "signatures": ["SBS1", "SBS5", "SBS36"],
        },
        "switching": {
            "enabled": True,
            "branch_length_scaling": True,
            "units": [
                {
                    "signatures": ["SBS1"],
                    "root_prob": 1.0,
                    "lambda_on": 0.0,
                    "lambda_off": 0.0,
                },
                {
                    "signatures": ["SBS5"],
                    "root_prob": 1.0,
                    "lambda_on": 0.0,
                    "lambda_off": 0.0,
                },
                {
                    "signatures": ["SBS36"],
                    "root_prob": 0.3,
                    "lambda_on": 0.6,
                    "lambda_off": 0.3,
                },
            ],
        },
        "levels": {
            "enabled": True,
            "walk": "dirichlet",
            "concentration": 50.0,
            "branch_length_scaling": True,
            "concentration_floor": 1.0,
            "root_concentration": 1.0,
            "activation_pseudocount": 0.02,
        },
        "forest": {
            "n_trees": 4,
            "nodes_per_tree": [4, 12],
            "depth": [2, 5],
            "max_attempts": 200,
            "branch_lengths": {
                "distribution": "lognormal",
                "params": {"mu": 0.0, "sigma": 0.6},
            },
        },
        "burden": {"mean": 120, "distribution": "negbin", "dispersion": 2.0},
        "counts": {"model": "dirichlet_multinomial", "kappa": 50.0},
    }
    for key, value in overrides.items():
        config[key] = value
    return config


def _deep_set(config: dict, path: str, value) -> dict:
    """Return a deep-copied config with config[a][b][...] = value."""
    config = copy.deepcopy(config)
    *parents, leaf = path.split(".")
    node = config
    for p in parents:
        node = node[p]
    node[leaf] = value
    return config


# shapes


def test_shapes():
    gen = TreeSwitchDriftGenerator(_base_config(), seed=RNG_SEED)

    K = 3
    n_nodes = len(gen.get_true_activities())
    assert n_nodes > 0

    assert gen.get_true_signatures().shape == (K, 96)
    assert gen.get_true_activities().shape == (n_nodes, K)
    assert gen.get_true_active_sets().shape == (n_nodes, K)
    assert gen.get_mutation_count_matrix().shape[1] == 96
    assert gen.get_mutation_count_matrix().shape[0] <= n_nodes

    edges = gen.get_tree_edges()
    assert list(edges.columns) == ["tumour", "parent", "child", "length"]
    # one edge per non-root node
    n_roots = gen.get_true_activities().index.str.endswith("_1").sum()
    assert len(edges) == n_nodes - n_roots

    assert list(gen.get_true_activities().columns) == ["SBS1", "SBS5", "SBS36"]
    assert list(gen.get_true_active_sets().columns) == ["SBS1", "SBS5", "SBS36"]
    assert list(gen.get_true_signatures().index) == ["SBS1", "SBS5", "SBS36"]


# byte-for-byte seed determinism


def test_seed_determinism():
    config = _base_config()
    gen_a = TreeSwitchDriftGenerator(config, seed=RNG_SEED)
    gen_b = TreeSwitchDriftGenerator(config, seed=RNG_SEED)

    pd.testing.assert_frame_equal(
        gen_a.get_true_signatures(), gen_b.get_true_signatures()
    )
    pd.testing.assert_frame_equal(
        gen_a.get_true_activities(), gen_b.get_true_activities()
    )
    pd.testing.assert_frame_equal(
        gen_a.get_true_active_sets(), gen_b.get_true_active_sets()
    )
    pd.testing.assert_frame_equal(
        gen_a.get_mutation_count_matrix(), gen_b.get_mutation_count_matrix()
    )
    pd.testing.assert_frame_equal(gen_a.get_tree_edges(), gen_b.get_tree_edges())
    assert gen_a.get_newick_forest() == gen_b.get_newick_forest()
    assert gen_a.get_ground_truth_params() == gen_b.get_ground_truth_params()


# loader validation rules


def test_repertoire_duplicate_signatures_rejected():
    config = _deep_set(
        _base_config(), "repertoire.signatures", ["SBS1", "SBS1", "SBS5"]
    )
    with pytest.raises(ValueError, match="duplicate"):
        parse_generator_config(config)


def test_repertoire_missing_from_catalogue_rejected():
    config = _base_config()
    config["repertoire"]["signatures"] = ["SBS1", "SBS5", "NOT_A_SIGNATURE"]
    # units still reference SBS36 -> partition fails first unless we fix units too
    config["switching"]["units"] = [
        {"signatures": ["SBS1"], "root_prob": 1.0, "lambda_on": 0.0, "lambda_off": 0.0},
        {"signatures": ["SBS5"], "root_prob": 1.0, "lambda_on": 0.0, "lambda_off": 0.0},
        {
            "signatures": ["NOT_A_SIGNATURE"],
            "root_prob": 0.3,
            "lambda_on": 0.6,
            "lambda_off": 0.3,
        },
    ]
    with pytest.raises(ValueError, match="not found in catalogue"):
        TreeSwitchDriftGenerator(config, seed=RNG_SEED)


def test_switching_units_required_when_enabled():
    config = _deep_set(_base_config(), "switching.units", None)
    with pytest.raises(ValueError, match="units is required"):
        parse_generator_config(config)


def test_switching_units_missing_signature_rejected():
    config = copy.deepcopy(_base_config())
    config["switching"]["units"] = [
        {"signatures": ["SBS1"], "root_prob": 1.0, "lambda_on": 0.0, "lambda_off": 0.0},
        {"signatures": ["SBS5"], "root_prob": 1.0, "lambda_on": 0.0, "lambda_off": 0.0},
        # SBS36 omitted entirely
    ]
    with pytest.raises(ValueError, match="partition"):
        parse_generator_config(config)


def test_switching_units_overlap_rejected():
    config = copy.deepcopy(_base_config())
    config["switching"]["units"] = [
        {
            "signatures": ["SBS1", "SBS36"],
            "root_prob": 1.0,
            "lambda_on": 0.0,
            "lambda_off": 0.0,
        },
        {"signatures": ["SBS5"], "root_prob": 1.0, "lambda_on": 0.0, "lambda_off": 0.0},
        {
            "signatures": ["SBS36"],
            "root_prob": 0.3,
            "lambda_on": 0.6,
            "lambda_off": 0.3,
        },
    ]
    with pytest.raises(ValueError, match="overlap"):
        parse_generator_config(config)


def test_switching_root_prob_out_of_range_rejected():
    config = copy.deepcopy(_base_config())
    config["switching"]["units"][2]["root_prob"] = 1.5
    with pytest.raises(ValueError, match="root_prob"):
        parse_generator_config(config)


def test_switching_requires_always_on_unit():
    config = copy.deepcopy(_base_config())
    for unit in config["switching"]["units"]:
        unit["root_prob"] = 0.5
        unit["lambda_off"] = 0.1
    with pytest.raises(ValueError, match="always-on"):
        parse_generator_config(config)


def test_switching_rates_scaling_true_allows_rate_above_one():
    config = copy.deepcopy(_base_config())
    config["switching"]["units"][2]["lambda_on"] = 5.0  # a rate, fine when scaling
    parse_generator_config(config)  # should not raise


def test_switching_rates_scaling_true_rejects_negative():
    config = copy.deepcopy(_base_config())
    config["switching"]["units"][2]["lambda_on"] = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        parse_generator_config(config)


def test_switching_rates_scaling_false_requires_probability_bounds():
    config = copy.deepcopy(_base_config())
    config["switching"]["branch_length_scaling"] = False
    config["switching"]["units"][2]["lambda_on"] = 1.5  # not a valid probability
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        parse_generator_config(config)


def test_switching_rates_scaling_false_accepts_probability_bounds():
    config = copy.deepcopy(_base_config())
    config["switching"]["branch_length_scaling"] = False
    config["switching"]["units"][2]["lambda_on"] = 0.5
    config["switching"]["units"][2]["lambda_off"] = 0.5
    parse_generator_config(config)  # should not raise


def test_levels_concentration_floor_above_concentration_rejected_when_scaling():
    config = _deep_set(_base_config(), "levels.concentration_floor", 100.0)
    with pytest.raises(ValueError, match="concentration_floor"):
        parse_generator_config(config)


def test_levels_concentration_floor_above_concentration_ok_when_not_scaling():
    config = _deep_set(_base_config(), "levels.concentration_floor", 100.0)
    config = _deep_set(config, "levels.branch_length_scaling", False)
    parse_generator_config(config)  # inert in this mode, should not raise


def test_levels_activation_pseudocount_must_be_positive():
    config = _deep_set(_base_config(), "levels.activation_pseudocount", 0.0)
    with pytest.raises(ValueError, match="activation_pseudocount"):
        parse_generator_config(config)


def test_forest_nodes_per_tree_min_above_max_rejected():
    config = _deep_set(_base_config(), "forest.nodes_per_tree", [10, 5])
    with pytest.raises(ValueError, match="nodes_per_tree"):
        parse_generator_config(config)


def test_forest_branch_length_distribution_invalid_rejected():
    config = _deep_set(
        _base_config(), "forest.branch_lengths", {"distribution": "uniform"}
    )
    with pytest.raises(ValueError, match="lognormal"):
        parse_generator_config(config)


def test_counts_dirichlet_multinomial_requires_kappa():
    config = _deep_set(_base_config(), "counts", {"model": "dirichlet_multinomial"})
    with pytest.raises(ValueError, match="kappa"):
        parse_generator_config(config)


def test_counts_multinomial_does_not_require_kappa():
    config = _deep_set(_base_config(), "counts", {"model": "multinomial"})
    parse_generator_config(config)  # should not raise


# renormalise-then-scale on all four edge kinds


def test_carry_forward_neither_activation_nor_deactivation():
    e_parent = np.array([0.6, 0.4, 0.0])
    a_child = np.array([True, True, False])
    base = carry_forward(e_parent, a_child, activation_pseudocount=0.02)
    np.testing.assert_allclose(base, [0.6, 0.4, 0.0])  # already sums to 1


def test_carry_forward_activation_only():
    e_parent = np.array([0.6, 0.4, 0.0])
    a_child = np.array([True, True, True])  # signature 2 newly on
    base = carry_forward(e_parent, a_child, activation_pseudocount=0.02)
    expected = np.array([0.6, 0.4, 0.02])
    expected = expected / expected.sum()
    np.testing.assert_allclose(base, expected)
    assert base.sum() == pytest.approx(1.0)


def test_carry_forward_deactivation_only():
    e_parent = np.array([0.5, 0.3, 0.2])
    a_child = np.array([True, True, False])  # signature 2 switches off
    base = carry_forward(e_parent, a_child, activation_pseudocount=0.02)
    expected = np.array([0.5, 0.3, 0.0])
    expected = expected / expected.sum()
    np.testing.assert_allclose(base, expected)
    assert base[2] == 0.0


def test_carry_forward_both_activation_and_deactivation():
    e_parent = np.array([0.5, 0.5, 0.0, 0.0])
    a_child = np.array([True, False, True, False])  # sig 1 off, sig 2 newly on
    base = carry_forward(e_parent, a_child, activation_pseudocount=0.02)
    expected = np.array([0.5, 0.0, 0.02, 0.0])
    expected = expected / expected.sum()
    np.testing.assert_allclose(base, expected)
    assert base[1] == 0.0
    assert base[3] == 0.0


# concentration floor at extreme branch length


def test_concentration_floor_at_extreme_length():
    levels_cfg = LevelsConfig(
        enabled=True,
        concentration=50.0,
        branch_length_scaling=True,
        concentration_floor=1.0,
        root_concentration=1.0,
        activation_pseudocount=0.02,
    )
    # an enormous branch length would otherwise push concentration to ~0
    conc = effective_concentration(levels_cfg, length=1e9, l_median=1.0)
    assert conc == pytest.approx(levels_cfg.concentration_floor)


def test_concentration_not_floored_at_typical_length():
    levels_cfg = LevelsConfig(
        enabled=True,
        concentration=50.0,
        branch_length_scaling=True,
        concentration_floor=1.0,
        root_concentration=1.0,
        activation_pseudocount=0.02,
    )
    conc = effective_concentration(levels_cfg, length=1.0, l_median=1.0)
    assert conc == pytest.approx(50.0)


def test_concentration_constant_when_scaling_disabled():
    levels_cfg = LevelsConfig(
        enabled=True,
        concentration=50.0,
        branch_length_scaling=False,
        concentration_floor=1.0,
        root_concentration=1.0,
        activation_pseudocount=0.02,
    )
    for length in [0.01, 1.0, 1e6]:
        conc = effective_concentration(levels_cfg, length=length, l_median=1.0)
        assert conc == pytest.approx(50.0)


# rejection sampler: retry and hard failure


def test_rejection_sampler_retries_and_succeeds():
    """A tight-but-reachable depth window fails on attempt 1 but succeeds
    given enough attempts, at the same seed -- proof the sampler actually
    retries rather than only trying once."""
    branch_cfg = BranchLengthConfig(
        distribution="lognormal", params={"mu": 0.0, "sigma": 0.6}
    )
    tight_cfg = ForestConfig(
        n_trees=1,
        nodes_per_tree=(4, 20),
        depth=(4, 4),
        max_attempts=1,
        branch_lengths=branch_cfg,
    )
    with pytest.raises(RuntimeError):
        sample_forest(tight_cfg, np.random.default_rng(0))

    loose_cfg = ForestConfig(
        n_trees=1,
        nodes_per_tree=(4, 20),
        depth=(4, 4),
        max_attempts=50,
        branch_lengths=branch_cfg,
    )
    trees = sample_forest(loose_cfg, np.random.default_rng(0))
    assert len(trees) == 1


def test_rejection_sampler_hard_failure_names_ranges_and_attempts():
    branch_cfg = BranchLengthConfig(
        distribution="lognormal", params={"mu": 0.0, "sigma": 0.6}
    )
    # 2-3 nodes is below the minimum achievable network size (4 nodes, 2 leaves).
    impossible_cfg = ForestConfig(
        n_trees=1,
        nodes_per_tree=(2, 3),
        depth=(1, 10),
        max_attempts=5,
        branch_lengths=branch_cfg,
    )
    with pytest.raises(
        RuntimeError, match=r"nodes_per_tree.*\[2, 3\].*depth.*5 attempts"
    ):
        sample_forest(impossible_cfg, np.random.default_rng(0))


# mean-preservation of the Dirichlet step


def test_dirichlet_step_mean_preserving():
    levels_cfg = LevelsConfig(
        enabled=True,
        concentration=50.0,
        branch_length_scaling=True,
        concentration_floor=1.0,
        root_concentration=1.0,
        activation_pseudocount=0.02,
    )
    base = np.array([0.5, 0.3, 0.2])
    rng = np.random.default_rng(RNG_SEED)
    n_draws = 20000
    draws = np.array(
        [dirichlet_step(base, 20.0, levels_cfg, rng) for _ in range(n_draws)]
    )
    np.testing.assert_allclose(draws.mean(axis=0), base, atol=0.02)


def test_dirichlet_step_disabled_returns_base_unchanged():
    levels_cfg = LevelsConfig(
        enabled=False,
        concentration=50.0,
        branch_length_scaling=True,
        concentration_floor=1.0,
        root_concentration=1.0,
        activation_pseudocount=0.02,
    )
    base = np.array([0.5, 0.3, 0.2])
    rng = np.random.default_rng(RNG_SEED)
    e = dirichlet_step(base, 20.0, levels_cfg, rng)
    np.testing.assert_array_equal(e, base)


# activation preserves continuing signatures' relative proportions


def test_activation_preserves_continuing_ratios():
    e_parent = np.array([0.6, 0.4, 0.0])  # sig 0:1 ratio is 1.5
    a_child = np.array([True, True, True])  # sig 2 newly activates
    base = carry_forward(e_parent, a_child, activation_pseudocount=0.05)
    assert base[0] / base[1] == pytest.approx(e_parent[0] / e_parent[1])


def test_activation_preserves_ratios_with_three_continuing_signatures():
    e_parent = np.array([0.2, 0.5, 0.3, 0.0])
    a_child = np.array([True, True, True, True])
    base = carry_forward(e_parent, a_child, activation_pseudocount=0.1)
    ratios_before = e_parent[:3] / e_parent[0]
    ratios_after = base[:3] / base[0]
    np.testing.assert_allclose(ratios_after, ratios_before)


def test_activation_then_dirichlet_step_preserves_ratios_in_expectation():
    """carry_forward -> renormalise -> dirichlet_step, composed across an
    activation edge, preserves the continuing signatures' relative
    proportions in expectation, not just in the deterministic base. Closes
    the gap between test_dirichlet_step_mean_preserving (drift only, no
    activation, no carry_forward) and test_activation_preserves_continuing_
    ratios (carry_forward only, no draw)."""
    e_parent = np.array([0.6, 0.3, 0.0])  # sig 0:1 ratio is 2.0, sig 2 off
    a_child = np.array([True, True, True])  # sig 2 newly activates
    base = carry_forward(e_parent, a_child, activation_pseudocount=0.05)

    levels_cfg = LevelsConfig(
        enabled=True,
        concentration=50.0,
        branch_length_scaling=True,
        concentration_floor=1.0,
        root_concentration=1.0,
        activation_pseudocount=0.05,
    )
    rng = np.random.default_rng(RNG_SEED)
    n_draws = 20000
    draws = np.array(
        [dirichlet_step(base, 20.0, levels_cfg, rng) for _ in range(n_draws)]
    )
    mean_e = draws.mean(axis=0)
    np.testing.assert_allclose(mean_e, base, atol=0.02)

    parent_ratio = e_parent[0] / e_parent[1]
    sampled_ratio = mean_e[0] / mean_e[1]
    assert sampled_ratio == pytest.approx(parent_ratio, rel=0.05)


# off-signatures exactly zero and agree with the active set


def test_off_signatures_exactly_zero_and_match_active_set():
    gen = TreeSwitchDriftGenerator(_base_config(), seed=RNG_SEED)
    activities = gen.get_true_activities().to_numpy()
    active_sets = gen.get_true_active_sets().to_numpy()

    assert np.array_equal(activities == 0.0, active_sets == 0)
    assert np.all(activities[active_sets == 1] > 0.0)
    assert np.all(activities[active_sets == 0] == 0.0)
    # every row is a valid simplex point
    np.testing.assert_allclose(activities.sum(axis=1), 1.0)


def test_off_signatures_hold_with_levels_disabled():
    config = _deep_set(_base_config(), "levels.enabled", False)
    gen = TreeSwitchDriftGenerator(config, seed=RNG_SEED)
    activities = gen.get_true_activities().to_numpy()
    active_sets = gen.get_true_active_sets().to_numpy()
    assert np.array_equal(activities == 0.0, active_sets == 0)
