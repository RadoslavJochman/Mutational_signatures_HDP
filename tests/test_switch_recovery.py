"""Unit tests for the switch-recovery scoring layer: the calibration helpers
in src/analysis/analysis.py, node_variable_rows, and scripts/switch_recovery.py's
metric functions (switch_model_plan.md section 6.3)."""

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from switch_recovery import (  # noqa: E402
    binary_metrics,
    score_nodes,
    stratified_accuracy,
)

from src.analysis.analysis import (  # noqa: E402
    build_forest,
    expected_calibration_error,
    node_variable_rows,
    nodes_by_depth,
    reliability_bins,
)

# ---------------------------------------------------------------------------
# ECE and reliability bins
# ---------------------------------------------------------------------------


def _pairs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(int)
    return rng, y


def test_ece_perfect_predictions_is_zero():
    _, y = _pairs()
    assert expected_calibration_error(y.astype(float), y) == 0.0


def test_ece_inverted_predictions_is_one():
    _, y = _pairs()
    assert expected_calibration_error(1.0 - y, y) == pytest.approx(1.0)


def test_ece_constant_prediction_equals_prevalence_gap():
    _, y = _pairs()
    p = np.full_like(y, 0.5, dtype=float)
    assert expected_calibration_error(p, y) == pytest.approx(abs(y.mean() - 0.5))
    assert expected_calibration_error(p, np.ones_like(y)) == pytest.approx(0.5)


def test_ece_empty_is_nan():
    assert np.isnan(expected_calibration_error(np.array([]), np.array([])))


def test_reliability_bins_shape_counts_and_edges():
    rng, y = _pairs()
    p = rng.random(y.size)
    p[0], p[1] = 0.0, 1.0
    tbl = reliability_bins(p, y, n_bins=10)
    assert list(tbl.columns) == ["bin", "lo", "hi", "n", "mean_pred", "obs_freq"]
    assert len(tbl) == 10
    assert tbl["n"].sum() == y.size
    assert tbl.loc[9, "n"] >= 1  # p = 1 lands in the last bin, not an 11th
    for _, r in tbl[tbl["n"] > 0].iterrows():
        assert r["lo"] <= r["mean_pred"] <= r["hi"] + 1e-12


# ---------------------------------------------------------------------------
# node_variable_rows
# ---------------------------------------------------------------------------

NEWICK = "((C:0.5,D:0.7)B:0.3,E:1.0)A:0.0;"


def test_node_variable_rows_aligns_and_maps_labels():
    dr = nodes_by_depth(build_forest(NEWICK))
    by_depth = {}
    for (d, r), label in dr.items():
        by_depth.setdefault(d, {})[r] = label
    K, chains, draws = 3, 2, 5
    rng = np.random.default_rng(1)
    perm = np.array([2, 0, 1])
    post = {}
    truth = {}
    for d, rows in by_depth.items():
        n = len(rows)
        base = rng.random((draws, n, K))
        # chain 1 carries the same rows permuted along K
        post[f"a_prob_level_{d}"] = np.stack([base, base[..., perm]])
        for r in range(n):
            truth[rows[r]] = base[:, r, :].mean(axis=0)
    post["sigma"] = rng.random((chains, draws))
    idata = az.from_dict(posterior=post)

    perms = np.stack([np.arange(K), np.argsort(perm)])
    labels, A = node_variable_rows(idata.posterior, "a_prob_level", NEWICK, perms)
    assert set(labels) == {"A", "B", "C", "D", "E"}
    assert A.shape == (5, chains, K)
    for i, label in enumerate(labels):
        np.testing.assert_allclose(A[i, 0], truth[label])
        np.testing.assert_allclose(A[i, 1], truth[label])

    labels_kept, A_kept = node_variable_rows(
        idata.posterior, "a_prob_level", NEWICK, perms, keep={"A", "C"}
    )
    assert set(labels_kept) == {"A", "C"}
    assert A_kept.shape == (2, chains, K)


# ---------------------------------------------------------------------------
# binary_metrics / stratified_accuracy / score_nodes
# ---------------------------------------------------------------------------


def test_binary_metrics_perfect():
    y = np.array([0, 1, 1, 0, 1])
    m = binary_metrics(y.astype(float), y)
    for k in ("auroc", "auprc", "precision", "recall", "f1", "accuracy"):
        assert m[k] == pytest.approx(1.0), k
    assert m["brier"] == pytest.approx(0.0)
    assert m["ece"] == pytest.approx(0.0)


def test_binary_metrics_one_class_gives_nan_not_exception():
    y = np.ones(6, dtype=int)
    p = np.array([0.9, 0.8, 0.95, 0.7, 0.99, 0.6])
    m = binary_metrics(p, y)
    assert np.isnan(m["auroc"]) and np.isnan(m["auprc"])
    assert m["accuracy"] == pytest.approx(1.0)
    assert np.isfinite(m["brier"]) and np.isfinite(m["ece"])
    # all-off truth with all-off predictions: precision undefined -> NaN
    m0 = binary_metrics(np.zeros(4), np.zeros(4, dtype=int))
    assert np.isnan(m0["precision"])
    assert m0["accuracy"] == 1.0


def test_stratified_accuracy_bins():
    level = np.array([0.0, 0.0, 0.03, 0.05, 0.2, 0.9])
    y = (level > 0).astype(int)
    p = np.array([0.1, 0.9, 0.9, 0.1, 0.9, 0.9])  # one error in level0, one in low
    acc = stratified_accuracy(p, y, level)
    assert acc["acc_level0"] == pytest.approx(0.5)
    assert acc["acc_level_low"] == pytest.approx(0.5)
    assert acc["acc_level_high"] == pytest.approx(1.0)
    acc_empty = stratified_accuracy(p[:2], y[:2], level[:2])
    assert np.isnan(acc_empty["acc_level_high"])


def test_score_nodes_rows_and_always_on_signature_is_nan():
    rng = np.random.default_rng(2)
    n_nodes, n_chains, K = 12, 2, 3
    Y = (rng.random((n_nodes, K)) < 0.5).astype(int)
    Y[:, 0] = 1  # signature 0 on everywhere in truth
    L = Y * rng.uniform(0.01, 0.5, size=(n_nodes, K))
    P = np.clip(
        Y[:, None, :] * 0.8 + rng.normal(scale=0.1, size=(n_nodes, n_chains, K)), 0, 1
    )
    df = score_nodes(P, Y, L, ["S0", "S1", "S2"])
    assert list(df["group"]) == ["overall", "S0", "S1", "S2"]
    s0 = df[df["group"] == "S0"].iloc[0]
    assert np.isnan(s0["auroc_mean"]) and np.isnan(s0["auprc_mean"])
    assert np.isfinite(s0["accuracy_mean"])
    overall = df[df["group"] == "overall"].iloc[0]
    assert overall["auroc_best"] >= overall["auroc_mean"] >= overall["auroc_worst"]
    assert overall["brier_best"] <= overall["brier_mean"] <= overall["brier_worst"]
    assert {"acc_level0_mean", "acc_level_low_mean", "acc_level_high_mean"} <= set(
        df.columns
    )
