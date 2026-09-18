"""Unit tests for scripts/scaling_metrics.py's convergence variable selection.

_convergence_vars is inclusion-based (matches activity_var_<d> and any
variable name containing 'sigma'), so a new variable added to the model is
silently in or out depending on its name -- not a decision anyone made. These
tests pin the deliberate choice made when TreeHDP's mu_level was added: it is
a single forest-pooled baseline every root deviates from, genuinely
identifiable (unlike the non-centred eta_level/z_level/z_root increments),
and so is monitored for convergence alongside sigma.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import arviz as az  # noqa: E402
import numpy as np  # noqa: E402
from scaling_metrics import (  # noqa: E402
    _convergence_vars,
    _drop_constant,
    convergence_row,
)


class _FakePosterior:
    """Stand-in for an xarray posterior: _convergence_vars only reads .data_vars."""

    def __init__(self, names):
        self.data_vars = names


def test_convergence_vars_fixed_sig():
    """Fixed-sig trace: e_level_* and sigma are kept; eta/z-level excluded,
    mu_level kept."""
    post = _FakePosterior(
        [
            "e_level_0",
            "e_level_1",
            "eta_level_0",
            "eta_level_1",
            "z_level_1",
            "sigma",
            "mu_level",
        ]
    )
    kept = _convergence_vars(post, "e_level", None)
    assert set(kept) == {"e_level_0", "e_level_1", "sigma", "mu_level"}


def test_convergence_vars_denovo():
    """De novo trace: same rule; z_root_* and the latent signatures matrix
    are excluded, mu_level is kept."""
    post = _FakePosterior(
        [
            "e_level_0",
            "e_level_1",
            "e_level_2",
            "eta_level_0",
            "eta_level_1",
            "eta_level_2",
            "z_root_0",
            "z_level_1",
            "z_level_2",
            "sigma",
            "mu_level",
            "signatures",
        ]
    )
    kept = _convergence_vars(post, "e_level", None)
    assert set(kept) == {"e_level_0", "e_level_1", "e_level_2", "sigma", "mu_level"}


def test_convergence_vars_override():
    """An explicit --conv-vars list bypasses the name-matching entirely."""
    post = _FakePosterior(["e_level_0", "sigma", "mu_level"])
    kept = _convergence_vars(post, "e_level", ["mu_level"])
    assert kept == ["mu_level"]


def test_convergence_vars_switch_model():
    """Switch-model trace: lambda_on/lambda_off/pi_root are kept;
    a_prob_level_* (bounded, can be constant) is excluded."""
    post = _FakePosterior(
        [
            "e_level_0",
            "e_level_1",
            "a_prob_level_0",
            "a_prob_level_1",
            "eta_level_0",
            "z_root_0",
            "sigma",
            "mu_level",
            "lambda_on",
            "lambda_off",
            "pi_root",
        ]
    )
    kept = _convergence_vars(post, "e_level", None)
    assert set(kept) == {
        "e_level_0",
        "e_level_1",
        "sigma",
        "mu_level",
        "lambda_on",
        "lambda_off",
        "pi_root",
    }


def _fake_idata(seed=0):
    rng = np.random.default_rng(seed)
    chains, draws = 2, 40
    return az.from_dict(
        posterior={
            "sigma": rng.lognormal(size=(chains, draws)),
            "e_level_0": rng.dirichlet(np.ones(3), size=(chains, draws, 2)),
            # exactly constant, as a_prob can be for an always-on signature
            "a_prob_level_0": np.ones((chains, draws, 2, 3)),
            # one constant element inside an otherwise varying variable
            "lambda_on": np.concatenate(
                [rng.lognormal(size=(chains, draws, 2)), np.ones((chains, draws, 1))],
                axis=-1,
            ),
        }
    )


def test_drop_constant_removes_only_fully_constant_variables():
    idata = _fake_idata()
    kept, dropped = _drop_constant(
        idata.posterior, ["sigma", "e_level_0", "a_prob_level_0", "lambda_on"]
    )
    assert dropped == ["a_prob_level_0"]
    assert kept == ["sigma", "e_level_0", "lambda_on"]


def test_convergence_row_is_finite_despite_constants():
    """A fully constant variable and a constant element must not turn
    max_rhat / min_ess into NaN."""
    idata = _fake_idata()
    row = convergence_row(idata, ["sigma", "e_level_0", "a_prob_level_0", "lambda_on"])
    assert np.isfinite(row["max_rhat"])
    assert np.isfinite(row["min_ess"])
    assert row["min_ess"] > 0
