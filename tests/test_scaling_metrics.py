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

from scaling_metrics import _convergence_vars  # noqa: E402


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
