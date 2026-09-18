# Tests

pytest suite for the library and pipeline. New features ship with their tests here (see the
Testing section of CLAUDE.md).

## Layout

Mirror `src/` and the scripts under test, for example:

- `test_analysis_metrics.py` for the metrics in `src/analysis/analysis.py`
- `test_walk_transforms.py` for the softmax and walk round-trips
- `test_alignment.py` for `chain_perms_to_true`
- `test_simulator.py` for `TreeSwitchDriftGenerator` (`src/models/hdp_simulator.py`)
- `test_smoke.py` for the tiny end-to-end generate, infer, score run
- `test_switch_pruning.py` for `src/models/switch_pruning.py`: the brute-force oracle over
  every joint on/off assignment of two small forests (the correctness anchor), the Kronecker
  checks of the axis-wise contraction, the stable log-sum-exp, `always_on` against the exact
  limit of the full grid, and the `tree_coupled: false` independence check
- `test_inference_switch.py` for `TreeHDP(..., switching=...)`: the bridge test (everything
  forced on equals the plain multinomial), finite logp/dlogp in both backends, the
  signature-axis registry, validation errors, and `walk_branch_length_scaling`
- `test_align_trace.py` for the registry-driven de novo alignment in `run_inference.py`
- `test_scaling_metrics.py` for the convergence variable set and its NaN guard
- `test_switch_posterior.py` for `src/analysis/switch_posterior.py`: FFBS marginals and
  edge events against the oracle, `always_on` and `tree_coupled` in the sampler
- `test_switch_recovery.py` for `node_variable_rows`, ECE and reliability bins, and
  `scripts/switch_recovery.py`'s metric functions
- `test_smoke_switch.py` for the switch model end to end in both modes, including the
  Rao-Blackwellised versus sampled cross-check
- `test_switch_convergence.py` (slow) for the switch model's statistical gate: an easy
  hand-built forest, 2 chains x 500 draws, r_hat, ESS, divergences and node-level on/off
  accuracy

## Running

- Fast gate (default, excludes slow tests): `pytest`
- Slow statistical tests only: `pytest -m slow`
- Everything: `pytest -m ""`

Unit tests are deterministic: seed the simulator and pass `random_seed` to `pm.sample`. Assert
on shapes, bounds, invariants, and generous tolerances, never exact posterior numbers.
