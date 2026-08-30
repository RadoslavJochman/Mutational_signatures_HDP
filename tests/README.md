# Tests

pytest suite for the library and pipeline. New features ship with their tests here (see the
Testing section of CLAUDE.md).

## Layout

Mirror `src/` and the scripts under test, for example:

- `test_analysis_metrics.py` for the metrics in `src/analysis/analysis.py`
- `test_walk_transforms.py` for the softmax and walk round-trips
- `test_alignment.py` for `chain_perms_to_true`
- `test_simulator.py` for `TreeSignatureGenerator`
- `test_smoke.py` for the tiny end-to-end generate, infer, score run

## Running

- Fast gate (default, excludes slow tests): `pytest`
- Slow statistical tests only: `pytest -m slow`
- Everything: `pytest -m ""`

Unit tests are deterministic: seed the simulator and pass `random_seed` to `pm.sample`. Assert
on shapes, bounds, invariants, and generous tolerances, never exact posterior numbers.
