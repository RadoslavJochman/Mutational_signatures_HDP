"""Shared pytest fixtures. Feature tests live alongside in tests/, mirroring src/."""

import pytest

RNG_SEED = 20240607  # fixed seed for deterministic unit and smoke tests


@pytest.fixture
def tmp_workdir(tmp_path, monkeypatch):
    """Run a test in an isolated working directory.

    The pipeline writes data and results relative to the current directory, so
    tests that exercise generate, infer, or score should run inside this fixture
    to avoid touching the repo's data/ and results/.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path
