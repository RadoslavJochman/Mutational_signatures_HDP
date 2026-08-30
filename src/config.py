"""
config.py

Utilities for loading YAML experiment configs and resolving output paths.

Usage
-----
    from src.config import load_config, make_output_dir

    cfg = load_config("configs/fixed_sig_experiment.yaml")
    out_dir = make_output_dir(cfg)   # e.g. results/fixed_sig_experiment_2025-04-19/
"""

from __future__ import annotations

from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Any, Dict

import pymc as pm
import pytensor.tensor as pt
import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML config file and return it as a plain dict.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open() as f:
        return yaml.safe_load(f)


def make_output_dir(
    base_dir: str | Path = "experiments",
    experiment_name: str = "experiment",
    subdir: str | None = None,
) -> Path:
    """
    Create and return an experiment's output directory, or one of its
    subdirectories.

    The directory is ``<base_dir>/<experiment_name>``, or
    ``<base_dir>/<experiment_name>/<subdir>`` when `subdir` is given.
    `experiment_name` may itself contain `/` (a sweep replicate's name, e.g.
    ``"corr_sweep/rep03"``), which joins as extra path segments the same way.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment (config `experiment_name`).
    base_dir : str or Path
        Root experiments directory (config `experiment_root`).
    subdir : str, optional
        `data`, `results`, or `plots` -- the one caller-specific
        subdirectory each script owns within the experiment directory.

    Returns
    -------
    Path
        Absolute path to the created directory.
    """
    out_dir = Path(base_dir) / f"{experiment_name}"
    if subdir is not None:
        out_dir = out_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_prior(config: dict, prior_name: str, dim: int = None):
    """
    Retrieves and instantiates a PyMC prior distribution based on a configuration
    dictionary.

    Args:
        config (dict): A configuration dictionary containing the distribution name
            and its required parameters.
        prior_name (str): The key used to look up the desired distribution name
            within the `config` dictionary.
        dim(str): The dimensionality of the distribution.
    Returns:
        Returns a partial pm.Distribution configured with the provided parameters
        that requires only a 'name' to be instantiated.

    Raises:
        KeyError: If the distribution specified in `config[prior_name]` is not
            found in the supported `priors` dictionary.
    """
    priors = {
        "Dir": pm.Dirichlet,
        "Norm": pm.Normal,
        "LogNorm": pm.LogNormal,
        "Exp": pm.Exponential,
        "Beta": pm.Beta,
        "Gamma": pm.Gamma,
        "Fixed": pm.Deterministic,
    }
    dist_type = config.get(prior_name)
    if dist_type not in priors:
        raise KeyError(f"'{dist_type}' prior not found.")

    raw_params = config[f"{prior_name}_parm"]
    parsed_params = {}

    for param_name, param_value in raw_params.items():
        base_val = float(Fraction(str(param_value)))

        if dim != 1:
            val = [base_val] * int(dim)
        else:
            val = base_val
        parsed_params[param_name] = pt.as_tensor_variable(val)

    return partial(priors[dist_type], **parsed_params)
