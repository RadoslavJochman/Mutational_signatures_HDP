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

import os
import datetime
from pathlib import Path
from typing import Any, Dict
import pymc as pm
import yaml
from fractions import Fraction
from functools import partial

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
    base_dir: str | Path = "results",
    experiment_name: str = "experiment"
) -> Path:
    """
    Create and return a dated experiment output directory.

    The directory name is ``<experiment_name>``, If not provided,
    the directory is named ``experiment``.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    base_dir : str or Path
        Root results directory (default: ``results/``).

    Returns
    -------
    Path
        Absolute path to the created directory.
    """
    out_dir = Path(base_dir) / f"{experiment_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def get_prior(config: dict, prior_name: str, dim: int=None):
    """
        Retrieves and instantiates a PyMC prior distribution based on a configuration dictionary.

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
        }
    dist_type = config.get(prior_name)
    if dist_type not in priors:
        raise KeyError(f"'{dist_type}' prior not found.")

    raw_params = config[f"{prior_name}_parm"]
    parsed_params = {}

    for param_name, param_value in raw_params.items():
        base_val = float(Fraction(str(param_value)))

        if dim != 1:
            parsed_params[param_name] = [base_val] * int(dim)
        else:
            parsed_params[param_name] = base_val

    return partial(priors[dist_type], **parsed_params)