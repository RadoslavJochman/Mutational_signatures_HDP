"""First real-data TreeHDP runs on slice D: fixed-signature mode
(treehdp-v1, S known, the shared ILR walk), on the real snv/cna/consensus
trees and the real spectra, comparing settings and trees.

Not part of the generate-infer-score pipeline (that is config-driven,
`experiment_root`/`experiment_name`); this is a one-off real-data run
against fixed local files, with a fixed CLI instead. Trees are loaded
through TreeHDP's own Newick loader (constructing the model does the
parsing, exactly as check_tree_inputs.py already verified); spectra are
loaded only through `read_data_matrix` (src/analysis/analysis.py), never a
bare `pd.read_csv`, for the reason its own docstring gives.

Seeding, one deliberate departure from `_BaseTreeHDP.sample()`
--------------------------------------------------------------
`model.sample()` never forwards a `random_seed` to `pm.sample` (it always
passes `random_seed=None`, i.e. OS entropy), so a fixed, reproducible seed
is not reachable through that method as it stands. Rather than edit the
model file (out of scope here, and the task this script serves must NOT
edit it), `build_and_sample` below builds the model exactly as
`model.sample()` would (same `nuts_sampler="numpyro"`, same
`nuts={"max_tree_depth": ...}`) but calls `pm.sample` directly with an
explicit `random_seed`, then sets `model.trace` itself so every other
`_BaseTreeHDP` accessor (`get_posterior_mean`, `get_node_activity_posterior`)
still works unchanged. Confirmed reproducible: two runs at the same seed
give bit-identical posterior arrays.

Settings
--------
    main          spectra (a) (all confidently present SNVs), signatures
                  SBS1, SBS5, SBS95 -- SBS95 is a deliberate artefact
                  sink (refit_cosmic.py's forward selection picked it as
                  the full catalogue's best single addition to SBS1+SBS5,
                  and it is collinear with the candidate-set SBS18 at
                  cosine 0.900 -- see refit_cosmic.py's collinearity
                  report).
    sensitivity   spectra (c) (VAF >= 0.15 and alt reads >= 3, the
                  stricter confidence subset), signatures SBS1, SBS5 only.
    full          the full triple-negative breast repertoire
                  (FULL_REPERTOIRE: SBS1, SBS2, SBS3, SBS5, SBS8, SBS13,
                  SBS17a, SBS17b, SBS18, SBS40a), on both spectra (a) and
                  (c). No SBS95: it is near-identical to SBS18 (cosine
                  0.900, refit_cosmic.py's collinearity report), and the
                  two together would be unidentifiable.
    baseline      the independent/no-tree comparator used in the
                  tree-vs-pooling ablation -- checked for at import time
                  (BASELINE_AVAILABLE); as of this run the codebase does
                  not have one yet (CLAUDE.md's "Direction of travel"
                  lists it as still to be built), so every baseline
                  function here is a no-op that says so rather than
                  building a new comparator.

90% interval means the equal-tailed 5th/95th percentile interval of the
flattened (chains x draws) posterior samples, NOT arviz's KDE-based HDI --
simpler and fully reproducible from raw samples.

Outputs (to --outdir, one directory per (setting, tree))
    trace.nc                 the full InferenceData
    activity_summary.csv     node, kind (observed/latent), signature,
                              mean, p5, p95 -- every graph node, not just
                              observed ones
    convergence.csv          max_rhat, min_ess_bulk, min_ess_tail,
                              n_divergences, runtime_seconds -- over
                              e_level_* and sigma only (CLAUDE.md:
                              eta_level_*/z_level_*/z_root_* excluded)
    posterior_predictive.csv cluster, cosine_similarity (observed
                              spectrum vs posterior-mean reconstruction)
    run_info.json             tree, signatures, seed and sampler settings,
                              for the record

Usage (one run)
---------------
    PYTENSOR_FLAGS='cxx=' python realdata/scripts/run_treehdp_realdata.py \\
        --tree snv --spectra realdata/local_tree_input/spectra.csv \\
        --signatures SBS1 SBS5 SBS95 \\
        --outdir realdata/local_tree_input/treehdp_runs/main_snv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "realdata" / "scripts"))

import arviz as az  # noqa: E402
import pymc as pm  # noqa: E402
from check_tree_inputs import CONSENSUS_NEWICK  # noqa: E402

from src.analysis.analysis import DEFAULT_PRIORS, cosine, read_data_matrix  # noqa: E402
from src.models.hdp_inference import TreeHDP  # noqa: E402

# Project-standard NUTS settings for a fixed-signature run (see
# configs/config_fixed_realistic_test.yaml, fixed-sig-v2 tag; the same
# priors, draws, tune, chains, cores and target_accept a real fixed-sig
# run has used before, now on the shared ILR walk instead of the retired
# anchored softmax).
DRAWS = 1000
TUNE = 1000
CHAINS = 4
CORES = 4
TARGET_ACCEPT = 0.9
MAX_TREEDEPTH = 10
SEED = 20260927

TREE_NAMES = ("snv", "cna", "consensus")

# The full triple-negative breast repertoire (the "full" setting). No
# SBS95: collinear with SBS18 at cosine 0.900 (refit_cosmic.py's
# collinearity report), so the two together would be unidentifiable.
FULL_REPERTOIRE = [
    "SBS1",
    "SBS2",
    "SBS3",
    "SBS5",
    "SBS8",
    "SBS13",
    "SBS17a",
    "SBS17b",
    "SBS18",
    "SBS40a",
]

# The tree-free/independent-node ablation used in the tree-vs-pooling
# comparison (CLAUDE.md's "Direction of travel", item 2) does not exist in
# this codebase yet -- checked by search, not assumed. Every baseline_*
# function below reports this rather than building a new comparator.
BASELINE_AVAILABLE = False


def load_tree(name: str, tree_dir: Path) -> str:
    """Newick string for one of 'snv', 'cna', 'consensus'. snv/cna are
    read from `tree_dir`; consensus is CONSENSUS_NEWICK, imported from
    check_tree_inputs.py rather than retyped."""
    if name == "snv":
        return (tree_dir / "snv_tree.nwk").read_text().strip()
    if name == "cna":
        return (tree_dir / "cna_tree.nwk").read_text().strip()
    if name == "consensus":
        return CONSENSUS_NEWICK
    raise ValueError(f"unknown tree name {name!r}; use one of {TREE_NAMES}")


def select_signatures(catalogue: pd.DataFrame, names: List[str]) -> np.ndarray:
    """(K, C) fixed-signature matrix, rows in exactly ``names``' order --
    component k of the model is ``names[k]``, never re-sorted. Raises
    ``ValueError`` naming any signature absent from ``catalogue``, rather
    than silently dropping it (unlike refit_cosmic.py's candidate-set
    availability check, which is exploratory; this selects the model's
    actual signature set, so a typo or a missing name must be fatal)."""
    missing = [name for name in names if name not in catalogue.index]
    if missing:
        raise ValueError(
            f"signature(s) not in catalogue: {missing}. Available: "
            f"{sorted(catalogue.index)}"
        )
    return catalogue.loc[names].to_numpy()


def build_and_sample(
    newick_string: str,
    spectra: pd.DataFrame,
    signature_matrix: np.ndarray,
    draws: int = DRAWS,
    tune: int = TUNE,
    chains: int = CHAINS,
    cores: int = CORES,
    target_accept: float = TARGET_ACCEPT,
    max_treedepth: int = MAX_TREEDEPTH,
    seed: int = SEED,
) -> Tuple[TreeHDP, float]:
    """Build TreeHDP (S known) and sample with a FIXED random_seed -- see
    the module docstring for why this calls ``pm.sample`` directly rather
    than ``model.sample()``. Returns ``(model, elapsed_seconds)``, with
    ``model.trace`` already set."""
    model = TreeHDP(
        newick_string=newick_string,
        data_matrix=spectra,
        priors=DEFAULT_PRIORS,
        fixed_signatures=signature_matrix,
    )
    start = time.time()
    with model.model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            nuts_sampler="numpyro",
            nuts={"max_tree_depth": max_treedepth},
            random_seed=seed,
        )
    elapsed = time.time() - start
    model.trace = trace
    return model, elapsed


def convergence_diagnostics(trace, elapsed: float) -> Dict[str, Optional[float]]:
    """max r-hat and min bulk/tail ESS over the identifiable activity
    variables (``e_level_*``) and ``sigma`` only -- CLAUDE.md: the raw
    walk increments (``eta_level_*``, ``z_level_*``, ``z_root_*``) are
    excluded, uninformative by construction. Plus the divergence count and
    wall-clock runtime."""
    var_names = [v for v in trace.posterior.data_vars if v.startswith("e_level_")]
    var_names.append("sigma")
    rhat = az.rhat(trace, var_names=var_names)
    ess_bulk = az.ess(trace, var_names=var_names, method="bulk")
    ess_tail = az.ess(trace, var_names=var_names, method="tail")
    max_rhat = max(float(rhat[v].max()) for v in rhat.data_vars)
    min_ess_bulk = min(float(ess_bulk[v].min()) for v in ess_bulk.data_vars)
    min_ess_tail = min(float(ess_tail[v].min()) for v in ess_tail.data_vars)
    n_divergences = (
        int(trace.sample_stats["diverging"].sum().item())
        if "diverging" in trace.sample_stats
        else None
    )
    return {
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "min_ess_tail": min_ess_tail,
        "n_divergences": n_divergences,
        "runtime_seconds": elapsed,
    }


def activity_summary(model: TreeHDP, signature_names: List[str]) -> pd.DataFrame:
    """One row per (node, signature): posterior mean and the 5%/95%
    percentile interval, for every node in the graph -- observed clusters
    and every latent node, including germline and any hidden g<k>."""
    rows = []
    for node in model.graph.nodes():
        samples = model.get_node_activity_posterior(node)  # (chains, draws, K)
        flat = samples.reshape(-1, samples.shape[-1])
        mean = flat.mean(axis=0)
        p5 = np.percentile(flat, 5, axis=0)
        p95 = np.percentile(flat, 95, axis=0)
        kind = "observed" if node in model.observed_labels else "latent"
        for k, signature in enumerate(signature_names):
            rows.append(
                {
                    "node": node,
                    "kind": kind,
                    "signature": signature,
                    "mean": float(mean[k]),
                    "p5": float(p5[k]),
                    "p95": float(p95[k]),
                }
            )
    return pd.DataFrame(rows)


def posterior_predictive_check(
    model: TreeHDP, signature_matrix: np.ndarray, spectra: pd.DataFrame
) -> pd.DataFrame:
    """Cosine similarity between each observed cluster's raw spectrum and
    the posterior-mean reconstruction (posterior mean activity @ the
    fixed signature matrix)."""
    rows = []
    for node in sorted(model.observed_labels):
        mean_activity = model.get_posterior_mean(node)  # (K,)
        reconstruction = mean_activity @ signature_matrix
        observed = spectra.loc[node].to_numpy()
        rows.append(
            {"cluster": node, "cosine_similarity": cosine(observed, reconstruction)}
        )
    return pd.DataFrame(rows)


def run_one(
    tree_name: str,
    newick_string: str,
    spectra: pd.DataFrame,
    signature_names: List[str],
    signature_matrix: np.ndarray,
    outdir: Path,
    draws: int = DRAWS,
    tune: int = TUNE,
    chains: int = CHAINS,
    cores: int = CORES,
    target_accept: float = TARGET_ACCEPT,
    max_treedepth: int = MAX_TREEDEPTH,
    seed: int = SEED,
) -> Dict[str, Optional[float]]:
    """Build, sample, and write every output for one (tree, setting)
    combination. Returns the convergence diagnostics dict (also written
    to convergence.csv)."""
    outdir.mkdir(parents=True, exist_ok=True)
    model, elapsed = build_and_sample(
        newick_string,
        spectra,
        signature_matrix,
        draws,
        tune,
        chains,
        cores,
        target_accept,
        max_treedepth,
        seed,
    )
    model.trace.to_netcdf(str(outdir / "trace.nc"))

    diagnostics = convergence_diagnostics(model.trace, elapsed)
    pd.DataFrame([diagnostics]).to_csv(outdir / "convergence.csv", index=False)

    activity_summary(model, signature_names).to_csv(
        outdir / "activity_summary.csv", index=False
    )
    posterior_predictive_check(model, signature_matrix, spectra).to_csv(
        outdir / "posterior_predictive.csv", index=False
    )

    with open(outdir / "run_info.json", "w") as f:
        json.dump(
            {
                "tree": tree_name,
                "signatures": signature_names,
                "seed": seed,
                "draws": draws,
                "tune": tune,
                "chains": chains,
                "cores": cores,
                "target_accept": target_accept,
                "max_treedepth": max_treedepth,
            },
            f,
            indent=2,
        )

    return diagnostics


# --------------------------------------------------------------------------- #
# Comparison report (run after one or more run directories exist)
# --------------------------------------------------------------------------- #


def load_run(outdir: Path) -> dict:
    """Read one run directory's own CSV/JSON outputs back -- the
    comparison report works from these, not from re-loaded traces."""
    return {
        "activity": pd.read_csv(outdir / "activity_summary.csv"),
        "convergence": pd.read_csv(outdir / "convergence.csv"),
        "ppc": pd.read_csv(outdir / "posterior_predictive.csv"),
        "info": json.loads((outdir / "run_info.json").read_text()),
    }


def activity_comparison_table(
    runs: Dict[str, dict], signatures: Tuple[str, ...] = ("SBS1", "SBS5")
) -> pd.DataFrame:
    """Report item 1: per cluster, ``signatures``' activity (mean, 90%
    interval) across every run in ``runs`` (keyed by an arbitrary run
    label, e.g. ``"main_snv"``), as one long table."""
    rows = []
    for run_label, run in runs.items():
        observed = run["activity"][run["activity"]["kind"] == "observed"]
        for cluster in sorted(observed["node"].unique()):
            for signature in signatures:
                match = observed[
                    (observed["node"] == cluster) & (observed["signature"] == signature)
                ]
                if len(match) == 0:
                    continue
                row = match.iloc[0]
                rows.append(
                    {
                        "run": run_label,
                        "cluster": cluster,
                        "signature": signature,
                        "mean": row["mean"],
                        "p5": row["p5"],
                        "p95": row["p95"],
                    }
                )
    return pd.DataFrame(rows)


def tree_sensitivity_table(
    runs: Dict[str, dict],
    setting: str,
    trees: Tuple[str, ...] = TREE_NAMES,
    signatures: Tuple[str, ...] = ("SBS1", "SBS5"),
) -> pd.DataFrame:
    """Report item 2: for ``setting`` (e.g. ``"main"``), across whichever
    of ``trees`` have a run in ``runs`` (keyed ``"<setting>_<tree>"``), the
    largest difference in each cluster/signature's activity mean between
    any two trees, and whether the two most different trees' 90%
    intervals overlap."""
    tree_runs = {
        tree: runs[f"{setting}_{tree}"] for tree in trees if f"{setting}_{tree}" in runs
    }
    if len(tree_runs) < 2:
        return pd.DataFrame(
            columns=[
                "cluster",
                "signature",
                "max_diff",
                "tree_max",
                "tree_min",
                "intervals_overlap",
            ]
        )
    any_activity = next(iter(tree_runs.values()))["activity"]
    clusters = sorted(any_activity[any_activity["kind"] == "observed"]["node"].unique())

    rows = []
    for cluster in clusters:
        for signature in signatures:
            values = {}
            for tree, run in tree_runs.items():
                act = run["activity"]
                match = act[
                    (act["node"] == cluster)
                    & (act["signature"] == signature)
                    & (act["kind"] == "observed")
                ]
                if len(match):
                    row = match.iloc[0]
                    values[tree] = (row["mean"], row["p5"], row["p95"])
            if len(values) < 2:
                continue
            means = {tree: v[0] for tree, v in values.items()}
            tree_max = max(means, key=means.get)
            tree_min = min(means, key=means.get)
            lo_max, hi_max = values[tree_max][1], values[tree_max][2]
            lo_min, hi_min = values[tree_min][1], values[tree_min][2]
            overlap = not (hi_max < lo_min or hi_min < lo_max)
            rows.append(
                {
                    "cluster": cluster,
                    "signature": signature,
                    "max_diff": means[tree_max] - means[tree_min],
                    "tree_max": tree_max,
                    "tree_min": tree_min,
                    "intervals_overlap": overlap,
                }
            )
    return pd.DataFrame(rows)


def spectra_sensitivity_table(
    runs: Dict[str, dict],
    tree: str,
    main_label: str = "main",
    sensitivity_label: str = "sensitivity",
) -> pd.DataFrame:
    """Report item 3: for ``tree``, SBS1 and SBS5 in the main run vs the
    sensitivity run (main renormalised to SBS1+SBS5 for a like-for-like
    comparison against the sensitivity run's 2-signature fit), and the
    SBS95 share in the main run."""
    main_key, sensitivity_key = f"{main_label}_{tree}", f"{sensitivity_label}_{tree}"
    if main_key not in runs or sensitivity_key not in runs:
        return pd.DataFrame(
            columns=[
                "cluster",
                "main_sbs1_renorm",
                "sensitivity_sbs1",
                "main_sbs5_renorm",
                "sensitivity_sbs5",
                "main_sbs95_share",
            ]
        )
    main_obs = runs[main_key]["activity"].pipe(lambda d: d[d["kind"] == "observed"])
    sensitivity_obs = runs[sensitivity_key]["activity"].pipe(
        lambda d: d[d["kind"] == "observed"]
    )

    def _value(df: pd.DataFrame, cluster, signature) -> Optional[float]:
        match = df[(df["node"] == cluster) & (df["signature"] == signature)]
        return float(match.iloc[0]["mean"]) if len(match) else None

    rows = []
    for cluster in sorted(main_obs["node"].unique()):
        m1 = _value(main_obs, cluster, "SBS1")
        m5 = _value(main_obs, cluster, "SBS5")
        m95 = _value(main_obs, cluster, "SBS95") or 0.0
        s1 = _value(sensitivity_obs, cluster, "SBS1")
        s5 = _value(sensitivity_obs, cluster, "SBS5")
        if m1 is None or m5 is None or s1 is None or s5 is None:
            continue
        total = m1 + m5
        rows.append(
            {
                "cluster": cluster,
                "main_sbs1_renorm": m1 / total if total > 0 else float("nan"),
                "sensitivity_sbs1": s1,
                "main_sbs5_renorm": m5 / total if total > 0 else float("nan"),
                "sensitivity_sbs5": s5,
                "main_sbs95_share": m95,
            }
        )
    return pd.DataFrame(rows)


def active_signature_summary(
    activity: pd.DataFrame, threshold: float = 0.02
) -> pd.DataFrame:
    """Per signature, over one run's activity_summary.csv: how many
    observed clusters have their 90% interval's lower bound (``p5``)
    above ``threshold`` (clearly active, not just consistent with zero),
    and the largest posterior mean among those clusters."""
    observed = activity[activity["kind"] == "observed"]
    rows = []
    for signature in observed["signature"].unique():
        rows_for_sig = observed[observed["signature"] == signature]
        active = rows_for_sig[rows_for_sig["p5"] > threshold]
        rows.append(
            {
                "signature": signature,
                "n_clusters_active": len(active),
                "n_clusters_total": len(rows_for_sig),
                "max_mean": float(rows_for_sig["mean"].max()),
            }
        )
    return pd.DataFrame(rows)


def baseline_comparison_table(*_args, **_kwargs) -> pd.DataFrame:
    """Report item 4: no-op. The independent/no-tree ablation used in the
    tree-vs-pooling comparison does not exist in this codebase (checked by
    search, see BASELINE_AVAILABLE) -- returns an empty table rather than
    building a new comparator, per instruction."""
    return pd.DataFrame(
        columns=["cluster", "tree_interval_width", "baseline_interval_width"]
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tree", required=True, choices=TREE_NAMES)
    p.add_argument("--tree-dir", type=Path, default=Path("realdata/local_tree_input"))
    p.add_argument("--spectra", type=Path, required=True)
    p.add_argument(
        "--catalogue",
        type=Path,
        default=Path("COSMIC_sig/cosmic_v3.4_sbs96_grch37_full.csv"),
    )
    p.add_argument("--signatures", nargs="+", required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--draws", type=int, default=DRAWS)
    p.add_argument("--tune", type=int, default=TUNE)
    p.add_argument("--chains", type=int, default=CHAINS)
    p.add_argument("--cores", type=int, default=CORES)
    p.add_argument("--target-accept", type=float, default=TARGET_ACCEPT)
    p.add_argument("--max-treedepth", type=int, default=MAX_TREEDEPTH)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    newick_string = load_tree(args.tree, args.tree_dir)
    spectra = read_data_matrix(args.spectra)
    catalogue = pd.read_csv(args.catalogue, index_col=0)
    signature_matrix = select_signatures(catalogue, args.signatures)

    diagnostics = run_one(
        args.tree,
        newick_string,
        spectra,
        args.signatures,
        signature_matrix,
        args.outdir,
        args.draws,
        args.tune,
        args.chains,
        args.cores,
        args.target_accept,
        args.max_treedepth,
        args.seed,
    )
    print(
        f"max_rhat={diagnostics['max_rhat']:.4f} "
        f"min_ess_bulk={diagnostics['min_ess_bulk']:.1f} "
        f"min_ess_tail={diagnostics['min_ess_tail']:.1f} "
        f"n_divergences={diagnostics['n_divergences']} "
        f"runtime_seconds={diagnostics['runtime_seconds']:.1f}"
    )
    print(f"Wrote outputs to {args.outdir}")


if __name__ == "__main__":
    main()
