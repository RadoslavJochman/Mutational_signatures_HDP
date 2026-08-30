"""
make_replicate_configs.py

Expand one base config into R replicate configs that differ only in the data
seed and in the output identity, so the replicates of a setting are independent
datasets that can be averaged into an error bar. For each replicate it bumps
`simulation.seed`, gives it a nested `experiment_name` (`<sweep>/rep<NN>`), and
repoints the `inference.data.*` paths at its own generated data directory. The
`simulation.repertoire.path` is left untouched so every replicate loads the same
frozen signature matrix; only the forest and activities vary.

Every replicate config is written into its own experiment directory,
`<experiment_root>/<sweep>/rep<NN>/config.yaml`, where `<sweep>` is the base
config's own `experiment_name` and `experiment_root` is read from the base
config itself -- there is no separate `--outdir`, so a replicate's config can
never disagree with where its own data and results actually land.

Run once per setting (per overlap rho or per tree count); point each invocation
at that setting's base config and pass `--setting-value` so the aggregation
manifest groups correctly. Use `--append` to accumulate one manifest across
settings.

Outputs (under <experiment_root>/<sweep>/)
    rep<NN>/config.yaml       one config per replicate, NN zero-padded to fit
                              n_reps - 1 (at least 2 digits)
    manifest_configs.txt      config paths, one per line (for the sbatch
                              array; set --array=0-(N-1))
    manifest_agg.csv          setting, rep, dir for aggregate_replicates.py
                              (dir is the per-replicate results/recovery directory)
    manifest_agg_nmf.csv      same, for the per-replicate results/nmf directory

Run from scripts/, like every other pipeline script, since the base config's
`experiment_root` is a scripts-relative path (e.g. `../experiments/`) and this
script now resolves it for real (mkdir, not just string-building).

Usage
    cd scripts
    python make_replicate_configs.py \
        --base ../experiments/corr_sweep/base_config_denovo_corr_0.3_20_trees.yaml \
        --n-reps 15 --setting-value 0.3 --draws 2000 --tune 2000
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


def repoint_data(data: dict, experiment_root: str, new_name: str) -> dict:
    """Point every data file at this replicate's own generated directory,
    <experiment_root>/<new_name>/data/<filename>. Reconstructing from the
    directory rather than string-replacing the old name makes this immune to
    a base config whose data paths do not already match its experiment_name
    (a common copy-paste slip), and to one that points at a different
    dataset entirely."""
    base_dir = f"{experiment_root.rstrip('/')}/{new_name}/data"
    return {
        k: (f"{base_dir}/{Path(v).name}" if isinstance(v, str) else v)
        for k, v in data.items()
    }


def make_one(
    base: dict, seed: int, new_name: str, draws: int | None, tune: int | None
) -> dict:
    """One replicate config: bumped seed, nested name, repointed data paths,
    optional sampler overrides. Everything else copied from the base."""
    cfg = copy.deepcopy(base)
    cfg["experiment_name"] = new_name
    cfg["simulation"]["seed"] = seed
    cfg["inference"]["data"] = repoint_data(
        cfg["inference"]["data"], base["experiment_root"], new_name
    )
    if draws is not None:
        cfg["inference"]["draws"] = draws
    if tune is not None:
        cfg["inference"]["tune"] = tune
    return cfg


def write_config(cfg: dict, experiment_root: Path, header: str) -> Path:
    """Dump one replicate's config into its own experiment directory,
    <experiment_root>/<sweep>/rep<NN>/config.yaml."""
    exp_dir = experiment_root / cfg["experiment_name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    path = exp_dir / "config.yaml"
    with path.open("w") as f:
        f.write(header)
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--n-reps", type=int, required=True)
    ap.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="seed of replicate 0; default is the base config's seed. "
        "replicate r uses base-seed + r",
    )
    ap.add_argument(
        "--setting-value",
        default=None,
        help="the controllable-axis value for this base (e.g. the "
        "overlap rho or the tree count); tags the agg manifest",
    )
    ap.add_argument("--draws", type=int, default=None)
    ap.add_argument("--tune", type=int, default=None)
    ap.add_argument(
        "--append",
        action="store_true",
        help="append to existing manifests instead of overwriting",
    )
    a = ap.parse_args()

    base = yaml.safe_load(Path(a.base).read_text())
    sweep_name = base["experiment_name"]
    experiment_root = Path(base["experiment_root"])
    base_seed = (
        a.base_seed if a.base_seed is not None else int(base["simulation"]["seed"])
    )
    setting = a.setting_value if a.setting_value is not None else sweep_name

    sweep_dir = experiment_root / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    pad = max(2, len(str(a.n_reps - 1)))
    exp_root_str = str(experiment_root).rstrip("/")
    cfg_lines, agg_rows, agg_rows_nmf = [], [], []
    for r in range(a.n_reps):
        rep_label = f"rep{r:0{pad}d}"
        new_name = f"{sweep_name}/{rep_label}"
        seed = base_seed + r
        cfg = make_one(base, seed, new_name, a.draws, a.tune)
        header = (
            f"# generated by make_replicate_configs.py from {a.base}\n"
            f"# replicate {r}, seed {seed}, setting {setting}\n"
        )
        path = write_config(cfg, experiment_root, header)
        cfg_lines.append(str(path))
        res_dir = f"{exp_root_str}/{new_name}/results"
        agg_rows.append(f"{setting},{r},{res_dir}/recovery")
        agg_rows_nmf.append(f"{setting},{r},{res_dir}/nmf")

    mode = "a" if a.append else "w"
    cfg_manifest = sweep_dir / "manifest_configs.txt"
    agg_manifest = sweep_dir / "manifest_agg.csv"
    nmf_manifest = sweep_dir / "manifest_agg_nmf.csv"
    with cfg_manifest.open(mode) as f:
        f.write("\n".join(cfg_lines) + "\n")
    for man, rows in ((agg_manifest, agg_rows), (nmf_manifest, agg_rows_nmf)):
        write_header = not (a.append and man.exists())
        with man.open(mode) as f:
            if write_header:
                f.write("setting,rep,dir\n")
            f.write("\n".join(rows) + "\n")

    print(f"wrote {a.n_reps} configs to {sweep_dir}")
    print(f"  sbatch manifest:      {cfg_manifest}  (--array=0-{a.n_reps - 1})")
    print(f"  aggregation manifest: {agg_manifest} (Tree-HDP), {nmf_manifest} (NMF)")
    print(f"  seeds: {base_seed}..{base_seed + a.n_reps - 1}, setting={setting}")


if __name__ == "__main__":
    main()
