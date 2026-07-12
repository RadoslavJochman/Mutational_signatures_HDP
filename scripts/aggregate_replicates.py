"""
aggregate_replicates.py

Collapse many independent replicate runs into the mean and spread that the
error-bar plots need. Each replicate is one dataset (one `simulation.seed`) fit
and scored on its own; this groups them by experimental setting and reports the
across-replicate mean, standard deviation, standard error and count for every
metric, so a point on a recovery-versus-setting plot carries a real error bar.

Inputs
    --manifest  CSV with columns: setting, rep, dir
                `setting` is the controllable axis value (e.g. the overlap rho
                or the tree count), `rep` the replicate index, `dir` the run
                directory holding the recovery_*.csv that recovery_vs_truth.py
                or nmf_baseline.py wrote.
    --kind      summary     -> per-setting scalars (one row per replicate in
                              recovery_summary.csv); gives the dataset-level
                              error bars (median activity distance, mean
                              signature distance, exposure error, ...).
                signatures  -> per-(setting, signature) rows from
                              recovery_signatures.csv; gives a per-signature
                              error bar at each setting. Signature index is
                              comparable across replicates because the
                              signatures are frozen.
    --filename  override the per-replicate CSV name (default chosen by --kind).

Output (to --outdir)
    aggregated_<kind>.csv   mean / std / sem / n for every numeric column,
                            grouped by setting (and signature, for `signatures`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load(manifest: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Read one per-replicate CSV from every run dir, tagged with setting/rep."""
    frames = []
    for _, r in manifest.iterrows():
        path = Path(r["dir"]) / filename
        if not path.exists():
            print(f"missing, skipping: {path}")
            continue
        df = pd.read_csv(path)
        df["setting"] = r["setting"]
        df["rep"] = r["rep"]
        frames.append(df)
    if not frames:
        raise SystemExit("no replicate files found; check the manifest paths")
    return pd.concat(frames, ignore_index=True)


def _aggregate(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """mean / std / sem / n for every numeric column over the group."""
    value_cols = [c for c in df.select_dtypes("number").columns
                  if c not in group_cols and c != "rep"]
    g = df.groupby(group_cols)[value_cols]
    out = g.agg(["mean", "std", "sem", "count"])
    out.columns = [f"{c}_{stat}" for c, stat in out.columns]
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--kind", choices=["summary", "signatures"], default="summary")
    ap.add_argument("--filename", default=None)
    ap.add_argument("--outdir", default="aggregated")
    a = ap.parse_args()

    manifest = pd.read_csv(a.manifest)
    for col in ("setting", "rep", "dir"):
        if col not in manifest.columns:
            raise SystemExit(f"manifest needs a '{col}' column")

    default_name = {"summary": "recovery_summary.csv",
                    "signatures": "recovery_signatures.csv"}[a.kind]
    df = _load(manifest, a.filename or default_name)

    group = ["setting"] if a.kind == "summary" else ["setting", "signature"]
    agg = _aggregate(df, group)

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    dest = out / f"aggregated_{a.kind}.csv"
    agg.to_csv(dest, index=False)

    n_per = df.groupby(group).size()
    print(f"wrote {dest}")
    print(f"settings: {sorted(df['setting'].unique())}")
    print(f"replicates per group: min {int(n_per.min())}, max {int(n_per.max())}")
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 60)
    print(agg.round(4).to_string(index=False))


if __name__ == "__main__":
    main()