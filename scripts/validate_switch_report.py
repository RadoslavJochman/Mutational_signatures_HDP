"""
validate_switch_report.py

Summarise the pre-merge validation run of the switch model
(experiments/validate_switch/, driven by scripts/validate_switch.sh) into
experiments/validate_switch/report.md. Reads every replicate's config, trace
and scoring CSVs; computes nothing that the pipeline scripts did not already
score, apart from the convergence row (scaling_metrics.convergence_row on the
saved trace), the divergence count from sample_stats, and the across-chain
disagreement of a_prob_level_* (max over nodes and signatures of the spread
of the per-chain means).

Pass criteria (stated in the report, evaluated on the chain-mean metrics;
the worst chain is shown alongside):
    r_hat <= 1.01, ESS >= 400, no divergences, a_prob chain disagreement < 0.05,
    accuracy at true level 0 >= 0.95, accuracy at true level > 0.05 >= 0.9,
    ECE <= 0.1, v2 recovery not worse than v1 (median node cosine >= v1's and
    median node L1 <= v1's on the same dataset). The (0, 0.05] level bin is
    reported, not gated. Numbers are reported as they are.

Usage
    python scripts/validate_switch_report.py --root ../experiments/validate_switch
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytensor
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scaling_metrics import _convergence_vars, convergence_row  # noqa: E402

CRITERIA = {
    "max_rhat": ("<=", 1.01),
    "min_ess": (">=", 400),
    "divergences": ("==", 0),
    "a_prob_disagreement": ("<", 0.05),
    "acc_level0": (">=", 0.95),
    "acc_level_high": (">=", 0.90),
    "ece": ("<=", 0.10),
}


def _passes(value, rule):
    op, thr = rule
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return {
        "<=": value <= thr,
        ">=": value >= thr,
        "<": value < thr,
        "==": value == thr,
    }[op]


def _mark(ok):
    return "n/a" if ok is None else ("PASS" if ok else "FAIL")


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "nan"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{nd}f}"


def _md_table(rows, columns):
    head = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def summarise_arm(cfg_path: Path, root: Path, fig_dir: Path) -> dict:
    cfg = yaml.safe_load(cfg_path.read_text())
    rep_dir = cfg_path.parent
    arm = rep_dir.parent.name
    rep = rep_dir.name
    res = rep_dir / "results"
    mode = cfg["inference"]["model"]
    switching = bool((cfg["inference"].get("switching") or {}).get("enabled", False))
    chains, draws, tune = (cfg["inference"][k] for k in ("chains", "draws", "tune"))
    out = {
        "arm": arm,
        "rep": rep,
        "seed": cfg["simulation"]["seed"],
        "burden": cfg["simulation"]["burden"]["mean"],
        "model": ("v2 " if switching else "v1 ") + mode,
        "switching": switching,
        "git_tag": cfg.get("git_tag"),
    }

    counts_path = rep_dir / "data" / "mutation_count_matrix.csv"
    out["data_md5"] = hashlib.md5(counts_path.read_bytes()).hexdigest()[:10]
    out["n_nodes"] = len(pd.read_csv(counts_path, index_col=0))

    trace_path = res / ("trace_aligned.nc" if mode == "denovo" else "trace.nc")
    idata = az.from_netcdf(trace_path)
    post = idata.posterior
    out.update(convergence_row(idata, _convergence_vars(post, "e_level", None)))
    out["divergences"] = (
        int(idata.sample_stats["diverging"].values.sum())
        if "sample_stats" in idata.groups() and "diverging" in idata.sample_stats
        else None
    )
    secs = float((res / "infer_seconds.txt").read_text().strip())
    out["wall_s"] = secs
    out["s_per_draw"] = secs / (chains * draws)
    out["s_per_iter"] = secs / (chains * (draws + tune))

    if switching:
        spreads = []
        for var in post.data_vars:
            if str(var).startswith("a_prob_level_"):
                cm = post[var].values.mean(axis=1)  # (chains, n, K)
                spreads.append((cm.max(axis=0) - cm.min(axis=0)).max())
        out["a_prob_disagreement"] = float(max(spreads))

        summ = pd.read_csv(res / "switch" / "switch_summary.csv")
        overall = summ[summ["group"] == "overall"].iloc[0]
        for m in (
            "accuracy",
            "auroc",
            "auprc",
            "brier",
            "ece",
            "acc_level0",
            "acc_level_low",
            "acc_level_high",
        ):
            out[m] = float(overall[f"{m}_mean"])
            out[f"{m}_worst"] = float(overall[f"{m}_worst"])
        out["per_signature"] = summ[summ["group"] != "overall"][
            ["group", "auroc_mean", "accuracy_mean", "ece_mean", "acc_level_high_mean"]
        ]
        edges = pd.read_csv(res / "switch" / "switch_edge_summary.csv")
        out["edges"] = edges[edges["group"] == "overall"][
            ["event", "n_edges", "auroc", "auprc", "accuracy", "brier", "ece"]
        ]
        npz = np.load(res / "switch" / "state_samples.npz")
        out["state_draws"] = int(npz["draw_idx"].shape[0])

        fig_dir.mkdir(parents=True, exist_ok=True)
        src = res / "switch" / "switch_recovery.png"
        if src.exists():
            dst = fig_dir / f"{arm}_{rep}_switch_recovery.png"
            shutil.copyfile(src, dst)
            out["figure"] = dst.relative_to(root).as_posix()

    rec = pd.read_csv(res / "recovery" / "recovery_summary.csv").iloc[0]
    out["cosine_median"] = float(rec["act_cosine_median_mean"])
    out["l1_median"] = 2.0 * float(rec["act_tv_median_mean"])  # tv is half the L1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../experiments/validate_switch")
    ap.add_argument("--thin", type=int, default=None)
    a = ap.parse_args()
    root = Path(a.root)
    fig_dir = root / "figures"

    cfgs = sorted(root.glob("*/rep*/config.yaml"))
    arms = [summarise_arm(p, root, fig_dir) for p in cfgs]
    df = pd.DataFrame(arms)

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    backend = (
        "PyTensor Python backend (cxx empty; no C compilation)"
        if not pytensor.config.cxx
        else f"PyTensor C backend ({pytensor.config.cxx})"
    )

    lines = []
    lines.append("# Switch model validation (pre-merge, treehdp-v2)\n")
    lines.append(
        f"Generated {date.today().isoformat()} by scripts/validate_switch_report.py "
        f"at commit `{commit}`; configs carry `git_tag: {df['git_tag'].iloc[0]}`.\n"
    )
    lines.append(
        f"Backend: {backend}. Sampling itself runs in JAX through numpyro "
        "(`nuts_sampler='numpyro'`), so the backend affects PyTensor's own "
        "compiled functions (the post-hoc state sampler, scoring), not the "
        "NUTS trajectories. Sampler: 4 chains, 1000 tune, 1000 draws, "
        "target_accept 0.9 for every arm."
        + (f" State samples use every {a.thin}th draw." if a.thin else "")
        + "\n"
    )
    lines.append(
        "Repertoire: the catalogue holds ten signatures and none of SBS2, SBS13, "
        "SBS3, so the breast repertoire's unit structure is kept with stand-ins: "
        "SBS1/SBS5 clock (always on); [SBS36, SBS112] as the co-switching unit "
        "(for SBS2/SBS13); SBS40a (cosine 0.81 to SBS5) as the flat per-patient "
        "signature (for SBS3). Rates are simulator_spec.md section 3's. Burdens: "
        "970 (comfortable; the most-used historical sweep value) and 300 (low; "
        "the current smoke value). 4 trees of 6 to 10 nodes each per dataset.\n"
    )

    lines.append("## Pass criteria\n")
    lines.append(
        "Evaluated per arm on the chain-mean metrics (worst chain shown too): "
        "r_hat <= 1.01; ESS >= 400; no divergences; a_prob chain disagreement "
        "(max over nodes and signatures of the spread of per-chain means) < 0.05; "
        "accuracy at true level 0 >= 0.95; accuracy at true level > 0.05 >= 0.90; "
        "ECE <= 0.10; v2 recovery not worse than v1 on the same dataset (median "
        "node cosine >= v1's and median node L1 <= v1's). The (0, 0.05] level bin "
        "is reported, not gated. Convergence variables: e_level_*, sigma, mu_level, "
        "lambda_on, lambda_off, pi_root (a_prob_level_* excluded, see CLAUDE.md).\n"
    )

    lines.append("## Datasets\n")
    ds_rows = []
    for (burden, seed), g in df.groupby(["burden", "seed"]):
        ds_rows.append(
            {
                "burden": burden,
                "seed": seed,
                "nodes": int(g["n_nodes"].iloc[0]),
                "arms": ", ".join(f"{r.arm}/{r.rep}" for r in g.itertuples()),
                "count matrices identical": "yes"
                if g["data_md5"].nunique() == 1
                else "NO",
            }
        )
    lines.append(_md_table(ds_rows, list(ds_rows[0].keys())) + "\n")

    lines.append("## Convergence and cost (every arm)\n")
    conv_rows = []
    for r in df.itertuples():
        conv_rows.append(
            {
                "arm": r.arm,
                "rep": r.rep,
                "burden": r.burden,
                "model": r.model,
                "max r_hat": _fmt(r.max_rhat, 4),
                "min ESS": _fmt(r.min_ess, 0),
                "divergences": _fmt(r.divergences),
                "wall s": _fmt(r.wall_s, 0),
                "s / draw": _fmt(r.s_per_draw, 3),
                "a_prob disagreement": _fmt(getattr(r, "a_prob_disagreement", np.nan)),
            }
        )
    lines.append(_md_table(conv_rows, list(conv_rows[0].keys())) + "\n")
    lines.append(
        "wall s is the whole run_inference.py call (model build, JAX compile, "
        "tuning, sampling, trace writing); s / draw divides it by chains x draws.\n"
    )

    lines.append("## On/off recovery (v2 arms)\n")
    sw = df[df["switching"]]
    onoff_rows = []
    for r in sw.itertuples():
        onoff_rows.append(
            {
                "arm": r.arm,
                "rep": r.rep,
                "accuracy (worst)": f"{_fmt(r.accuracy)} ({_fmt(r.accuracy_worst)})",
                "AUROC": _fmt(r.auroc),
                "AUPRC": _fmt(r.auprc),
                "Brier": _fmt(r.brier),
                "ECE (worst)": f"{_fmt(r.ece)} ({_fmt(r.ece_worst)})",
                "acc level 0 (worst)": (
                    f"{_fmt(r.acc_level0)} ({_fmt(r.acc_level0_worst)})"
                ),
                "acc (0, 0.05]": _fmt(r.acc_level_low),
                "acc > 0.05 (worst)": (
                    f"{_fmt(r.acc_level_high)} ({_fmt(r.acc_level_high_worst)})"
                ),
            }
        )
    lines.append(_md_table(onoff_rows, list(onoff_rows[0].keys())) + "\n")

    lines.append("### Per signature (chain means)\n")
    for r in sw.itertuples():
        lines.append(f"**{r.arm}/{r.rep}**\n")
        ps = r.per_signature.copy()
        ps.columns = ["signature", "AUROC", "accuracy", "ECE", "acc > 0.05"]
        lines.append(
            _md_table(
                [
                    {k: _fmt(v) if isinstance(v, float) else v for k, v in row.items()}
                    for row in ps.to_dict("records")
                ],
                list(ps.columns),
            )
            + "\n"
        )

    lines.append("### Edge events (from the state samples, pooled over chains)\n")
    for r in sw.itertuples():
        lines.append(f"**{r.arm}/{r.rep}** ({r.state_draws} state draws per chain)\n")
        ed = r.edges.copy()
        lines.append(
            _md_table(
                [
                    {k: _fmt(v) if isinstance(v, float) else v for k, v in row.items()}
                    for row in ed.to_dict("records")
                ],
                list(ed.columns),
            )
            + "\n"
        )

    lines.append("## Activity recovery, v2 versus v1 on the same dataset\n")
    rec_rows = []
    fixed = df[df["model"].str.endswith("fixed")]
    for (burden, seed), g in fixed.groupby(["burden", "seed"]):
        v1 = g[~g["switching"]]
        v2 = g[g["switching"]]
        if len(v1) != 1 or len(v2) != 1:
            continue
        v1, v2 = v1.iloc[0], v2.iloc[0]
        rec_rows.append(
            {
                "burden": burden,
                "seed": seed,
                "cosine v1": _fmt(v1["cosine_median"]),
                "cosine v2": _fmt(v2["cosine_median"]),
                "delta cosine": _fmt(v2["cosine_median"] - v1["cosine_median"]),
                "L1 v1": _fmt(v1["l1_median"]),
                "L1 v2": _fmt(v2["l1_median"]),
                "delta L1": _fmt(v2["l1_median"] - v1["l1_median"]),
                "v2 not worse": _mark(
                    v2["cosine_median"] >= v1["cosine_median"]
                    and v2["l1_median"] <= v1["l1_median"]
                ),
                "_v2_arm": f"{v2['arm']}/{v2['rep']}",
            }
        )
    lines.append(
        _md_table(rec_rows, [c for c in rec_rows[0].keys() if not c.startswith("_")])
        + "\n"
    )
    lines.append(
        "Median over nodes of the per-node cosine and L1 (simplex L1, 0..2) of the "
        "chain-mean activity to the true activity, from recovery_vs_truth.py; "
        "for the v2 arms this scores the state-mixed e_level.\n"
    )
    denovo = df[df["model"].str.endswith("denovo")]
    if len(denovo):
        dn_rows = [
            {
                "arm": r.arm,
                "rep": r.rep,
                "cosine median": _fmt(r.cosine_median),
                "L1 median": _fmt(r.l1_median),
                "max r_hat": _fmt(r.max_rhat, 4),
                "min ESS": _fmt(r.min_ess, 0),
            }
            for r in denovo.itertuples()
        ]
        lines.append("De novo v2 (S latent, aligned to the truth before scoring):\n")
        lines.append(_md_table(dn_rows, list(dn_rows[0].keys())) + "\n")

    lines.append("## Verdict per arm\n")
    not_worse = {r["_v2_arm"]: r["v2 not worse"] for r in rec_rows}
    verdict_rows, failing = [], []
    for r in df.itertuples():
        row = {"arm": r.arm, "rep": r.rep, "model": r.model}
        checks = {
            "r_hat": _passes(r.max_rhat, CRITERIA["max_rhat"]),
            "ESS": _passes(r.min_ess, CRITERIA["min_ess"]),
            "divergences": _passes(r.divergences, CRITERIA["divergences"]),
        }
        if r.switching:
            checks["a_prob agreement"] = _passes(
                r.a_prob_disagreement, CRITERIA["a_prob_disagreement"]
            )
            checks["acc level 0"] = _passes(r.acc_level0, CRITERIA["acc_level0"])
            checks["acc > 0.05"] = _passes(r.acc_level_high, CRITERIA["acc_level_high"])
            checks["ECE"] = _passes(r.ece, CRITERIA["ece"])
            key = f"{r.arm}/{r.rep}"
            if key in not_worse:
                checks["not worse than v1"] = not_worse[key] == "PASS"
        for k, v in checks.items():
            row[k] = _mark(v)
        arm_ok = all(v is not False for v in checks.values())
        row["arm"] = r.arm
        row["verdict"] = "PASS" if arm_ok else "FAIL"
        if not arm_ok:
            failing.append(f"{r.arm}/{r.rep}")
        verdict_rows.append(row)
    cols = [
        "arm",
        "rep",
        "model",
        "r_hat",
        "ESS",
        "divergences",
        "a_prob agreement",
        "acc level 0",
        "acc > 0.05",
        "ECE",
        "not worse than v1",
        "verdict",
    ]
    lines.append(_md_table(verdict_rows, cols) + "\n")
    if failing:
        lines.append(
            f"**Overall: FAIL.** Failing arms: {', '.join(failing)}. "
            "Nothing was tuned; the numbers above are as produced.\n"
        )
    else:
        lines.append("**Overall: PASS** on every arm.\n")

    lines.append("## Reliability diagrams (v2 arms)\n")
    for r in sw.itertuples():
        fig = getattr(r, "figure", None)
        if isinstance(fig, str):
            lines.append(f"**{r.arm}/{r.rep}**\n\n![{r.arm} {r.rep}]({fig})\n")

    (root / "report.md").write_text("\n".join(lines))
    print(root / "report.md")


if __name__ == "__main__":
    main()
