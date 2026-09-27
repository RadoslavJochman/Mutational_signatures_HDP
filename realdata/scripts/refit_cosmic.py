"""Refit each cluster's observed 96-channel spectrum against COSMIC
signatures, to help decide which signatures the real slice D run should
actually condition on.

Not part of the generate-infer-score pipeline: a one-off check before
choosing `simulation.repertoire.signatures` for a real-data config (see the
"Diagnostic scripts" precedent in CLAUDE.md -- interpretation goes to
stdout, the CSV stays clean).

Method
------
Two fits per cluster, both by non-negative least squares (`scipy.optimize.
nnls`: exposures x >= 0 minimising ||spectrum - x @ signatures||):

    candidate   the triple-negative breast repertoire named in the plan
                (SBS1, SBS2, SBS3, SBS5, SBS8, SBS13, SBS17a, SBS17b, SBS18,
                SBS40a), restricted to whichever of these actually exist in
                cosmic_signatures.csv -- report the rest as missing, do not
                fail on them.
    full        every signature in cosmic_signatures.csv, to catch anything
                the candidate set misses (including sequencing-artefact
                signatures COSMIC also catalogues).

Exposures are reported as fractions of the fitted total (zero if the fit is
degenerate), alongside the cosine similarity between the observed spectrum
and the fit's reconstruction. Per-signature stability comes from a
multinomial bootstrap: resample the observed spectrum's total mutation
count from its own channel proportions `--n-bootstrap` times (200), refit
each replicate, and report the fraction of replicates in which that
signature's exposure exceeds `--stability-threshold` (5%) -- a signature
that is only sometimes needed to explain sampling noise is unstable, not
load-bearing.

Two flags follow directly from these fits, printed, never chosen here:
a signature above the stability threshold in the full fit but absent from
the candidate set (the candidate repertoire may be missing something real),
and a candidate-set signature that never clears the threshold in any
cluster's candidate fit (the candidate repertoire may be carrying dead
weight). Cluster 10 (the largest private SNV count in the plan) gets a
direct comparison against the other clusters' spectra, since a real
biological difference there would need its own signature rather than
forcing it onto the shared repertoire.

Outputs (to --outdir)
    refit_exposures.csv   long format: cluster, fit_set, signature,
                           exposure_fraction, bootstrap_stability,
                           cosine_similarity (repeated per row of its
                           cluster/fit_set group, for a self-contained row)
    refit_summary.csv     per cluster/fit_set: cosine_similarity,
                           n_signatures_above_threshold

Usage
-----
    python realdata/scripts/refit_cosmic.py \\
        --spectra realdata/local_tree_input/spectra.csv \\
        --cosmic-signatures COSMIC_sig/cosmic_signatures.csv \\
        --outdir realdata/local_tree_input/refit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.analysis import cosine, read_data_matrix  # noqa: E402

BREAST_CANDIDATE_SET = [
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
N_BOOTSTRAP = 200
STABILITY_THRESHOLD = 0.05


def split_available(
    cosmic: pd.DataFrame, names: List[str]
) -> Tuple[List[str], List[str]]:
    """``(present, missing)``: which of ``names`` are actually rows of
    ``cosmic`` (by signature name), preserving ``names``' own order."""
    present = [n for n in names if n in cosmic.index]
    missing = [n for n in names if n not in cosmic.index]
    return present, missing


def nnls_fit(spectrum: np.ndarray, signatures: pd.DataFrame) -> Tuple[pd.Series, float]:
    """Fit ``spectrum`` (96,) as a non-negative combination of
    ``signatures``' rows. Returns (exposure fractions indexed by signature
    name, summing to 1 unless the fit is degenerate; cosine similarity
    between ``spectrum`` and the reconstruction)."""
    basis = signatures.to_numpy().T  # (96, K): one column per signature
    exposures, _residual = nnls(basis, spectrum.astype(float))
    total = exposures.sum()
    fractions = exposures / total if total > 0 else exposures
    reconstruction = basis @ exposures
    similarity = cosine(spectrum, reconstruction)
    return pd.Series(fractions, index=signatures.index), similarity


def bootstrap_stability(
    spectrum: np.ndarray,
    signatures: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
    threshold: float = STABILITY_THRESHOLD,
    seed: int = 0,
) -> pd.Series:
    """Fraction of ``n_bootstrap`` multinomial resamples of ``spectrum``
    (same total count, drawn from its own channel proportions) whose NNLS
    exposure fraction for each signature exceeds ``threshold``. Zero
    resamples requested, or an all-zero spectrum, gives all-zero stability
    rather than dividing by zero.
    """
    n_total = int(round(spectrum.sum()))
    signature_names = list(signatures.index)
    if n_bootstrap <= 0 or n_total <= 0:
        return pd.Series(np.zeros(len(signature_names)), index=signature_names)
    rng = np.random.default_rng(seed)
    probs = spectrum / spectrum.sum()
    above = np.zeros(len(signature_names))
    for _ in range(n_bootstrap):
        resampled = rng.multinomial(n_total, probs)
        fractions, _ = nnls_fit(resampled, signatures)
        above += (fractions.to_numpy() > threshold).astype(float)
    return pd.Series(above / n_bootstrap, index=signature_names)


def fit_cluster(
    spectrum: np.ndarray,
    signatures: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
    threshold: float = STABILITY_THRESHOLD,
    seed: int = 0,
) -> Tuple[pd.Series, float, pd.Series]:
    """``(exposure_fractions, cosine_similarity, bootstrap_stability)`` for
    one cluster against one signature set -- the fit plus its stability,
    computed together since the stability bootstrap needs the same fit
    function."""
    fractions, similarity = nnls_fit(spectrum, signatures)
    stability = bootstrap_stability(spectrum, signatures, n_bootstrap, threshold, seed)
    return fractions, similarity, stability


def flag_missing_from_candidate(
    exposures: pd.DataFrame,
    candidate_names: List[str],
    threshold: float = STABILITY_THRESHOLD,
) -> List[Tuple[str, str, float]]:
    """``(cluster, signature, exposure_fraction)`` triples where the FULL
    fit needed a signature above ``threshold`` that the candidate set does
    not have at all."""
    full = exposures[exposures["fit_set"] == "full"]
    flagged = full[
        (full["exposure_fraction"] > threshold)
        & (~full["signature"].isin(candidate_names))
    ]
    return list(
        flagged[["cluster", "signature", "exposure_fraction"]].itertuples(
            index=False, name=None
        )
    )


def flag_dead_weight_candidates(
    exposures: pd.DataFrame, threshold: float = STABILITY_THRESHOLD
) -> List[str]:
    """Candidate-set signatures whose candidate-fit exposure never exceeds
    ``threshold`` in ANY cluster -- carried by the candidate repertoire but
    never actually used."""
    candidate = exposures[exposures["fit_set"] == "candidate"]
    max_by_sig = candidate.groupby("signature")["exposure_fraction"].max()
    return sorted(max_by_sig.index[max_by_sig <= threshold])


def compare_cluster_to_others(
    spectra: pd.DataFrame, cluster: str
) -> Tuple[float, float]:
    """``(cosine to the mean of the other clusters, mean pairwise cosine
    among the other clusters)`` -- both on channel-proportion-normalised
    spectra, so a cluster's own total SNV count does not skew the
    comparison. The second number is the baseline typical similarity;
    ``cluster`` is an outlier if its own similarity to the others falls
    well below it.
    """
    proportions = spectra.div(spectra.sum(axis=1), axis=0)
    others = proportions.drop(index=cluster)
    to_mean = cosine(
        proportions.loc[cluster].to_numpy(), others.mean(axis=0).to_numpy()
    )
    pairwise = [
        cosine(others.loc[a].to_numpy(), others.loc[b].to_numpy())
        for i, a in enumerate(others.index)
        for b in others.index[i + 1 :]
    ]
    baseline = float(np.mean(pairwise)) if pairwise else float("nan")
    return to_mean, baseline


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--spectra", type=Path, default=Path("realdata/local_tree_input/spectra.csv")
    )
    p.add_argument(
        "--cosmic-signatures",
        type=Path,
        default=Path("COSMIC_sig/cosmic_signatures.csv"),
    )
    p.add_argument(
        "--outdir", type=Path, default=Path("realdata/local_tree_input/refit")
    )
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--stability-threshold", type=float, default=STABILITY_THRESHOLD)
    p.add_argument("--focus-cluster", default="10")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    spectra = read_data_matrix(args.spectra)
    cosmic = pd.read_csv(args.cosmic_signatures, index_col=0)

    present_candidates, missing_candidates = split_available(
        cosmic, BREAST_CANDIDATE_SET
    )
    print("## Candidate set availability")
    print(f"present in cosmic_signatures.csv: {present_candidates}")
    if missing_candidates:
        print(f"MISSING from cosmic_signatures.csv: {missing_candidates}")
    candidate_signatures = cosmic.loc[present_candidates]

    rows = []
    summary_rows = []
    for cluster in spectra.index:
        spectrum = spectra.loc[cluster].to_numpy()
        for fit_name, signatures in (
            ("candidate", candidate_signatures),
            ("full", cosmic),
        ):
            fractions, similarity, stability = fit_cluster(
                spectrum,
                signatures,
                n_bootstrap=args.n_bootstrap,
                threshold=args.stability_threshold,
                seed=args.seed,
            )
            n_above = int((fractions > args.stability_threshold).sum())
            summary_rows.append(
                {
                    "cluster": cluster,
                    "fit_set": fit_name,
                    "cosine_similarity": similarity,
                    "n_signatures_above_threshold": n_above,
                }
            )
            for signature in signatures.index:
                rows.append(
                    {
                        "cluster": cluster,
                        "fit_set": fit_name,
                        "signature": signature,
                        "exposure_fraction": float(fractions[signature]),
                        "bootstrap_stability": float(stability[signature]),
                        "cosine_similarity": similarity,
                    }
                )

    exposures = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows)

    args.outdir.mkdir(parents=True, exist_ok=True)
    exposures.to_csv(args.outdir / "refit_exposures.csv", index=False)
    summary.to_csv(args.outdir / "refit_summary.csv", index=False)

    print("\n## Per-cluster fit summary")
    print(summary.to_string(index=False))

    print("\n## Per-cluster exposures above threshold (candidate fit)")
    candidate_exposures = exposures[exposures["fit_set"] == "candidate"]
    for cluster in spectra.index:
        row = candidate_exposures[candidate_exposures["cluster"] == cluster]
        above = row[row["exposure_fraction"] > args.stability_threshold]
        parts = ", ".join(
            f"{r.signature}={r.exposure_fraction:.2f} "
            f"(stability {r.bootstrap_stability:.2f})"
            for r in above.itertuples()
        )
        print(f"  clone{cluster}: {parts or '(none above threshold)'}")

    missing_flags = flag_missing_from_candidate(
        exposures, present_candidates, args.stability_threshold
    )
    print(
        "\n## Flag: signature above threshold in the full fit, "
        "absent from candidate set"
    )
    if missing_flags:
        for cluster, signature, fraction in missing_flags:
            print(f"  clone{cluster}: {signature} = {fraction:.2f}")
    else:
        print("  none")

    dead_weight = flag_dead_weight_candidates(exposures, args.stability_threshold)
    print("\n## Flag: candidate-set signature never above threshold in any cluster")
    print(f"  {dead_weight or '(none)'}")

    if args.focus_cluster in spectra.index:
        to_mean, baseline = compare_cluster_to_others(spectra, args.focus_cluster)
        print(f"\n## Cluster {args.focus_cluster} vs the others")
        print(
            f"  cosine(clone{args.focus_cluster}, mean of others) = {to_mean:.4f}; "
            f"mean pairwise cosine among the OTHER clusters = {baseline:.4f}"
        )
        if to_mean < baseline - 0.02:
            print(
                f"  clone{args.focus_cluster} is noticeably less similar to the "
                "others than they are to each other -- consistent with a "
                "distinct signature profile, not just more mutations."
            )
        else:
            print(
                f"  clone{args.focus_cluster} is about as similar to the others "
                "as they are to each other -- its private SNV count looks like "
                "more of the same profile, not a distinct one."
            )
    else:
        print(
            f"\ncluster {args.focus_cluster!r} not found in spectra; "
            "skipping the comparison"
        )

    print(f"\nWrote refit_exposures.csv and refit_summary.csv to {args.outdir}.")
    print("No signature set is chosen here -- read the flags above and decide.")


if __name__ == "__main__":
    main()
