"""Refit each cluster's observed 96-channel spectrum against COSMIC
signatures, to help decide which signatures the real slice D run should
actually condition on.

Not part of the generate-infer-score pipeline: a one-off check before
choosing `simulation.repertoire.signatures` for a real-data config (see the
"Diagnostic scripts" precedent in CLAUDE.md -- interpretation goes to
stdout, the CSV stays clean).

Method
------
Two fits per cluster:

    candidate         non-negative least squares (`scipy.optimize.nnls`:
                       exposures x >= 0 minimising ||spectrum - x @
                       signatures||) against the triple-negative breast
                       repertoire named in the plan (SBS1, SBS2, SBS3, SBS5,
                       SBS8, SBS13, SBS17a, SBS17b, SBS18, SBS40a),
                       restricted to whichever of these actually exist in
                       the catalogue in use -- report the rest as missing,
                       do not fail on them.
    full (forward     against the full catalogue (`--catalogue`, the
    selection)         converted COSMIC v3.4 SBS GRCh37 file by default --
                       see COSMIC_sig/README_full_catalogue.md), NOT plain
                       NNLS over all ~85 signatures at once (an
                       underdetermined fit against 96 channels): starting
                       from SBS1 and SBS5, greedily add whichever remaining
                       signature most improves NNLS cosine similarity,
                       stopping once the best available gain falls below
                       `--min-gain` (0.01). Reported in
                       refit_selection_order.csv: which signature entered
                       at which step, its gain, and the cosine after adding
                       it -- the direct evidence for whether a signature is
                       load-bearing or marginal.

Exposures are reported as fractions of the fitted total (zero if the fit is
degenerate, or if a full-catalogue signature was never selected), alongside
the cosine similarity between the observed spectrum and the fit's
reconstruction. Per-signature stability comes from a multinomial bootstrap:
resample the observed spectrum's total mutation count from its own channel
proportions `--n-bootstrap` times (200), refit each replicate against the
SAME signature set the cluster's own fit landed on (the candidate set, or
the full fit's own selected subset -- forward selection itself is not
rerun per replicate), and report the fraction of replicates in which that
signature's exposure exceeds `--stability-threshold` (5%).

Two flags follow directly from these fits, printed, never chosen here:
a signature above the stability threshold in the full fit but absent from
the candidate set (the candidate repertoire may be missing something
real), and a candidate-set signature that never clears the threshold in
any cluster's candidate fit (the candidate repertoire may be carrying dead
weight). Any selected full-fit signature that is a known COSMIC sequencing
artefact (`KNOWN_ARTEFACT_SIGNATURES`) is reported separately, per cluster
-- selection does not mean acceptance. Cluster 10 (the largest private SNV
count in the plan) gets a direct comparison against the other clusters'
spectra, since a real biological difference there would need its own
signature rather than forcing it onto the shared repertoire.

Also reports pairwise cosine similarity among `--collinearity-signatures`
(default SBS1, SBS3, SBS5, SBS18, SBS40a, SBS95, SBS2, SBS13, from
`--catalogue`), flagging pairs above `--collinearity-threshold` (0.8) --
a check on whether the candidate/full fits' components could be
substituting for each other rather than each explaining something real.

Outputs (to --outdir)
    refit_collinearity.csv    the pairwise cosine matrix among
                               --collinearity-signatures
    refit_exposures.csv       long format: cluster, fit_set, signature,
                               exposure_fraction, bootstrap_stability,
                               cosine_similarity (repeated per row of its
                               cluster/fit_set group, for a self-contained
                               row)
    refit_summary.csv         per cluster/fit_set: cosine_similarity,
                               n_signatures_above_threshold
    refit_selection_order.csv full-fit forward selection only: cluster,
                               step, signature, gain, cosine_after

Usage
-----
    python realdata/scripts/refit_cosmic.py \\
        --spectra realdata/local_tree_input/spectra.csv \\
        --catalogue COSMIC_sig/cosmic_v3.4_sbs96_grch37_full.csv \\
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

# COSMIC's own "Proposed aetiology: Possible sequencing artefact" signatures
# (cancer.sanger.ac.uk/signatures/sbs/, checked 2026-09-27). Not a control
# value here -- reported alongside a selected signature, never used to
# exclude it; COSMIC's own list may grow with later releases.
KNOWN_ARTEFACT_SIGNATURES = [
    "SBS27",
    "SBS43",
    "SBS45",
    "SBS46",
    "SBS47",
    "SBS48",
    "SBS49",
    "SBS50",
    "SBS51",
    "SBS52",
    "SBS53",
    "SBS54",
    "SBS55",
    "SBS56",
    "SBS57",
    "SBS58",
    "SBS59",
    "SBS60",
    "SBS95",
]

FORWARD_SELECTION_START = ["SBS1", "SBS5"]
N_BOOTSTRAP = 200
STABILITY_THRESHOLD = 0.05
MIN_GAIN = 0.01
COLLINEARITY_THRESHOLD = 0.8
COLLINEARITY_SIGNATURES = [
    "SBS1",
    "SBS3",
    "SBS5",
    "SBS18",
    "SBS40a",
    "SBS95",
    "SBS2",
    "SBS13",
]


def split_available(
    cosmic: pd.DataFrame, names: List[str]
) -> Tuple[List[str], List[str]]:
    """``(present, missing)``: which of ``names`` are actually rows of
    ``cosmic`` (by signature name), preserving ``names``' own order."""
    present = [n for n in names if n in cosmic.index]
    missing = [n for n in names if n not in cosmic.index]
    return present, missing


def pairwise_signature_cosine(
    catalogue: pd.DataFrame, names: List[str]
) -> pd.DataFrame:
    """Symmetric cosine-similarity matrix among ``names``' rows in
    ``catalogue`` (whichever of them are actually present -- missing names
    are dropped, not raised on). Diagonal is 1.0 by construction."""
    present = [n for n in names if n in catalogue.index]
    matrix = pd.DataFrame(index=present, columns=present, dtype=float)
    for a in present:
        for b in present:
            matrix.loc[a, b] = cosine(
                catalogue.loc[a].to_numpy(), catalogue.loc[b].to_numpy()
            )
    return matrix


def flag_collinear_pairs(
    matrix: pd.DataFrame, threshold: float = COLLINEARITY_THRESHOLD
) -> List[Tuple[str, str, float]]:
    """``(signature_a, signature_b, cosine)`` for every off-diagonal pair
    whose similarity exceeds ``threshold``, each pair reported once."""
    names = list(matrix.index)
    return [
        (names[i], names[j], float(matrix.iloc[i, j]))
        for i in range(len(names))
        for j in range(i + 1, len(names))
        if matrix.iloc[i, j] > threshold
    ]


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


def forward_selection(
    spectrum: np.ndarray,
    signatures: pd.DataFrame,
    start: List[str] = FORWARD_SELECTION_START,
    min_gain: float = MIN_GAIN,
) -> Tuple[List[str], List[dict]]:
    """Greedily grow a signature subset for ``spectrum`` against the full
    ``signatures`` catalogue: always start from ``start`` (SBS1, SBS5), then
    repeatedly add whichever remaining signature most improves NNLS cosine
    similarity, stopping once the best available gain is below
    ``min_gain``. Plain NNLS over the whole catalogue at once is
    underdetermined against 96 channels; this is the fix.

    Returns ``(selected, order)``: ``selected`` is ``start`` plus every
    signature added, in entry order; ``order`` is one dict per addition
    (``signature``, ``gain``, ``cosine_after``), NOT including ``start``
    itself (there is nothing to compare its addition against).
    """
    selected = list(start)
    remaining = [name for name in signatures.index if name not in selected]
    _, cosine_so_far = nnls_fit(spectrum, signatures.loc[selected])
    order: List[dict] = []
    while remaining:
        best_signature, best_cosine = None, cosine_so_far
        for candidate in remaining:
            _, trial_cosine = nnls_fit(spectrum, signatures.loc[selected + [candidate]])
            if trial_cosine > best_cosine:
                best_signature, best_cosine = candidate, trial_cosine
        if best_signature is None or best_cosine - cosine_so_far < min_gain:
            break
        selected.append(best_signature)
        remaining.remove(best_signature)
        order.append(
            {
                "signature": best_signature,
                "gain": best_cosine - cosine_so_far,
                "cosine_after": best_cosine,
            }
        )
        cosine_so_far = best_cosine
    return selected, order


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
        "--catalogue",
        type=Path,
        default=Path("COSMIC_sig/cosmic_v3.4_sbs96_grch37_full.csv"),
        help="the full signature catalogue for the candidate-availability "
        "check and the forward-selection full fit",
    )
    p.add_argument(
        "--outdir", type=Path, default=Path("realdata/local_tree_input/refit")
    )
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--stability-threshold", type=float, default=STABILITY_THRESHOLD)
    p.add_argument("--min-gain", type=float, default=MIN_GAIN)
    p.add_argument("--focus-cluster", default="10")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--collinearity-signatures",
        nargs="+",
        default=COLLINEARITY_SIGNATURES,
        help="signatures to report a pairwise cosine matrix for, from "
        "--catalogue, before fitting anything",
    )
    p.add_argument(
        "--collinearity-threshold", type=float, default=COLLINEARITY_THRESHOLD
    )
    args = p.parse_args()

    spectra = read_data_matrix(args.spectra)
    catalogue = pd.read_csv(args.catalogue, index_col=0)

    collinearity = pairwise_signature_cosine(catalogue, args.collinearity_signatures)
    print("## Pairwise signature cosine similarity (collinearity check)")
    print(collinearity.round(3).to_string())
    collinear_pairs = flag_collinear_pairs(collinearity, args.collinearity_threshold)
    print(f"\nFlag: pairs above {args.collinearity_threshold}")
    if collinear_pairs:
        for a, b, sim in collinear_pairs:
            print(f"  {a} / {b}: {sim:.3f}")
    else:
        print("  none")

    present_candidates, missing_candidates = split_available(
        catalogue, BREAST_CANDIDATE_SET
    )
    print("\n## Candidate set availability")
    print(f"present in {args.catalogue.name}: {present_candidates}")
    if missing_candidates:
        print(f"MISSING from {args.catalogue.name}: {missing_candidates}")
    candidate_signatures = catalogue.loc[present_candidates]

    rows = []
    summary_rows = []
    selection_rows = []
    for cluster in spectra.index:
        spectrum = spectra.loc[cluster].to_numpy()

        # candidate: plain NNLS over the (small) restricted repertoire.
        fractions, similarity, stability = fit_cluster(
            spectrum,
            candidate_signatures,
            n_bootstrap=args.n_bootstrap,
            threshold=args.stability_threshold,
            seed=args.seed,
        )
        summary_rows.append(
            {
                "cluster": cluster,
                "fit_set": "candidate",
                "cosine_similarity": similarity,
                "n_signatures_above_threshold": int(
                    (fractions > args.stability_threshold).sum()
                ),
            }
        )
        for signature in candidate_signatures.index:
            rows.append(
                {
                    "cluster": cluster,
                    "fit_set": "candidate",
                    "signature": signature,
                    "exposure_fraction": float(fractions[signature]),
                    "bootstrap_stability": float(stability[signature]),
                    "cosine_similarity": similarity,
                }
            )

        # full: forward selection over the whole catalogue (plain NNLS over
        # ~85 signatures against 96 channels is underdetermined), then the
        # bootstrap over the resulting fixed subset only.
        selected, order = forward_selection(
            spectrum, catalogue, FORWARD_SELECTION_START, args.min_gain
        )
        for step, entry in enumerate(order, start=1):
            selection_rows.append({"cluster": cluster, "step": step, **entry})
        full_fractions, full_similarity = nnls_fit(spectrum, catalogue.loc[selected])
        full_stability = bootstrap_stability(
            spectrum,
            catalogue.loc[selected],
            n_bootstrap=args.n_bootstrap,
            threshold=args.stability_threshold,
            seed=args.seed,
        )
        full_fractions = full_fractions.reindex(catalogue.index, fill_value=0.0)
        full_stability = full_stability.reindex(catalogue.index, fill_value=0.0)
        summary_rows.append(
            {
                "cluster": cluster,
                "fit_set": "full",
                "cosine_similarity": full_similarity,
                "n_signatures_above_threshold": int(
                    (full_fractions > args.stability_threshold).sum()
                ),
            }
        )
        for signature in catalogue.index:
            rows.append(
                {
                    "cluster": cluster,
                    "fit_set": "full",
                    "signature": signature,
                    "exposure_fraction": float(full_fractions[signature]),
                    "bootstrap_stability": float(full_stability[signature]),
                    "cosine_similarity": full_similarity,
                }
            )

    exposures = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows)
    selection_order = pd.DataFrame(selection_rows)

    args.outdir.mkdir(parents=True, exist_ok=True)
    collinearity.to_csv(args.outdir / "refit_collinearity.csv")
    exposures.to_csv(args.outdir / "refit_exposures.csv", index=False)
    summary.to_csv(args.outdir / "refit_summary.csv", index=False)
    selection_order.to_csv(args.outdir / "refit_selection_order.csv", index=False)

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

    print(
        f"\n## Full-fit forward selection (start {FORWARD_SELECTION_START}, "
        f"min gain {args.min_gain})"
    )
    all_selected: set = set()
    for cluster in spectra.index:
        cluster_order = selection_order[selection_order["cluster"] == cluster]
        entered = ", ".join(
            f"{r.signature} (+{r.gain:.3f} -> {r.cosine_after:.4f})"
            for r in cluster_order.itertuples()
        )
        print(
            f"  clone{cluster}: {FORWARD_SELECTION_START} then "
            f"{entered or '(nothing else)'}"
        )
        all_selected.update(FORWARD_SELECTION_START)
        all_selected.update(cluster_order["signature"])

    print(f"\nselected in at least one cluster: {sorted(all_selected)}")

    artefacts_selected = sorted(all_selected & set(KNOWN_ARTEFACT_SIGNATURES))
    print("\n## Flag: known sequencing-artefact signature selected in the full fit")
    print(f"  {artefacts_selected or '(none)'}")

    for name in ("SBS4", "SBS105"):
        if name not in catalogue.index:
            print(
                f"\n{name}: absent from the full v3.4 catalogue -- cannot be selected"
            )
        elif name in all_selected:
            print(f"\n{name}: SURVIVES -- selected in at least one cluster's full fit")
        else:
            print(f"\n{name}: does not survive -- never selected in any cluster")

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
