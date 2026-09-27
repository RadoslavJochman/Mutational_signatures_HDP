"""Compare the full-catalogue signature refit across three confidence
thresholds on the same confidently-present SNV calls, to check whether the
full fit's flat (SBS3, SBS40a) and artefact (SBS95) components come from
weak read support rather than real biology.

Not part of the generate-infer-score pipeline: a one-off check (see the
"Diagnostic scripts" precedent in CLAUDE.md -- interpretation goes to
stdout, the CSV stays clean).

Subsets (each its own spectra.csv, generated on Euler by
realdata/scripts/euler/confidence_subset_spectra.sbatch -- nothing is
reparsed here, this script only refits already-binned spectra; see that
sbatch script's own docstring for exactly which build_snv_tree.py functions
it reuses):

    a   the current default (alt >= 2, vaf >= 0.05) -- spectra.csv as
        stage 8 already writes it.
    b   alt >= 4 (vaf floor unchanged at 0.05) -- drops low-alt-read calls
        classify_snv_state would still call present.
    c   vaf >= 0.15 and alt >= 3 -- a stricter VAF floor on top.

Each subset's spectrum is refit two ways, reusing refit_cosmic.py's own
nnls_fit and forward_selection (not reimplemented):

    fixed_candidate_plus_sbs95   NNLS against the candidate set plus SBS95
                                  (the artefact signature the plain
                                  full-catalogue fit selected -- see
                                  refit_cosmic.py) -- a FIXED basis, so
                                  the same signatures' shares are directly
                                  comparable across subsets a, b, c.
    forward_selection             refit_cosmic.py's own greedy forward
                                  selection over the full catalogue,
                                  reported separately since its own
                                  selected set can differ per subset.

If the SBS95/SBS3/SBS40a shares (from the fixed-basis fit) fall as the
read-support bar rises from subset a to subset c, that is evidence those
components are propped up by weak calls; if they hold steady, the
components look real regardless of the confidence bar.

Outputs (to --outdir)
    confidence_exposures.csv       long format: subset, cluster, fit_set,
                                    signature, exposure_fraction,
                                    cosine_similarity, snv_count
    confidence_selection_order.csv forward-selection fit only: subset,
                                    cluster, step, signature, gain,
                                    cosine_after
    confidence_shares.csv          per cluster: SBS95/SBS3/SBS40a share
                                    (fixed-basis fit) in subsets a, b, c,
                                    and the a -> c change

Usage
-----
    python realdata/scripts/refit_confidence_subsets.py \\
        --subset-a realdata/local_tree_input/confidence_subsets/a/spectra.csv \\
        --subset-b realdata/local_tree_input/confidence_subsets/b/spectra.csv \\
        --subset-c realdata/local_tree_input/confidence_subsets/c/spectra.csv \\
        --catalogue COSMIC_sig/cosmic_v3.4_sbs96_grch37_full.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "realdata" / "scripts"))

from refit_cosmic import (  # noqa: E402
    BREAST_CANDIDATE_SET,
    FORWARD_SELECTION_START,
    MIN_GAIN,
    forward_selection,
    nnls_fit,
    split_available,
)

from src.analysis.analysis import read_data_matrix  # noqa: E402

TRACKED_SIGNATURES = ["SBS95", "SBS3", "SBS40a"]
SUBSETS = ["a", "b", "c"]


def fixed_basis_fit(spectrum, catalogue: pd.DataFrame, extra_signatures: List[str]):
    """NNLS against the candidate set plus ``extra_signatures`` (whichever
    of either actually exist in ``catalogue``) -- a fixed basis, so its
    exposures are directly comparable across subsets."""
    present_candidates, _ = split_available(catalogue, BREAST_CANDIDATE_SET)
    present_extra, _ = split_available(catalogue, extra_signatures)
    names = present_candidates + [
        n for n in present_extra if n not in present_candidates
    ]
    return nnls_fit(spectrum, catalogue.loc[names])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--subset-a",
        type=Path,
        default=Path("realdata/local_tree_input/confidence_subsets/a/spectra.csv"),
    )
    p.add_argument(
        "--subset-b",
        type=Path,
        default=Path("realdata/local_tree_input/confidence_subsets/b/spectra.csv"),
    )
    p.add_argument(
        "--subset-c",
        type=Path,
        default=Path("realdata/local_tree_input/confidence_subsets/c/spectra.csv"),
    )
    p.add_argument(
        "--catalogue",
        type=Path,
        default=Path("COSMIC_sig/cosmic_v3.4_sbs96_grch37_full.csv"),
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("realdata/local_tree_input/confidence_subsets/refit"),
    )
    p.add_argument("--min-gain", type=float, default=MIN_GAIN)
    args = p.parse_args()

    catalogue = pd.read_csv(args.catalogue, index_col=0)
    subset_spectra = {
        "a": read_data_matrix(args.subset_a),
        "b": read_data_matrix(args.subset_b),
        "c": read_data_matrix(args.subset_c),
    }

    exposure_rows = []
    selection_rows = []
    for subset_name, spectra in subset_spectra.items():
        for cluster in spectra.index:
            spectrum = spectra.loc[cluster].to_numpy()
            snv_count = int(spectrum.sum())

            fixed_fractions, fixed_cosine = fixed_basis_fit(
                spectrum, catalogue, ["SBS95"]
            )
            for signature in fixed_fractions.index:
                exposure_rows.append(
                    {
                        "subset": subset_name,
                        "cluster": cluster,
                        "fit_set": "fixed_candidate_plus_sbs95",
                        "signature": signature,
                        "exposure_fraction": float(fixed_fractions[signature]),
                        "cosine_similarity": fixed_cosine,
                        "snv_count": snv_count,
                    }
                )

            selected, order = forward_selection(
                spectrum, catalogue, FORWARD_SELECTION_START, args.min_gain
            )
            for step, entry in enumerate(order, start=1):
                selection_rows.append(
                    {"subset": subset_name, "cluster": cluster, "step": step, **entry}
                )
            fs_fractions, fs_cosine = nnls_fit(spectrum, catalogue.loc[selected])
            fs_fractions = fs_fractions.reindex(catalogue.index, fill_value=0.0)
            for signature in catalogue.index:
                exposure_rows.append(
                    {
                        "subset": subset_name,
                        "cluster": cluster,
                        "fit_set": "forward_selection",
                        "signature": signature,
                        "exposure_fraction": float(fs_fractions[signature]),
                        "cosine_similarity": fs_cosine,
                        "snv_count": snv_count,
                    }
                )

    exposures = pd.DataFrame(exposure_rows)
    selection_order = pd.DataFrame(selection_rows)

    # Tracked signatures' share across subsets, from the fixed-basis fit
    # only (a variable forward-selection basis is not comparable this way).
    fixed = exposures[exposures["fit_set"] == "fixed_candidate_plus_sbs95"]
    share_rows = []
    for cluster in subset_spectra["a"].index:
        row = {"cluster": cluster}
        for signature in TRACKED_SIGNATURES:
            shares = {}
            for subset_name in SUBSETS:
                match = fixed[
                    (fixed["cluster"] == cluster)
                    & (fixed["subset"] == subset_name)
                    & (fixed["signature"] == signature)
                ]
                shares[subset_name] = (
                    float(match["exposure_fraction"].iloc[0]) if len(match) else 0.0
                )
            row[f"{signature}_a"] = shares["a"]
            row[f"{signature}_b"] = shares["b"]
            row[f"{signature}_c"] = shares["c"]
            row[f"{signature}_change_a_to_c"] = shares["c"] - shares["a"]
        share_rows.append(row)
    shares = pd.DataFrame(share_rows)

    args.outdir.mkdir(parents=True, exist_ok=True)
    exposures.to_csv(args.outdir / "confidence_exposures.csv", index=False)
    selection_order.to_csv(args.outdir / "confidence_selection_order.csv", index=False)
    shares.to_csv(args.outdir / "confidence_shares.csv", index=False)

    print("## SNV count per cluster per subset")
    counts = exposures.drop_duplicates(["subset", "cluster"])[
        ["subset", "cluster", "snv_count"]
    ].pivot(index="cluster", columns="subset", values="snv_count")
    print(counts[SUBSETS].to_string())

    print("\n## Tracked signature shares (fixed candidate+SBS95 basis), a -> b -> c")
    print(shares.to_string(index=False))

    print("\n## Forward selection: entered order per subset per cluster")
    for subset_name in SUBSETS:
        print(f"\nsubset {subset_name}:")
        for cluster in subset_spectra[subset_name].index:
            cluster_order = selection_order[
                (selection_order["subset"] == subset_name)
                & (selection_order["cluster"] == cluster)
            ]
            entered = ", ".join(
                f"{r.signature} (+{r.gain:.3f} -> {r.cosine_after:.4f})"
                for r in cluster_order.itertuples()
            )
            print(
                f"  clone{cluster}: {FORWARD_SELECTION_START} then "
                f"{entered or '(nothing else)'}"
            )

    print(
        f"\nWrote confidence_exposures.csv, confidence_selection_order.csv, "
        f"confidence_shares.csv to {args.outdir}."
    )
    print("No signature set is chosen here -- read the shares above and decide.")


if __name__ == "__main__":
    main()
