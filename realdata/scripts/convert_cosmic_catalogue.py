"""Convert COSMIC's full SBS96 reference catalogue into this repo's
positional Channel_0..Channel_95 order (rows = signatures, matching
COSMIC_sig/cosmic_signatures.csv's own layout), and validate it against
that file for every signature the two share.

Not part of the generate-infer-score pipeline: a one-off conversion run
once the raw catalogue is in hand, kept for the record (see the
"Diagnostic scripts" precedent in CLAUDE.md -- interpretation goes to
stdout, the CSV stays clean).

WHY A SEPARATE FILE, NOT cosmic_signatures.csv ITSELF
------------------------------------------------------
cosmic_signatures.csv is the small, fixed signature set the simulations are
pinned to (CLAUDE.md: do not change it). This script only ever reads it,
for the validation step, and writes the full catalogue to its own file.

SOURCE
------
COSMIC Mutational Signatures v3.4, SBS, GRCh37 -- the genome build slice D
and this repo's Mutect2 calling both use. Fetched directly, no login
needed (confirmed: the downloads page states this signatures module does
not require one):

    https://cancer.sanger.ac.uk/signatures/documents/2123/COSMIC_v3.4_SBS_GRCh37.txt
    (redirects to
    https://cog.sanger.ac.uk/cosmic-signatures-production/documents/COSMIC_v3.4_SBS_GRCh37.txt)

See COSMIC_sig/README_full_catalogue.md for the fetch date and the
licence note this download carries -- COSMIC's data is free for academic,
non-commercial use, but redistribution (e.g. committing the converted copy
here) is not addressed by that summary and is not assumed cleared; the raw
and converted files are gitignored until confirmed.

CONVERSION
----------
The raw file is (96 rows x 86 columns): one "Type" column (COSMIC's own
"A[C>A]A" .. "T[T>G]T" trinucleotide labels) plus one column per signature
-- rows are channels, columns are signatures, the transpose of this repo's
convention. Each Type label is parsed into (five, ref, alt, three) and
passed through build_snv_tree.py's own `pyrimidine_normalise` +
`channel_index` (imported, not reimplemented) to place it at the correct
position on the 96-wide Channel_0..Channel_95 axis -- keyed by the label
itself, never by the raw file's row order (which happens to already be
alphabetical, i.e. already in the target order, but this script does not
rely on that coincidence).

Usage
-----
    python realdata/scripts/convert_cosmic_catalogue.py \\
        --raw COSMIC_sig/COSMIC_v3.4_SBS_GRCh37.txt \\
        --existing COSMIC_sig/cosmic_signatures.csv \\
        --out COSMIC_sig/cosmic_v3.4_sbs96_grch37_full.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "realdata" / "scripts" / "euler"))

import build_snv_tree as bt  # noqa: E402

from src.analysis.analysis import cosine  # noqa: E402

TYPE_PATTERN = re.compile(r"^([ACGT])\[([ACGT])>([ACGT])\]([ACGT])$")


def parse_type_label(label: str) -> int:
    """Channel index for one COSMIC ``Type`` label (e.g. ``"A[C>A]A"``),
    via build_snv_tree.py's own ``pyrimidine_normalise`` + ``channel_index``
    -- never by row position."""
    match = TYPE_PATTERN.match(label)
    if not match:
        raise ValueError(f"unrecognised COSMIC Type label: {label!r}")
    five, ref, alt, three = match.groups()
    five, ref, alt, three = bt.pyrimidine_normalise(five, ref, alt, three)
    return bt.channel_index(five, ref, alt, three)


def convert_catalogue(raw: pd.DataFrame) -> pd.DataFrame:
    """``raw`` is COSMIC's own layout (index = Type label, columns =
    signature names). Returns (signatures x Channel_0..Channel_95), each
    Type's row placed by its own label's channel index, never by position.
    """
    channel_of = {label: parse_type_label(label) for label in raw.index}
    n_distinct = len(set(channel_of.values()))
    if n_distinct != bt.N_CHANNELS:
        raise ValueError(
            f"expected {bt.N_CHANNELS} distinct channels, got {n_distinct} "
            f"from {len(raw.index)} Type labels -- a label parsed to the "
            "same channel as another, or the file is not the expected shape"
        )
    out = pd.DataFrame(index=raw.columns, columns=bt.channel_labels(), dtype=float)
    for label, channel in channel_of.items():
        out.iloc[:, channel] = raw.loc[label].to_numpy()
    out.index.name = None
    return out


def compare_to_existing(
    converted: pd.DataFrame, existing: pd.DataFrame
) -> pd.DataFrame:
    """Cosine similarity per signature name shared by both catalogues."""
    shared = sorted(set(converted.index) & set(existing.index))
    rows = [
        {
            "signature": name,
            "cosine_to_full_catalogue": cosine(
                existing.loc[name].to_numpy(), converted.loc[name].to_numpy()
            ),
        }
        for name in shared
    ]
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw", type=Path, required=True, help="COSMIC's own raw SBS GRCh37 txt file"
    )
    p.add_argument(
        "--existing", type=Path, default=Path("COSMIC_sig/cosmic_signatures.csv")
    )
    p.add_argument(
        "--out", type=Path, default=Path("COSMIC_sig/cosmic_v3.4_sbs96_grch37_full.csv")
    )
    args = p.parse_args()

    raw = pd.read_csv(args.raw, sep="\t", index_col=0)
    converted = convert_catalogue(raw)

    existing = pd.read_csv(args.existing, index_col=0)
    comparison = compare_to_existing(converted, existing)
    print(
        "## Cosine similarity: cosmic_signatures.csv vs the full v3.4 GRCh37 catalogue"
    )
    print(comparison.sort_values("signature").to_string(index=False))

    missing_from_full = sorted(set(existing.index) - set(converted.index))
    if missing_from_full:
        print(
            "\nin cosmic_signatures.csv but absent from the full v3.4 catalogue "
            f"(a later COSMIC release, presumably): {missing_from_full}"
        )
    low_similarity = comparison[comparison["cosine_to_full_catalogue"] < 0.999]
    if len(low_similarity):
        print("\nSHARED signatures that do not match closely (cosine < 0.999):")
        print(low_similarity.to_string(index=False))
    else:
        print("\nevery shared signature matches to floating-point precision")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    converted.to_csv(args.out)
    print(
        f"\nWrote {converted.shape[0]} signatures x {converted.shape[1]} channels "
        f"to {args.out}"
    )


if __name__ == "__main__":
    main()
