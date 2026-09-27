# Full COSMIC v3.4 SBS GRCh37 catalogue

Not `cosmic_signatures.csv` (the small, fixed set the simulations are pinned to --
CLAUDE.md: do not change it). This is the full COSMIC SBS reference catalogue, kept
separately, fetched for the real slice D signature refit
(`realdata/scripts/refit_cosmic.py`).

## Source

COSMIC Mutational Signatures v3.4, SBS, GRCh37 -- the genome build slice D and this
repo's Mutect2 calling both use. Fetched directly, no login needed (the downloads page
states this signatures module does not require one):

```
https://cancer.sanger.ac.uk/signatures/documents/2123/COSMIC_v3.4_SBS_GRCh37.txt
```

which redirects to COSMIC's own storage:

```
https://cog.sanger.ac.uk/cosmic-signatures-production/documents/COSMIC_v3.4_SBS_GRCh37.txt
```

Fetched 2026-09-27. 96 rows (COSMIC's own `Type` trinucleotide labels, `A[C>A]A` ..
`T[T>G]T`) x 86 columns (one `Type` column plus 85 SBS signatures -- SBS1..SBS99 minus
gaps, e.g. SBS3 and SBS7a exist, some numbers such as SBS61-83 do not).

## Files

- `COSMIC_v3.4_SBS_GRCh37.txt` -- the raw download, unmodified, COSMIC's own layout
  (rows = channels, columns = signatures).
- `cosmic_v3.4_sbs96_grch37_full.csv` -- converted to this repo's layout (rows =
  signatures, columns = `Channel_0`..`Channel_95` in `cosmic_signatures.csv`'s positional
  order), by `realdata/scripts/convert_cosmic_catalogue.py`, keyed by each row's own
  `Type` label via `build_snv_tree.py`'s `pyrimidine_normalise` + `channel_index` --
  never by row position.

## Licence -- not yet cleared for committing either file

COSMIC's licensing page (`cancer.sanger.ac.uk/cosmic/license`, redirecting to
`cosmickb.org/licensing`) states the data is free for the academic community
("educating and conducting research in non-profit settings"), with a separate
commercial licence required for commercial products, services, or for-profit R&D.
It does not address redistribution (e.g. committing a derived copy into a git
repository) one way or the other.

Both files are therefore gitignored (`realdata/local_tree_input/`-style: real
downloaded data, not committed pending a decision) until confirmed either way. If
you have an institutional COSMIC agreement or another basis for committing them,
say so and this note updates; otherwise treat both as local-only.

## Validation against cosmic_signatures.csv

`convert_cosmic_catalogue.py` compares every signature the two files share by cosine
similarity, after conversion. Result (2026-09-27 fetch): all 8 shared signatures
(SBS1, SBS4, SBS5, SBS36, SBS37, SBS40a, SBS44, SBS92) match to floating-point
exactness (cosine 1.0). `SBS105` and `SBS112` are in `cosmic_signatures.csv` but not in
v3.4 at all (presumably added in a later COSMIC release) -- not a mismatch, an absence.
