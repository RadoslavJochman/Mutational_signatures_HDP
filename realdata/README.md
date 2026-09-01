# realdata

Upstream tooling that turns real single-cell sequencing data into the per-node mutation
count matrix Tree-HDP consumes. This is separate from the model and simulator work in
`src/`/`scripts/`/`experiments/`: it does not import from or get imported by them, and its
output is a `mutation_count_matrix.csv`-shaped file (nodes/subclones x 96 trinucleotide
channels) that plugs into the existing `run_inference.py` pipeline the same way a
simulated dataset does.

## What this does

SECEDO (github.com/ratschlab/secedo) clusters single cells from 10x scDNA-seq data into
subclones and calls variants per cluster, given a reference genome. Its companion repo,
secedo-evaluation, documents the recipe for a public triple-negative breast cancer dataset:
download, preprocessing, pileup creation, and running SECEDO. Neither repo produces
signature-ready spectra; the final step, binning each subclone's somatic VCF into a 96
trinucleotide-context count vector, is ours to write.

## Layout

- `external/` -- clones of the SECEDO repos. Not our code, not tracked (see `.gitignore`).
- `scripts/` -- our own code: VCF-to-96-channel binning and any glue between SECEDO's
  outputs and this pipeline's inputs.
- `recipe/` -- the documented, numbered run plan from raw data to a per-cluster somatic
  VCF, with each stage flagged as scripted-in-the-repo or ours to write.
- `data/` -- raw downloads, intermediate pileups/VCFs, and final outputs. Gitignored;
  regenerate from the recipe rather than expecting it to be there.

## Status

Structure and recipe only so far; see `recipe/breast_cancer_plan.md` for the plan and the
data-cost accounting before any download or run.