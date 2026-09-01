# Breast cancer dataset: raw 10x data to signature-ready spectra

Read in full from `realdata/external/secedo-evaluation/breast_cancer/` (README, the three
`preprocessing/*.sh` scripts, `clustering.sh`, `mutect.sh`, `split.sh`) and
`realdata/external/secedo/README.md`. Numbered stages below; each flagged **[scripted]**
(runs as-is, LSF-flavoured but adaptable) or **[ours]** (nothing in either repo does this).

Four things this reading corrected against what we assumed going in, flagged here rather
than smoothed over -- see "Corrections" at the end before reading the stages.

## Stages

1. **[scripted]** Download the 5 BAM+BAI slices (10x breast tissue nuclei, sections A-E,
   ~2000 cells each). One `wget` loop in `breast_cancer/README.md`. See data cost below.

2. **[scripted, but unused downstream]** Download the CHISEL clone/copy-number mapping
   data (`mapping_<section>.tsv.gz`) from the `chisel-data` GitHub repo. Also one `wget`
   loop in the README. Nothing in this repo's scripts reads these files -- see
   Corrections.

3. **[scripted]** Preprocessing, three steps, each a standalone script taking the previous
   step's output:
   - `preprocessing/1_run_filtering.sh <bam>`: `samtools view -f 0x2 -F 0x500`, keeping
     properly-paired, primary, non-duplicate reads.
   - `preprocessing/2_run_filteringCB.sh <bam>`: drops reads without a 10x cell-barcode
     (`CB`) tag.
   - `preprocessing/3_run_splittingByCB.sh <bam>` + `split_by_CBtag.py`: sorts by `CB` tag
     and splits into ~2000 per-cell BAMs (`cell_bams/`).
   - **[ours, small]** step 3 reads a `PER_CELL_SUMMARY_FILE` (`*_summary_metrics.csv`,
     an allowed-barcode list) that no script here generates -- it's expected to already
     exist, presumably Cell Ranger's own per-cell summary shipped alongside the 10x
     dataset. We'd need to source or reconstruct this list, not derive it from scratch.

4. **[scripted]** Split each per-cell BAM by chromosome (`split.sh`, driven by
   `clustering.sh`'s `split_bams` step): one BAM per cell per chromosome, for
   parallelising pileup creation.

5. **[scripted]** Pileup creation: `clustering.sh`'s `create_pileup` step calls our
   just-built `pileup` binary once per chromosome (1-22, X; Y intentionally skipped),
   writing SECEDO's binary pileup format.

6. **[scripted]** Run SECEDO: `clustering.sh`'s `variant_calling` step calls the `secedo`
   binary against the pileups with `--reference_genome`, `flags_breast`
   (`homozygous_filtered_rate=0.5`, `seq_error_rate=0.05` in the actual invocation vs.
   `0.001` in the checked-in `flags_breast` file -- the script overrides it on the command
   line), and a `--merge_file` (`breast_group_*`, a precomputed cell-to-slice-group
   mapping). Outputs a clustering plus one VCF per cluster (`cluster_<n>.vcf`) against the
   reference -- **all variants relative to reference, not yet somatic-filtered**.

7. **Germline/somatic filtering** -- this is where the repo's actual approach differs from
   a CHISEL-PoN design (see Corrections):
   - **[scripted, but a different method than a CHISEL PoN]** `mutect.sh` runs MuTect 1.1.4
     (old, Java 6) per cluster per chromosome, using **one SECEDO cluster (`clone19`) as
     the matched "normal"** rather than a CHISEL-identified normal-cell panel, against
     GRCh37 plus dbSNP and COSMIC VCFs for filtering.
   - **[ours, if we want the CHISEL-based design specifically]** a genuine
     CHISEL-normal-cell-identified Panel-of-Normals filter -- reading the downloaded
     CHISEL clone mapping to pick which cells/clusters CHISEL calls copy-number-normal,
     building a PoN from those, and filtering each cluster's SECEDO VCF against it. Nothing
     in either repo does this; it would need designing and writing from scratch, or we
     adopt the repo's own Mutect+pseudo-normal method instead (simpler, but a materially
     different filter than what we originally described).

8. **[ours]** VCF -> 96-channel trinucleotide binning: turn each cluster's somatic VCF into
   one row of a `mutation_count_matrix.csv`-shaped table (subclone/cluster x 96 contexts).
   Will live in `realdata/scripts/`. Needs a reference FASTA for the trinucleotide context
   lookup (same GRCh37 build used upstream, for consistency).

**Not addressed by either repo, worth flagging even though out of scope for this pass**:
Tree-HDP needs a tree over nodes (a Newick string), not just a flat per-cluster count
table. Neither repo builds a phylogeny over SECEDO's clusters -- that's a further piece of
"ours" beyond the binning step, not covered here.

## Corrections against what we assumed going in

- **Reference genome is GRCh37, not GRCh38.** `clustering.sh` and `mutect.sh` both hardcode
  `GRCh37.p13.genome.fa`. GRCh38 appears only in the *synthetic* (`varsim/`) evaluation
  pipeline, a separate dataset from the breast-cancer one. If GRCh38 is wanted for this
  dataset specifically, that's a deviation from the repo's own recipe, not a following of
  it -- worth deciding deliberately rather than defaulting to what the scripts do.
- **The germline/PoN filtering is not CHISEL-based in this repo.** The breast-cancer
  README downloads CHISEL clone data, but no script consumes it. The actual somatic
  filter used (`mutect.sh`) is Mutect1 against a SECEDO-clustered pseudo-normal
  (`clone19`), a different design from a CHISEL-identified-normal-cells Panel of Normals.
  Building the CHISEL-PoN version specifically is ours to write (stage 7).
- **Zenodo record 10.5281/zenodo.6516955 is source code, not data.** Checked its metadata
  directly: one file, `ratschlab/secedo-v.1.0.7.zip` (8.45 MB), a SECEDO release archive.
  No BAMs, pileups, VCFs, or CHISEL/PoN inputs are in it. The CHISEL mapping files come
  from the separate `chisel-data` GitHub repo (stage 2); everything else comes from 10x
  directly (stage 1) or is generated locally by the pipeline.
- **No phylogeny is produced by either repo.** SECEDO outputs a flat clustering plus one
  VCF per cluster; nothing here builds a tree over the clusters. Tree-HDP needs a Newick
  string over nodes, not just a flat per-cluster count table -- building that tree is a
  further "ours" piece, separate from and in addition to the VCF-to-96-channel binning
  (stage 8), and out of scope for this pass.

## Data cost (see the chat report for the full breakdown)

- 5 BAM+BAI slices: ~944 GB total (measured via HTTP HEAD on the real download URLs, not
  estimated).
- Reference genome (GRCh37, matching what the scripts actually use): Ensembl's GRCh37
  primary-assembly FASTA is 870 MB gzipped (measured via HTTP HEAD), ~3 GB uncompressed
  (needed uncompressed for SECEDO/samtools use).
- CHISEL mapping data: negligible (~60 KB per section).
- Not counted here: intermediate pileups, per-cell BAMs, and per-chromosome splits, which
  multiply the raw BAM footprint substantially during processing (SECEDO's own README
  cites ~35 GB just for streamed pileup output at 8000 cells/0.5x coverage; our directory
  fan-out from splitting ~2000 cells x 24 chromosomes per slice adds meaningfully more
  before that's cleaned up).