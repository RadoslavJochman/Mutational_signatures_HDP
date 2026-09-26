# Shared paths and settings for the slice-D-only Euler run. Sourced by every job script
# below, so the pipeline stages can never disagree on where things live.
#
# Adjust PERSIST_DIR before submitting -- not fetchable by any script in either SECEDO
# repo (see realdata/recipe/euler_slice_d_plan.md, "Prerequisites you supply").

set -euo pipefail

SLICE="D"

# --- scratch (transient, unbacked-up, purged -- heavy intermediates only) ---
SCRATCH_ROOT="${SCRATCH:?SCRATCH not set -- are you on an Euler login/compute node?}/secedo_slice_d"
REF_DIR="${SCRATCH_ROOT}/ref"
RAW_DIR="${SCRATCH_ROOT}/raw"
PREPROC_DIR="${SCRATCH_ROOT}/preprocess"
CELL_BAMS_DIR="${SCRATCH_ROOT}/cell_bams"
CELL_BAMS_SPLIT_DIR="${SCRATCH_ROOT}/cell_bams_split"
PILEUP_DIR="${SCRATCH_ROOT}/pileups"
CLUSTERING_DIR="${SCRATCH_ROOT}/clustering"
CLUSTER_BAMS_DIR="${SCRATCH_ROOT}/cluster_bams"
MUTECT_DIR="${SCRATCH_ROOT}/mutect"
CNV_DIR="${SCRATCH_ROOT}/cnv"
LOG_DIR="${SCRATCH_ROOT}/logs"

# --- persistent output (survives scratch purge -- confirm this path before submitting;
# it is a placeholder pointing at $HOME, adjust to your group's /cluster/work storage if
# you'd rather keep it off $HOME's quota) ---
PERSIST_DIR="${PERSIST_DIR:-${HOME}/secedo_runs/slice_D}"

# --- repo-relative paths (built secedo binaries, this repo's own scripts) ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SECEDO_BUILD="${REPO_ROOT}/realdata/external/secedo/build"
PILEUP_BIN="${SECEDO_BUILD}/pileup"
SECEDO_BIN="${SECEDO_BUILD}/secedo"
# SCICoNE build directory: holds the scicone-* binaries (breakpoint_detection, inference,
# ...) that the pyscicone wrapper resolves by name; stage 09 passes it to scicone.SCICoNE.
SCICONE_BUILD_DIR="${SCICONE_BUILD_DIR:-${REPO_ROOT}/realdata/external/SCICoNE/build}"
# LICHeE checkout (stage 08's optional comparison only, RUN_LICHEE=1 -- Dollo
# parsimony is the sole source of snv_tree.nwk). Stage 08 runs it as `java -cp
# release/lichee.jar:lib/* lineage.LineageEngine`, not through release/lichee, whose
# launcher cannot find its dependencies on JDK 11; it needs the jar and lib/ from here.
LICHEE_HOME="${LICHEE_HOME:-${REPO_ROOT}/realdata/external/lichee/LICHeE}"
EULER_SCRIPTS="${REPO_ROOT}/realdata/scripts/euler"

# --- 10x download URLs (slice D only -- do not loop over A/B/C/E) ---
BAM_URL="https://s3-us-west-2.amazonaws.com/10x.files/samples/cell-dna/1.1.0/breast_tissue_D_2k/breast_tissue_D_2k_possorted_bam.bam"
BAI_URL="https://s3-us-west-2.amazonaws.com/10x.files/samples/cell-dna/1.1.0/breast_tissue_D_2k/breast_tissue_D_2k_possorted_bam.bam.bai"
SUMMARY_URL="https://s3-us-west-2.amazonaws.com/10x.files/samples/cell-dna/1.1.0/breast_tissue_D_2k/breast_tissue_D_2k_per_cell_summary_metrics.csv"

# --- CellRanger DNA's own per-cell CNV output, stage 09's (build_cna_tree.py)
# input: read by pyscicone's read_10x (GC-corrected counts, unmappable bins,
# chromosome stops, outlier-cell filtering), so this stage does not reimplement any of
# that. Confirmed hosted alongside the BAM (HTTP 200: cnv_data.h5 ~2.5 GB,
# node_cnv_calls.bed ~32 MB -- CellRanger DNA's own CNV segment calls, kept for
# cross-check only, not consumed by build_cna_tree.py). ---
CNV_H5_URL="https://cf.10xgenomics.com/samples/cell-dna/1.1.0/breast_tissue_D_2k/breast_tissue_D_2k_cnv_data.h5"
NODE_CNV_CALLS_URL="https://cf.10xgenomics.com/samples/cell-dna/1.1.0/breast_tissue_D_2k/breast_tissue_D_2k_node_cnv_calls.bed"
CNV_H5="${RAW_DIR}/breast_tissue_D_2k_cnv_data.h5"
NODE_CNV_CALLS_BED="${RAW_DIR}/breast_tissue_D_2k_node_cnv_calls.bed"

# --- reference genome (GRCh37, matching what clustering.sh/mutect.sh hardcode) ---
REF_FASTA_GZ_URL="https://ftp.ensembl.org/pub/grch37/release-113/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz"
REF_FASTA="${REF_DIR}/GRCh37.p13.genome.fa"

# --- dbSNP and COSMIC: staged for the original MuTect1 recipe, not hosted by either repo
# (see the plan doc). Neither is consumed by GATK4 Mutect2/FilterMutectCalls -- MuTect1's
# --dbsnp/--cosmic were inputs to its own LOD-threshold classifier (stricter at dbSNP
# sites, relaxed at COSMIC hotspots), a mechanism GATK4 replaced with germline-resource
# population-AF filtering and Panels of Normals, neither of which COSMIC or this dbSNP
# file can serve as (they'd need an AF INFO field; COSMIC's is a mutation catalogue, not
# population frequencies). Left defined and staged (harmless, and dbSNP or a like-shaped
# resource may still be useful for --germline-resource once we add one) but unused by
# 06_mutect.sbatch as of the Mutect2 switch -- see the plan doc's Mutect1->Mutect2 note. ---
DBSNP_VCF_GZ_URL="https://storage.googleapis.com/gcp-public-data--broad-references/hg19/v0/dbsnp_138.b37.vcf.gz"
DBSNP_VCF="${REF_DIR}/dbsnp_138.b37.vcf"
COSMIC_VCF="${COSMIC_VCF:-${REF_DIR}/cosmic_v94_hg37_coding_and_noncoding.vcf}"

# --- force-call presence thresholds (stage 06b's pass 2, read by build_tree.py): a
# force-called site counts as present in a cluster only at or above both of these. ---
PRESENCE_MIN_VAF="${PRESENCE_MIN_VAF:-0.05}"
PRESENCE_MIN_ALT_READS="${PRESENCE_MIN_ALT_READS:-2}"

# --- force-call absence thresholds (same stage): a zero-ALT-read call counts as
# confidently absent only if a real mutation at ABSENT_EXPECTED_VAF would, at the
# site's depth, have been this unlikely (ABSENT_ALPHA) to show zero reads by chance.
# Everything else at a genotyped site is unknown, never absent -- see
# build_snv_tree.py's module docstring. ---
ABSENT_EXPECTED_VAF="${ABSENT_EXPECTED_VAF:-0.25}"
ABSENT_ALPHA="${ABSENT_ALPHA:-0.05}"

# --- GATK4 (Mutect2 + FilterMutectCalls + SelectVariants): runs on modern Java, so unlike
# MuTect1 it needs no special JDK fetch. Confirmed interactively on an Euler login node:
# gatk lives under the stack/2024-06 software stack, not the default one, so a bare
# `module load gatk` silently no-ops. The working chain is
#   module load stack/2024-06 gcc/12.2.0
#   module load gatk/4.4.0.0
# which also resolves samtools (1.17) the same way; both 00_download.sbatch and
# 06_mutect.sbatch load this chain explicitly. If Euler's module tree changes and this
# stops resolving, fall back to installing gatk4 into the pipeline's own conda/venv
# (`conda install -c bioconda gatk4` or `pip install gatk`) and drop the `module load`
# lines in those two scripts. ---

# --- pseudo-normal cluster: back to tumour-vs-pseudo-normal calling (gnomAD dropped,
# see the plan doc's "revert to tumour-vs-pseudo-normal" entry). Stage 05 excludes this
# cluster from the tumour task list; stage 06/06b call it as Mutect2's -normal. Cluster 4
# is slice D's matched pseudo-normal, identified from stage 04/05's clustering output. ---
NORMAL_CLUSTER_ID="${NORMAL_CLUSTER_ID:-4}"

# --- stage 09 (CNA tree): patient sex sets SCICoNE's neutral copy number for chrX/chrY
# and so the tree root. Slice D is FEMALE; build_cna_tree.py requires --sex explicitly and
# has no default, so this is the one place it is recorded. ---
PATIENT_SEX="${PATIENT_SEX:-female}"

mkdir -p "${REF_DIR}" "${RAW_DIR}" "${PREPROC_DIR}" "${CELL_BAMS_DIR}" \
    "${CELL_BAMS_SPLIT_DIR}" "${PILEUP_DIR}" "${CLUSTERING_DIR}" "${CLUSTER_BAMS_DIR}" \
    "${MUTECT_DIR}" "${CNV_DIR}" "${LOG_DIR}"