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
FLAGS_BREAST="${REPO_ROOT}/realdata/external/secedo-evaluation/breast_cancer/flags_breast"
EULER_SCRIPTS="${REPO_ROOT}/realdata/scripts/euler"

# --- 10x download URLs (slice D only -- do not loop over A/B/C/E) ---
BAM_URL="https://cf.10xgenomics.com/samples/cell-dna/1.1.0/breast_tissue_D_2k/breast_tissue_D_2k_possorted_bam.bam"
BAI_URL="https://cf.10xgenomics.com/samples/cell-dna/1.1.0/breast_tissue_D_2k/breast_tissue_D_2k_possorted_bam.bam.bai"
SUMMARY_URL="https://cf.10xgenomics.com/samples/cell-dna/1.1.0/breast_tissue_D_2k/breast_tissue_D_2k_per_cell_summary_metrics.csv"

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

# --- GATK4 (Mutect2 + FilterMutectCalls + SelectVariants): runs on modern Java, so unlike
# MuTect1 it needs no special JDK fetch -- Euler's own openjdk module stack (11/17/21)
# covers it. Try a module first (`module spider gatk` on a login node); if Euler has none,
# install into the pipeline's own conda/venv instead (`conda install -c bioconda gatk4` or
# `pip install gatk`) and drop the `module load` line in 06_mutect.sbatch. Not resolved
# here since it needs checking against the live module tree at submission time. ---

# --- pseudo-normal cluster: the original script hardcodes "clone19" for slice B; which
# cluster stands in for the matched normal is data-dependent and cannot be known before
# stage 04/05 produce slice D's actual clustering. Inspect CLUSTERING_DIR/clustering and
# CLUSTER_BAMS_DIR (cluster sizes, and which one looks diploid/background) after stage 05,
# then set this before submitting stage 06. Left unset deliberately -- 06 refuses to run
# without it rather than guessing. ---
NORMAL_CLUSTER_ID="${NORMAL_CLUSTER_ID:-}"

mkdir -p "${REF_DIR}" "${RAW_DIR}" "${PREPROC_DIR}" "${CELL_BAMS_DIR}" \
    "${CELL_BAMS_SPLIT_DIR}" "${PILEUP_DIR}" "${CLUSTERING_DIR}" "${CLUSTER_BAMS_DIR}" \
    "${MUTECT_DIR}" "${LOG_DIR}"