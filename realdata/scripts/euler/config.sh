# Shared paths and settings for the slice-D-only Euler run. Sourced by every job script
# below, so the pipeline stages can never disagree on where things live.
#
# Adjust PERSIST_DIR, COSMIC_VCF, MUTECT_JAR, and MUTECT_JAVA before submitting -- these
# four are not fetchable by any script in either SECEDO repo (see
# realdata/recipe/euler_slice_d_plan.md, "Prerequisites you supply").

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

# --- MuTect1 reference inputs: not hosted by either repo, see the plan doc.
# dbSNP is a public substitute (b138 supersedes b132); COSMIC needs your own COSMIC
# account and cannot be fetched unattended -- stage it at COSMIC_VCF yourself first. ---
DBSNP_VCF_GZ_URL="https://storage.googleapis.com/gcp-public-data--broad-references/hg19/v0/dbsnp_138.b37.vcf.gz"
DBSNP_VCF="${REF_DIR}/dbsnp_138.b37.vcf"
COSMIC_VCF="${COSMIC_VCF:-${REF_DIR}/cosmic_v94_hg37_coding_and_noncoding.vcf}"

# --- MuTect1 itself needs Java 7 specifically, not Java 6 as the original mutect.sh's own
# ~/jre1.6.0_45 assumption implied (GATK/MuTect-era code relies on sun.reflect/sun.misc
# internals that Java 9+'s module system hides -- verified this isn't just a class-file-
# version mismatch a newer JVM would tolerate). Euler's module stack has no Java 6/7/8
# (only openjdk 11/17/21) -- but MuTect1 was never going to be a module anyway, it's a
# plain `java -jar`. Oracle's own JRE/JDK archive now gates old versions behind a login;
# Azul's Zulu builds of OpenJDK don't, and cover Java 7. Verified: downloaded and ran
# ZULU_JDK7_URL's tarball, reports "openjdk version 1.7.0_352". 00_download fetches and
# unpacks it into ZULU_JDK7_DIR (under $HOME, not scratch -- a small reusable tool, not
# run data, and $HOME survives a scratch purge). ---
ZULU_JDK7_URL="https://cdn.azul.com/zulu/bin/zulu7.56.0.11-ca-jdk7.0.352-linux_x64.tar.gz"
ZULU_JDK7_DIR="${HOME}/secedo_tools/zulu7"
MUTECT_JAVA="${MUTECT_JAVA:-${ZULU_JDK7_DIR}/bin/java}"
# MuTect 1.1.4 jar: needs your own copy (Broad login), exactly as the original mutect.sh
# assumed a personal ~/mutect install. Expected at MUTECT_JAR; override to point elsewhere.
MUTECT_JAR="${MUTECT_JAR:-${HOME}/mutect/muTect-1.1.4.jar}"

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