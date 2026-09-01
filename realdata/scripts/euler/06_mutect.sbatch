#!/bin/bash
#SBATCH --job-name=secedo-D-mutect
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=%x-%A_%a.log
#
# Translates mutect.sh (LSF bsub-per-cluster-per-chromosome -> one SLURM array task per
# (cluster, chromosome) pair, read from tasks.tsv written by stage 05). Same MuTect1
# invocation, same normal-vs-tumour design (one SECEDO cluster as pseudo-normal, not a
# CHISEL panel of normals -- see realdata/recipe/breast_cancer_plan.md). GRCh37
# throughout, matching the original.
#
# One deliberate addition: the original mutect.sh requests only --out (MuTect1's
# tab-delimited call-stats .txt) and --coverage_file, not a VCF. Per-cluster somatic
# VCFs are the actual target of this run, and MuTect1 supports emitting one directly
# (--vcf) from the same call with no change to the calling logic or inputs -- added
# below rather than post-processing the call-stats file ourselves.
#
# --array bound below is a placeholder -- set it to $(wc -l < tasks.tsv) - 1 after stage
# 05 writes tasks.tsv (its exact size depends on how many clusters slice D produces).

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

if [ -z "${NORMAL_CLUSTER_ID}" ]; then
    echo "NORMAL_CLUSTER_ID not set in config.sh -- see stage 05." >&2
    exit 1
fi
if [ ! -x "${MUTECT_JAVA}" ]; then
    echo "MUTECT_JAVA (${MUTECT_JAVA}) not found or not executable -- MuTect1 needs" \
         "Java 7, not provided as an Euler module. Stage 00 fetches a Zulu JDK 7 into" \
         "\$HOME automatically (see config.sh); rerun it if this is missing." >&2
    exit 1
fi
if [ ! -f "${COSMIC_VCF}" ]; then
    echo "COSMIC_VCF (${COSMIC_VCF}) not found -- see stage 00's note; COSMIC needs your" \
         "own account and cannot be fetched unattended." >&2
    exit 1
fi

tasks_file="${MUTECT_DIR}/tasks.tsv"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${tasks_file}")
cluster=$(echo "${line}" | cut -f1)
chrom=$(echo "${line}" | cut -f2)
if [ -z "${cluster}" ]; then
    echo "No task at line $((SLURM_ARRAY_TASK_ID + 1)) of ${tasks_file}."
    exit 0
fi

normal_bam="${CLUSTER_BAMS_DIR}/clone${NORMAL_CLUSTER_ID}.bam"
tumor_bam="${CLUSTER_BAMS_DIR}/clone${cluster}.bam"
out_callstats="${MUTECT_DIR}/clone${cluster}_${chrom}.txt"
out_cov="${MUTECT_DIR}/clone${cluster}_${chrom}.coverage"
out_vcf="${MUTECT_DIR}/clone${cluster}_${chrom}.vcf"

echo "MuTect1: cluster ${cluster} vs normal cluster ${NORMAL_CLUSTER_ID}, chromosome ${chrom}"
"${MUTECT_JAVA}" -Xmx6g -jar "${MUTECT_JAR}" --analysis_type MuTect \
    --reference_sequence "${REF_FASTA}" \
    --dbsnp "${DBSNP_VCF}" \
    --cosmic "${COSMIC_VCF}" \
    --input_file:normal "${normal_bam}" \
    --input_file:tumor "${tumor_bam}" \
    --out "${out_callstats}" \
    --coverage_file "${out_cov}" \
    --vcf "${out_vcf}" \
    --intervals "${chrom}"