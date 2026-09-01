#!/bin/bash
#SBATCH --job-name=secedo-D-copyout
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=%x-%j.log
#
# Must-have (a): scratch is transient and unbacked-up, so copy the run's outputs to
# persistent storage as soon as stage 06 finishes -- do not leave the only copy here.
# Copies the per-cluster somatic VCFs (the actual deliverable) plus the supporting
# call-stats/coverage files, the secedo clustering assignment and per-cluster
# reference-relative VCFs, and this run's logs, so the run is reproducible-by-inspection
# without needing scratch to still exist.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

mkdir -p "${PERSIST_DIR}/mutect_vcfs" "${PERSIST_DIR}/secedo_clustering" "${PERSIST_DIR}/logs"

echo "Copying per-cluster somatic VCFs and call-stats to ${PERSIST_DIR}/mutect_vcfs..."
cp "${MUTECT_DIR}"/clone*.vcf "${MUTECT_DIR}"/clone*.txt "${MUTECT_DIR}"/clone*.coverage \
    "${PERSIST_DIR}/mutect_vcfs/"

echo "Copying secedo clustering assignment and reference-relative VCFs to" \
     "${PERSIST_DIR}/secedo_clustering..."
cp "${CLUSTERING_DIR}/clustering" "${CLUSTERING_DIR}"/cluster_*.vcf \
    "${PERSIST_DIR}/secedo_clustering/" 2>/dev/null || true
cp "${PILEUP_DIR}/chromosome_1.map" "${PERSIST_DIR}/secedo_clustering/"

echo "Copying logs..."
cp "${LOG_DIR}"/*.log "${SCRATCH_ROOT}"/*.log "${PERSIST_DIR}/logs/" 2>/dev/null || true

echo "Verifying copy..."
n_scratch=$(find "${MUTECT_DIR}" -maxdepth 1 -name 'clone*.vcf' | wc -l)
n_persist=$(find "${PERSIST_DIR}/mutect_vcfs" -maxdepth 1 -name 'clone*.vcf' | wc -l)
if [ "${n_scratch}" -ne "${n_persist}" ]; then
    echo "MISMATCH: ${n_scratch} VCFs in scratch but ${n_persist} copied to ${PERSIST_DIR}." >&2
    exit 1
fi
echo "Copied ${n_persist} per-cluster somatic VCFs to ${PERSIST_DIR}/mutect_vcfs. Done."
du -sh "${PERSIST_DIR}"

# The two numbers that matter for deciding whether this data fits the sparse per-node
# count regime Tree-HDP targets: how many clusters SECEDO found, and the somatic SNV
# count per cluster (summed across chromosomes). Written to a CSV rather than left in
# the log, so it survives independently of this job's stdout.
echo "Cluster summary (somatic SNVs per cluster, summed across chromosomes):"
summary_csv="${PERSIST_DIR}/cluster_summary.csv"
echo "cluster_id,n_somatic_snvs" > "${summary_csv}"
n_clusters_final=0
for bam in "${CLUSTER_BAMS_DIR}"/clone*.bam; do
    [ -e "${bam}" ] || continue
    cid=$(basename "${bam}" .bam | sed 's/^clone//')
    [ "${cid}" = "${NORMAL_CLUSTER_ID}" ] && continue
    n=$(cat "${PERSIST_DIR}/mutect_vcfs"/clone${cid}_*.vcf 2>/dev/null | grep -vc '^#' || true)
    echo "  clone${cid}: ${n}"
    echo "${cid},${n}" >> "${summary_csv}"
    n_clusters_final=$((n_clusters_final + 1))
done
echo "${n_clusters_final} tumour clusters (normal cluster ${NORMAL_CLUSTER_ID} excluded)." \
     "Full table: ${summary_csv}"