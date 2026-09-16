#!/bin/bash
#SBATCH --job-name=secedo-D-clusterbams
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=%x-%j.log
#
# Ours -- see build_cluster_bams.py's docstring. Not in either repo: mutect.sh assumes
# per-cluster whole-genome BAMs already exist; this is how they get built, from
# secedo's own clustering + pileup .map output plus the per-cell BAMs from stage 01.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/config.sh"
module load stack/2024-06 gcc/12.2.0 2>/dev/null || true
module load samtools 2>/dev/null || true

# No --min-cluster-size here: SECEDO's own --min_cluster_size (500) already applied
# during clustering (stage 04), as its stop-splitting threshold, not a keep-cluster
# filter -- its finished clusters are all smaller than that. Re-applying 500 as a
# floor here discarded every real cluster and built zero BAMs. Every populated
# cluster gets a BAM (the python default --min-cluster-size 1).
python3 "${EULER_SCRIPTS}/build_cluster_bams.py" \
    --map-file "${PILEUP_DIR}/chromosome_1.map" \
    --clustering-file "${CLUSTERING_DIR}/clustering" \
    --cell-bams-dir "${CELL_BAMS_DIR}" \
    --out-dir "${CLUSTER_BAMS_DIR}"

ls -la "${CLUSTER_BAMS_DIR}"
n_clusters=$(find "${CLUSTER_BAMS_DIR}" -maxdepth 1 -name 'clone*.bam' | wc -l)
echo "SECEDO found ${n_clusters} populated clusters (cluster 0/no-cluster excluded)."

# Task list for stage 06 (mutect): one line per (cluster, chromosome), every populated
# cluster except NORMAL_CLUSTER_ID -- tumour-vs-pseudo-normal calling (see
# 06_mutect.sbatch) needs that cluster held back as Mutect2's -normal, not called as a
# tumour task itself. NORMAL_CLUSTER_ID must be set in config.sh (inspect this stage's
# own output first if it isn't yet known for a fresh clustering run).
if [ -z "${NORMAL_CLUSTER_ID}" ]; then
    echo "NORMAL_CLUSTER_ID is not set in config.sh -- inspect ${CLUSTER_BAMS_DIR} above" \
         "and set it, then rerun this stage to regenerate tasks.tsv with the pseudo-normal" \
         "cluster excluded." >&2
    exit 1
fi
tasks_file="${MUTECT_DIR}/tasks.tsv"
: > "${tasks_file}"
for bam in "${CLUSTER_BAMS_DIR}"/clone*.bam; do
    cluster_id=$(basename "${bam}" .bam | sed 's/^clone//')
    [ "${cluster_id}" = "${NORMAL_CLUSTER_ID}" ] && continue
    for chrom in {1..22} X Y; do
        printf '%s\t%s\n' "${cluster_id}" "${chrom}" >> "${tasks_file}"
    done
done
echo "Wrote $(wc -l < "${tasks_file}") mutect tasks to ${tasks_file}" \
     "(pseudo-normal cluster ${NORMAL_CLUSTER_ID} excluded)."