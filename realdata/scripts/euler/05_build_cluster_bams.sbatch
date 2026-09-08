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
module load samtools 2>/dev/null || true

python3 "${EULER_SCRIPTS}/build_cluster_bams.py" \
    --map-file "${PILEUP_DIR}/chromosome_1.map" \
    --clustering-file "${CLUSTERING_DIR}/clustering" \
    --cell-bams-dir "${CELL_BAMS_DIR}" \
    --out-dir "${CLUSTER_BAMS_DIR}" \
    --min-cluster-size 500

ls -la "${CLUSTER_BAMS_DIR}"
n_clusters=$(find "${CLUSTER_BAMS_DIR}" -maxdepth 1 -name 'clone*.bam' | wc -l)
echo "SECEDO found ${n_clusters} clusters at >= 500 cells (cluster 0/no-cluster excluded)."

# Task list for stage 06 (mutect): one line per (tumour cluster, chromosome), excluding
# NORMAL_CLUSTER_ID -- set NORMAL_CLUSTER_ID in config.sh first by inspecting the cluster
# BAMs/sizes just written above (see the comment in config.sh).
if [ -z "${NORMAL_CLUSTER_ID}" ]; then
    echo "NORMAL_CLUSTER_ID is not set -- inspect ${CLUSTER_BAMS_DIR} above, decide which" \
         "cluster is the pseudo-normal, set NORMAL_CLUSTER_ID in config.sh, then rerun" \
         "this stage (it's cheap -- samtools merge only) to regenerate tasks.tsv."
    exit 1
fi
tasks_file="${MUTECT_DIR}/tasks.tsv"
: > "${tasks_file}"
for bam in "${CLUSTER_BAMS_DIR}"/clone*.bam; do
    cluster_id=$(basename "${bam}" .bam | sed 's/^clone//')
    if [ "${cluster_id}" = "${NORMAL_CLUSTER_ID}" ]; then
        continue
    fi
    for chrom in {1..22} X Y; do
        printf '%s\t%s\n' "${cluster_id}" "${chrom}" >> "${tasks_file}"
    done
done
echo "Wrote $(wc -l < "${tasks_file}") mutect tasks to ${tasks_file}."