#!/bin/bash
#SBATCH --job-name=secedo-D-splitchrom
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-38%20
#SBATCH --output=%x-%A_%a.log
#
# Translates split.sh's split_bams step (LSF bsub-per-100-files -> one SLURM array task
# per batch of CELLS_PER_TASK cells, batched rather than one task per cell to stay well
# under the QOS per-user submit limit -- see "batching" note below). Splits each per-cell
# BAM into per-chromosome BAMs (1-22, X, Y), for parallelising pileup creation exactly as
# the original does.
#
# Array size: ceil(n_cells / CELLS_PER_TASK) tasks, 0-indexed -- must be set at submit
# time from the actual per-cell BAM count (check stage 01's log; the 10x "2k cells"
# slices are ~2000 cells, not exactly). --array bound below (0-38%20) assumes 1916 cells
# at the default CELLS_PER_TASK=50 (ceil(1916/50) = 39 tasks, indices 0-38); recompute
# for the actual count before submitting if it differs. %20 throttles concurrent tasks;
# adjust to taste.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/config.sh"
module load stack/2024-06 gcc/12.2.0 2>/dev/null || true
module load samtools 2>/dev/null || true

CELLS_PER_TASK="${CELLS_PER_TASK:-50}"

mapfile -t cell_bams < <(find "${CELL_BAMS_DIR}" -maxdepth 1 -name '*.bam' | sort)
n=${#cell_bams[@]}
start=$((SLURM_ARRAY_TASK_ID * CELLS_PER_TASK))
if [ "${start}" -ge "${n}" ]; then
    echo "Task start index ${start} >= ${n} cell BAMs found; nothing to do for this task."
    exit 0
fi
end=$((start + CELLS_PER_TASK - 1))
if [ "${end}" -ge "${n}" ]; then
    end=$((n - 1))
fi

for idx in $(seq "${start}" "${end}"); do
    source_bam="${cell_bams[${idx}]}"
    filename=$(basename -- "${source_bam}")
    pref="${filename%.*}"

    echo "Splitting ${source_bam}..."
    [ -f "${source_bam}.bai" ] || samtools index "${source_bam}"
    for chrom in {1..22} X Y; do
        chunk="${CELL_BAMS_SPLIT_DIR}/${pref}_chr${chrom}.bam"
        samtools view -b "${source_bam}" "${chrom}" > "${chunk}"
        samtools index "${chunk}"
    done
    echo "${source_bam} split into 24 chromosome files."
done