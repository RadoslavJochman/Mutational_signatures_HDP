#!/bin/bash
#SBATCH --job-name=secedo-D-splitchrom
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-1999%200
#SBATCH --output=%x-%A_%a.log
#
# Translates split.sh's split_bams step (LSF bsub-per-100-files -> one SLURM array task
# per cell; --array bound below assumes ~2000 cells, matching the 10x "2k cells" slices --
# check the actual per-cell BAM count from stage 01's log and adjust the array range
# before submitting if it differs). Splits each per-cell BAM into per-chromosome BAMs
# (1-22, X, Y), for parallelising pileup creation exactly as the original does.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
module load samtools 2>/dev/null || true

mapfile -t cell_bams < <(find "${CELL_BAMS_DIR}" -maxdepth 1 -name '*.bam' | sort)
n=${#cell_bams[@]}
idx=${SLURM_ARRAY_TASK_ID}
if [ "${idx}" -ge "${n}" ]; then
    echo "Task index ${idx} >= ${n} cell BAMs found; nothing to do for this task."
    exit 0
fi

source_bam="${cell_bams[${idx}]}"
filename=$(basename -- "${source_bam}")
pref="${filename%.*}"

echo "Splitting ${source_bam}..."
for chrom in {1..22} X Y; do
    chunk="${CELL_BAMS_SPLIT_DIR}/${pref}_chr${chrom}.bam"
    samtools view -b "${source_bam}" "${chrom}" > "${chunk}"
    samtools index "${chunk}"
done
echo "${source_bam} split into 24 chromosome files."