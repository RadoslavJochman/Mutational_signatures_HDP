#!/bin/bash
#SBATCH --job-name=secedo-D-pileup
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=500M
#SBATCH --array=0-22
#SBATCH --output=%x-%a.log
#
# Translates clustering.sh's create_pileup step (LSF bsub-per-chromosome -> one SLURM
# array task per chromosome). Chromosomes 1-22, X (23 tasks, array indices 0-22); Y is
# skipped, matching the original comment ("maybe it confuses things"). Each task copies
# only its own chromosome's per-cell split BAMs into a private scratch tmp dir before
# running the pileup binary, then removes the copy -- same pattern as the original,
# which keeps any one task's transient footprint to one chromosome's slice of the data
# rather than the whole per-chromosome-split directory.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/config.sh"
module load openblas 2>/dev/null || true  # adjust to the module name/version on Euler

chroms=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X)
chrom="${chroms[${SLURM_ARRAY_TASK_ID}]}"

tmp_dir=$(mktemp -d -t "pileup-chr${chrom}-XXXXXXXXXX" --tmpdir="${SCRATCH_ROOT}")
mkdir -p "${tmp_dir}"

echo "Copying chromosome ${chrom} split BAMs into ${tmp_dir}..."
cp "${CELL_BAMS_SPLIT_DIR}"/*_chr${chrom}.bam* "${tmp_dir}/"
n=$(find "${tmp_dir}" -name '*.bam' | wc -l)
echo "Found ${n} per-cell BAMs for chromosome ${chrom}."

echo "Running pileup for chromosome ${chrom}..."
"${PILEUP_BIN}" -i "${tmp_dir}/" -o "${PILEUP_DIR}/chromosome" --num_threads 20 \
    --log_level=trace --min_base_quality 20 --max_coverage 1000 --chromosomes "${chrom}"

rm -rf "${tmp_dir}"
echo "Chromosome ${chrom} pileup done."