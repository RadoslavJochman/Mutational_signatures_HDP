#!/bin/bash
#SBATCH --job-name=secedo-D-preprocess
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2200M
#SBATCH --output=%x-%j.log
#
# Translates preprocessing/{1_run_filtering,2_run_filteringCB,3_run_splittingByCB}.sh
# (LSF -> SLURM, single slice D, paths repointed at scratch). Faithful to the original
# filters and to the CB-tag split; the only substitution is the allowed-barcode source
# (see the plan doc) -- 10x does publish slice D's own per_cell_summary_metrics.csv, so
# no synthetic tag list is needed.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
module load samtools 2>/dev/null || true

RAW_BAM="${RAW_DIR}/breast_tissue_D_2k_possorted_bam.bam"
POS_SORTED="${PREPROC_DIR}/breast_tissue_D_2k_possorted_bam_pos_sorted.bam"
CB_FILTERED="${PREPROC_DIR}/breast_tissue_D_2k_possorted_bam_pos_sorted_CB.bam"
CB_SORTED="${PREPROC_DIR}/breast_tissue_D_2k_possorted_bam_pos_sorted_CB_sorted.bam"
TAGS_FILE="${PREPROC_DIR}/allowedTags"
SUMMARY_CSV="${RAW_DIR}/breast_tissue_D_2k_per_cell_summary_metrics.csv"

echo "[1/3] Filtering: proper pair, primary, non-duplicate..."
samtools view -h -b -f 0x2 -F 0x500 "${RAW_BAM}" > "${POS_SORTED}"
echo "Reads before: $(samtools view -c "${RAW_BAM}")  after: $(samtools view -c "${POS_SORTED}")"

echo "[2/3] Filtering: require CB tag..."
samtools view -h "${POS_SORTED}" | grep -E '^@|CB:Z:' | samtools view -b > "${CB_FILTERED}"
samtools index "${CB_FILTERED}"
echo "Reads with CB tag: $(samtools view -c "${CB_FILTERED}")"

echo "[3/3] Splitting by CB tag..."
samtools sort -t CB -m 2G -@ 10 -T "${PREPROC_DIR}/sort_tmp" "${CB_FILTERED}" \
    > "${CB_SORTED}"

# allowed-barcode list: the header-stripped first column of 10x's own per-cell summary
# for this slice (same source 3_run_splittingByCB.sh expects; downloaded in stage 00).
tail -n +2 "${SUMMARY_CSV}" | cut -d ',' -f1 > "${TAGS_FILE}"
echo "$(wc -l < "${TAGS_FILE}") allowed barcodes."

mkdir -p "${CELL_BAMS_DIR}"
python3 "${REPO_ROOT}/realdata/external/secedo-evaluation/breast_cancer/preprocessing/split_by_CBtag.py" \
    -f "${CB_SORTED}" -o "${CELL_BAMS_DIR}/" -t "${TAGS_FILE}"

n_cells=$(find "${CELL_BAMS_DIR}" -name '*.bam' | wc -l)
echo "Wrote ${n_cells} per-cell BAMs to ${CELL_BAMS_DIR}."

# housekeeping: the pre-sort CB-filtered copy is superseded by CB_SORTED once split
# succeeds; drop it to keep the preprocess/ footprint to one full-size copy.
rm -f "${CB_FILTERED}" "${CB_FILTERED}.bai"