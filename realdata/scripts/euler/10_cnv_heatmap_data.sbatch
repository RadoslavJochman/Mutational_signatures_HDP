#!/bin/bash
#SBATCH --job-name=secedo-D-cnvheatmap
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=%x-%j.log
#
# Ours -- data for the slice D copy-number heatmap (Figure 6). Reads cnv_data.h5
# the way stage 09 does and rebuilds each SCICoNE node's copy number from stage
# 09's saved outputs (no SCICoNE rerun); see cnv_heatmap_data.py's docstring.
# The plot is made locally by realdata/scripts/plot_cnv_heatmap.py from the npz.
#
# Needs stage 09's tree_input/ (cna_cell_nodes.csv, cna_tree_diagnostics.txt,
# scicone_breakpoints.npz) and the same venv with pyscicone. Memory: read_10x
# holds the unfiltered and filtered cells x bins matrices (~2 GB each, float64)
# plus a copy while the bin filter is replayed; 16 GB covers it.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/config.sh"

module load stack/2024-06 gcc/12.2.0 2>/dev/null || true

PY="${REPO_ROOT}/.venv/bin/python"
TREE_OUT_DIR="${PERSIST_DIR}/tree_input"
OUT="${TREE_OUT_DIR}/figures_data/cnv_heatmap.npz"

for f in "${CNV_H5}" "${TREE_OUT_DIR}/cna_cell_nodes.csv" \
         "${TREE_OUT_DIR}/cna_tree_diagnostics.txt" \
         "${TREE_OUT_DIR}/scicone_breakpoints.npz"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: ${f} not found (stage 00 or 09 output)." >&2
        exit 1
    fi
done

"${PY}" "${EULER_SCRIPTS}/cnv_heatmap_data.py" \
    --cnv-h5 "${CNV_H5}" \
    --tree-dir "${TREE_OUT_DIR}" \
    --scicone-build-dir "${SCICONE_BUILD_DIR}" \
    --sex "${PATIENT_SEX}" \
    --block-mb "${BLOCK_MB:-1}" \
    --out "${OUT}"

ls -lh "${OUT}"
