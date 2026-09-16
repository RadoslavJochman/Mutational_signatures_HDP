#!/bin/bash
#SBATCH --job-name=secedo-D-buildcnatree
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=%x-%j.log
#
# Ours -- builds the CNA-side tree TreeHDP consumes: SCICoNE's copy-number event history
# over single cells, collapsed onto the same SECEDO clusters the SNV tree (stage 08) uses.
# See build_cna_tree.py's module docstring for the input path (CellRanger DNA's own
# cnv_data.h5, stage 00's download), the Newick contract it shares with the SNV tree, and
# the caveat that cnv_data.h5's exact schema and SCICoNE's exact CLI/output format are not
# confirmed against a real run -- CONFIRM THESE FIRST (h5py against the downloaded
# cnv_data.h5; scicone --help / its README against the built binary) before trusting this
# stage's output.
#
# Independent of stages 06-08 (SNV calling and tree): only needs stage 00's cnv_data.h5
# download and stage 04/05's SECEDO clustering + pileup .map file. Not chained into
# submit_all.sh -- run by hand once those are available, same as stage 08.
#
# Needs a python environment with numpy, networkx, h5py installed (the repo's own .venv
# works if present on this cluster; h5py is pinned in requirements.txt). Also needs the
# SCICoNE binary built at realdata/external/SCICoNE/build/scicone (see SCICONE_BIN in
# config.sh) -- clone cbg-ethz/SCICoNE into realdata/external/ and follow its README's
# cmake build (comparable to SECEDO's own build; see config.sh's GCC-flags note for that).
#
# --mem-per-cpu raised above the other stages' 4G: cnv_data.h5's raw counts matrix is
# ~2000 cells x genome-wide bins, loaded fully into memory by load_cnv_counts_matrix.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/config.sh"

module load stack/2024-06 gcc/12.2.0 2>/dev/null || true

PY="${REPO_ROOT}/.venv/bin/python"
if [ ! -x "${PY}" ]; then
    echo "ERROR: ${PY} not found. Create the repo venv with numpy/networkx/h5py, or edit this to point at the right interpreter." >&2
    exit 1
fi

if [ ! -f "${CNV_H5}" ]; then
    echo "ERROR: ${CNV_H5} not found -- run 00_download.sbatch first." >&2
    exit 1
fi

TREE_OUT_DIR="${PERSIST_DIR}/tree_input"
mkdir -p "${TREE_OUT_DIR}"

PSEUDOBULK_FALLBACK="${PSEUDOBULK_FALLBACK:-0}"
PSEUDOBULK_ARGS=()
if [ "${PSEUDOBULK_FALLBACK}" = "1" ]; then
    PSEUDOBULK_ARGS=(--pseudobulk-fallback)
fi

"${PY}" "${EULER_SCRIPTS}/build_cna_tree.py" \
    --cnv-h5 "${CNV_H5}" \
    --map-file "${PILEUP_DIR}/chromosome_1.map" \
    --clustering-file "${CLUSTERING_DIR}/clustering" \
    --out-dir "${TREE_OUT_DIR}" \
    --scicone-bin "${SCICONE_BIN}" \
    ${PSEUDOBULK_ARGS[@]+"${PSEUDOBULK_ARGS[@]}"}

echo "Wrote cna_tree.nwk and cna_tree_diagnostics.txt to ${TREE_OUT_DIR}."