#!/bin/bash
#SBATCH --job-name=secedo-D-buildcnatree
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=6G
#SBATCH --output=%x-%j.log
#
# Ours -- builds the CNA-side tree TreeHDP consumes: SCICoNE's copy-number event history
# over single cells, collapsed onto the same SECEDO clusters the SNV tree (stage 08) uses.
# See build_cna_tree.py's module docstring for the pyscicone-based flow, the Newick
# contract it shares with the SNV tree, and the three things worth auditing on a first
# real run (filtered-cell barcodes, region neutral states, the cluster-4 fold).
#
# Independent of stages 06-08 (SNV calling and tree): only needs stage 00's cnv_data.h5
# download and stage 04/05's SECEDO clustering + pileup .map file. Not chained into
# submit_all.sh -- run by hand once those are available, same as stage 08.
#
# Resources (UNMEASURED, tune after the first real run): SCICoNE's learn_tree runs
# n_reps=10 tree searches in a multiprocessing Pool, hence 10 CPUs; the 40000-iteration
# cluster-tree search and breakpoint detection over a 200-cell subsample dominate the
# wall time, hence 24h. --mem-per-cpu covers read_10x, which holds the unfiltered and
# filtered cells x bins matrices (~2000 x ~10^5 float64 each) at once.
#
# Needs the repo .venv with pyscicone installed (pip install
# realdata/external/SCICoNE/pyscicone/, see requirements.txt) and the SCICoNE binaries
# built in SCICONE_BUILD_DIR (clone cbg-ethz/SCICoNE into realdata/external/ and follow
# its README's cmake build). The import pre-flight below fails fast if either is broken.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/config.sh"

module load stack/2024-06 gcc/12.2.0 2>/dev/null || true

PY="${REPO_ROOT}/.venv/bin/python"
if [ ! -x "${PY}" ]; then
    echo "ERROR: ${PY} not found. Create the repo venv (see requirements.txt) or edit this to point at the right interpreter." >&2
    exit 1
fi

# Fail before the job, not four hours in, if pyscicone (and PhenoGraph, imported at its
# module top) is missing or broken in this environment.
if ! "${PY}" -c "import scicone"; then
    echo "ERROR: 'import scicone' failed in ${PY}. Install pyscicone and its dependencies: pip install ${REPO_ROOT}/realdata/external/SCICoNE/pyscicone/" >&2
    exit 1
fi

if [ ! -d "${SCICONE_BUILD_DIR}" ]; then
    echo "ERROR: ${SCICONE_BUILD_DIR} not found -- build SCICoNE first (see header)." >&2
    exit 1
fi

if [ ! -f "${CNV_H5}" ]; then
    echo "ERROR: ${CNV_H5} not found -- run 00_download.sbatch first." >&2
    exit 1
fi

TREE_OUT_DIR="${PERSIST_DIR}/tree_input"
mkdir -p "${TREE_OUT_DIR}"

# Breakpoint detection and tree-search knobs; defaults live in build_cna_tree.py, override
# by exporting before sbatch. Empty means "use the script's default".
EXTRA_ARGS=()
[ -n "${BP_MAX_CELLS:-}" ] && EXTRA_ARGS+=(--bp-max-cells "${BP_MAX_CELLS}")
[ -n "${BP_WINDOW_SIZE:-}" ] && EXTRA_ARGS+=(--bp-window-size "${BP_WINDOW_SIZE}")
[ -n "${N_REPS:-}" ] && EXTRA_ARGS+=(--n-reps "${N_REPS}")
[ -n "${COPY_NUMBER_LIMIT:-}" ] && EXTRA_ARGS+=(--copy-number-limit "${COPY_NUMBER_LIMIT}")
[ -n "${CLUSTER_TREE_N_ITERS:-}" ] && EXTRA_ARGS+=(--cluster-tree-n-iters "${CLUSTER_TREE_N_ITERS}")
if [ "${PSEUDOBULK_FALLBACK:-0}" = "1" ]; then
    EXTRA_ARGS+=(--pseudobulk-fallback)
fi

"${PY}" "${EULER_SCRIPTS}/build_cna_tree.py" \
    --cnv-h5 "${CNV_H5}" \
    --map-file "${PILEUP_DIR}/chromosome_1.map" \
    --clustering-file "${CLUSTERING_DIR}/clustering" \
    --out-dir "${TREE_OUT_DIR}" \
    --scicone-build-dir "${SCICONE_BUILD_DIR}" \
    --sex "${PATIENT_SEX}" \
    --normal-cluster-id "${NORMAL_CLUSTER_ID}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "Wrote cna_tree.nwk and cna_tree_diagnostics.txt to ${TREE_OUT_DIR}."
