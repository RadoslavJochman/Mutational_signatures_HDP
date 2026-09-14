#!/bin/bash
#SBATCH --job-name=secedo-D-buildtree
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=%x-%j.log
#
# Ours -- turns stage 06b's per-cluster force-called VCFs into the two inputs TreeHDP
# consumes: a rooted Newick tree (an implicit, spectrum-less germline root --
# GERMLINE_ROOT_ID, not a pseudo-normal cluster -- labelled-internal-node form) and a
# per-cluster 96-channel spectra matrix in cosmic_signatures.csv's channel order. See
# build_tree.py's module docstring for the tree-construction and channel-binning
# contracts, and the plan doc's stage-8 entry for what this reads and writes.
#
# SCITE is the primary tree builder (build_tree.py exits non-zero if it is
# unavailable or fails); mutation-set containment is a cross-check reported
# in tree_diagnostics.txt only, not what gets emitted as tree.nwk.
#
# Needs a python environment with pandas, numpy, networkx, phylox, and pysam
# installed (the repo's own .venv works if present on this cluster). pysam is
# the one dependency this stage adds beyond what run_inference.py needs --
# pinned in requirements.txt. Also needs the SCITE binary built at
# realdata/external/scite/scite (see SCITE_BIN below).

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/config.sh"

TREE_OUT_DIR="${PERSIST_DIR}/tree_input"
mkdir -p "${TREE_OUT_DIR}"

# pysam's htslib libs, and SCITE itself, may need the stack at runtime (SCITE
# was built against it), same as the other stages.
module load stack/2024-06 gcc/12.2.0 2>/dev/null || true

# SLURM jobs don't inherit an interactively-activated conda/venv, so use the
# repo's own venv interpreter explicitly rather than ambient python3.
PY="${REPO_ROOT}/.venv/bin/python"
if [ ! -x "${PY}" ]; then
    echo "ERROR: ${PY} not found. Create the repo venv with pandas/numpy/networkx/phylox/pysam, or edit this to point at the right interpreter." >&2
    exit 1
fi

# SCITE is the primary tree builder; build_tree.py refuses to fall back to the
# containment tree if it is missing or fails. Override any of these from the
# environment if the defaults don't suit a given run.
#
# build_tree.py only feeds SCITE the topology-informative SNVs (present in
# 2..k-1 of the tumour clusters; see filter_informative_snvs), which is what
# makes these defaults workable -- the unfiltered ~7500-SNV set at the old
# 5 x 1,000,000 defaults ran past 8h. SCITE_MAX_MUTATIONS is an additional
# safety valve on top of that filter, unset (no cap) unless overridden.
SCITE_BIN="${SCITE_BIN:-${REPO_ROOT}/realdata/external/scite/scite}"
SCITE_FD="${SCITE_FD:-1e-3}"
SCITE_AD="${SCITE_AD:-0.15}"
SCITE_RESTARTS="${SCITE_RESTARTS:-3}"
SCITE_CHAIN_LENGTH="${SCITE_CHAIN_LENGTH:-100000}"
SCITE_MAX_MUTATIONS="${SCITE_MAX_MUTATIONS:-}"

SCITE_MAX_MUTATIONS_ARGS=()
if [ -n "${SCITE_MAX_MUTATIONS}" ]; then
    SCITE_MAX_MUTATIONS_ARGS=(--scite-max-mutations "${SCITE_MAX_MUTATIONS}")
fi

"${PY}" "${EULER_SCRIPTS}/build_tree.py" \
    --vcf-dir "${PERSIST_DIR}/mutect_vcfs" \
    --presence-min-vaf "${PRESENCE_MIN_VAF}" \
    --presence-min-alt-reads "${PRESENCE_MIN_ALT_READS}" \
    --ref-fasta "${REF_FASTA}" \
    --cosmic-signatures "${REPO_ROOT}/COSMIC_sig/cosmic_signatures.csv" \
    --out-dir "${TREE_OUT_DIR}" \
    --scite-bin "${SCITE_BIN}" \
    --scite-fd "${SCITE_FD}" \
    --scite-ad "${SCITE_AD}" \
    --scite-restarts "${SCITE_RESTARTS}" \
    --scite-chain-length "${SCITE_CHAIN_LENGTH}" \
    ${SCITE_MAX_MUTATIONS_ARGS[@]+"${SCITE_MAX_MUTATIONS_ARGS[@]}"}

echo "Wrote tree.nwk, spectra.csv, clone_snv_matrix.csv, tree_diagnostics.txt" \
     "to ${TREE_OUT_DIR}."
