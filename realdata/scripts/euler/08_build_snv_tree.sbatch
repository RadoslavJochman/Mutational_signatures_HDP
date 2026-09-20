#!/bin/bash
#SBATCH --job-name=secedo-D-buildsnvtree
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=%x-%j.log
#
# Ours -- turns stage 06b's per-cluster force-called VCFs into the SNV-side inputs
# TreeHDP consumes: a rooted Newick tree (an implicit, spectrum-less germline root --
# GERMLINE_ROOT_ID, not the pseudo-normal cluster -- labelled-internal-node form) and a
# per-cluster 96-channel spectra matrix in cosmic_signatures.csv's channel order. See
# build_snv_tree.py's module docstring for the tree-construction and channel-binning
# contracts. (The CNA-side tree is built separately by stage 09/build_cna_tree.py.)
#
# LICHeE is the primary tree builder; an in-house Camin-Sokal parsimony search is the
# automatic fallback (not debug-only) if LICHeE is unavailable, fails, or its
# .trees.txt output cannot be parsed and joined to the clusters. LICHeE is run through
# java -cp (see build_snv_tree.py's module docstring) and its .trees.txt is parsed;
# -dot is never used, so no display or Xvfb is needed. It builds a tree over
# mutation-presence groups, so several clusters can share one node: fewer distinct
# nodes than clusters is expected, not an error.
#
# Needs a python environment with pandas, numpy, networkx, phylox, and pysam
# installed (the repo's own .venv works if present on this cluster). pysam is
# the one dependency this stage adds beyond what run_inference.py needs --
# pinned in requirements.txt. Also needs a JDK (JAVA_MODULE below) and LICHeE built at
# LICHEE_HOME (config.sh): release/lichee.jar and lib/. Pass --skip-lichee
# (SKIP_LICHEE=1 below) to go straight to Camin-Sokal if it isn't available yet.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/config.sh"

TREE_OUT_DIR="${PERSIST_DIR}/tree_input"
mkdir -p "${TREE_OUT_DIR}"

# pysam's htslib libs need the stack at runtime, same as the other stages. LICHeE needs a
# JDK: openjdk/11 is confirmed to work; override with JAVA_MODULE if Euler's naming moves.
JAVA_MODULE="${JAVA_MODULE:-openjdk/11}"
module load stack/2024-06 gcc/12.2.0 2>/dev/null || true
module load "${JAVA_MODULE}" 2>/dev/null || true

# SLURM jobs don't inherit an interactively-activated conda/venv, so use the
# repo's own venv interpreter explicitly rather than ambient python3.
PY="${REPO_ROOT}/.venv/bin/python"
if [ ! -x "${PY}" ]; then
    echo "ERROR: ${PY} not found. Create the repo venv with pandas/numpy/networkx/phylox/pysam, or edit this to point at the right interpreter." >&2
    exit 1
fi

SKIP_LICHEE="${SKIP_LICHEE:-0}"

SKIP_LICHEE_ARGS=()
if [ "${SKIP_LICHEE}" = "1" ]; then
    SKIP_LICHEE_ARGS=(--skip-lichee)
else
    if ! command -v java >/dev/null 2>&1; then
        echo "ERROR: java not on PATH (JAVA_MODULE=${JAVA_MODULE}); load a JDK or set SKIP_LICHEE=1." >&2
        exit 1
    fi
    # Fail the job now, not partway through, if LICHeE cannot launch. Judged on the
    # error text, not on -h's exit status, which is not reliable. The lib/* glob is one
    # -cp argument for Java to expand, hence the quotes.
    lichee_check=$(java -cp "${LICHEE_HOME}/release/lichee.jar:${LICHEE_HOME}/lib/*" \
        lineage.LineageEngine -h 2>&1) || true
    if echo "${lichee_check}" | grep -q -E "Could not find or load main class|NoClassDefFoundError|ClassNotFoundException"; then
        echo "ERROR: LICHeE cannot launch from ${LICHEE_HOME} (are release/lichee.jar and lib/ present?). Output:" >&2
        echo "${lichee_check}" >&2
        exit 1
    fi
fi

"${PY}" "${EULER_SCRIPTS}/build_snv_tree.py" \
    --vcf-dir "${PERSIST_DIR}/mutect_vcfs" \
    --presence-min-vaf "${PRESENCE_MIN_VAF}" \
    --presence-min-alt-reads "${PRESENCE_MIN_ALT_READS}" \
    --ref-fasta "${REF_FASTA}" \
    --cosmic-signatures "${REPO_ROOT}/COSMIC_sig/cosmic_signatures.csv" \
    --out-dir "${TREE_OUT_DIR}" \
    --lichee-home "${LICHEE_HOME}" \
    ${SKIP_LICHEE_ARGS[@]+"${SKIP_LICHEE_ARGS[@]}"}

echo "Wrote snv_tree.nwk, spectra.csv, clone_snv_matrix.csv, snv_tree_diagnostics.txt" \
     "to ${TREE_OUT_DIR}."