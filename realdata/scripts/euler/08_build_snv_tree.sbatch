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
# Dollo parsimony over trees with hidden internal nodes is the sole source of
# snv_tree.nwk: real slice D data needs a hidden ancestor (dropout, not independent
# gains, explains the dominant near-complete presence patterns), which neither LICHeE
# nor Camin-Sokal can represent. LICHeE is an optional comparison only (RUN_LICHEE=1
# below), never used for snv_tree.nwk.
#
# Needs a python environment with pandas, numpy, networkx, phylox, and pysam
# installed (the repo's own .venv works if present on this cluster). pysam is
# the one dependency this stage adds beyond what run_inference.py needs --
# pinned in requirements.txt. A JDK and LICHeE built at LICHEE_HOME (config.sh:
# release/lichee.jar and lib/) are needed only when RUN_LICHEE=1.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/config.sh"

TREE_OUT_DIR="${PERSIST_DIR}/tree_input"
mkdir -p "${TREE_OUT_DIR}"

# pysam's htslib libs need the stack at runtime, same as the other stages.
module load stack/2024-06 gcc/12.2.0 2>/dev/null || true

# SLURM jobs don't inherit an interactively-activated conda/venv, so use the
# repo's own venv interpreter explicitly rather than ambient python3.
PY="${REPO_ROOT}/.venv/bin/python"
if [ ! -x "${PY}" ]; then
    echo "ERROR: ${PY} not found. Create the repo venv with pandas/numpy/networkx/phylox/pysam, or edit this to point at the right interpreter." >&2
    exit 1
fi

# Dollo search/bootstrap knobs; defaults live in build_snv_tree.py, override by
# exporting before sbatch. Empty means "use the script's default".
DOLLO_ARGS=()
[ -n "${DOLLO_MAX_CLUSTERS:-}" ] && DOLLO_ARGS+=(--dollo-max-clusters "${DOLLO_MAX_CLUSTERS}")
[ -n "${DOLLO_BOOTSTRAP:-}" ] && DOLLO_ARGS+=(--dollo-bootstrap "${DOLLO_BOOTSTRAP}")
[ -n "${DOLLO_MIN_CLADE_SUPPORT:-}" ] && DOLLO_ARGS+=(--dollo-min-clade-support "${DOLLO_MIN_CLADE_SUPPORT}")
[ -n "${SEED:-}" ] && DOLLO_ARGS+=(--seed "${SEED}")

# Compare against the CNA tree automatically once stage 9 has produced one.
CNA_TREE="${TREE_OUT_DIR}/cna_tree.nwk"
if [ -f "${CNA_TREE}" ]; then
    DOLLO_ARGS+=(--compare-tree "${CNA_TREE}")
fi

# LICHeE: optional comparison only (RUN_LICHEE=1), never snv_tree.nwk's source. Needs
# a JDK: openjdk/11 is confirmed to work; override with JAVA_MODULE if Euler's naming
# moves.
RUN_LICHEE="${RUN_LICHEE:-0}"
LICHEE_ARGS=()
if [ "${RUN_LICHEE}" = "1" ]; then
    LICHEE_ARGS+=(--run-lichee --lichee-home "${LICHEE_HOME}")
    [ -n "${LICHEE_MIN_VAF_PRESENT:-}" ] && LICHEE_ARGS+=(--lichee-min-vaf-present "${LICHEE_MIN_VAF_PRESENT}")
    [ -n "${LICHEE_MAX_VAF_ABSENT:-}" ] && LICHEE_ARGS+=(--lichee-max-vaf-absent "${LICHEE_MAX_VAF_ABSENT}")
    [ -n "${LICHEE_MIN_CLUSTER_SIZE:-}" ] && LICHEE_ARGS+=(--lichee-min-cluster-size "${LICHEE_MIN_CLUSTER_SIZE}")
    [ -n "${LICHEE_ERROR_MARGIN:-}" ] && LICHEE_ARGS+=(--lichee-error-margin "${LICHEE_ERROR_MARGIN}")

    JAVA_MODULE="${JAVA_MODULE:-openjdk/11}"
    module load "${JAVA_MODULE}" 2>/dev/null || true
    if ! command -v java >/dev/null 2>&1; then
        echo "ERROR: java not on PATH (JAVA_MODULE=${JAVA_MODULE}); load a JDK or unset RUN_LICHEE." >&2
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
    --absent-expected-vaf "${ABSENT_EXPECTED_VAF}" \
    --absent-alpha "${ABSENT_ALPHA}" \
    --ref-fasta "${REF_FASTA}" \
    --cosmic-signatures "${REPO_ROOT}/COSMIC_sig/cosmic_signatures.csv" \
    --out-dir "${TREE_OUT_DIR}" \
    --normal-cluster-id "${NORMAL_CLUSTER_ID}" \
    ${DOLLO_ARGS[@]+"${DOLLO_ARGS[@]}"} \
    ${LICHEE_ARGS[@]+"${LICHEE_ARGS[@]}"}

echo "Wrote snv_tree.nwk, spectra.csv, clone_snv_matrix.csv, clone_snv_states.csv," \
     "clone_snv_unknown_reason.csv, snv_tree_diagnostics.txt to ${TREE_OUT_DIR}."
