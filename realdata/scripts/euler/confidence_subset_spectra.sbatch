#!/bin/bash
#SBATCH --job-name=secedo-D-confidence-subsets
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=%x-%j.log
#
# Reruns build_snv_tree.py three times at different presence thresholds, to
# check whether the full-catalogue refit's flat (SBS3/SBS40a) and artefact
# (SBS95) components come from weak read support rather than real biology
# (see realdata/scripts/refit_confidence_subsets.py, run locally afterwards).
# Each run is the exact same code path as the normal stage 8 run -- the
# same parse_forced_vcf_calls, classify_cluster_states and
# bin_cluster_spectra, nothing reparsed differently here -- just with a
# tighter --presence-min-alt-reads/--presence-min-vaf, writing its own
# spectra.csv (plus clone_snv_matrix.csv, diagnostics, etc.) to its own
# subdirectory under tree_input/confidence_subsets/. The Dollo search still
# runs (it is not worth adding a flag to skip it for five clusters), but
# its bootstrap is turned off (--dollo-bootstrap 0) since only spectra.csv
# is needed here.
#
# Subsets:
#   a  presence-min-alt-reads=2, presence-min-vaf=0.05  (today's default,
#      i.e. spectra.csv as stage 8 already writes it -- rerun here too so
#      all three sit side by side under the same confidence_subsets/ root)
#   b  presence-min-alt-reads=4, presence-min-vaf=0.05  (drop low-alt-read
#      calls classify_snv_state would still call present)
#   c  presence-min-alt-reads=3, presence-min-vaf=0.15  (a stricter VAF
#      floor on top)
#
# Only spectra.csv from each subdirectory needs copying down locally --
# never the VCFs.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/config.sh"

module load stack/2024-06 gcc/12.2.0 2>/dev/null || true

PY="${REPO_ROOT}/.venv/bin/python"
if [ ! -x "${PY}" ]; then
    echo "ERROR: ${PY} not found. Create the repo venv with pandas/numpy/networkx/phylox/pysam." >&2
    exit 1
fi

SUBSET_ROOT="${PERSIST_DIR}/tree_input/confidence_subsets"

run_subset () {
    local name="$1" min_alt="$2" min_vaf="$3"
    local out_dir="${SUBSET_ROOT}/${name}"
    mkdir -p "${out_dir}"
    echo "== subset ${name}: presence-min-alt-reads=${min_alt} presence-min-vaf=${min_vaf} =="
    "${PY}" "${EULER_SCRIPTS}/build_snv_tree.py" \
        --vcf-dir "${PERSIST_DIR}/mutect_vcfs" \
        --presence-min-vaf "${min_vaf}" \
        --presence-min-alt-reads "${min_alt}" \
        --ref-fasta "${REF_FASTA}" \
        --cosmic-signatures "${REPO_ROOT}/COSMIC_sig/cosmic_signatures.csv" \
        --out-dir "${out_dir}" \
        --normal-cluster-id "${NORMAL_CLUSTER_ID}" \
        --dollo-bootstrap 0
}

run_subset a 2 0.05
run_subset b 4 0.05
run_subset c 3 0.15

echo
echo "Wrote spectra.csv (and clone_snv_matrix.csv, diagnostics) for subsets" \
     "a/b/c under ${SUBSET_ROOT}."
echo "Copy back only spectra.csv from each, e.g. from a machine with scp access:"
echo "  scp <user>@euler.ethz.ch:${SUBSET_ROOT}/a/spectra.csv realdata/local_tree_input/confidence_subsets/a/spectra.csv"
echo "  scp <user>@euler.ethz.ch:${SUBSET_ROOT}/b/spectra.csv realdata/local_tree_input/confidence_subsets/b/spectra.csv"
echo "  scp <user>@euler.ethz.ch:${SUBSET_ROOT}/c/spectra.csv realdata/local_tree_input/confidence_subsets/c/spectra.csv"
