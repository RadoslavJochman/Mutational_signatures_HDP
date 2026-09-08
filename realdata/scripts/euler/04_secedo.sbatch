#!/bin/bash
#SBATCH --job-name=secedo-D-cluster
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=1200M
#SBATCH --output=%x-%j.log
#
# Translates clustering.sh's variant_calling step for slice D alone. Same
# homozygous_filtered_rate/seq_error_rate override as the original (the command-line
# values, not flags_breast's file defaults), same clustering_type and max_coverage.
#
# Deliberate deviation: no --merge_file. That flag pools multiple cells' pileups to
# artificially inflate coverage; secedo's own flag help says it is "only works on
# synthetic data where we know consecutive cells are part of the same cluster" --
# not applicable to slice D's real, unmerged single cells, and there is no
# breast_group_D_* file in secedo-evaluation to begin with (only ABCDE/B-specific
# ones exist, built for their multi-slice coverage-pooling runs). Omitting it makes
# secedo use the identity grouping (one group per real cell), which is the correct,
# faithful behaviour for a single real slice, not a workaround.
#
# ~16GB was reported sufficient for a single slice at the default pileup
# --min_different=3 (see clustering.sh's own comment); 24GB (20 x 1200M) leaves margin.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/config.sh"
module load openblas 2>/dev/null || true  # adjust to the module name/version on Euler

"${SECEDO_BIN}" -i "${PILEUP_DIR}/" -o "${CLUSTERING_DIR}/" --num_threads 20 \
    --log_level=trace --flagfile "${FLAGS_BREAST}" \
    --homozygous_filtered_rate=0.5 --seq_error_rate=0.05 --min_cluster_size 500 \
    --reference_genome="${REF_FASTA}" \
    --clustering_type SPECTRAL6 --max_coverage 300

echo "Clustering and variant calling done. Contents of ${CLUSTERING_DIR}:"
ls -la "${CLUSTERING_DIR}"