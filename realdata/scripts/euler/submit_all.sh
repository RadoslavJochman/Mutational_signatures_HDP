#!/bin/bash
# Orchestrates the slice-D pipeline as a chain of dependent sbatch submissions. NOT run
# automatically by anything -- review config.sh (PERSIST_DIR, NORMAL_CLUSTER_ID after
# stage 05, COSMIC_VCF, MUTECT_JAVA/MUTECT_JAR) and this file, then submit by hand, one
# stage at a time or all at once via --dependency chaining as written here.
#
# NORMAL_CLUSTER_ID cannot be known before stage 05 runs, so this script pauses there:
# rerun it (cheap, samtools merge only) after setting NORMAL_CLUSTER_ID to continue into
# mutect. 06's --array bound is likewise computed here from tasks.tsv, not hardcoded.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

j00=$(sbatch --parsable 00_download.sbatch)
j01=$(sbatch --parsable --dependency=afterok:${j00} 01_preprocess.sbatch)

n_cells=$(find "${SCRATCH}/secedo_slice_d/cell_bams" -maxdepth 1 -name '*.bam' 2>/dev/null | wc -l)
n_cells=${n_cells:-1999}  # placeholder until stage 01 has actually run
j02=$(sbatch --parsable --dependency=afterok:${j01} --array=0-$((n_cells - 1))%200 \
    02_split_chrom.sbatch)

j03=$(sbatch --parsable --dependency=afterok:${j02} 03_pileup.sbatch)
j04=$(sbatch --parsable --dependency=afterok:${j03} 04_secedo.sbatch)
j05=$(sbatch --parsable --dependency=afterok:${j04} 05_build_cluster_bams.sbatch)

echo "Submitted through stage 05 (job ${j05})."
echo "Once it completes: inspect \${SCRATCH}/secedo_slice_d/cluster_bams, set" \
     "NORMAL_CLUSTER_ID in config.sh, then run:"
echo
echo '  sbatch 05_build_cluster_bams.sbatch   # regenerates tasks.tsv with the normal excluded'
echo '  n=$(wc -l < "${SCRATCH}/secedo_slice_d/mutect/tasks.tsv")'
echo '  j06=$(sbatch --parsable --array=0-$((n - 1)) 06_mutect.sbatch)'
echo '  sbatch --dependency=afterany:${j06} 07_copy_out.sbatch'
echo
echo "07_copy_out.sbatch uses --dependency=afterany, not afterok, so the must-have copy" \
     "step still runs (and reports what did/didn't finish) even if some mutect array" \
     "tasks fail -- never leaves a completed subset stranded in scratch."