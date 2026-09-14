#!/bin/bash
#SBATCH --job-name=secedo-D-forcecall
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=%x-%A_%a.log
#
# Pass 2 of tumour-only + gnomAD somatic calling (see 06_mutect.sbatch's header for pass
# 1, which this depends on). Every cluster's pass-1 discovery is independent, so a
# mutation private to one cluster there could just as well be a true positive another
# cluster's own assembly happened to miss, as a genuine absence -- the clone x mutation
# presence matrix build_tree.py needs to nest requires every cluster to be genotyped at
# the SAME candidate sites, not "called independently, then compared." This stage unions
# pass 1's PASS SNP sites per chromosome, then force-calls every cluster at every site in
# that union, so presence/absence at each site comes from the same read-support test in
# every cluster.
#
# SUB-DECISION -- two SLURM-level phases, ONE script, dispatched by its first positional
# argument, run as two separate sbatch submissions with an explicit dependency between
# them. This repo's existing stages are already one-script-per-pipeline-stage with array
# bounds computed and dependencies chained externally (see submit_all.sh); SLURM has no
# clean way to make some tasks of one job array wait on others finishing first, so this
# follows that same externally-chained pattern rather than inventing an intra-array
# ordering hack (a sentinel file plus a poll-and-sleep loop in every one of 144 tasks was
# the alternative considered, and is strictly worse: it wastes allocated task time
# spinning, and still needs a timeout to fail loudly, which a real SLURM dependency
# already gives for free):
#
#   sbatch --parsable 06b_forcecall.sbatch union              # 1 task: builds all 24 chromosomes' union_sites_<chrom>.vcf
#   j=<the job id printed above>
#   sbatch --dependency=afterok:${j} --array=0-143 06b_forcecall.sbatch forcecall
#
# "union" ignores SLURM_ARRAY_TASK_ID and loops over all 24 chromosomes itself -- cheap
# text processing over already-small SNP-only VCFs, not worth its own array. "forcecall"
# reads tasks.tsv exactly as stage 06 does: the same 144 (cluster, chromosome) pairs.
#
# Force-calling fixed sites is much faster than pass 1's de novo assembly search (no
# variant discovery, just genotyping known positions), hence the shorter --time than
# 06_mutect.sbatch's 08:00:00.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/config.sh"

module load stack/2024-06 gcc/12.2.0 2>/dev/null || true
module load gatk/4.4.0.0 2>/dev/null || true

if ! command -v gatk >/dev/null 2>&1; then
    echo "gatk not found on PATH -- see config.sh's GATK note (module or conda install)." >&2
    exit 1
fi
if [ ! -f "${REF_FASTA%.fa}.dict" ]; then
    echo "Reference sequence dictionary (${REF_FASTA%.fa}.dict) not found -- GATK4 needs" \
         "it alongside the .fai. Stage 00 now creates this; rerun it if missing." >&2
    exit 1
fi
if [ ! -f "${GNOMAD_VCF}" ]; then
    echo "GNOMAD_VCF (${GNOMAD_VCF}) not found -- see config.sh's note." >&2
    exit 1
fi

mode="${1:-}"

case "${mode}" in
union)
    # SUB-DECISION: union sites are built with plain shell text processing (grep/awk/
    # sort/uniq), not bcftools -- bcftools is not among the modules this repo has
    # confirmed on Euler (only stack/2024-06, gatk, samtools; see config.sh's GATK note),
    # and GATK4 dropped CombineVariants (GATK3-only) and has nothing built for merging
    # different single-sample VCFs' sites into one union. This also runs no equivalent of
    # `bcftools norm`, so a site where two clusters' pass-1 calls disagree on REF (a
    # reference-build mismatch, which should not happen since both come from the same
    # REF_FASTA and Mutect2 build) would surface as a near-duplicate line at the same
    # position rather than being reconciled -- a real limitation of this approach, flagged
    # here rather than silently assumed away.
    for chrom in {1..22} X Y; do
        union_vcf="${MUTECT_DIR}/union_sites_${chrom}.vcf"
        {
            echo "##fileformat=VCFv4.2"
            printf '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n'
            for cluster_bam in "${CLUSTER_BAMS_DIR}"/clone*.bam; do
                cluster_id=$(basename "${cluster_bam}" .bam | sed 's/^clone//')
                disc_vcf="${MUTECT_DIR}/clone${cluster_id}_${chrom}.disc.vcf"
                [ -f "${disc_vcf}" ] || continue
                grep -v '^#' "${disc_vcf}" \
                    | awk -F'\t' -v OFS='\t' '{print $1, $2, ".", $4, $5, ".", ".", "."}'
            done | sort -k1,1 -k2,2n -k4,4 -k5,5 -u
        } > "${union_vcf}"
        echo "chrom ${chrom}: $(grep -vc '^#' "${union_vcf}") union sites -> ${union_vcf}"
    done
    ;;

forcecall)
    tasks_file="${MUTECT_DIR}/tasks.tsv"
    line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${tasks_file}")
    cluster=$(echo "${line}" | cut -f1)
    chrom=$(echo "${line}" | cut -f2)
    if [ -z "${cluster}" ]; then
        echo "No task at line $((SLURM_ARRAY_TASK_ID + 1)) of ${tasks_file}."
        exit 0
    fi

    # SUB-DECISION: union files live directly under MUTECT_DIR, named union_sites_<chrom>
    # -- one per chromosome, shared by every cluster's force-call task for that
    # chromosome, alongside (not under a separate subdirectory of) the per-(cluster,
    # chrom) VCFs already there, since nothing else currently needs a dedicated namespace
    # for them.
    union_vcf="${MUTECT_DIR}/union_sites_${chrom}.vcf"
    if [ ! -f "${union_vcf}" ]; then
        echo "ERROR: ${union_vcf} not found -- run and wait for" \
             "'sbatch 06b_forcecall.sbatch union' before the force-call array." >&2
        exit 1
    fi

    tumor_bam_tagged="${MUTECT_DIR}/clone${cluster}.tagged.bam"
    if [ ! -f "${tumor_bam_tagged}" ]; then
        echo "ERROR: ${tumor_bam_tagged} not found -- stage 06 (pass 1) retags each" \
             "cluster's BAM before calling; run it first." >&2
        exit 1
    fi

    raw_vcf="${MUTECT_DIR}/clone${cluster}_${chrom}.forced_raw.vcf.gz"
    forced_vcf="${MUTECT_DIR}/clone${cluster}_${chrom}.forced.vcf"

    echo "Mutect2 (force-call): cluster ${cluster}, chromosome ${chrom}, sites from ${union_vcf}"
    # SUB-DECISION: --force-call-filtered-alleles is the flag name given in the task spec;
    # not independently confirmed against gatk/4.4.0.0's own --help here (no cluster
    # access from this environment). If Mutect2 rejects it as an unrecognised argument,
    # check `gatk Mutect2 --help` on Euler for this build's actual flag for the same
    # intent -- force a genotype call at every --alleles site even if it would otherwise
    # be filtered (some GATK4 versions have used --genotype-filtered-alleles for this).
    gatk --java-options "-Xmx6g" Mutect2 \
        -R "${REF_FASTA}" \
        -I "${tumor_bam_tagged}" \
        --germline-resource "${GNOMAD_VCF}" \
        --intervals "${chrom}" \
        --max-mnp-distance 0 \
        --alleles "${union_vcf}" \
        --force-call-filtered-alleles \
        -O "${raw_vcf}"

    # FilterMutectCalls still runs (annotates FILTER, useful context), but unlike pass 1
    # there is no SelectVariants --exclude-filtered afterwards: this pass exists to
    # genotype every cluster at every union site, including sites where the cluster lacks
    # the mutation (force-called there NOT PASS, or with low/zero AF). build_tree.py reads
    # AF and AD directly off every record, PASS or not, to decide presence -- see its
    # module docstring and PRESENCE_MIN_VAF/PRESENCE_MIN_ALT_READS in config.sh.
    echo "FilterMutectCalls (force-called, PASS not enforced): cluster ${cluster}, chromosome ${chrom}"
    gatk --java-options "-Xmx6g" FilterMutectCalls \
        -R "${REF_FASTA}" \
        -V "${raw_vcf}" \
        --stats "${raw_vcf}.stats" \
        -O "${forced_vcf}"
    ;;

*)
    echo "Usage: sbatch 06b_forcecall.sbatch {union|forcecall}" >&2
    echo "  union:     one task, builds all 24 chromosomes' union_sites_<chrom>.vcf" \
         "from pass 1's *.disc.vcf files. Run and wait for this first." >&2
    echo "  forcecall: array task (0-143, one per tasks.tsv line), force-calls one" \
         "(cluster, chromosome) pair at that chromosome's union sites." >&2
    exit 1
    ;;
esac
