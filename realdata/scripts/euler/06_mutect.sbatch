#!/bin/bash
#SBATCH --job-name=secedo-D-mutect2
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=%x-%A_%a.log
#
# Translates mutect.sh (LSF bsub-per-cluster-per-chromosome -> one SLURM array task per
# (cluster, chromosome) pair, read from tasks.tsv written by stage 05). GATK4 Mutect2
# (Jack approved -- MuTect1's jar is discontinued and no longer obtainable). GRCh37
# throughout, matching the original.
#
# This is pass 1 (discovery) of tumour-only + gnomAD germline-resource calling, NOT the
# tumour-vs-pseudo-normal design this stage used to run: the pseudo-normal (sibling
# cluster) design made 85% of SNVs cluster-private, unable to support a tree -- there is no
# guarantee a sibling clone is genuinely diploid/background, so its "germline" subtraction
# was really just another tumour clone's private mutations being subtracted from
# everyone else. Tumour-only calling against GNOMAD_VCF (population allele frequencies)
# gives each cluster's absolute somatic status instead of a status relative to whichever
# cluster got picked as normal, which is what a phylogeny needs. Absolute status alone
# still does not make the six clusters' calls MUTUALLY CONSISTENT, though -- one cluster's
# independent assembly can miss a variant another cluster does support -- so pass 2
# (06b_forcecall.sbatch) unions this pass's PASS SNP sites per chromosome and force-calls
# every cluster at every union site, which is what actually lets the clone x mutation
# matrix nest. Every cluster runs through this pass (stage 05's tasks.tsv no longer
# excludes anything); NORMAL_CLUSTER_ID is vestigial (see config.sh).
#
# Three things the GATK4 swap needed that the MuTect1 recipe didn't -- see the plan doc's
# "MuTect1 -> Mutect2" note for the fuller writeup of what this changes in the output:
#
#  1. Per-cluster read-group retagging. build_cluster_bams.py (stage 05) merges per-cell
#     BAMs that all carry the *same* original 10x @RG SM tag (splitting/merging by cell
#     barcode never touches read groups). A single-sample Mutect2 run still wants a clean,
#     non-ambiguous SM on its one -I input rather than 10x's original shared tag, so each
#     cluster is still retagged once into clone<id>.tagged.bam with a unique synthetic SM
#     before calling Mutect2 -- flock-guarded since every chromosome task for the same
#     cluster (up to ~24 of them) races to do this first. (Under the old tumour-vs-normal
#     design this also kept Mutect2's -normal matching from misreading two same-SM BAMs as
#     both "normal"; -normal is gone now, but the retag is still the right thing to feed a
#     single-sample caller.)
#  2. FilterMutectCalls. Mutect2's raw output is unfiltered candidate calls with FILTER
#     annotations, not a finished call set the way MuTect1's was -- PASS variants only are
#     kept.
#  3. SNP-only restriction. Mutect2 is assembly-based and calls indels too, which
#     MuTect1's pileup-based test never could and which the 96-channel SBS binning this
#     pipeline feeds has no use for -- filtered out here so stage 07's somatic-count
#     summary keeps meaning what it meant before (somatic SNVs, not SNVs+indels).
#     --max-mnp-distance 0 (below) turns off Mutect2's merging of adjacent phased
#     substitutions into a single MNP record, so the SNP-only select only ever removes
#     indels -- an MNP record would carry two SBS events at once, and dropping it wholesale
#     would have quietly lost both from the 96-channel spectra.
#
# COSMIC_VCF and DBSNP_VCF are still not used by this stage: neither maps onto anything in
# the Mutect2/FilterMutectCalls argument set (see config.sh's note). GNOMAD_VCF is the
# germline resource now (--germline-resource below) -- population allele frequencies,
# which is what GATK4 actually expects there; DBSNP_VCF's b37 dbSNP file has no AF field
# and could not serve this role even if substituted in.
#
# Every gatk call below pins the JVM heap with --java-options "-Xmx6g": left to its
# default, the GATK JVM sizes its heap from the node's total RAM, not the SLURM cgroup's
# share of it, so an unpinned heap can overshoot the 8G (2 cpus x 4G/cpu) this job
# actually gets and be killed for exceeding its cgroup memory limit.
#
# --time raised from the tumour-vs-normal design's 04:00:00 to 08:00:00: tumour-only
# discovery has no matched-normal read pileup to help Mutect2's assembly step prune
# candidate haplotypes early, so a chromosome task here can run longer than the old
# tumour-vs-normal equivalent did.
#
# --array bound below is a placeholder -- set it to $(wc -l < tasks.tsv) - 1 after stage
# 05 writes tasks.tsv (144 lines: every populated cluster x 24 chromosomes).

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/config.sh"

module load stack/2024-06 gcc/12.2.0 2>/dev/null || true
module load gatk/4.4.0.0 2>/dev/null || true
module load samtools 2>/dev/null || true  # needed here for the read-group retagging step

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
    echo "GNOMAD_VCF (${GNOMAD_VCF}) not found -- see config.sh's note; it lives on" \
         "scratch and is not fetched by any script here yet." >&2
    exit 1
fi

tasks_file="${MUTECT_DIR}/tasks.tsv"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${tasks_file}")
cluster=$(echo "${line}" | cut -f1)
chrom=$(echo "${line}" | cut -f2)
if [ -z "${cluster}" ]; then
    echo "No task at line $((SLURM_ARRAY_TASK_ID + 1)) of ${tasks_file}."
    exit 0
fi

tumor_bam="${CLUSTER_BAMS_DIR}/clone${cluster}.bam"

# Give each cluster's merged BAM its own read-group SM before Mutect2 sees it (see point 1
# above). Cached under MUTECT_DIR and flock-guarded so the ~24 chromosome tasks sharing a
# cluster don't race to retag/index the same file concurrently. Also read by
# 06b_forcecall.sbatch (pass 2), which reuses this same tagged BAM.
tag_bam() {
    local cluster_id="$1" src="$2"
    local tagged="${MUTECT_DIR}/clone${cluster_id}.tagged.bam"
    local lock="${MUTECT_DIR}/.clone${cluster_id}.tag.lock"
    (
        flock 9
        if [ ! -f "${tagged}" ]; then
            local rg_line="@RG"$'\t'"ID:clone${cluster_id}"$'\t'"SM:clone${cluster_id}"
            samtools addreplacerg -r "${rg_line}" -O bam -o "${tagged}.tmp" "${src}"
            samtools index "${tagged}.tmp"
            mv "${tagged}.tmp" "${tagged}"
            mv "${tagged}.tmp.bai" "${tagged}.bai"
        fi
    ) 9>"${lock}"
    echo "${tagged}"
}
tumor_bam_tagged=$(tag_bam "${cluster}" "${tumor_bam}")

raw_vcf="${MUTECT_DIR}/clone${cluster}_${chrom}.raw.vcf.gz"
filtered_vcf="${MUTECT_DIR}/clone${cluster}_${chrom}.filtered.vcf.gz"
out_vcf="${MUTECT_DIR}/clone${cluster}_${chrom}.disc.vcf"

echo "Mutect2 (tumour-only, pass 1 discovery): cluster ${cluster}, chromosome ${chrom}"
gatk --java-options "-Xmx6g" Mutect2 \
    -R "${REF_FASTA}" \
    -I "${tumor_bam_tagged}" \
    --germline-resource "${GNOMAD_VCF}" \
    --intervals "${chrom}" \
    --max-mnp-distance 0 \
    -O "${raw_vcf}"

echo "FilterMutectCalls: cluster ${cluster}, chromosome ${chrom}"
gatk --java-options "-Xmx6g" FilterMutectCalls \
    -R "${REF_FASTA}" \
    -V "${raw_vcf}" \
    --stats "${raw_vcf}.stats" \
    -O "${filtered_vcf}"

# PASS-only, SNPs only -- see point 3 above. --max-mnp-distance 0 on the Mutect2 call
# keeps adjacent phased substitutions as separate SNP records instead of merging them
# into MNPs, so this select now only removes indels; nothing SNP-typed is dropped.
# ".disc.vcf" (discovery), not the old cluster-vs-normal ".vcf" name -- 06b_forcecall.sbatch
# unions these per chromosome and force-calls every cluster at the union, writing
# ".forced.vcf", which is what build_tree.py now reads.
gatk --java-options "-Xmx6g" SelectVariants \
    -R "${REF_FASTA}" \
    -V "${filtered_vcf}" \
    --exclude-filtered \
    --select-type-to-include SNP \
    -O "${out_vcf}"
