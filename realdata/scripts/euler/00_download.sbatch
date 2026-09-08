#!/bin/bash
#SBATCH --job-name=secedo-D-download
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=%x-%j.log
#
# Downloads slice D (BAM + BAI + per-cell summary CSV) and the GRCh37 reference into
# scratch. Slice D only -- do not adapt this loop back to "A B C D E".

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

echo "Downloading slice ${SLICE} BAM/BAI/summary into ${RAW_DIR}..."
wget -c -O "${RAW_DIR}/breast_tissue_D_2k_possorted_bam.bam" "${BAM_URL}"
wget -c -O "${RAW_DIR}/breast_tissue_D_2k_possorted_bam.bam.bai" "${BAI_URL}"
wget -c -O "${RAW_DIR}/breast_tissue_D_2k_per_cell_summary_metrics.csv" "${SUMMARY_URL}"

echo "Downloading GRCh37 reference into ${REF_DIR}..."
wget -c -O "${REF_DIR}/GRCh37.p13.genome.fa.gz" "${REF_FASTA_GZ_URL}"
gunzip -f "${REF_DIR}/GRCh37.p13.genome.fa.gz"
mv "${REF_DIR}/Homo_sapiens.GRCh37.dna.primary_assembly.fa" "${REF_FASTA}" 2>/dev/null || true
# gunzip preserves the archive member's own name; normalise it to what clustering.sh/
# mutect.sh expect (GRCh37.p13.genome.fa) if it came out under the Ensembl filename.
if [ ! -f "${REF_FASTA}" ]; then
    found="$(find "${REF_DIR}" -maxdepth 1 -name '*.fa' | head -n1)"
    mv "${found}" "${REF_FASTA}"
fi

module load samtools 2>/dev/null || true  # adjust to the module name/version on Euler
samtools faidx "${REF_FASTA}"

echo "Downloading dbSNP (public GRCh37 substitute for the original build-132 file --" \
     "see the plan doc) into ${REF_DIR}..."
wget -c -O "${DBSNP_VCF}.gz" "${DBSNP_VCF_GZ_URL}"
gunzip -f "${DBSNP_VCF}.gz"
# Not consumed by 06_mutect.sbatch any more (GATK4 Mutect2 has no --dbsnp argument);
# staged anyway in case it's wanted for a future --germline-resource-style refinement.

module load gatk 2>/dev/null || true   # adjust to the module name/version on Euler; if
                                        # Euler has none, install gatk4 into the pipeline
                                        # conda/venv instead -- see config.sh
if command -v gatk >/dev/null 2>&1; then
    gatk --version
    if [ ! -f "${REF_FASTA%.fa}.dict" ]; then
        echo "Creating the reference sequence dictionary GATK4 needs alongside the .fai..."
        gatk CreateSequenceDictionary -R "${REF_FASTA}"
    fi
else
    echo "gatk not found on PATH after 'module load gatk' -- install it into the" \
         "pipeline conda/venv (see config.sh) before running stage 06." >&2
fi

if [ ! -f "${COSMIC_VCF}" ]; then
    echo "COSMIC_VCF not found at ${COSMIC_VCF}."
    echo "COSMIC requires your own registered account -- download"
    echo "cosmic_v94_hg37_coding_and_noncoding.vcf (or your closest available GRCh37"
    echo "COSMIC release) by hand and place it there, or export COSMIC_VCF=<path>."
    echo "Not fetched automatically. Note: unused by 06_mutect.sbatch as of the Mutect2"
    echo "switch (no equivalent to MuTect1's --cosmic) -- staged for provenance/future use"
    echo "only, no longer a prerequisite for stage 06 to run."
fi

echo "Slice ${SLICE} download complete."
du -sh "${RAW_DIR}" "${REF_DIR}"