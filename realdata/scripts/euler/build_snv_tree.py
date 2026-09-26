"""Turn stage 06/06b's per-cluster somatic VCFs into the two SNV-side inputs
TreeHDP consumes: a rooted Newick tree and a per-cluster 96-channel spectra
matrix. (The CNA-side tree -- SCICoNE over copy-number bins, collapsed onto
the same SECEDO clusters -- is built separately by build_cna_tree.py; the two
trees share the Newick contract below and are run through TreeHDP
independently for comparison.)

Ours: neither secedo nor secedo-evaluation builds a phylogeny over SECEDO's
clusters or bins VCFs into COSMIC's 96 SBS channels. This is that step (recipe
stage 8, `realdata/recipe/euler_slice_d_plan.md`).

Contract (from `src/models/hdp_inference.py`'s `_BaseTreeHDP`, do not "fix" these)
    - Newick is labelled-internal-node form, e.g. ``((c2,c3)c1)germline;``.
      Observed clusters sit at internal nodes carrying their SECEDO cluster-ID
      label when a tool's own node happens to coincide with one (LICHeE's
      integration, kept as an optional comparison -- see below); Dollo, the
      primary method, always makes clusters leaves, since no SECEDO cluster
      is ancestral to another (each is a sampled population), and every
      internal node is a hidden ancestor, labelled ``g<k>`` and passed to
      ``verify_newick`` via ``hidden_ids``. Either way, every node that is
      NOT in ``hidden_ids`` is a real, spectrum-bearing cluster.
    - The tree is rooted at ``GERMLINE_ROOT_ID``, an implicit, spectrum-less
      latent node standing for the germline/empty-mutation state -- not a
      pseudo-normal cluster (see below). It carries no spectrum and is never a
      column in the presence matrix or the spectra table, which the model
      handles natively.
    - Node labels in the Newick, the spectra index, and the SECEDO cluster IDs
      are one ID system throughout.
    - The 96 spectra columns are in cosmic_signatures.csv's exact channel
      order, because the model does ``dot(activities, signatures)`` and aligns
      observed counts to signature columns positionally. ``main`` asserts this
      before doing anything else and fails loudly if it does not hold.

Calling design: tumour-vs-pseudo-normal (reverted from Attempt 2's tumour-only
+ gnomAD design, which made 85% of SNVs cluster-private and could not support
a tree). One SECEDO cluster (``NORMAL_CLUSTER_ID``, config.sh, asserted absent
from the discovered cluster set by ``--normal-cluster-id``) is Mutect2's
-normal and never called as a tumour cluster itself -- it has no VCF, is
absent from ``spectra.csv``, and is not a column in the presence matrix.
Because independently-assembled clusters can each miss a variant another
cluster's own assembly supports, stage 06b unions every tumour cluster's
pass-1 PASS SNP sites per chromosome and force-calls every cluster at that
union, so presence (and VAF) at every site comes from the same read-support
test everywhere. This script reads that force-called output
(``*.forced.vcf``), not stage 06's pass-1 VCFs directly -- with two
qualifications confirmed on a real run:

    Sample selection by name. The forced VCFs are tumour-vs-normal Mutect2
    output, so each has TWO sample columns (the tumour ``clone<id>`` and the
    pseudo-normal), in GATK's sorted-sample-name order -- not tumour-first.
    ``parse_forced_vcf_calls`` selects the tumour's column BY NAME from the
    ``#CHROM`` header, never by position: reading a fixed column silently
    reads the pseudo-normal's genotype for any cluster ID sorting after
    ``NORMAL_CLUSTER_ID``.

    Union restriction. Force-calling adds the union alleles to Mutect2's
    interval, but Mutect2 still emits its own discovery calls on top, and the
    forced VCF keeps non-PASS records -- so a forced VCF's own records are not
    already restricted to the union. Presence is decided only at union sites,
    the whole purpose of the force-call design, so this script rebuilds that
    union itself from pass-1's PASS VCFs (already in ``--vcf-dir``, exactly as
    stage 06b builds its own union, so this does not depend on stage 06b's
    ``union_sites_<chrom>.vcf`` under scratch) and ignores any forced record
    outside it (``restrict_calls_to_union``), recording how many were dropped
    per cluster in ``snv_tree_diagnostics.txt``. ``assert_presence_within_union``
    then checks no cluster's present count exceeds the union size, as a sign
    the restriction actually took.

A site counts as PRESENT in a cluster if its force-called record has VAF >=
``--presence-min-vaf`` and ALT read depth >= ``--presence-min-alt-reads``
(defaults from config.sh's PRESENCE_MIN_VAF/PRESENCE_MIN_ALT_READS); ABSENT
otherwise.

Tree construction: Dollo parsimony over trees with hidden internal nodes
(``dollo_tree``) is the SOLE source of ``snv_tree.nwk``. Neither SCITE nor the
mutation-set containment heuristic this module used to carry is used any
more (SCITE is designed for single-cell genotype matrices, mismatched to
SECEDO's pseudobulk clusters; containment assumes a perfect phylogeny real
calls do not satisfy -- see git history for the retired
``build_clone_tree``/``containment_fraction`` code), and neither are LICHeE
or Camin-Sokal, both tried and rejected on real slice D data before Dollo:

    LICHeE (Popic et al. 2015), at the tau this pipeline's presence calls
    already use (0.05), found 0 valid trees: pervasive dropout (a trunk or
    clade mutation missed by one cluster's own force-called genotype) makes
    its hard absence/presence constraints unsatisfiable. Widening its own
    ambiguity band (``-maxVAFAbsent`` above ``-minVAFPresent``) does not
    rescue it -- it instead turns every cluster's low-level leakage into
    "shared", collapsing the whole tree to one node.

    An in-house Camin-Sokal parsimony search (kept until this change,
    exhaustive over every rooted topology with clusters as its only
    candidate internal nodes) returned a star: it can only place an
    OBSERVED cluster as an internal node and forbids losses entirely, so it
    cannot represent a hidden ancestor -- exactly what this presence matrix
    needs. Its dominant patterns are "all but one cluster" (e.g. 4 of 5
    tumour clusters sharing a mutation the fifth's own force-called
    genotype missed), which look like dropout from a real clade, not
    independent gains in every cluster but one.

LICHeE is kept as an optional COMPARISON only (``--run-lichee``, default
off): it never produces ``snv_tree.nwk``, and its own verdict (including a
literal "0 valid trees"-style line, quoted verbatim via
``extract_lichee_verdict``) is recorded in ``snv_tree_diagnostics.txt``
alongside its tree, when it runs. Its own VAF thresholds, ``-minClusterSize``
and ``-e`` (error margin) are independently configurable
(``--lichee-min-vaf-present``/``--lichee-max-vaf-absent``/
``--lichee-min-cluster-size``/``--lichee-error-margin``), since exploring
them is now the whole point of running it. Everything about how it is
invoked and how ``build_lichee_clone_tree`` attaches clusters to its output
(via the shared ``attach_option_a``, also used by ``build_cna_tree.py``'s
SCICoNE integration, so a node shared between two clusters is always
resolved the same way) is unchanged from when it was primary; see
``run_lichee``, ``parse_lichee_trees`` and ``build_lichee_clone_tree`` for
the mechanics. Camin-Sokal's code is gone (git history keeps it).

Dollo (``dollo_tree``): clusters are leaves -- no SECEDO cluster is
ancestral to another, each is a sampled population -- and every internal
node is a hidden ancestor, found by exhaustive search over every rooted
BINARY topology (``enumerate_dollo_topologies``, ``(2n-3)!!`` of them, 105
at n=5, capped at ``--dollo-max-clusters`` (7) rather than hang). A
mutation's Dollo cost on one topology is one gain at its presence pattern's
LCA, plus one loss per maximal fully-absent clade beneath that LCA
(``dollo_pattern_cost``) -- Dollo, unlike Camin-Sokal, allows a mutation to
be lost, exactly the dropout this data shows. Costs are precomputed once per
topology per distinct pattern (``dollo_cost_table``): a bootstrap replicate
only ever reweights this same fixed pattern universe, never introduces a
new one, so scoring 1000 replicates is a fast weighted sum, not 1000 fresh
tree walks.

Ties at the minimum cost are never broken arbitrarily: their STRICT
CONSENSUS is emitted (``strict_consensus_clades`` -- clades not common to
every optimal topology collapse into a polytomy; the intersection of
laminar clade families is itself laminar, so this is always a well-defined
tree, down to a star when nothing beyond the full leaf set is common).
Support for each consensus clade comes from the bootstrap: resample SNVs
with replacement ``--dollo-bootstrap`` times (a multinomial draw over the
fixed pattern universe, mathematically equivalent to resampling rows and
far cheaper) and recompute that replicate's own optimal consensus; a clade's
support is the fraction of replicates it survives in. Any clade below
``--dollo-min-clade-support`` (0.7) is dropped before the final tree is
built (dropping members from a laminar family keeps it laminar, so this
needs no separate graph-collapse pass -- ``build_tree_from_clades`` is
called once, on the already-filtered set). The diagnostics also report,
per bootstrap replicate, how often the optimal topology and the runner-up
each win outright -- the direct instability signal between the two closest
topologies -- and a per-edge gain/loss breakdown on the emitted tree
(``dollo_edge_report``, generalised to the polytomies consensus/collapse
can produce), trunk mutations landing on ``germline`` -> the MRCA.

Verified against the real slice D pattern counts (5 tumour clusters, 15
distinct presence patterns): the unique optimum is
``render_topology``'s ``(3,(10,((7,8),9)))``, cost 1996 above the baseline
every topology shares (singleton and full-set patterns cost 1 -- one gain,
no losses -- on every topology, so they never affect which one wins); the
runner-up, 30 more, swaps whether 9 or 10 joins the ``{7,8}`` clade first.
Confirmed independently by hand and by script before trusting the
implementation.

``--compare-tree <path>`` (e.g. stage 9's ``cna_tree.nwk``) reports
clade-level agreement against another tree over the same clusters
(``compare_tree_clades``), ignoring hidden-node names on both sides -- a
clade is just a leaf-set.

Channel ordering was recovered empirically (cosmic_signatures.csv carries no
channel labels, only ``Channel_0..Channel_95``): SBS1's four dominant channels
sit 24 apart, at the position matching NCG>NTG (C>T at CpG, all four 5' flanks),
and SBS4/SBS5/SBS92's dominant channels match their known C>A/T>C aetiology
under the same hypothesis. That fixes the axis as index = five*24 + subtype*4 +
three, five/three in A,C,G,T order and subtype in C>A,C>G,C>T,T>A,T>C,T>G order
-- the alphabetical sort of COSMIC's own ``A[C>A]A`` .. ``T[T>G]T`` labels.
``main`` still asserts cosmic_signatures.csv's columns match before trusting it.
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Hashable, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import phylox

N_CHANNELS = 96
BASES = ["A", "C", "G", "T"]
SUBTYPES = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
BASE_IDX = {b: i for i, b in enumerate(BASES)}
SUBTYPE_IDX = {s: i for i, s in enumerate(SUBTYPES)}
COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}

# The tree's root: an implicit, spectrum-less latent node for the germline/
# empty-mutation state. Not a real cluster (no VCF, no BAM, no column in the
# presence matrix or spectra table) -- the pseudo-normal cluster (config.sh's
# NORMAL_CLUSTER_ID) is Mutect2's -normal, never itself called, so this plays
# the tree's unique root without being an actual SECEDO cluster.
GERMLINE_ROOT_ID = "germline"

SNVKey = Tuple[str, int, str, str]  # chrom, 1-based pos, ref, alt

_FORCED_VCF_NAME_RE = re.compile(
    r"^clone(?P<cluster>[^_]+)_(?P<chrom>.+)\.forced\.vcf$"
)
_PASS1_VCF_NAME_RE = re.compile(r"^clone(?P<cluster>[^_]+)_(?P<chrom>[^.]+)\.vcf$")


# --------------------------------------------------------------------------- #
# Channel axis
# --------------------------------------------------------------------------- #


def channel_labels() -> List[str]:
    """The 96 column names as they appear in cosmic_signatures.csv."""
    return [f"Channel_{i}" for i in range(N_CHANNELS)]


def channel_index(five: str, ref: str, alt: str, three: str) -> int:
    """Index into the 96-channel axis for a pyrimidine-normalised substitution.

    index = five*24 + subtype*4 + three, matching the alphabetical order of
    COSMIC's own "A[C>A]A" .. "T[T>G]T" trinucleotide labels (see module
    docstring for how this was recovered).
    """
    try:
        return BASE_IDX[five] * 24 + SUBTYPE_IDX[f"{ref}>{alt}"] * 4 + BASE_IDX[three]
    except KeyError as exc:
        raise ValueError(
            f"unrecognised base or substitution: five={five!r} ref={ref!r} "
            f"alt={alt!r} three={three!r}"
        ) from exc


def pyrimidine_normalise(
    five: str, ref: str, alt: str, three: str
) -> Tuple[str, str, str, str]:
    """Reverse-complement onto the pyrimidine strand when ref is a purine (A/G)."""
    if ref in ("A", "G"):
        return COMPLEMENT[three], COMPLEMENT[ref], COMPLEMENT[alt], COMPLEMENT[five]
    return five, ref, alt, three


def snv_channel(five: str, ref: str, alt: str, three: str) -> int:
    """Channel index for one SNV, normalising to the pyrimidine strand first."""
    five, ref, alt, three = pyrimidine_normalise(five, ref, alt, three)
    return channel_index(five, ref, alt, three)


# --------------------------------------------------------------------------- #
# VCF parsing
# --------------------------------------------------------------------------- #


def _safe_float(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(x: str) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def parse_format_values(format_str: str, sample_str: str) -> Dict[str, str]:
    """{FORMAT key: raw sample value}, as found in one VCF genotype column pair
    (the ``FORMAT`` and single-sample columns of a single-sample VCF record)."""
    return dict(zip(format_str.split(":"), sample_str.split(":")))


def parse_forced_vcf_calls(
    vcf_path: Path, cluster_id: str
) -> Dict[SNVKey, Tuple[float, int]]:
    """Parse one force-called VCF (stage 06b's pass 2) into ``{snv_key: (vaf,
    alt_reads)}`` for every single-base-substitution ALT allele at every
    record.

    The forced VCFs come from tumour-vs-normal Mutect2 (stage 06's ``-normal``
    design) and so carry TWO sample columns, tumour ``clone<cluster_id>`` and
    the pseudo-normal, in whatever order GATK wrote them (sorted sample name,
    which puts a cluster ID below the normal's ahead of it and one above
    behind it). The tumour column is therefore selected BY NAME from the
    ``#CHROM`` header, never by position: reading a fixed column silently
    reads the pseudo-normal's genotype for any cluster whose ID sorts after
    ``NORMAL_CLUSTER_ID`` (confirmed on a real run -- clone7/8/9 came back
    with implausibly low presence, clone3/10 implausibly high, because 3 and
    10 sort ahead of the normal's column and 7/8/9 sort after it). Raises
    ValueError, naming the file and the header's sample columns, if
    ``clone<cluster_id>`` is not among them.

    Every record, PASS or not: pass 2 force-calls every cluster at every
    union site regardless of whether that cluster independently supports it,
    and presence is decided in Python directly off VAF/ALT-read depth (see
    ``resolve_presence_calls``), not off the FILTER column. Mutect2 still
    emits its own discovery calls alongside the forced ones, so this alone
    is not enough to restrict to the union of candidate sites -- see
    ``restrict_calls_to_union``, applied by the caller.

    SUB-DECISION: AF/AD extraction assumes GATK4 Mutect2's own FORMAT layout
    -- AD is ``ref_depth,alt_depth_1[,alt_depth_2...]`` (one more entry than
    ALT alleles, ref first) and AF is ``alt_af_1[,alt_af_2...]`` (one entry
    per ALT allele, no ref entry) -- paired positionally with the record's own
    (possibly multi-allelic) ALT list. A record whose AD or AF cannot be
    parsed this way (wrong field count, non-numeric, "." for no coverage) is
    treated as zero VAF / zero ALT reads rather than raising: force-calling a
    site with no supporting reads at all is an expected "absent here"
    outcome, not a malformed file.
    """
    sample_name = f"clone{cluster_id}"
    calls: Dict[SNVKey, Tuple[float, int]] = {}
    sample_col: Optional[int] = None
    with open(vcf_path) as fh:
        for line in fh:
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                if sample_name not in samples:
                    raise ValueError(
                        f"{vcf_path} has no sample column {sample_name!r}; "
                        f"samples found: {samples}"
                    )
                sample_col = 9 + samples.index(sample_name)
                continue
            if line.startswith("#"):
                continue
            if sample_col is None:
                raise ValueError(f"{vcf_path} has records before a #CHROM header")
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= sample_col:
                continue
            chrom, pos, _id, ref, alt_field = fields[:5]
            format_str, sample_str = fields[8], fields[sample_col]
            ref = ref.upper()
            if len(ref) != 1 or ref not in "ACGT":
                continue
            fmt = parse_format_values(format_str, sample_str)
            ad_raw = fmt.get("AD", "").split(",")
            af_raw = fmt.get("AF", "").split(",")
            for i, alt in enumerate(alt_field.split(",")):
                alt = alt.upper()
                if len(alt) != 1 or alt not in "ACGT":
                    continue
                vaf = _safe_float(af_raw[i]) if i < len(af_raw) else 0.0
                alt_reads = _safe_int(ad_raw[i + 1]) if i + 1 < len(ad_raw) else 0
                calls[(chrom, int(pos), ref, alt)] = (vaf, alt_reads)
    return calls


def resolve_presence_calls(
    cluster_to_calls: Dict[str, Dict[SNVKey, Tuple[float, int]]],
    min_vaf: float,
    min_alt_reads: int,
) -> Dict[str, Set[SNVKey]]:
    """A force-called site is PRESENT in a cluster if its VAF >= ``min_vaf``
    AND ALT read depth >= ``min_alt_reads``; ABSENT otherwise -- including
    sites the cluster's own record never reached these thresholds at.
    """
    return {
        cluster: {
            key
            for key, (vaf, alt_reads) in calls.items()
            if vaf >= min_vaf and alt_reads >= min_alt_reads
        }
        for cluster, calls in cluster_to_calls.items()
    }


def discover_forced_cluster_vcfs(vcf_dir: Path) -> Dict[str, List[Path]]:
    """Group ``clone<cluster>_<chrom>.forced.vcf`` files (stage 06b's pass-2
    force-call output) by cluster. The pseudo-normal cluster is never a key
    here: it has no VCF (excluded from stage 05's tasks.tsv, never called).
    """
    out: Dict[str, List[Path]] = defaultdict(list)
    for f in sorted(vcf_dir.glob("clone*_*.forced.vcf")):
        m = _FORCED_VCF_NAME_RE.match(f.name)
        if not m:
            continue
        out[m.group("cluster")].append(f)
    return dict(out)


def discover_pass1_cluster_vcfs(vcf_dir: Path) -> Dict[str, List[Path]]:
    """Group stage 06's plain ``clone<cluster>_<chrom>.vcf`` files (PASS,
    SNP-only pass-1 output) by cluster, excluding stage 06b's
    ``*.forced.vcf`` (``_PASS1_VCF_NAME_RE``'s chromosome group excludes
    dots, so it cannot match a ``.forced.vcf`` name). Read only for their
    site keys -- see ``build_union_sites`` -- and to report each cluster's
    own pass-1 PASS count in the diagnostics.
    """
    out: Dict[str, List[Path]] = defaultdict(list)
    for f in sorted(vcf_dir.glob("clone*_*.vcf")):
        if f.name.endswith(".forced.vcf"):
            continue
        m = _PASS1_VCF_NAME_RE.match(f.name)
        if not m:
            continue
        out[m.group("cluster")].append(f)
    return dict(out)


def parse_vcf_site_keys(vcf_path: Path) -> Set[SNVKey]:
    """Every single-base-substitution ``(chrom, pos, ref, alt)`` key in a
    VCF, ignoring genotype columns entirely. Used to reconstruct stage 06b's
    union of candidate sites from pass-1's own PASS VCFs (see
    ``build_union_sites``), matching the site key stage 06b's own union
    build uses (``CHROM``, ``POS``, ``REF``, one ``ALT`` allele).
    """
    keys: Set[SNVKey] = set()
    with open(vcf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                continue
            chrom, pos, _id, ref, alt_field = fields[:5]
            ref = ref.upper()
            if len(ref) != 1 or ref not in "ACGT":
                continue
            for alt in alt_field.split(","):
                alt = alt.upper()
                if len(alt) != 1 or alt not in "ACGT":
                    continue
                keys.add((chrom, int(pos), ref, alt))
    return keys


def build_union_sites(pass1_vcfs: Dict[str, List[Path]]) -> Set[SNVKey]:
    """Reconstruct stage 06b's per-chromosome union of pass-1 PASS SNP sites
    directly from the pass-1 VCFs already in ``--vcf-dir``, rather than
    depending on stage 06b's own ``union_sites_<chrom>.vcf`` under scratch
    (``MUTECT_DIR``, which stage 07 does not copy out). Raises ValueError if
    ``pass1_vcfs`` is empty.
    """
    if not pass1_vcfs:
        raise ValueError(
            "no pass-1 clone*_*.vcf files to rebuild the union stage 06b "
            "force-called against"
        )
    union: Set[SNVKey] = set()
    for files in pass1_vcfs.values():
        for f in files:
            union |= parse_vcf_site_keys(f)
    return union


def restrict_calls_to_union(
    calls: Dict[SNVKey, Tuple[float, int]], union_sites: Set[SNVKey]
) -> Tuple[Dict[SNVKey, Tuple[float, int]], int]:
    """Keep only the calls whose site key is in ``union_sites``. Force-calling
    adds the union alleles to Mutect2's own interval, but Mutect2 still emits
    its own discovery calls on top, and the forced VCF keeps non-PASS
    records -- presence must be decided only at union sites, which is the
    whole purpose of the force-call design. Returns ``(kept, n_dropped)``.
    """
    kept = {k: v for k, v in calls.items() if k in union_sites}
    return kept, len(calls) - len(kept)


def assert_presence_within_union(
    cluster_to_snvs: Dict[str, Set[SNVKey]], union_sites: Set[SNVKey]
) -> None:
    """Sanity guard: no cluster's present-SNV count can exceed the union
    size, since presence is resolved only from calls already restricted to
    it. Raises ValueError, naming the cluster and both counts, if it ever
    does -- the sign that the union restriction was not actually applied.
    """
    for cluster, snvs in cluster_to_snvs.items():
        if len(snvs) > len(union_sites):
            raise ValueError(
                f"cluster {cluster} has {len(snvs)} present SNVs, more than "
                f"the union's {len(union_sites)} sites -- presence must "
                "never exceed the union"
            )


# --------------------------------------------------------------------------- #
# clone x SNV matrix
# --------------------------------------------------------------------------- #


def _sort_key(cluster_id: Hashable) -> Tuple[int, object]:
    """Numeric sort for plain-integer cluster IDs, lexicographic fallback otherwise."""
    try:
        return (0, int(cluster_id))
    except (TypeError, ValueError):
        return (1, str(cluster_id))


def snv_key_str(key: SNVKey) -> str:
    chrom, pos, ref, alt = key
    return f"{chrom}:{pos}:{ref}>{alt}"


def build_snv_presence_matrix(cluster_to_snvs: Dict[str, Set[SNVKey]]) -> pd.DataFrame:
    """clone x SNV binary presence matrix: rows the union of SNVs, columns the
    clusters."""
    clusters = sorted(cluster_to_snvs, key=_sort_key)
    all_snvs = sorted(
        {snv_key_str(k) for snvs in cluster_to_snvs.values() for k in snvs}
    )
    matrix = pd.DataFrame(0, index=all_snvs, columns=clusters, dtype=int)
    for cluster, snvs in cluster_to_snvs.items():
        rows = [snv_key_str(k) for k in snvs]
        matrix.loc[rows, cluster] = 1
    return matrix


def mutation_sets_from_matrix(matrix: pd.DataFrame) -> Dict[str, Set[str]]:
    """Column -> set of row labels present (value == 1), read back off the
    presence matrix."""
    return {col: set(matrix.index[matrix[col] == 1]) for col in matrix.columns}


def three_gamete_violations(mutation_sets: Dict[str, Set[Hashable]]) -> int:
    """Count SNV pairs violating the perfect-phylogeny (three-gamete) test.

    Two SNVs are incompatible with a single mutation history if, among the
    clusters, all three of "only i", "only j", and "both" occur (the fourth
    gamete, "neither", is irrelevant to infinite-sites compatibility). Each
    SNV's presence across clusters is bit-packed into one integer so a pair
    check is O(1); overall cost is O(n_snvs^2), fine at the SNV counts this
    pipeline expects (sparse tens-to-hundreds of mutations per cluster).

    A general presence-matrix compatibility diagnostic, independent of which
    tree method is used -- reported in snv_tree_diagnostics.txt regardless.
    """
    clusters = sorted(mutation_sets)
    cluster_bit = {c: i for i, c in enumerate(clusters)}
    snv_masks: Dict[Hashable, int] = {}
    for cluster, muts in mutation_sets.items():
        bit = 1 << cluster_bit[cluster]
        for m in muts:
            snv_masks[m] = snv_masks.get(m, 0) | bit

    masks = list(snv_masks.values())
    violations = 0
    for i in range(len(masks)):
        mi = masks[i]
        for j in range(i + 1, len(masks)):
            mj = masks[j]
            if (mi & ~mj) and (mj & ~mi) and (mi & mj):
                violations += 1
    return violations


# --------------------------------------------------------------------------- #
# Newick I/O and the model loader's own contract
# --------------------------------------------------------------------------- #


def digraph_to_newick(tree: nx.DiGraph, root: Hashable) -> str:
    """Rooted, labelled-internal-node Newick string for a clone tree."""

    def _recurse(node: Hashable) -> str:
        children = sorted(tree.successors(node), key=_sort_key)
        if not children:
            return str(node)
        inner = ",".join(_recurse(c) for c in children)
        return f"({inner}){node}"

    return _recurse(root) + ";"


def parse_newick_like_model(newick_str: str) -> nx.DiGraph:
    """Parse a Newick string exactly the way ``_BaseTreeHDP.__init__`` does.

    Splits on ";", parses each tree with phylox, and relabels every node to
    its Newick label -- the same two steps the model loader runs before it
    ever looks at node IDs.
    """
    graph = nx.DiGraph()
    for s in newick_str.split(";"):
        if not s.strip():
            continue
        tree = phylox.DiNetwork.from_newick(s)
        mapping = {n: tree.nodes[n].get("label", str(n)) for n in tree.nodes()}
        graph = nx.compose(graph, nx.relabel_nodes(tree, mapping))
    return graph


def verify_newick(
    newick_str: str,
    cluster_ids: Set[str],
    normal_id: str,
    hidden_ids: Optional[Set[str]] = None,
) -> None:
    """Round-trip check: parse the way the model does and confirm the contract holds.

    ``normal_id`` is the tree-root label (``GERMLINE_ROOT_ID`` in ``main``),
    not a real cluster. ``hidden_ids`` are the hidden group nodes LICHeE's
    shared nodes need (``g<id>``): extra labelled nodes with no spectra row,
    which the model treats as latent, like the root. They must not collide with
    a cluster ID. Raises AssertionError if the root is not the unique root, or
    if the parsed node labels are not exactly the cluster IDs, the root and the
    hidden IDs, each appearing once.
    """
    hidden = set(hidden_ids or ())
    clash = hidden & (set(cluster_ids) | {normal_id})
    if clash:
        raise AssertionError(
            f"hidden node IDs collide with real labels: {sorted(clash)}"
        )
    graph = parse_newick_like_model(newick_str)
    expected = set(cluster_ids) | {normal_id} | hidden
    labels = list(graph.nodes())
    if len(labels) != len(expected):
        raise AssertionError(
            f"parsed {len(labels)} node labels, expected {len(expected)} "
            f"(cluster IDs plus the germline root and any hidden group nodes); "
            f"a label collision or missing cluster is likely: {sorted(labels)} "
            f"vs {sorted(expected)}"
        )
    if set(labels) != expected:
        raise AssertionError(
            f"parsed labels {sorted(labels)} != expected {sorted(expected)}"
        )
    roots = [n for n, d in graph.in_degree() if d == 0]
    if len(roots) != 1:
        raise AssertionError(f"expected exactly one root, got {roots}")
    if roots[0] != normal_id:
        raise AssertionError(
            f"root is {roots[0]!r}, expected germline root {normal_id!r}"
        )


# --------------------------------------------------------------------------- #
# 96-channel binning
# --------------------------------------------------------------------------- #


def bin_cluster_spectra(
    cluster_to_snvs: Dict[str, Set[SNVKey]], fasta
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Bin each cluster's SNVs into the 96 SBS channels using trinucleotide context.

    ``fasta`` is an open pysam.FastaFile over REF_FASTA. Skips (and counts) any
    SNV whose trinucleotide context hits an ambiguous base, or whose middle
    base disagrees with the VCF's REF allele (a reference-build mismatch).
    """
    clusters = sorted(cluster_to_snvs, key=_sort_key)
    labels = channel_labels()
    counts = np.zeros((len(clusters), N_CHANNELS), dtype=int)
    skip_counts = {"ambiguous_context": 0, "ref_mismatch": 0}

    for row, cluster in enumerate(clusters):
        for chrom, pos, ref, alt in cluster_to_snvs[cluster]:
            context = fasta.fetch(chrom, pos - 2, pos + 1).upper()
            if len(context) != 3 or any(b not in "ACGT" for b in context):
                skip_counts["ambiguous_context"] += 1
                continue
            five, mid, three = context[0], context[1], context[2]
            if mid != ref:
                skip_counts["ref_mismatch"] += 1
                continue
            try:
                idx = snv_channel(five, ref, alt, three)
            except ValueError:
                skip_counts["ambiguous_context"] += 1
                continue
            counts[row, idx] += 1

    spectra = pd.DataFrame(counts, index=clusters, columns=labels)
    return spectra, skip_counts


# --------------------------------------------------------------------------- #
# Dollo parsimony over trees with hidden internal nodes (primary tree builder)
# --------------------------------------------------------------------------- #

# Hidden ancestor / group node label prefix, shared with the LICHeE
# integration below (``attach_option_a``'s own hidden group nodes use the
# same scheme, so both mechanisms' hidden ids look alike).
_GROUP_NODE_PREFIX = "g"

# A Dollo topology is a nested, canonical structure with no independent node
# identity: a leaf is its own cluster-id string; an internal node is a
# ``frozenset`` of exactly its two children (each itself a leaf id or a
# further nested frozenset). Two structurally identical topologies, however
# built, compare equal and hash the same -- this is what lets
# ``strict_consensus_clades`` and the cost-table lookups work by value.


@lru_cache(maxsize=None)
def _dollo_topologies_over(leaves: Tuple[str, ...]) -> frozenset:
    """Every distinct rooted binary topology over ``leaves`` (as canonical
    nested frozensets), memoized on the sorted leaf tuple: ``(2n-3)!!``
    results for ``n`` leaves (105 for n=5, 10395 at the n=7 default cap).
    Generated by recursively splitting the leaf set into two nonempty parts
    and combining each part's own topologies; every split is visited from
    both sides (once as (A, B), once as (B, A)), but since a node is an
    unordered pair, the duplicate is naturally absorbed by the returned set.
    """
    if len(leaves) == 1:
        return frozenset([leaves[0]])
    leafset = frozenset(leaves)
    result = set()
    for k in range(1, len(leaves)):
        for combo in itertools.combinations(leaves, k):
            left = frozenset(combo)
            right = leafset - left
            for lt in _dollo_topologies_over(tuple(sorted(left, key=_sort_key))):
                for rt in _dollo_topologies_over(tuple(sorted(right, key=_sort_key))):
                    result.add(frozenset([lt, rt]))
    return frozenset(result)


def enumerate_dollo_topologies(leaves: List[str], max_clusters: int = 7) -> List:
    """Every rooted binary topology over ``leaves``, sorted by
    ``render_topology`` for a deterministic iteration order.

    Raises ValueError past ``max_clusters`` (default 7, 10395 topologies) --
    exhaustive over every rooted binary tree, tractable at the
    handful-of-clusters scale one SECEDO run produces, not in general, and
    also raises if ``leaves`` is empty.
    """
    if len(leaves) > max_clusters:
        raise ValueError(
            f"enumerate_dollo_topologies: {len(leaves)} clusters exceeds the "
            f"exhaustive search cap of {max_clusters} (see --dollo-max-clusters)"
        )
    if not leaves:
        raise ValueError("enumerate_dollo_topologies: no clusters to build a tree over")
    topologies = list(_dollo_topologies_over(tuple(sorted(leaves, key=_sort_key))))
    topologies.sort(key=render_topology)
    return topologies


@lru_cache(maxsize=None)
def _dollo_leaves_under(node) -> frozenset:
    if isinstance(node, str):
        return frozenset([node])
    c1, c2 = tuple(node)
    return _dollo_leaves_under(c1) | _dollo_leaves_under(c2)


def render_topology(node) -> str:
    """Canonical, human-readable structural rendering of one topology (no
    root label): children ordered by their own leaf sets (``_sort_key``),
    the same convention ``digraph_to_newick`` uses, so identical topologies
    always render identically regardless of how they were built."""
    if isinstance(node, str):
        return node
    c1, c2 = tuple(node)
    kids = sorted([c1, c2], key=lambda n: sorted(_dollo_leaves_under(n), key=_sort_key))
    return "(" + ",".join(render_topology(k) for k in kids) + ")"


def _dollo_losses_below(node, S: frozenset) -> int:
    """Loss events needed within ``node``'s own subtree so every leaf under
    it that is NOT in ``S`` ends up absent: one loss per MAXIMAL fully-absent
    clade (Dollo allows a single loss to cover an entire absent subtree at
    once), found by recursing only into subtrees that still mix present and
    absent leaves."""
    if isinstance(node, str):
        return 0 if node in S else 1
    leaves = _dollo_leaves_under(node)
    if leaves.isdisjoint(S):
        return 1
    c1, c2 = tuple(node)
    return _dollo_losses_below(c1, S) + _dollo_losses_below(c2, S)


def _dollo_lca_children(node, S: frozenset) -> Tuple[object, object]:
    """The two children of the node where ``S`` (``2 <= |S| <= n-1``) first
    spans both -- the LCA of ``S`` in this topology -- returned as its two
    children (for loss-counting): the nested-frozenset representation gives
    the LCA no independent identity of its own."""
    c1, c2 = tuple(node)
    if S <= _dollo_leaves_under(c1):
        return _dollo_lca_children(c1, S)
    if S <= _dollo_leaves_under(c2):
        return _dollo_lca_children(c2, S)
    return c1, c2


def dollo_pattern_cost(topology, pattern: frozenset) -> Tuple[int, int]:
    """``(cost, losses)`` of one presence pattern on one topology: one gain
    at the pattern's LCA plus one loss per maximal absent clade beneath it.
    A singleton or the full leaf set costs 1 (one gain, no losses) on EVERY
    topology -- structurally invariant, so it never affects which topology
    is chosen, though it is still reported (a private mutation, or the
    trunk)."""
    if len(pattern) == 1 or pattern == _dollo_leaves_under(topology):
        return 1, 0
    c1, c2 = _dollo_lca_children(topology, pattern)
    losses = _dollo_losses_below(c1, pattern) + _dollo_losses_below(c2, pattern)
    return 1 + losses, losses


def dollo_cost_table(topologies: List, patterns: Iterable[frozenset]):
    """``{topology: {pattern: cost}}`` for every topology and every one of
    ``patterns``, computed once: a bootstrap replicate only ever reweights
    this SAME fixed set of observed patterns (resampling changes counts,
    never introduces a new distinct pattern), so scoring every replicate is
    then a fast weighted sum against this table rather than a fresh tree
    walk each time."""
    patterns = list(patterns)
    return {t: {p: dollo_pattern_cost(t, p)[0] for p in patterns} for t in topologies}


def dollo_best_topologies(
    topologies: List, pattern_counts: Dict[frozenset, int], cost_table
) -> Tuple[int, List]:
    """``(min_cost, [topologies achieving it])``, cost summed over
    ``pattern_counts`` weighted against ``cost_table`` (a bootstrap
    replicate's counts are always a subset of the same pattern universe the
    table was built for)."""
    totals = []
    for t in topologies:
        table_t = cost_table[t]
        total = sum(table_t[p] * n for p, n in pattern_counts.items() if p in table_t)
        totals.append((total, t))
    min_cost = min(c for c, _ in totals)
    best = sorted((t for c, t in totals if c == min_cost), key=render_topology)
    return min_cost, best


def dollo_clades(topology) -> Set[frozenset]:
    """Every internal node's leaf-set (including the full leaf set, the
    MRCA), excluding trivial singleton leaves."""
    if isinstance(topology, str):
        return set()
    c1, c2 = tuple(topology)
    return {_dollo_leaves_under(topology)} | dollo_clades(c1) | dollo_clades(c2)


def strict_consensus_clades(topologies: List) -> Set[frozenset]:
    """Clades common to every topology in ``topologies`` -- the strict
    consensus, never an arbitrary pick among ties. Always nonempty (at
    least the full leaf set) and always a valid laminar family, since the
    intersection of laminar families is itself laminar."""
    clade_sets = [dollo_clades(t) for t in topologies]
    common = clade_sets[0]
    for c in clade_sets[1:]:
        common = common & c
    return common


def build_tree_from_clades(
    clades: Set[frozenset],
    leaves: List[str],
    germline_id: str = GERMLINE_ROOT_ID,
) -> Tuple[nx.DiGraph, Dict[frozenset, str], Set[str]]:
    """Turn a laminar family of clades (``strict_consensus_clades``'s
    output, optionally with low-support ones filtered out first -- removing
    members from a laminar family keeps it laminar, so filtering before
    building is simpler than building then collapsing) into the actual tree:
    the full leaf set becomes the MRCA, germline's only child (trunk
    mutations gain on that edge); every other clade a hidden node; every
    leaf hangs off the smallest clade containing it (directly off germline
    if none does, e.g. a single-cluster tree). Hidden ids are assigned
    deterministically, ``g1, g2, ...`` in ascending (size, sorted leaves)
    order, so identical input always gives identical labels.

    Returns ``(tree, label_of_clade, hidden_ids)``.
    """
    ordered = sorted(clades, key=lambda c: (len(c), sorted(c, key=_sort_key)))
    label_of_clade: Dict[frozenset, str] = {
        c: f"{_GROUP_NODE_PREFIX}{i + 1}" for i, c in enumerate(ordered)
    }
    hidden_ids = set(label_of_clade.values())

    parent_of_clade: Dict[frozenset, Optional[frozenset]] = {}
    placed: List[frozenset] = []
    for c in sorted(clades, key=lambda c: -len(c)):
        supersets = [p for p in placed if c < p]
        parent_of_clade[c] = min(supersets, key=len) if supersets else None
        placed.append(c)

    tree = nx.DiGraph()
    tree.add_node(germline_id)
    for c, parent in parent_of_clade.items():
        parent_label = germline_id if parent is None else label_of_clade[parent]
        tree.add_edge(parent_label, label_of_clade[c])

    for leaf in leaves:
        supersets = [c for c in clades if leaf in c]
        if supersets:
            parent_label = label_of_clade[min(supersets, key=len)]
        else:
            parent_label = germline_id
        tree.add_edge(parent_label, leaf)

    return tree, label_of_clade, hidden_ids


def pattern_counts_from_matrix(matrix: pd.DataFrame) -> Dict[frozenset, int]:
    """Counter of each distinct row's presence pattern (as a frozenset of
    the clusters present), read off the clone x SNV binary matrix. This is
    what Dollo scores against: which distinct pattern occurs how many
    times, not each SNV individually -- the same information, far fewer
    entries to weight-sum over."""
    if matrix.shape[0] == 0:
        return {}
    clusters = list(matrix.columns)
    weights = np.array([1 << i for i in range(len(clusters))], dtype=np.int64)
    masks = matrix.to_numpy(dtype=np.int64) @ weights
    counts = Counter(int(m) for m in masks)
    return {
        frozenset(c for i, c in enumerate(clusters) if mask & (1 << i)): n
        for mask, n in counts.items()
        if mask != 0
    }


def _all_leaves_under(tree: nx.DiGraph, root: str) -> Dict[str, frozenset]:
    """Post-order leaf-set cache for every node of an ACTUAL (possibly
    non-binary, post-consensus) emitted tree -- generalises
    ``_dollo_leaves_under`` (which only works on the binary search-space
    representation) to any ``nx.DiGraph`` following this pipeline's Newick
    contract (hidden internal nodes, real clusters as leaves): reused for
    Dollo's own per-edge report and for ``compare_tree_clades`` against a
    tree built some other way (e.g. the CNA tree)."""
    result: Dict[str, frozenset] = {}

    def visit(node: str) -> frozenset:
        children = list(tree.successors(node))
        leaves = frozenset([node]) if not children else frozenset()
        for c in children:
            leaves |= visit(c)
        result[node] = leaves
        return leaves

    visit(root)
    return result


def dollo_edge_report(
    tree: nx.DiGraph, germline_id: str, pattern_counts: Dict[frozenset, int]
) -> Dict[Tuple[str, str], Dict[str, int]]:
    """Mutations gained and lost on every edge of the EMITTED tree (which,
    after consensus and low-support collapse, may have polytomies -- this
    generalises the binary LCA/loss logic above to arbitrary branching). A
    singleton pattern's gain lands on the edge into its own leaf; the full
    leaf set's gain lands on the trunk (``germline`` -> the MRCA node).
    """
    leaves_under = _all_leaves_under(tree, germline_id)

    def find_lca(node: str, S: frozenset) -> str:
        for c in tree.successors(node):
            if S <= leaves_under[c]:
                return find_lca(c, S)
        return node

    def mark_losses(node: str, S: frozenset, count: int, report: dict) -> None:
        parent = next(iter(tree.predecessors(node)))
        if leaves_under[node].isdisjoint(S):
            report[(parent, node)]["losses"] += count
            return
        for c in tree.successors(node):
            mark_losses(c, S, count, report)

    report: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: {"gains": 0, "losses": 0}
    )
    for S, count in pattern_counts.items():
        lca = find_lca(germline_id, S)
        parent = next(iter(tree.predecessors(lca)))
        report[(parent, lca)]["gains"] += count
        for c in tree.successors(lca):
            mark_losses(c, S, count, report)
    return dict(report)


@dataclass
class DolloResult:
    """The Dollo tree plus everything the diagnostics report."""

    tree: nx.DiGraph
    hidden_ids: Set[str]
    best_cost: int
    runner_up_cost: Optional[int]
    n_topologies: int
    n_ties: int  # topologies achieving best_cost, before consensus
    optimal_render: str
    runner_up_render: Optional[str]
    edge_report: Dict[Tuple[str, str], Dict[str, int]]
    clade_support: Dict[frozenset, float]  # every clade of the pre-collapse consensus
    collapsed_clades: List[frozenset]  # support < min_clade_support, dropped
    top2_win_rate: Dict[str, float]  # {"optimal", "runner_up"}: bootstrap win fraction
    n_bootstrap: int


def dollo_tree(
    pattern_counts: Dict[frozenset, int],
    leaves: List[str],
    max_clusters: int = 7,
    n_bootstrap: int = 1000,
    min_clade_support: float = 0.7,
    seed: int = 0,
    germline_id: str = GERMLINE_ROOT_ID,
) -> DolloResult:
    """Dollo parsimony over trees with hidden internal nodes: the primary
    SNV-tree method (see module docstring for why). Clusters are leaves --
    each a sampled population, never ancestral to another -- and every
    internal node is a hidden ancestor, found by exhaustive search over
    every rooted binary topology (``enumerate_dollo_topologies``), unlike
    the retired Camin-Sokal search, which could only place observed
    clusters as internal nodes and so could never represent one.

    Ties at the minimum cost are never broken arbitrarily: their STRICT
    CONSENSUS is emitted (edges not common to every optimal topology
    collapse into polytomies). Support for each of that consensus's clades
    is estimated by the bootstrap (resample SNVs with replacement,
    ``n_bootstrap`` times, recompute the optimal consensus per replicate --
    a multinomial draw over the fixed pattern universe is exactly equivalent
    to resampling rows and much cheaper, and is what is actually done here);
    any clade below ``min_clade_support`` is then dropped before the final
    tree is built (a laminar family stays laminar with members removed, so
    this needs no separate graph-collapse step). ``n_bootstrap=0`` skips
    estimation entirely and emits the raw consensus, unfiltered.
    """
    if len(leaves) == 1:
        tree = nx.DiGraph([(germline_id, leaves[0])])
        return DolloResult(
            tree=tree,
            hidden_ids=set(),
            best_cost=0,
            runner_up_cost=None,
            n_topologies=1,
            n_ties=1,
            optimal_render=leaves[0],
            runner_up_render=None,
            edge_report=dollo_edge_report(tree, germline_id, pattern_counts),
            clade_support={},
            collapsed_clades=[],
            top2_win_rate={},
            n_bootstrap=0,
        )

    topologies = enumerate_dollo_topologies(leaves, max_clusters=max_clusters)
    cost_table = dollo_cost_table(topologies, pattern_counts.keys())
    best_cost, best = dollo_best_topologies(topologies, pattern_counts, cost_table)

    def _total(t, counts):
        table_t = cost_table[t]
        return sum(table_t[p] * n for p, n in counts.items() if p in table_t)

    all_costs = sorted({_total(t, pattern_counts) for t in topologies})
    runner_up_cost = all_costs[1] if len(all_costs) > 1 else None
    runner_up = (
        [t for t in topologies if _total(t, pattern_counts) == runner_up_cost]
        if runner_up_cost is not None
        else []
    )

    consensus_clades = strict_consensus_clades(best)
    full_leaf_set = frozenset(leaves)

    clade_support: Dict[frozenset, float] = {}
    top2_win_rate: Dict[str, float] = {}
    if n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        pattern_order = sorted(pattern_counts, key=lambda p: sorted(p, key=_sort_key))
        counts_arr = np.array([pattern_counts[p] for p in pattern_order], dtype=float)
        total_n = int(counts_arr.sum())
        probs = counts_arr / counts_arr.sum()
        support_counts: Dict[frozenset, int] = defaultdict(int)
        win_optimal = 0
        win_runner_up = 0
        for _ in range(n_bootstrap):
            draw = rng.multinomial(total_n, probs)
            rep_counts = {p: int(c) for p, c in zip(pattern_order, draw) if c > 0}
            _, rep_best = dollo_best_topologies(topologies, rep_counts, cost_table)
            for clade in strict_consensus_clades(rep_best):
                support_counts[clade] += 1
            if any(t in rep_best for t in best):
                win_optimal += 1
            if any(t in rep_best for t in runner_up):
                win_runner_up += 1
        for clade in consensus_clades:
            if clade != full_leaf_set:
                clade_support[clade] = support_counts.get(clade, 0) / n_bootstrap
        top2_win_rate = {
            "optimal": win_optimal / n_bootstrap,
            "runner_up": win_runner_up / n_bootstrap,
        }

    kept_clades = {
        c
        for c in consensus_clades
        if c == full_leaf_set or clade_support.get(c, 1.0) >= min_clade_support
    }
    collapsed_clades = sorted(
        (c for c in consensus_clades if c not in kept_clades),
        key=lambda c: sorted(c, key=_sort_key),
    )

    tree, _label_of_clade, hidden_ids = build_tree_from_clades(
        kept_clades, leaves, germline_id=germline_id
    )

    return DolloResult(
        tree=tree,
        hidden_ids=hidden_ids,
        best_cost=best_cost,
        runner_up_cost=runner_up_cost,
        n_topologies=len(topologies),
        n_ties=len(best),
        optimal_render=render_topology(best[0]),
        runner_up_render=render_topology(runner_up[0]) if runner_up else None,
        edge_report=dollo_edge_report(tree, germline_id, pattern_counts),
        clade_support=clade_support,
        collapsed_clades=collapsed_clades,
        top2_win_rate=top2_win_rate,
        n_bootstrap=n_bootstrap,
    )


def compare_tree_clades(
    tree: nx.DiGraph, root: str, other_newick_path: Path
) -> Dict[str, List[List[str]]]:
    """Clade-level agreement between ``tree`` (rooted at ``root``) and
    another tree read from ``other_newick_path`` (e.g. the CNA tree),
    ignoring hidden-node names -- a clade is just a leaf-set, however its
    own internal node happens to be labelled -- over the cluster set the two
    share. Returns ``{"both", "only_first", "only_second"}``, each a sorted
    list of sorted cluster-id lists.
    """
    other_str = Path(other_newick_path).read_text()
    other_graph = parse_newick_like_model(other_str)
    other_roots = [n for n, d in other_graph.in_degree() if d == 0]
    if len(other_roots) != 1:
        raise ValueError(
            f"{other_newick_path}: expected exactly one root, got {other_roots}"
        )
    other_root = other_roots[0]

    own_leaves_under = _all_leaves_under(tree, root)
    other_leaves_under = _all_leaves_under(other_graph, other_root)
    shared = own_leaves_under[root] & other_leaves_under[other_root]

    def clades(leaves_under: Dict[str, frozenset], root: str) -> Set[frozenset]:
        all_leaves = leaves_under[root]
        result = set()
        for node, leaves in leaves_under.items():
            if node == root:
                continue
            restricted = leaves & shared
            if 2 <= len(restricted) < len(all_leaves & shared):
                result.add(restricted)
        return result

    own = clades(own_leaves_under, root)
    other = clades(other_leaves_under, other_root)

    def render(clades: Set[frozenset]) -> List[List[str]]:
        return sorted((sorted(c, key=_sort_key) for c in clades), key=str)

    return {
        "both": render(own & other),
        "only_first": render(own - other),
        "only_second": render(other - own),
    }


# --------------------------------------------------------------------------- #
# LICHeE (optional comparison, never the primary tree builder)
# --------------------------------------------------------------------------- #


LICHEE_MAIN_CLASS = "lineage.LineageEngine"
LICHEE_OUT_NAME = "lichee_out.trees.txt"
_CLUSTER_COLUMN_PREFIX = "c"
_DECOMPOSITION_DOTS_PER_LEVEL = 5


def resolve_lichee_home(explicit: Optional[str] = None) -> Tuple[Path, Path]:
    """Locate LICHeE and return ``(jar, lib_dir)``: ``explicit`` (the
    ``--lichee-home`` CLI arg), else ``$LICHEE_HOME``, else the repo's own
    checkout at ``realdata/external/lichee/LICHeE``.

    The ``release/lichee`` launcher is not used (it cannot find its
    dependencies on JDK 11), so what is needed is ``release/lichee.jar``, the
    ``lib/`` directory and ``java`` on ``PATH``. Raises FileNotFoundError,
    naming every one that is missing.
    """
    home = Path(
        explicit
        or os.environ.get("LICHEE_HOME")
        or Path(__file__).resolve().parents[2] / "external" / "lichee" / "LICHeE"
    )
    jar = home / "release" / "lichee.jar"
    lib = home / "lib"
    missing = []
    if not jar.is_file():
        missing.append(f"jar {jar}")
    if not lib.is_dir():
        missing.append(f"lib directory {lib}")
    if shutil.which("java") is None:
        missing.append("java on PATH")
    if missing:
        raise FileNotFoundError(
            f"LICHeE home {home} is unusable: missing {'; '.join(missing)}"
        )
    return jar, lib


def write_lichee_input(
    cluster_to_calls: Dict[str, Dict[SNVKey, Tuple[float, int]]],
    path: Path,
    germline_label: str = GERMLINE_ROOT_ID,
) -> List[str]:
    """Write LICHeE's tab-separated VAF input: ``#chr position description
    <samples...>``, one row per SNV present in at least one cluster's calls,
    VAF from ``cluster_to_calls`` (0.0 for a cluster with no record at that
    site).

    Column order is this module's usual numeric-then-lexicographic cluster
    sort, headed ``c<cluster id>`` (LICHeE echoes a sample's header verbatim in
    its output, so ``parse_lichee_trees`` joins on it after stripping the
    ``c``). A synthetic all-zero ``germline_label`` column is prepended as
    LICHeE's required baseline/normal sample (``-n 0``) -- see module
    docstring for why this pipeline has no real one.

    Returns the column headers actually written (germline first, then the
    ``c<id>`` clusters), for ``parse_lichee_trees``.
    """
    clusters = sorted(cluster_to_calls, key=_sort_key)
    columns = [germline_label] + [f"{_CLUSTER_COLUMN_PREFIX}{c}" for c in clusters]
    all_sites = sorted({key for calls in cluster_to_calls.values() for key in calls})

    lines = ["#chr\tposition\tdescription\t" + "\t".join(columns)]
    for chrom, pos, ref, alt in all_sites:
        description = f"{ref}>{alt}"
        row = [chrom, str(pos), description, "0.0"]  # germline VAF always 0
        for cluster in clusters:
            vaf, _alt_reads = cluster_to_calls[cluster].get(
                (chrom, pos, ref, alt), (0.0, 0)
            )
            row.append(f"{vaf:.4f}")
        lines.append("\t".join(row))

    Path(path).write_text("\n".join(lines) + "\n")
    return columns


def run_lichee(
    input_path: Path,
    out_path: Path,
    log_path: Path,
    jar: Path,
    lib: Path,
    min_vaf_present: float,
    max_vaf_absent: float,
    normal_index: int = 0,
    min_cluster_size: Optional[int] = None,
    error_margin: Optional[float] = None,
    log_header: str = "",
) -> Path:
    """Run LICHeE's ``build`` step and return ``out_path``, its ``.trees.txt``.

    Invoked as ``java -cp <jar>:<lib>/* lineage.LineageEngine``: the ``lib/*``
    is ONE classpath entry that Java expands itself, so it is passed unglobbed.
    ``-s 1`` is passed and ``-dot`` never is (it needs a display and throws
    HeadlessException on compute nodes); the topology is read from the
    ``.trees.txt`` instead. ``min_vaf_present``/``max_vaf_absent`` are
    independently configurable (LICHeE is comparison-only now, see module
    docstring, so its own clustering knobs are worth exploring rather than
    fixed to one cutoff); ``min_cluster_size``/``error_margin`` (``-e``) are
    passed only when given. LICHeE's exit status is not reliable, so it is
    ignored: success is ``out_path`` existing afterwards (any stale copy is
    removed first). Raises FileNotFoundError, naming the file and the log,
    if it does not.
    """
    out_path = Path(out_path)
    out_path.unlink(missing_ok=True)
    cmd = [
        "java",
        "-cp", f"{jar}:{lib}/*",
        LICHEE_MAIN_CLASS,
        "-build",
        "-i", str(input_path),
        "-n", str(normal_index),
        "-s", "1",
        "-o", str(out_path),
        "-minVAFPresent", str(min_vaf_present),
        "-maxVAFAbsent", str(max_vaf_absent),
    ]  # fmt: skip
    if min_cluster_size is not None:
        cmd += ["-minClusterSize", str(min_cluster_size)]
    if error_margin is not None:
        cmd += ["-e", str(error_margin)]
    with open(log_path, "w") as log:
        if log_header:
            log.write(log_header)
        log.write("command: " + " ".join(cmd) + "\n\n")
        log.flush()
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)

    if not out_path.exists():
        raise FileNotFoundError(
            f"LICHeE wrote no {out_path}; see {log_path} for its output"
        )
    return out_path


_LICHEE_VERDICT_RE = re.compile(
    r"^.*\bvalid tree[s]?\b.*$", re.IGNORECASE | re.MULTILINE
)


def extract_lichee_verdict(log_path: Path) -> Optional[str]:
    """LICHeE's own stated verdict line (e.g. "Found 0 valid trees"),
    quoted verbatim from its captured log if present, for the diagnostics --
    LICHeE is comparison-only, so its raw wording matters more than any
    interpretation of it here."""
    text = Path(log_path).read_text(errors="replace")
    match = _LICHEE_VERDICT_RE.search(text)
    return match.group(0).strip() if match else None


@dataclass
class LicheeTrees:
    """The three blocks of a LICHeE ``.trees.txt``, as parsed."""

    profiles: Dict[str, str]  # node id -> presence profile over the columns
    vafs: Dict[str, List[float]]  # node id -> VAFs of its PRESENT columns, in order
    parent_of: Dict[str, str]  # Tree 0's {child: parent}, root excluded
    root: str  # Tree 0's root node id
    n_trees: int
    # sample name -> [(depth, profile, vaf)]; empty for a GL-only sample
    decomposition: Dict[str, List[Tuple[int, str, float]]]


_TREE_HEADER_RE = re.compile(r"^\*{4}Tree (\d+)\*{4}$")
_TREE_EDGE_RE = re.compile(r"^(\S+)\s*->\s*(\S+)$")
_DECOMP_HEADER_RE = re.compile(r"^Sample lineage decomposition:\s*(\S+)$")
_DECOMP_LINE_RE = re.compile(r"^(\.*)([01]+):\s*(\S+)\s*\[[^\]]*\]$")


def parse_lichee_trees(path: Path, column_order: List[str]) -> LicheeTrees:
    """Parse LICHeE's ``.trees.txt``: the ``Nodes:`` block, ``****Tree 0****``
    and ``Sample decomposition:``.

    ``column_order`` is the input header (germline first, then the ``c<id>``
    clusters); a node's profile is over those columns left to right, bit 0
    being the germline. A node's VAF vector covers only its PRESENT columns.
    Tree 0's edges come in no particular order, so they are read into a
    parent map. Only Tree 0 is used; ``n_trees`` says how many were written.
    Raises ValueError, naming what was expected, for a missing block, a
    profile of the wrong length or with the germline bit set, a VAF vector
    that does not match its profile, or a tree without a unique root ``0``.
    """
    profiles: Dict[str, str] = {}
    vafs: Dict[str, List[float]] = {}
    edges: List[Tuple[str, str]] = []
    decomposition: Dict[str, List[Tuple[int, str, float]]] = {}
    n_trees = 0
    section = None
    sample: Optional[str] = None

    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "Nodes:":
            section = "nodes"
            continue
        tree_header = _TREE_HEADER_RE.match(line)
        if tree_header:
            n_trees += 1
            section = "tree0" if tree_header.group(1) == "0" else "other_tree"
            continue
        if line.startswith("Sample decomposition"):
            section = "decomposition"
            continue
        if line.startswith("SNV info"):
            section = "ignored"
            continue

        if section == "nodes":
            fields = line.split("\t")
            if len(fields) < 3:
                raise ValueError(f"Nodes: line has fewer than 3 tab fields: {raw!r}")
            node, profile, vaf_text = (f.strip() for f in fields[:3])
            try:
                vaf_values = [float(v) for v in vaf_text.strip("[]").split()]
            except ValueError as e:
                raise ValueError(f"unreadable VAF vector in {raw!r}") from e
            if len(profile) != len(column_order):
                raise ValueError(
                    f"node {node} profile {profile!r} has {len(profile)} bits, "
                    f"expected {len(column_order)} (columns {column_order})"
                )
            if profile[0] != "0":
                raise ValueError(
                    f"node {node} profile {profile!r} has the germline bit set"
                )
            if len(vaf_values) != profile.count("1"):
                raise ValueError(
                    f"node {node}: {len(vaf_values)} VAFs for {profile.count('1')} "
                    f"present columns in profile {profile!r}"
                )
            profiles[node] = profile
            vafs[node] = vaf_values
        elif section == "tree0":
            edge = _TREE_EDGE_RE.match(line)
            if edge:
                edges.append((edge.group(1), edge.group(2)))
        elif section == "decomposition":
            header = _DECOMP_HEADER_RE.match(line)
            if header:
                sample = header.group(1)
                decomposition[sample] = []
            elif line == "GL":
                continue
            else:
                m = _DECOMP_LINE_RE.match(raw.rstrip())
                if m is None or sample is None:
                    raise ValueError(f"unreadable decomposition line: {raw!r}")
                dots = len(m.group(1))
                if dots == 0 or dots % _DECOMPOSITION_DOTS_PER_LEVEL:
                    raise ValueError(
                        f"decomposition line {raw!r} is indented {dots} dots, not a "
                        f"multiple of {_DECOMPOSITION_DOTS_PER_LEVEL}"
                    )
                decomposition[sample].append(
                    (
                        dots // _DECOMPOSITION_DOTS_PER_LEVEL,
                        m.group(2),
                        float(m.group(3)),
                    )
                )

    if not profiles:
        raise ValueError(f"{path} has no 'Nodes:' block with any nodes")
    if not edges:
        raise ValueError(f"{path} has no '****Tree 0****' block with any edges")
    if not decomposition:
        raise ValueError(f"{path} has no 'Sample decomposition:' block")

    parent_of: Dict[str, str] = {}
    for parent, child in edges:
        if child in parent_of:
            raise ValueError(f"Tree 0 gives node {child} two parents")
        parent_of[child] = parent
    roots = sorted({p for p in parent_of.values() if p not in parent_of})
    if roots != ["0"]:
        raise ValueError(
            f"Tree 0 must have the germline node 0 as its one root; roots found: "
            f"{roots}"
        )
    return LicheeTrees(profiles, vafs, parent_of, "0", n_trees, decomposition)


@dataclass
class LicheeResult:
    """The clone tree plus everything the diagnostics report."""

    tree: nx.DiGraph
    hidden_ids: Set[str]
    node_of_cluster: Dict[str, str]  # cluster -> LICHeE node it resolved to
    shared: List[List[str]]  # clusters sharing one node, one list per node
    absent: List[str]  # in no node: attached to the germline root
    at_root: List[str]  # tied only at the germline root: attached to it
    collapsed_nodes: List[str]  # LICHeE nodes carrying no cluster
    group_subtends: Dict[str, Tuple[int, int]]  # g<v> -> (direct, in subtree)
    n_trees: int
    n_nodes: int  # LICHeE nodes in Tree 0, germline excluded


def _lca(nodes: List[str], parent_of: Dict[str, str], root: str) -> str:
    def chain(n: str) -> List[str]:
        out = [n]
        while out[-1] != root:
            if out[-1] not in parent_of:
                raise ValueError(f"node {n} does not reach the root {root} in Tree 0")
            out.append(parent_of[out[-1]])
        return out

    chains = [chain(n) for n in nodes]
    common = set(chains[0]).intersection(*chains[1:])
    if not common:
        raise ValueError(f"nodes {nodes} share no ancestor in Tree 0")
    return next(a for a in chains[0] if a in common)  # nearest to the nodes


@dataclass
class OptionAAttachment:
    """The result of ``attach_option_a``: the emitted tree plus everything a
    caller's diagnostics report about how items were attached to it."""

    tree: nx.DiGraph
    hidden_ids: Set[str]
    shared: List[List[str]]  # items sharing one source node, one list per node
    collapsed_nodes: List[str]  # source nodes carrying no item
    group_subtends: Dict[str, Tuple[int, int]]  # g<v> -> (direct, in subtree)


def attach_option_a(
    parent_of: Dict[str, str],
    node_of_item: Dict[str, str],
    root: str,
    root_items: Iterable[str] = (),
    germline_id: str = GERMLINE_ROOT_ID,
    group_prefix: str = _GROUP_NODE_PREFIX,
) -> OptionAAttachment:
    """Collapse an arbitrary rooted tree (``parent_of``, a tool's own
    ``{child: parent}`` node space) onto a set of items, each already
    resolved to one node in that space (``node_of_item``), one labelled leaf
    per item ("option (a)"). Shared by build_snv_tree.py's LICHeE integration
    and build_cna_tree.py's SCICoNE integration, so a shared node is resolved
    the same way regardless of which tool produced it.

    ``C(v)``, the items resolving to a node ``v``: ``|C(v)|=1`` with
    children -- the item labels ``v``, ``v``'s child subtrees hang under it;
    ``|C(v)|=1`` no children -- leaf; ``|C(v)|>=2`` -- a hidden group node
    ``<group_prefix><v>`` stands in for ``v``, its items are sibling leaves
    under it (an unresolved polytomy) and ``v``'s children hang under it too;
    ``|C(v)|=0`` -- ``v`` is collapsed and its children lift to the nearest
    emitted ancestor.

    ``root_items`` are attached directly under ``germline_id`` rather than
    resolved through the tree (an item tied only at the root, or with no node
    of its own) -- passed straight through, never looked up in
    ``node_of_item``.

    Several items sharing one node is the normal case this attachment exists
    for, not an error (a node-per-sample tool would never need it): this
    raises ValueError only if a hidden group node's label collides with a
    real item ID or the germline ID.
    """
    children_of: Dict[str, List[str]] = defaultdict(list)
    for child, parent in parent_of.items():
        children_of[parent].append(child)

    by_node: Dict[str, List[str]] = defaultdict(list)
    for item, node in node_of_item.items():
        by_node[node].append(item)
    for items in by_node.values():
        items.sort(key=_sort_key)

    for node in by_node:
        group = f"{group_prefix}{node}"
        if group in node_of_item or group == germline_id:
            raise ValueError(f"group node label {group!r} collides with a real label")

    tree = nx.DiGraph()
    tree.add_node(germline_id)
    hidden_ids: Set[str] = set()
    collapsed: List[str] = []
    direct_count: Dict[str, int] = {}

    def place(node: str, attach_to: str) -> None:
        here = by_node.get(node, [])
        kids = sorted(children_of.get(node, []), key=_sort_key)
        if not here:
            collapsed.append(node)
            for kid in kids:
                place(kid, attach_to)
        elif len(here) == 1:
            tree.add_edge(attach_to, here[0])
            for kid in kids:
                place(kid, here[0])
        else:
            group = f"{group_prefix}{node}"
            hidden_ids.add(group)
            direct_count[group] = len(here)
            tree.add_edge(attach_to, group)
            for item in here:
                tree.add_edge(group, item)
            for kid in kids:
                place(kid, group)

    for kid in sorted(children_of.get(root, []), key=_sort_key):
        place(kid, germline_id)
    for item in sorted(root_items, key=_sort_key):
        tree.add_edge(germline_id, item)

    subtends = {
        g: (direct_count[g], len(nx.descendants(tree, g) - hidden_ids))
        for g in hidden_ids
    }
    return OptionAAttachment(
        tree=tree,
        hidden_ids=hidden_ids,
        shared=[items for _, items in sorted(by_node.items()) if len(items) > 1],
        collapsed_nodes=sorted(collapsed, key=_sort_key),
        group_subtends=subtends,
    )


def build_lichee_clone_tree(
    trees: LicheeTrees,
    column_order: List[str],
    tau: float,
    germline_id: str = GERMLINE_ROOT_ID,
) -> LicheeResult:
    """Build the SECEDO-cluster clone tree from a parsed ``.trees.txt``.

    Join. A cluster is "in" a node when the node's profile has the cluster's
    column bit set AND its entry in the node's VAF vector (matched by the bit's
    rank among the set bits) exceeds ``tau``. It attaches to the DEEPEST such
    node; if several tie for deepest, to their lowest common ancestor. That is
    cross-checked against the cluster's deepest decomposition line above
    ``tau`` (the same set of nodes must come out) and a disagreement raises.

    Attachment is ``attach_option_a`` -- see there. A cluster in no node (its
    decomposition is GL-only) or tied only at the germline root hangs directly
    off the germline root, as a root item. Several clusters sharing a node is
    the normal case, not an error: LICHeE groups SNVs by presence pattern, so
    it emits fewer nodes than there are clusters.
    """
    if column_order[0] != germline_id:
        raise ValueError(f"column 0 is {column_order[0]!r}, expected {germline_id!r}")
    cluster_of_column: Dict[str, str] = {}
    for column in column_order[1:]:
        if not column.startswith(_CLUSTER_COLUMN_PREFIX) or len(column) == 1:
            raise ValueError(f"cluster column {column!r} is not c<id>")
        cluster_of_column[column] = column[len(_CLUSTER_COLUMN_PREFIX) :]
    clusters = list(cluster_of_column.values())

    for name in trees.decomposition:
        if name != germline_id and name not in cluster_of_column:
            raise ValueError(
                f"decomposition sample {name!r} is neither {germline_id!r} nor a "
                f"c<id> column; known clusters {sorted(clusters, key=_sort_key)}"
            )

    parent_of, root = trees.parent_of, trees.root
    children_of: Dict[str, List[str]] = defaultdict(list)
    for child, parent in parent_of.items():
        children_of[parent].append(child)
    depth: Dict[str, int] = {root: 0}

    for start in parent_of:
        path, node = [], start
        while node not in depth:
            if node in path or node not in parent_of:
                raise ValueError(f"Tree 0 has a cycle or a broken chain at node {node}")
            path.append(node)
            node = parent_of[node]
        for step in reversed(path):
            depth[step] = depth[parent_of[step]] + 1

    tree_nodes = [n for n in parent_of if n in trees.profiles]
    profile_to_node: Dict[str, str] = {}
    for node in tree_nodes:
        if trees.profiles[node] in profile_to_node:
            raise ValueError(
                f"nodes {profile_to_node[trees.profiles[node]]} and {node} share "
                f"profile {trees.profiles[node]!r}; the join would be ambiguous"
            )
        profile_to_node[trees.profiles[node]] = node

    node_of_cluster: Dict[str, str] = {}
    absent: List[str] = []
    at_root: List[str] = []
    for j, column in enumerate(column_order):
        if j == 0:
            continue
        cid = cluster_of_column[column]
        in_nodes = []
        for node in tree_nodes:
            profile = trees.profiles[node]
            if profile[j] == "1":
                rank = profile[:j].count("1")
                if trees.vafs[node][rank] > tau:
                    in_nodes.append(node)

        if column not in trees.decomposition:
            raise ValueError(f"no 'Sample lineage decomposition: {column}' block")
        lines = [ln for ln in trees.decomposition[column] if ln[2] > tau]
        decomp_nodes: Set[str] = set()
        if lines:
            deepest_level = max(ln[0] for ln in lines)
            for _, profile, _ in (ln for ln in lines if ln[0] == deepest_level):
                if profile not in profile_to_node:
                    raise ValueError(
                        f"{column}: deepest decomposition profile {profile!r} matches "
                        f"no Nodes: entry (profiles {sorted(profile_to_node)})"
                    )
                decomp_nodes.add(profile_to_node[profile])

        if not in_nodes:
            if decomp_nodes:
                raise ValueError(
                    f"{column} is in no node by presence (bit set and VAF > {tau}) "
                    f"but its decomposition places it at node(s) {sorted(decomp_nodes)}"
                )
            absent.append(cid)
            continue
        top = max(depth[n] for n in in_nodes)
        deepest = {n for n in in_nodes if depth[n] == top}
        if deepest != decomp_nodes:
            raise ValueError(
                f"{column}: presence puts it at node(s) {sorted(deepest)} but its "
                f"deepest decomposition line puts it at {sorted(decomp_nodes)}"
            )
        node = (
            next(iter(deepest))
            if len(deepest) == 1
            else _lca(sorted(deepest), parent_of, root)
        )
        if node == root:
            at_root.append(cid)
        else:
            node_of_cluster[cid] = node

    attachment = attach_option_a(
        parent_of,
        node_of_cluster,
        root,
        root_items=absent + at_root,
        germline_id=germline_id,
    )
    placed = set(attachment.tree.nodes()) - attachment.hidden_ids - {germline_id}
    if placed != set(clusters):
        raise ValueError(
            f"emitted clusters {sorted(placed)} != input clusters {sorted(clusters)}"
        )
    return LicheeResult(
        tree=attachment.tree,
        hidden_ids=attachment.hidden_ids,
        node_of_cluster=node_of_cluster,
        shared=attachment.shared,
        absent=sorted(absent, key=_sort_key),
        at_root=sorted(at_root, key=_sort_key),
        collapsed_nodes=attachment.collapsed_nodes,
        group_subtends=attachment.group_subtends,
        n_trees=trees.n_trees,
        n_nodes=len(parent_of),
    )


# --------------------------------------------------------------------------- #
# Topology classification (chain vs branching)
# --------------------------------------------------------------------------- #


def is_unbranched_chain(tree: nx.DiGraph) -> bool:
    """True if every node in ``tree`` has at most one child -- a straight
    line down from the root, no branching anywhere."""
    return all(tree.out_degree(n) <= 1 for n in tree.nodes())


def chain_depth_if_linear(tree: nx.DiGraph) -> Optional[int]:
    """If ``tree`` is a single unbranched path from a unique root, return its
    depth (edge count from root to the tip); otherwise None."""
    if not is_unbranched_chain(tree):
        return None
    roots = [n for n, d in tree.in_degree() if d == 0]
    if len(roots) != 1:
        return None
    depth = 0
    node = roots[0]
    seen = {node}
    while True:
        children = list(tree.successors(node))
        if not children:
            return depth
        node = children[0]
        if node in seen:  # malformed/cyclic input guard; not expected from a tree
            return None
        seen.add(node)
        depth += 1


def classify_topology(tree: Optional[nx.DiGraph]) -> str:
    """'linear chain (depth N)', 'branching', or 'unavailable' (``tree`` is
    None) -- the one-line topology summary used throughout the diagnostics."""
    if tree is None:
        return "unavailable"
    depth = chain_depth_if_linear(tree)
    return f"linear chain (depth {depth})" if depth is not None else "branching"


def compare_topologies(
    tree_a: nx.DiGraph, tree_b: nx.DiGraph, cluster_ids: List[str]
) -> float:
    """Fraction of cluster pairs whose ancestor/descendant relation agrees
    between two trees -- a pairwise-relation summary, not a full tree-edit
    distance."""

    def relation(tree: nx.DiGraph, a: str, b: str) -> str:
        if a in tree and b in tree:
            if a in nx.ancestors(tree, b):
                return "ancestor"
            if b in nx.ancestors(tree, a):
                return "descendant"
        return "unrelated"

    ids = list(cluster_ids)
    total = agree = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            total += 1
            if relation(tree_a, ids[i], ids[j]) == relation(tree_b, ids[i], ids[j]):
                agree += 1
    return agree / total if total else float("nan")


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


def _lichee_lines(r: LicheeResult) -> List[str]:
    """Findings from a LICHeE run: node/cluster counts, shared nodes, absent
    clusters and the hidden group nodes (which add a random-walk step for the
    clusters below them in TreeHDP, so are named here to be auditable against
    the CNA-tree comparison)."""
    n_clusters = len(set(r.tree.nodes()) - r.hidden_ids - {GERMLINE_ROOT_ID})
    lines = [
        f"LICHeE Tree 0: {r.n_nodes} nodes carrying {n_clusters} clusters "
        f"({len(set(r.node_of_cluster.values()))} distinct nodes hold a cluster). "
        "Fewer nodes than clusters is expected: LICHeE groups SNVs by "
        "presence pattern, so it cannot tell some clusters apart.\n"
    ]
    if r.n_trees > 1:
        lines.append(
            f"FINDING: LICHeE emitted {r.n_trees} trees; Tree 0 (top-ranked) used.\n"
        )
    for cids in r.shared:
        lines.append(
            f"FINDING: clusters {','.join(cids)} indistinguishable by SNV profile, "
            "attached as siblings.\n"
        )
    if r.absent:
        lines.append(
            f"FINDING: cluster(s) {','.join(r.absent)} are in no LICHeE node (no SNV "
            f"above the VAF cutoff); attached as direct leaves of {GERMLINE_ROOT_ID}.\n"
        )
    if r.at_root:
        lines.append(
            f"FINDING: cluster(s) {','.join(r.at_root)} tie only at the germline "
            f"root; attached as direct leaves of {GERMLINE_ROOT_ID}.\n"
        )
    if r.collapsed_nodes:
        lines.append(
            f"LICHeE node(s) {','.join(r.collapsed_nodes)} carry no cluster and "
            "were collapsed.\n"
        )
    for group, (direct, below) in sorted(r.group_subtends.items()):
        lines.append(
            f"Hidden group node {group}: {direct} clusters directly, {below} in its "
            "subtree. It adds one random-walk step for them in TreeHDP.\n"
        )
    return lines


def _dollo_lines(r: DolloResult) -> List[str]:
    """Findings from the Dollo search: cost and runner-up, ties and
    consensus, per-clade bootstrap support (every clade, including ones
    collapsed for low support), the 9-vs-10-style win-rate between the top
    two topologies, and the per-edge gain/loss report."""
    lines = [
        f"Dollo search: {r.n_topologies} topologies, best cost {r.best_cost}"
        + (f", runner-up {r.runner_up_cost}" if r.runner_up_cost is not None else "")
        + f". Optimal topology: {r.optimal_render}\n"
    ]
    if r.runner_up_render is not None:
        lines.append(f"Runner-up topology: {r.runner_up_render}\n")
    if r.n_ties > 1:
        lines.append(
            f"FINDING: {r.n_ties} topologies tied at the minimum cost; the "
            "strict consensus (shared clades only) is emitted, not one "
            "picked arbitrarily.\n"
        )
    if r.n_bootstrap > 0:
        lines.append(f"\nBootstrap: {r.n_bootstrap} replicates (resampled SNVs).\n")
        lines.append(
            f"Top-two win rate: optimal topology won {r.top2_win_rate['optimal']:.1%} "
            f"of replicates, runner-up {r.top2_win_rate['runner_up']:.1%} -- this is "
            "the direct instability signal between the two closest topologies.\n"
        )
        lines.append("Clade support (fraction of replicates each clade appears in):\n")
        for clade, support in sorted(
            r.clade_support.items(), key=lambda kv: (-kv[1], sorted(kv[0]))
        ):
            collapsed = (
                " -- COLLAPSED (below threshold)" if clade in r.collapsed_clades else ""
            )
            label = ",".join(sorted(clade, key=_sort_key))
            lines.append(f"  {{{label}}}: {support:.1%}{collapsed}\n")
    else:
        lines.append("Bootstrap: skipped (--dollo-bootstrap 0).\n")

    lines.append("\nPer-edge gains and losses (trunk = germline -> the MRCA):\n")
    for (parent, child), counts in sorted(r.edge_report.items()):
        lines.append(
            f"  {parent} -> {child}: {counts['gains']} gained, "
            f"{counts['losses']} lost\n"
        )
    return lines


def _lichee_comparison_lines(
    lichee_result: Optional[LicheeResult],
    lichee_error: Optional[str],
    verdict: Optional[str],
) -> List[str]:
    """LICHeE's role is comparison only now (see module docstring): it
    never produces ``snv_tree.nwk``, so this reports its own verdict --
    including a literal "0 valid trees"-style line, if LICHeE printed one --
    without acting on it."""
    lines = ["\n## LICHeE (comparison only, not used for snv_tree.nwk)\n"]
    if verdict is not None:
        lines.append(f"LICHeE's own verdict: {verdict!r}\n")
    if lichee_result is not None:
        lines.append("LICHeE ran and its cluster attachments resolved.\n")
        lines.extend(_lichee_lines(lichee_result))
    elif lichee_error is not None:
        lines.append(f"LICHeE did not produce a usable tree: {lichee_error}\n")
    return lines


def write_diagnostics(
    path: Path,
    n_violations: int,
    skip_counts: Dict[str, int],
    dollo_result: DolloResult,
    lichee_result: Optional[LicheeResult] = None,
    lichee_error: Optional[str] = None,
    lichee_verdict: Optional[str] = None,
    tree_comparison: Optional[Dict[str, List[List[str]]]] = None,
    union_size: Optional[int] = None,
    cluster_union_stats: Optional[Dict[str, Dict[str, int]]] = None,
) -> None:
    """Dollo parsimony is the primary and only source of ``snv_tree.nwk``
    (see module docstring for why LICHeE and Camin-Sokal are not); LICHeE,
    when ``--run-lichee`` is passed, is reported here purely as an optional
    comparison.

    ``cluster_union_stats`` is ``{cluster: {"genotyped", "present",
    "pass1_pass", "dropped_outside_union"}}`` -- see ``main`` -- reported so a
    force-called record outside the union (Mutect2's own discovery calls, or
    a non-PASS record) is visible as a real number, not silently dropped.
    """
    lines = [
        "# Stage 08 (SNV tree) diagnostics\n\n",
        "## Method used: dollo\n",
    ]
    lines.extend(_dollo_lines(dollo_result))

    if (
        lichee_result is not None
        or lichee_error is not None
        or lichee_verdict is not None
    ):
        lines.extend(
            _lichee_comparison_lines(lichee_result, lichee_error, lichee_verdict)
        )

    lines.append(f"\nThree-gamete (perfect-phylogeny) violations: {n_violations}\n")
    total_skipped = sum(skip_counts.values())
    lines.append(f"SNVs skipped in binning: {total_skipped} {dict(skip_counts)}\n")

    if union_size is not None and cluster_union_stats:
        lines.append(f"\n## Union restriction ({union_size} sites in the union)\n")
        lines.append(
            "cluster: union sites genotyped, present, dropped outside union, "
            "pass-1 PASS count\n"
        )
        for cluster in sorted(cluster_union_stats, key=_sort_key):
            s = cluster_union_stats[cluster]
            lines.append(
                f"  clone{cluster}: genotyped {s['genotyped']}, "
                f"present {s['present']}, dropped {s['dropped_outside_union']}, "
                f"pass-1 PASS {s['pass1_pass']}\n"
            )
            if s["dropped_outside_union"] > 0:
                lines.append(
                    f"FINDING: clone{cluster} had {s['dropped_outside_union']} "
                    "forced record(s) outside the union (Mutect2's own discovery "
                    "calls, or a record the force-call pass never targeted); "
                    "ignored.\n"
                )

    lines.append("\n## Topology result\n")
    lines.append(f"Emitted tree: {classify_topology(dollo_result.tree)}\n")
    if classify_topology(dollo_result.tree).startswith("linear chain"):
        lines.append(
            "FINDING: the emitted tree is a non-branching chain -- the "
            "force-called SNVs do not support a branching phylogeny.\n"
        )

    if tree_comparison is not None:
        lines.append("\n## Comparison to --compare-tree (e.g. the CNA tree)\n")
        lines.append(f"Clades in both: {tree_comparison['both']}\n")
        lines.append(f"Clades only in the SNV tree: {tree_comparison['only_first']}\n")
        lines.append(
            f"Clades only in the other tree: {tree_comparison['only_second']}\n"
        )

    Path(path).write_text("".join(lines))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--vcf-dir",
        required=True,
        type=Path,
        help="directory of clone<c>_<chrom>.forced.vcf files (stage 06b's "
        "pass-2 force-call output, copied out by stage 07)",
    )
    p.add_argument("--ref-fasta", required=True, type=Path)
    p.add_argument(
        "--cosmic-signatures",
        required=True,
        type=Path,
        help="cosmic_signatures.csv, used only to assert the channel order",
    )
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--presence-min-vaf",
        type=float,
        default=0.05,
        help="a force-called site counts as present in a cluster only at or "
        "above this VAF (see config.sh's PRESENCE_MIN_VAF)",
    )
    p.add_argument(
        "--presence-min-alt-reads",
        type=int,
        default=2,
        help="a force-called site counts as present in a cluster only at or "
        "above this many ALT reads (see config.sh's PRESENCE_MIN_ALT_READS)",
    )
    p.add_argument(
        "--normal-cluster-id",
        required=True,
        help="pseudo-normal SECEDO cluster (NORMAL_CLUSTER_ID in config.sh); "
        "asserted absent from the discovered cluster set -- it must never "
        "enter the presence matrix or spectra",
    )
    p.add_argument(
        "--dollo-max-clusters",
        type=int,
        default=7,
        help="refuse the exhaustive Dollo topology search past this many "
        "clusters rather than hang (see enumerate_dollo_topologies)",
    )
    p.add_argument(
        "--dollo-bootstrap",
        type=int,
        default=1000,
        help="bootstrap replicates for clade support (0 skips estimation "
        "and emits the raw tie consensus, unfiltered)",
    )
    p.add_argument(
        "--dollo-min-clade-support",
        type=float,
        default=0.7,
        help="a consensus clade with bootstrap support below this is "
        "collapsed into a polytomy",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="Dollo bootstrap resampling seed"
    )
    p.add_argument(
        "--compare-tree",
        type=Path,
        default=None,
        help="another Newick tree over the same clusters (e.g. the CNA "
        "tree) to report clade-level agreement against, in the diagnostics",
    )
    p.add_argument(
        "--run-lichee",
        action="store_true",
        help="also run LICHeE as an optional comparison, reported in the "
        "diagnostics only -- it never produces snv_tree.nwk (see module "
        "docstring for why Dollo is primary)",
    )
    p.add_argument(
        "--lichee-home",
        default=None,
        help="LICHeE checkout (release/lichee.jar and lib/); defaults to "
        "$LICHEE_HOME, then realdata/external/lichee/LICHeE",
    )
    p.add_argument(
        "--lichee-min-vaf-present", type=float, default=None,
        help="LICHeE's own -minVAFPresent; defaults to --presence-min-vaf",
    )  # fmt: skip
    p.add_argument(
        "--lichee-max-vaf-absent", type=float, default=None,
        help="LICHeE's own -maxVAFAbsent; defaults to --presence-min-vaf",
    )  # fmt: skip
    p.add_argument(
        "--lichee-min-cluster-size", type=int, default=None,
        help="LICHeE's own -minClusterSize; omitted (LICHeE's own default) "
        "unless given",
    )  # fmt: skip
    p.add_argument(
        "--lichee-error-margin", type=float, default=None,
        help="LICHeE's own -e; omitted (LICHeE's own default) unless given",
    )  # fmt: skip
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    expected_cols = channel_labels()
    cosmic_cols = list(
        pd.read_csv(args.cosmic_signatures, index_col=0, nrows=0).columns
    )
    if cosmic_cols != expected_cols:
        sys.exit(
            f"{args.cosmic_signatures} channel order does not match the expected "
            f"{N_CHANNELS}-channel axis -- got {cosmic_cols[:4]}..., "
            f"expected {expected_cols[:4]}.... Refusing to emit spectra that would "
            "misalign against the model's signature columns."
        )

    cluster_vcfs = discover_forced_cluster_vcfs(args.vcf_dir)
    if not cluster_vcfs:
        sys.exit(f"no clone*_*.forced.vcf files found in {args.vcf_dir}")
    if args.normal_cluster_id in cluster_vcfs:
        sys.exit(
            f"pseudo-normal cluster {args.normal_cluster_id!r} has its own forced "
            f"VCF in {args.vcf_dir}; it must never enter the presence matrix or "
            "spectra (check --normal-cluster-id against config.sh's NORMAL_CLUSTER_ID)"
        )

    # The union stage 06b force-called against: reconstructed from pass-1's own PASS
    # VCFs, already in --vcf-dir, rather than depending on stage 06b's union file
    # under scratch. Presence is decided only at these sites -- see
    # restrict_calls_to_union below.
    pass1_vcfs = discover_pass1_cluster_vcfs(args.vcf_dir)
    union_sites = build_union_sites(pass1_vcfs)
    pass1_counts = {
        cluster: len(set().union(*(parse_vcf_site_keys(f) for f in files)))
        for cluster, files in pass1_vcfs.items()
    }

    cluster_to_calls: Dict[str, Dict[SNVKey, Tuple[float, int]]] = {}
    cluster_union_stats: Dict[str, Dict[str, int]] = {}
    for cluster, files in cluster_vcfs.items():
        raw_calls: Dict[SNVKey, Tuple[float, int]] = {}
        for f in files:
            raw_calls.update(parse_forced_vcf_calls(f, cluster))
        calls, n_dropped = restrict_calls_to_union(raw_calls, union_sites)
        cluster_to_calls[cluster] = calls
        cluster_union_stats[cluster] = {
            "genotyped": len(calls),
            "dropped_outside_union": n_dropped,
            "pass1_pass": pass1_counts.get(cluster, 0),
        }

    # Presence, per cluster: VAF and ALT-read thresholds applied directly to the
    # force-called genotype, not the FILTER column. This same call set backs
    # the presence matrix and the 96-channel spectra below.
    cluster_to_snvs = resolve_presence_calls(
        cluster_to_calls, args.presence_min_vaf, args.presence_min_alt_reads
    )
    assert_presence_within_union(cluster_to_snvs, union_sites)
    for cluster, snvs in cluster_to_snvs.items():
        cluster_union_stats[cluster]["present"] = len(snvs)

    snv_matrix = build_snv_presence_matrix(cluster_to_snvs)
    snv_matrix.to_csv(args.out_dir / "clone_snv_matrix.csv")

    mutation_sets = mutation_sets_from_matrix(snv_matrix)
    n_violations = three_gamete_violations(mutation_sets)

    # Dollo parsimony over trees with hidden internal nodes: the sole source of
    # snv_tree.nwk (see module docstring for why LICHeE and Camin-Sokal do not fit
    # this data). Clusters are leaves; every internal node is a hidden ancestor.
    pattern_counts = pattern_counts_from_matrix(snv_matrix)
    dollo_result = dollo_tree(
        pattern_counts,
        sorted(cluster_to_snvs, key=_sort_key),
        max_clusters=args.dollo_max_clusters,
        n_bootstrap=args.dollo_bootstrap,
        min_clade_support=args.dollo_min_clade_support,
        seed=args.seed,
    )

    lichee_result: Optional[LicheeResult] = None
    lichee_error: Optional[str] = None
    lichee_verdict: Optional[str] = None
    if args.run_lichee:
        try:
            jar, lib = resolve_lichee_home(args.lichee_home)
            lichee_input_path = args.out_dir / "lichee_input.txt"
            column_order = write_lichee_input(cluster_to_calls, lichee_input_path)
            log_path = args.out_dir / "lichee_run.log"
            trees_path = run_lichee(
                lichee_input_path,
                out_path=args.out_dir / LICHEE_OUT_NAME,
                log_path=log_path,
                jar=jar,
                lib=lib,
                min_vaf_present=args.lichee_min_vaf_present or args.presence_min_vaf,
                max_vaf_absent=args.lichee_max_vaf_absent or args.presence_min_vaf,
                min_cluster_size=args.lichee_min_cluster_size,
                error_margin=args.lichee_error_margin,
            )
            lichee_verdict = extract_lichee_verdict(log_path)
            lichee_result = build_lichee_clone_tree(
                parse_lichee_trees(trees_path, column_order),
                column_order,
                args.presence_min_vaf,
            )
        except (OSError, ValueError) as exc:
            lichee_error = str(exc)
            if (args.out_dir / "lichee_run.log").exists():
                lichee_verdict = extract_lichee_verdict(args.out_dir / "lichee_run.log")

    newick_str = digraph_to_newick(dollo_result.tree, GERMLINE_ROOT_ID)
    verify_newick(
        newick_str, set(cluster_to_snvs), GERMLINE_ROOT_ID, dollo_result.hidden_ids
    )
    (args.out_dir / "snv_tree.nwk").write_text(newick_str + "\n")

    import pysam

    with pysam.FastaFile(str(args.ref_fasta)) as fasta:
        spectra, skip_counts = bin_cluster_spectra(cluster_to_snvs, fasta)
    spectra.to_csv(args.out_dir / "spectra.csv")

    tree_comparison = None
    if args.compare_tree is not None:
        tree_comparison = compare_tree_clades(
            dollo_result.tree, GERMLINE_ROOT_ID, args.compare_tree
        )

    write_diagnostics(
        args.out_dir / "snv_tree_diagnostics.txt",
        n_violations,
        skip_counts,
        dollo_result,
        lichee_result,
        lichee_error,
        lichee_verdict,
        tree_comparison,
        len(union_sites),
        cluster_union_stats,
    )

    print(
        "Tumour clusters (tree tips + internal observed clones): "
        f"{len(cluster_to_snvs)}"
    )
    print(
        f"Dollo: best cost {dollo_result.best_cost}, runner-up "
        f"{dollo_result.runner_up_cost}, {dollo_result.n_ties} tied topologies"
    )
    print(f"Union size: {len(union_sites)} sites")
    print("Per cluster: union sites genotyped, present, pass-1 PASS count:")
    for cid in sorted(cluster_to_snvs, key=_sort_key):
        s = cluster_union_stats[cid]
        print(
            f"  clone{cid}: genotyped {s['genotyped']}, present {s['present']}, "
            f"pass-1 PASS {s['pass1_pass']}"
        )


if __name__ == "__main__":
    main()
