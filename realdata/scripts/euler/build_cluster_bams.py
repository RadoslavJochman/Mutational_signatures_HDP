"""Build one merged BAM per SECEDO cluster, ready for mutect.sh's per-cluster input.

Ours: neither secedo nor secedo-evaluation does this. `secedo` writes a flat
`clustering` file (comma-separated cluster id, one entry per cell index) plus, from
the earlier `pileup` step, a `<prefix>_<chromosome>.map` file (cell index -> original
per-cell BAM name, tab-separated, same alphabetical ordering `pileup` used when reading
its input directory). Joining the two gives cluster id -> set of per-cell BAM names;
this script does that join and merges each cluster's original (whole-genome, pre-split)
per-cell BAMs with `samtools merge`, writing `clone<cluster_id>.bam` -- the naming
mutect.sh expects (originally `clone<n>_1Y.bam`; we drop the `_1Y` suffix here since
this repo doesn't otherwise use it, and update mutect.sh's input filenames to match).
"""

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def reheader_to_coordinate_sorted(bam_path: Path) -> None:
    """Rewrite a merged BAM's @HD line to declare SO:coordinate.

    samtools merge leaves @HD SO:unknown on its output even when every input is
    coordinate-sorted and merged in order (as here), because it does not track
    sort order through the merge. GATK Mutect2 refuses to merge two such BAMs
    (tumour/normal) -- htsjdk's MergingSamRecordIterator throws
    NullPointerException: this.comparator is null -- unless each declares a sort
    order. This is a header-only correction, not a re-sort: the records are
    already coordinate-sorted, so we just say so.
    """
    header = subprocess.run(
        ["samtools", "view", "-H", str(bam_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    new_header, n = re.subn(
        r"^@HD.*$", "@HD\tVN:1.6\tSO:coordinate", header, count=1, flags=re.MULTILINE
    )
    if n == 0:
        # No @HD line at all: prepend one.
        new_header = "@HD\tVN:1.6\tSO:coordinate\n" + header

    hdr_file = bam_path.with_suffix(".hdr")
    reheadered = bam_path.with_suffix(".reh.bam")
    try:
        hdr_file.write_text(new_header)
        with open(reheadered, "wb") as out:
            subprocess.run(
                ["samtools", "reheader", str(hdr_file), str(bam_path)],
                check=True,
                stdout=out,
            )
        reheadered.replace(bam_path)
    finally:
        hdr_file.unlink(missing_ok=True)
        reheadered.unlink(missing_ok=True)


def read_map(map_file: Path) -> dict[int, str]:
    idx_to_name = {}
    for line in map_file.read_text().splitlines():
        if not line.strip():
            continue
        name, idx = line.rsplit("\t", 1)
        idx_to_name[int(idx)] = name
    return idx_to_name


def read_clustering(clustering_file: Path) -> list[int]:
    return [int(x) for x in clustering_file.read_text().strip().split(",")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map-file", required=True, type=Path)
    p.add_argument("--clustering-file", required=True, type=Path)
    p.add_argument("--cell-bams-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--min-cluster-size", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    idx_to_name = read_map(args.map_file)
    clusters = read_clustering(args.clustering_file)
    if len(clusters) != len(idx_to_name):
        sys.exit(
            f"clustering has {len(clusters)} cells but map file has {len(idx_to_name)} "
            "-- these must come from the same secedo run."
        )

    cluster_to_cells = defaultdict(list)
    for idx, cluster_id in enumerate(clusters):
        if cluster_id == 0:
            continue  # 0 means "no cluster" (secedo_main.cpp)
        cluster_to_cells[cluster_id].append(idx_to_name[idx])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for cluster_id, names in sorted(cluster_to_cells.items()):
        if len(names) < args.min_cluster_size:
            print(f"cluster {cluster_id}: {len(names)} cells, below minimum, skipping")
            continue
        bams = [args.cell_bams_dir / f"{name}.bam" for name in names]
        missing = [b for b in bams if not b.exists()]
        if missing:
            print(f"cluster {cluster_id}: {len(missing)} missing cell BAMs, e.g. {missing[0]}")
            continue
        out_bam = args.out_dir / f"clone{cluster_id}.bam"
        cmd = ["samtools", "merge", "-f", str(out_bam), *[str(b) for b in bams]]
        print(f"cluster {cluster_id}: merging {len(bams)} cells -> {out_bam}")
        if args.dry_run:
            print(" ".join(cmd))
            continue
        subprocess.run(cmd, check=True)
        reheader_to_coordinate_sorted(out_bam)
        subprocess.run(["samtools", "index", str(out_bam)], check=True)


if __name__ == "__main__":
    main()