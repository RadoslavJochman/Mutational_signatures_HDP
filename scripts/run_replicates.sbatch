#!/bin/bash
# run_replicates.sbatch
#
# One SLURM array task per replicate: generate the dataset from its config, fit
# it, and score recovery, each into its own directory. Replicates of a setting
# differ only in simulation.seed (and experiment_name), so grouping their
# recovery_*.csv by setting afterwards (aggregate_replicates.py) gives the
# across-dataset error bars.
#
# Layout assumed (paths are anchored to the project root, not to the cwd):
#     <root>/scripts/        generate_data.py, run_inference.py, scoring
#     <root>/experiments/    <sweep>/manifest_configs.txt, <sweep>/rep<NN>/config.yaml,
#                             <sweep>/rep<NN>/{data,results,plots}/ (this script's own output)
#     <root>/slurm_scripts/  this file; submit from here
# Submit from slurm_scripts/ so SLURM_SUBMIT_DIR points there, or export
# PROJECT_ROOT to override. The Python runs from scripts/ so the configs'
# ../experiments resolves correctly.
#
# Usage
#     sbatch run_replicates.sbatch experiments/corr_sweep/manifest_configs.txt denovo
#     sbatch run_replicates.sbatch <manifest> fixed
#     sbatch run_replicates.sbatch <manifest> denovo /cluster/scratch/$(id -un)/traces
# Manifest paths and the config paths inside it are taken relative to the project
# root (or absolute). Set --array to 0..(N-1) for N lines; %8 caps concurrency.
# The optional 3rd argument (or $TRACE_SCRATCH, or $SCRATCH) is where scored
# traces are moved.

#SBATCH --job-name=treehdp_rep
#SBATCH --array=0-14%8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rep_%A_%a.out

set -euo pipefail
export PYTHONUNBUFFERED=1   # flush Python stdout to the log live, not in blocks

# locate the project
ROOT="${PROJECT_ROOT:-$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}/.." && pwd)}"
SCRIPTS="$ROOT/scripts"
[ -f "$ROOT/.venv/bin/activate" ] && source "$ROOT/.venv/bin/activate"

# inputs
MANIFEST_IN=${1:?usage: sbatch run_replicates.sbatch <manifest> [fixed|denovo] [scratch]}
MODE=${2:-denovo}

# Resolve a path that may be absolute, root-relative, or scripts-relative. The
# manifest is usually root-relative (experiments/...), while the config paths
# inside it are absolute (make_replicate_configs.py writes them that way).
# Use whichever base exists.
resolve_path() {
    local p="$1" base
    case "$p" in /*) printf '%s' "$p"; return ;; esac
    for base in "$ROOT" "$SCRIPTS" "${SLURM_SUBMIT_DIR:-$PWD}"; do
        if [ -e "$base/$p" ]; then readlink -f "$base/$p"; return; fi
    done
    printf '%s' "$ROOT/$p"   # not found; return a path so the error message is clear
}

MANIFEST=$(resolve_path "$MANIFEST_IN")
[ -f "$MANIFEST" ] || { echo "manifest not found: $MANIFEST_IN (looked under $ROOT and $SCRIPTS)"; exit 1; }

CONFIG_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
[ -n "$CONFIG_LINE" ] || { echo "no config on line $((SLURM_ARRAY_TASK_ID + 1)) of $MANIFEST"; exit 1; }
CONFIG=$(resolve_path "$CONFIG_LINE")
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG_LINE (looked under $ROOT and $SCRIPTS)"; exit 1; }

cd "$SCRIPTS"   # so the configs' ../experiments resolves here
echo "task $SLURM_ARRAY_TASK_ID  mode=$MODE  root=$ROOT"
echo "config=$CONFIG"

# Stamp the requested mode into a per-task scratch copy of the config, so
# run_inference.py's required inference.model is set without mutating the
# checked-in config (the manifest's config may be reused or read by other
# tooling). generate_data.py keeps reading the original $CONFIG; it never
# looks at inference.model.
INF_CONFIG=$(mktemp --suffix=.yaml)
trap 'rm -f "$INF_CONFIG"' EXIT
python - "$CONFIG" "$MODE" "$INF_CONFIG" <<'PY'
import sys, yaml
cfg_path, mode, out_path = sys.argv[1:4]
cfg = yaml.safe_load(open(cfg_path))
cfg["inference"]["model"] = mode
yaml.safe_dump(cfg, open(out_path, "w"), sort_keys=False)
PY

# resolve the experiment name and its data/results dirs from the config.
# NAME can itself contain '/' for a sweep replicate (e.g. corr_sweep/rep03);
# read explicitly here rather than derived from $RES later, since $RES now
# always ends in '.../results' and basename would collide across every
# replicate.
read NAME DATA RES < <(python - "$CONFIG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
name = c.get("experiment_name", "experiment")
root = f'{c["experiment_root"].rstrip("/")}/{name}'
print(name, f'{root}/data', f'{root}/results')
PY
)
echo "NAME=$NAME  DATA=$DATA  RES=$RES  (resolved from $SCRIPTS)"

# generate -> infer -> score
python generate_data.py --config "$CONFIG"
python run_inference.py --config "$INF_CONFIG"

TRACE_FILE=trace_raw.nc
[ "$MODE" = fixed ] && TRACE_FILE=trace.nc

SIGFLAG=""
[ "$MODE" = denovo ] && SIGFLAG="--true-signatures $DATA/fixed_signatures.csv"
python recovery_vs_truth.py \
    --trace "$RES/$TRACE_FILE" \
    --true-activities "$DATA/true_activities.csv" \
    --newick "$DATA/newick_string.nwk" \
    $SIGFLAG \
    --metrics tv hellinger cosine \
    --outdir "$RES/recovery"

# de novo only: NMF baseline on the same counts, identical metrics
if [ "$MODE" = denovo ]; then
    python nmf_baseline.py \
        --counts "$DATA/mutation_count_matrix.csv" \
        --true-activities "$DATA/true_activities.csv" \
        --true-signatures "$DATA/fixed_signatures.csv" \
        --metrics tv hellinger cosine \
        --outdir "$RES/nmf"
fi

# once scored, move traces to scratch
SCRATCH_DIR=${3:-${TRACE_SCRATCH:-${SCRATCH:-/cluster/scratch/$(id -un)}/treehdp_traces}}
TRACE_DEST="$SCRATCH_DIR/$NAME"
mkdir -p "$TRACE_DEST"
for t in trace.nc trace_raw.nc trace_aligned.nc trace_raw.zarr trace_aligned.zarr; do
    if [ -e "$RES/$t" ]; then
        mv "$RES/$t" "$TRACE_DEST/"
        echo "moved $t -> $TRACE_DEST/"
    fi
done

echo "done $SLURM_ARRAY_TASK_ID"
