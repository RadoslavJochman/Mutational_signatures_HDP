#!/usr/bin/env bash
# validate_switch.sh -- drive the pre-merge validation of the switch model.
#
# Stages (argument, default "all"):
#   configs   expand the five base configs under experiments/validate_switch/
#             into replicate configs with make_replicate_configs.py
#             (3 seeds for high/low x v1/v2, 1 seed for the de novo arm)
#   run       for every replicate config: generate, infer (timed), recovery
#             _vs_truth, scaling_metrics, and for the v2 arms switch_states,
#             switch_recovery and plot_switch_recovery
#   report    write experiments/validate_switch/report.md
#   all       configs, run, report
#
# Run from anywhere; it cds into scripts/ like every other pipeline entry
# point. THIN (default 4) thins the posterior draws the state sampler uses.
# Set PYTENSOR_FLAGS in the calling shell if a backend override is needed;
# the report records which backend was used. Nothing here edits a config.
set -euo pipefail
cd "$(dirname "$0")"

ROOT=../experiments/validate_switch
STAGE=${1:-all}
THIN=${THIN:-4}

stage_configs() {
  for arm in high_v2 high_v1 low_v2 low_v1; do
    python make_replicate_configs.py --base "$ROOT/base_$arm.yaml" --n-reps 3 --setting-value "$arm"
  done
  python make_replicate_configs.py --base "$ROOT/base_high_v2_denovo.yaml" --n-reps 1 \
    --setting-value high_v2_denovo
}

cfg_get() {  # cfg_get <config> <python expression on cfg>
  python -c "import sys, yaml; cfg = yaml.safe_load(open(sys.argv[1])); print($2)" "$1"
}

run_arm() {
  local cfg=$1
  local dir data res mode switching rep trace
  dir=$(dirname "$cfg")
  data=$dir/data
  res=$dir/results
  rep=$(basename "$dir" | sed 's/^rep0*//; s/^$/0/')
  mode=$(cfg_get "$cfg" "cfg['inference']['model']")
  switching=$(cfg_get "$cfg" "bool((cfg['inference'].get('switching') or {}).get('enabled', False))")

  echo "=== $cfg (model=$mode, switching=$switching) ==="
  python generate_data.py --config "$cfg"

  local start=$SECONDS
  python run_inference.py --config "$cfg"
  mkdir -p "$res"
  echo $((SECONDS - start)) > "$res/infer_seconds.txt"

  local true_sig=()
  local fixed_sig=()
  if [ "$mode" = denovo ]; then
    trace=$res/trace_aligned.nc
    true_sig=(--true-signatures "$data/fixed_signatures.csv")
  else
    trace=$res/trace.nc
    fixed_sig=(--fixed-signatures "$data/fixed_signatures.csv")
  fi

  python recovery_vs_truth.py --trace "$trace" \
    --true-activities "$data/true_activities.csv" \
    --newick "$data/newick_string.nwk" "${true_sig[@]}" \
    --metrics cosine tv --outdir "$res/recovery"
  python scaling_metrics.py --trace "$trace" \
    --true-activities "$data/true_activities.csv" \
    --newick "$data/newick_string.nwk" \
    --n-trees "$rep" --out "$res/scaling_metrics.csv"

  if [ "$switching" = True ]; then
    python switch_states.py --trace "$trace" \
      --newick "$data/newick_string.nwk" \
      --counts "$data/mutation_count_matrix.csv" "${fixed_sig[@]}" \
      --thin "$THIN" --seed 0 --outdir "$res/switch"
    python switch_recovery.py --trace "$trace" \
      --true-active-sets "$data/true_active_sets.csv" \
      --true-activities "$data/true_activities.csv" \
      --newick "$data/newick_string.nwk" \
      --tree-edges "$data/tree_edges.csv" \
      --switch-edges "$res/switch/switch_edges.csv" "${true_sig[@]}" \
      --outdir "$res/switch"
    python plot_switch_recovery.py --nodes "$res/switch/switch_nodes.csv" \
      --calibration "$res/switch/switch_calibration.csv" --outdir "$res/switch"
  fi
}

stage_run() {
  for manifest in "$ROOT"/*/manifest_configs.txt; do
    while read -r cfg; do
      [ -n "$cfg" ] && run_arm "$cfg"
    done < "$manifest"
  done
}

stage_report() {
  python validate_switch_report.py --root "$ROOT" --thin "$THIN"
}

case "$STAGE" in
  configs) stage_configs ;;
  run) stage_run ;;
  report) stage_report ;;
  all) stage_configs; stage_run; stage_report ;;
  *) echo "unknown stage '$STAGE' (configs | run | report | all)" >&2; exit 2 ;;
esac
