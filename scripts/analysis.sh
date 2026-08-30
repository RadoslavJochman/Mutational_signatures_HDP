DATA=../data/config_denovo_easy_more_chains_draws_trees_nmf_init_corr_03   # from your config
RES=../results/config_denovo_easy_more_chains_draws_trees_nmf_init_corr_03                                  # wherever trace_*.nc live

python recovery_vs_truth.py \
  --trace            $RES/trace_raw.nc \
  --true-activities  $DATA/true_activities.csv \
  --newick           $DATA/newick_string.nwk \
  --true-signatures  $DATA/fixed_signatures.csv \
  --outdir           $RES/recovery

python diagnose_activity_identifiability.py \
  --trace      $RES/trace_aligned.nc \
  --true-sigs  $DATA/fixed_signatures.csv \
  --true-acts  $DATA/true_activities.csv \
  --counts     $DATA/mutation_count_matrix.csv \
  --newick     $DATA/newick_string.nwk \
  --per-camp-sigs \
  --outdir     $RES/activity_ident

python diagnose_camp_sig.py \
  --trace      $RES/trace_aligned.nc \
  --true-sigs  $DATA/fixed_signatures.csv \
  --outdir     $RES/camp_sig
