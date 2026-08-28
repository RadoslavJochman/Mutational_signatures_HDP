# CLAUDE.md

Conventions and layout for `Mutational_signatures_HDP` (Tree-HDP). Read before editing.

## Project

Tree-HDP infers mutational signature activities across tumour phylogenies. Activities follow a
random walk down each tree (a finite-dimensional approximation of a tree-structured HDP), node
spectra are `e_j S`, and counts are multinomial. Two inference models: `FixedSigHDP` (`S`
given) and `DeNovoHDP` (`S` inferred jointly with the activities). Developed and validated on
simulated forests; targets single-cell tumour data.

## Environment

Dependencies are pinned in `requirements.txt`. Create a virtual environment at the repo root
(`run_replicates.sbatch` sources `.venv/bin/activate`) and install:

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Core stack: PyMC 5.25.1 and PyTensor 2.31.7 for the models, ArviZ 0.23.4 with xarray for
traces, NumPy 2.2.6, pandas, SciPy and scikit-learn 1.7.2 (the NMF baseline), matplotlib and
seaborn for plots, NetworkX 3.4.2 and phylox 1.1.2 for trees, and JAX 0.4.38 with NumPyro for
the JAX sampling path (see `enable_local_gpu.sh`).

## Git and reproducibility

The project is a git repository, and experiments are pinned to tags. Every config records a
`git_tag` (in use so far: `fixed-sig-v1`, `fixed-sig-v2`, `denovo-v1`, `denovo-v1-ilr`,
`denovo-v2`), and reproducing a run means checking out that tag before generating and fitting.
Treat a tag as immutable: when the model changes in a way that would alter results, cut a new
tag and point the new configs at it rather than editing the model under an existing tag. The
planned unification is such a change and needs its own tag.

Working practice:

- Do the unification and evaluation redo on a branch, not on `main`, in small focused commits
  so each change is reviewable and reversible.
- Check `git status` and `git diff` before committing, and keep the working tree clean.
- Commit messages follow the writing rule: concise, plain, imperative, British spelling, no
  em-dashes. Do not add `Co-Authored-By` or `Generated with Claude Code` trailers.
- Do not commit run artefacts. `.gitignore` already covers `data/`, per-run `results/`, traces
  (`.nc`, `.zarr`), `.venv/`, `logs/`, `__pycache__/`, and `.DS_Store`. The tracked exceptions
  are the aggregated tables under `results/agg/` and `results/agg40/` and
  `results/scaling_results.csv`; those aggregated outputs are meant to be committed.

## Direction of travel (work in progress)

This repo predates two planned changes we are now making. When helping with a refactor, move
toward these targets rather than entrenching the current design, and expect the current configs
and parts of the models and evaluation to be replaced.

1. **One model, not two.** Merge `FixedSigHDP` and `DeNovoHDP` into a single model under the
   ILR activity walk, with signatures either clamped to a catalogue (`S` known) or given a
   Dirichlet prior and inferred jointly (`S` latent). The fixed case becomes the same model
   with `S` fixed, and the label-switching alignment applies only when `S` is latent. This
   also unifies the two runners. Rerun the fixed-signature results under the ILR walk so the
   reported numbers match the presented model.

2. **Evaluation redo.**
   - Metric: cosine similarity is primary (already in `recovery_vs_truth.py` and
     `nmf_baseline.py`), reported with the recovery rate at cosine >= 0.90; `L_1` for
     activities. Hellinger drops from the headline (`tv`/`hellinger` stay available as
     secondary).
   - Baselines: add PhySigs (tree-aware, known signatures), SigProfilerExtractor and MuSiCal
     (de novo, tree-free), and a tree-free ablation of Tree-HDP itself (the model with the walk
     removed, nodes independent). NMF stays as a simple reference. The tree-free ablation is
     the key comparison and should be emphasised over NMF.

3. **Calibrated simulator, replacing the correlation knob.** Draw signatures from the
   triple-negative breast repertoire (COSMIC SBS1, SBS5, SBS2, SBS13, SBS3, with SBS8/SBS18 in
   reserve), give activities biological structure (a clock baseline, an APOBEC edge shift per
   tree, per-patient HRD presence), calibrate tree shapes and per-node burdens to single-cell
   data, and make the low-count burden regime and the forest size the primary sweep axes. Demote
   the signature-overlap sweep to a secondary stress test that interpolates two flat signatures,
   and draw counts with over-dispersion (Dirichlet-multinomial) to avoid simulating from the
   inference model. The current `corr_sweep`, `size_sweep`, and `bases_*` configs will be
   replaced by configs for these axes. The full simulator design lives in `simulator_spec.md`
   (ask for it if it is not yet in the repo).

The sections below describe the code as it is now, so navigation stays accurate during the
transition.

## Repository layout

- `src/` is the library. Import from it; do not reimplement its pieces in scripts.
  - `src/models/hdp_inference.py`: `_BaseTreeHDP` (abstract), `FixedSigHDP`, `DeNovoHDP`
    (being merged into one model, see Direction of travel).
  - `src/models/hdp_simulator.py`: `TreeSignatureGenerator`, `synthesize_signatures`,
    `generate_random_forest` (the data generator).
  - `src/models/dirichlet_process.py`: `Measure`, `DirichletPrior`, `DirichletProcess`.
  - `src/analysis/analysis.py`: the shared helper library (metrics, alignment, tree/forest
    utilities, walk transforms). See below.
  - `src/plotting/figure_style.py`: `PALETTE`, `apply_style`, `save`. All figures go through
    these. `src/plotting/plots.py`: reusable plot builders.
  - `src/config.py`: `load_config`, `make_output_dir`, `get_prior` (YAML to PyMC priors).
- `scripts/` is a flat set of runnable entry points (pipeline plus one-off diagnostics).
- `configs/` holds base configs at the top level, plus one subdirectory per sweep
  (`size_sweep`, `corr_sweep`, `corr_sweep_40`, and the `bases_*` base configs the replicate
  configs are expanded from). Each sweep dir also carries `manifest_configs.txt` and
  `manifest_agg*.csv`. These correlation-knob sweeps are being replaced by the calibrated
  burden and forest-size sweeps (see Direction of travel).
- `data/`, `results/`, and `COSMIC_sig/` live at the repo root (not in git). Scripts run from
  `scripts/`, so the configs' `../data`, `../results`, `../COSMIC_sig` resolve to those.

## Running the pipeline

Run Python from `scripts/` (as `run_replicates.sbatch` does with `cd scripts`); the configs'
`../` paths depend on it. A single run is generate then infer then score:

```
cd scripts
python generate_data.py          --config ../configs/<cfg>.yaml   # -> ../data/<experiment_name>/
python run_inference.py          --config ../configs/<cfg>.yaml   # FIXED signatures
python run_unknown_inference.py  --config ../configs/<cfg>.yaml   # DE NOVO
python recovery_vs_truth.py --trace ../results/<name>/trace_raw.nc \
    --true-activities ../data/<name>/true_activities.csv \
    --newick ../data/<name>/newick_string.nwk \
    --true-signatures ../data/<name>/fixed_signatures.csv \
    --metrics tv hellinger cosine --outdir ../results/<name>/recovery
python nmf_baseline.py --counts ../data/<name>/mutation_count_matrix.csv \
    --true-activities ../data/<name>/true_activities.csv \
    --true-signatures ../data/<name>/fixed_signatures.csv \
    --metrics tv hellinger cosine --outdir ../results/<name>/nmf   # de novo comparison only
```

Rules that matter:

- De novo MUST use `run_unknown_inference.py`, never `run_inference.py`. The de novo runner
  adds the post-hoc per-draw alignment to chain 0 (`trace_aligned.nc`); without it the
  summary's `r_hat` and `ess` are meaningless, since chains merely order the signatures
  differently. `--true-signatures` is passed to scoring for de novo only (fixed-sig has no
  label switching, so component k already is true signature k).
- `inference.model` selects the de novo activity prior: `denovo` (default, and currently the
  only supported value, the random walk).
- Generation writes to `../data/<experiment_name>/`: `mutation_count_matrix.csv`,
  `newick_string.nwk`, `tree_edges.csv`, `fixed_signatures.csv`, `true_activities.csv`,
  `ground_truth_params.json`. Inference writes traces to `../results/<experiment_name>/`.
- Local GPU: `source enable_local_gpu.sh` (sets PyTensor to JAX mode).

## Config structure

Two top-level blocks. `simulation` drives `generate_data.py`; `inference` drives the runners.

```
experiment_name, git_tag
simulation:
  seed, results_dir (../data/), make_plots
  alpha            # activity-walk concentration e_j ~ Dir(alpha * e_parent)
  alpha_0          # baseline concentration e_0 ~ Dir((alpha_0/K) * 1_K)
  lam, nb_dispersion          # per-node burden; lam=970 matches the reference mean, nb=2.0
  activity_sparsity, signature_dropout
  signatures: {source: synthesize|load, correlation, path, num_signatures}
  forest: {num_trees, min_leaves, max_leaves, min_branch_length, max_branch_length}
inference:
  model                        # denovo (de novo runner only)
  num_signatures               # K
  data: {count_matrix, newick_string, tree_edges, fixed_signatures, true_activities,
         ground_truth_params}
  priors:
    sigma_prior, sigma_prior_parm   # walk-scale prior on sigma
    sigma_0                          # root-baseline std
    sigma_mu                         # de novo ILR only: forest-pooled usage-level std
    beta                             # de novo only: S_k ~ Dir(beta * 1_96)
  draws, tune, chains, cores, target_accept, max_treedepth
  results_dir (../results/)
```

Fixed-sig configs use `sigma_0` and no `sigma_mu`/`beta`. De novo ILR configs use `sigma_0`
plus `sigma_mu` and `beta`. `signatures.correlation` is ignored when `source: load`.

## Sweeps and replicates

A replicate is one dataset (one `simulation.seed`) fit and scored on its own; averaging
replicates of a setting gives the error bars.

- `make_replicate_configs.py` expands one `bases_*` base config into R replicate configs that
  differ only in `simulation.seed` and identity. `signatures.path` is left untouched so every
  replicate loads the same frozen signature matrix; only the forest and activities vary.
- `run_replicates.sbatch <manifest> [fixed|denovo] [scratch]` is one SLURM array task per
  replicate (generate, infer, score), moving traces to scratch afterwards.
- `aggregate_replicates.py` groups the per-replicate `recovery_*.csv` by setting into
  mean/sd/se/count. `sweep_aggregate.py` assembles a sweep into one master table.

## Shared library (`src/analysis/analysis.py`)

Reuse these rather than reimplementing. Metrics: `cosine`, `hellinger`, `total_variation`,
`jensen_shannon`, `bray_curtis`, `signature_distances`, `usage_distances`, `node_distances`,
`exposure_errors`, `across_chain`. Alignment: `align` and `chain_perms_to_true` (Hungarian on
signature cosine). Walk transforms: `softmax_last_zero`, `inv_softmax_last_zero`,
`forward_walk`, `inverse_walk`. Tree/forest: `build_forest`, `build_model`, `node_order`,
`node_depths`, `nodes_by_depth`. Mode analysis: `detect_camps`, `split_camps`,
`per_chain_activity`.

## Metrics and convergence

- Signature and activity recovery is scored by cosine similarity (primary); `tv` and
  `hellinger` are also available via `--metrics`. Signatures are matched to truth by Hungarian
  cosine before scoring, and the same permutation is applied to the activity components.
- For convergence (`scaling_metrics.py`), `max_rhat` and `min_ess` are taken over the
  identifiable activity variables (`e_level_*`) and `sigma`. The raw walk increments
  (`eta_level`, `z_level`) are excluded: they are uninformative by construction in the
  non-centred walk. Never report r_hat or ess over them.

## Diagnostic scripts (not the forward pipeline)

`camp_direction.py`, `diagnose_camp_path.py`, `diagnose_camp_sig.py`, `diagnose_modes.py`,
`diagnose_sampler_geometry.py`, `diagnose_pair_tradeoff.py`,
`diagnose_convergence_exposure.py`, `diagnose_activity_identifiability.py`, and their
`plot_camp_path.py` / `plot_mode_analysis.py` / `plot_activity_ident.py` /
`plot_convergence_exposure.py` are the chain-splitting and mixing-geometry investigation. That
material is cut from the paper. Keep them for the record, but they are not part of the
generate-infer-score pipeline; do not extend them when working on the paper's results.

## Analysis discipline

- Verify against the data before asserting. Check conclusions against the actual arrays or
  CSVs, not summaries or model output. Verdicts here have been overturned by the data before,
  and the data is the arbiter.
- Label-frame discipline. True-label frames (from `data_features.py`, `recovery_vs_truth.py`)
  and inferred-label frames (a de novo trace, chain 0's arbitrary labelling) must be aligned
  via `chain_perms_to_true` before any cross-referencing.
- Convergence is necessary but not sufficient. Per-chain recovery against ground truth is the
  arbiter, not r_hat alone; chains can agree on the worse of two near-equivalent modes.
- One variable at a time. Freeze everything but the swept axis (e.g. hold the signature matrix
  fixed across a tree-count sweep).
- Document corrections explicitly rather than smoothing them over.

## Script conventions

- Pipeline pattern is diagnose then CSV or plot then figures; each script does one stage.
- Scripts emit clean CSV outputs only. Interpretation lives in docstrings, not in `print`.
- Keep the CLI consistent with the existing scripts (`argparse`, `--config` for config-driven
  scripts, explicit `--trace`/`--outdir`/`--metrics` for scoring scripts).
- All figures go through `figure_style.apply_style()` and `figure_style.save()` (which writes
  both PDF and PNG). Use `PALETTE` roles (stiff for data-visible, soft for data-invisible,
  accent for a third series, grey for baselines), never ad hoc colours.

## Testing

Additions ship with tests. Every new function, model, metric, baseline, or simulator feature
comes with its tests in the same change, and a change is not done until they pass. Use pytest;
tests live in `tests/`, mirroring `src/` and the scripts they cover. Add `pytest` to the dev
dependencies.

Three levels:

- Unit (fast, deterministic, run by default). The pure functions in `src/analysis/analysis.py`
  and the simulator. Check bounds, symmetry, and known values for the metrics (`cosine`,
  `hellinger`, `total_variation`, `bray_curtis`); round-trips for the walk transforms
  (`softmax_last_zero`/`inv_softmax_last_zero`, `forward_walk`/`inverse_walk`); a known
  permutation for `chain_perms_to_true`; and shapes plus seed-determinism for
  `TreeSignatureGenerator`.
- Integration, the smoke config. A tiny end-to-end run (a couple of trees, roughly 50 to 100
  draws, one chain, fixed seed) through generate then infer then score, asserting the pipeline
  runs, writes the expected files, and produces the expected trace variables. Fast enough to run
  on every change.
- Statistical (slow, marked `@pytest.mark.slow`). On a seeded easy case, assert recovery is
  within a generous tolerance. Fix `random_seed` in `pm.sample` and keep these out of the
  default fast run.

Seed everything (the simulator and `pm.sample`) and assert on shapes, invariants, and rough
tolerances, never on exact posterior numbers. The smoke config doubles as the manual fast gate:
it is the smallest config that exercises generate then infer then score, and is what you run
after a refactor to confirm nothing broke before trusting recovery numbers.

## Formatting and linting

Ruff is the project standard for both formatting and linting. Run `ruff format .` and
`ruff check --fix .` before committing, and pin `ruff` in the dev dependencies. Configure it in
`pyproject.toml`; the code already fits an 88-column width (the ruff and black default), so keep
that. Start with a light rule set (`E`, `F`, `I`) and do not let the linter fight deliberate
numerical code: silence a specific rule inline with a targeted `# noqa` rather than restructuring
correct code. Land the first `ruff format` pass as its own commit, separate from any logic
change, so the mechanical reformat does not bury real diffs.

## Writing inside the repo

Docstrings, comments, README, commit messages. The LaTeX paper lives elsewhere and its style
rules do not apply here.

- Concise, plain, human-sounding; avoid em-dashes and AI-style phrasing.
- British spelling (tumour, reparameterising, factorisation).
- Interpretation goes in docstrings; stdout stays CSV.

## Do not

- Run de novo with `run_inference.py`, or trust r_hat/ess from an unaligned de novo trace.
- Print interpretation or verdicts to stdout.
- Duplicate helpers that exist in `src/analysis/analysis.py` or `figure_style.py`.
- Compare true-label and inferred-label frames without aligning first.
- Include `eta_level`/`z_level` in convergence statistics.
- Commit `data/`, per-run `results/`, or traces. The aggregated tables under `results/agg*/`
  and `results/scaling_results.csv` are the tracked exceptions.
- Add a feature without its tests, or mix a `ruff format` pass into a logic commit.
