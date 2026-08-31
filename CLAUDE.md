# CLAUDE.md

Conventions and layout for `Mutational_signatures_HDP` (Tree-HDP). Read before editing.

## Project

Tree-HDP infers mutational signature activities across tumour phylogenies. Activities follow a
random walk down each tree (a finite-dimensional approximation of a tree-structured HDP), node
spectra are `e_j S`, and counts are multinomial. One inference model, `TreeHDP`, under a shared
ILR activity walk: signatures are either clamped to a catalogue (`S` known) or given a
Dirichlet prior and inferred jointly (`S` latent). Developed and validated on simulated
forests; targets single-cell tumour data.

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
`denovo-v2`, `treehdp-v1`, `switch-drift-v1`, `switch-drift-v2`), and reproducing a run means
checking out that tag before generating and fitting.
Treat a tag as immutable: when the model changes in a way that would alter results, cut a new
tag and point the new configs at it rather than editing the model under an existing tag. The
planned unification is such a change and needs its own tag.

Working practice:

- Do the unification and evaluation redo on a branch, not on `main`, in small focused commits
  so each change is reviewable and reversible.
- Check `git status` and `git diff` before committing, and keep the working tree clean.
- Commit messages follow the writing rule: concise, plain, imperative, British spelling, no
  em-dashes. Do not add `Co-Authored-By` or `Generated with Claude Code` trailers.
- Do not commit run artefacts. `.gitignore` covers `data/`, `results/`, and `plots/` at any
  depth -- an experiment's own subdirectories, wherever they are nested (e.g.
  `experiments/<name>/data/`, `experiments/<sweep>/rep03/results/`) -- plus traces (`.nc`,
  `.zarr`), `.venv/`, `logs/`, `__pycache__/`, and `.DS_Store`. The tracked exceptions are
  every experiment's `config.yaml`, a sweep's `manifest_*` files and `agg/` directory, and
  `tables/` (`scaling_results.csv`, `agg/`, `agg40/`, the cross-cutting committed tables
  predating the per-sweep `agg/` convention); those are meant to be committed.

## Direction of travel (work in progress)

This repo predates two planned changes we are now making. When helping with a refactor, move
toward these targets rather than entrenching the current design, and expect the current configs
and parts of the models and evaluation to be replaced.

1. **One model, not two -- merge done, rerun still open.** `FixedSigHDP` and `DeNovoHDP` are
   merged into `TreeHDP`, a single model under the ILR activity walk, with signatures either
   clamped to a catalogue (`S` known) or given a Dirichlet prior and inferred jointly (`S`
   latent); the fixed case is the same model with `S` fixed, and the label-switching alignment
   applies only when `S` is latent. The two runners are merged into one `run_inference.py`,
   dispatching on the required `inference.model: fixed | denovo`. The tag this change needed
   is cut (`treehdp-v1`). Still open: migrate the fixed configs to the shared-walk
   convention, and rerun the fixed-signature results under the ILR walk so the reported
   numbers match the presented model.

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
   inference model. The old `corr_sweep`, `size_sweep`, and `bases_*` configs are already
   deleted (recoverable from the `treehdp-v1`/`switch-drift-v1` tags); this item's new sweep
   configs, once designed, replace them under `experiments/`. The full simulator design lives
   in `simulator_spec.md` (ask for it if it is not yet in the repo).

   `TreeSwitchDriftGenerator` threaded one `rng` through every draw, and `rng.multinomial`'s
   cost depends on the realized burden, so two configs differing only in an observation-side
   parameter (`burden.mean`, `counts.kappa`, `counts.model`) reshuffled every downstream node's
   switching and activity draws, not just its counts -- making the controlled comparisons this
   benchmark needs impossible. Fixed (`switch-drift-v2`): four independent generators spawned
   from one root seed via `SeedSequence(seed).spawn(4)`, one per stage (topology, switching,
   levels, observation), so a stage's stream position no longer depends on another stage's draw
   count. This changed what a given seed produces, hence the new tag; nothing tracked depended
   on `switch-drift-v1`'s exact output.

The sections below describe the code as it is now, so navigation stays accurate during the
transition.

## Repository layout

- `src/` is the library. Import from it; do not reimplement its pieces in scripts.
  - `src/models/hdp_inference.py`: `_BaseTreeHDP` (abstract), `TreeHDP` (`S` known or latent,
    see Direction of travel).
  - `src/models/hdp_simulator.py`: `TreeSwitchDriftGenerator` (the switch-plus-drift data
    generator; see `simulator_spec.md`).
  - `src/models/dirichlet_process.py`: `Measure`, `DirichletPrior`, `DirichletProcess`.
  - `src/analysis/analysis.py`: the shared helper library (metrics, alignment, tree/forest
    utilities, walk transforms). See below.
  - `src/plotting/figure_style.py`: `PALETTE`, `apply_style`, `save`. All figures go through
    these. `src/plotting/plots.py`: reusable plot builders.
  - `src/config.py`: `load_config`, `make_output_dir`, `get_prior` (YAML to PyMC priors).
- `scripts/` is a flat set of runnable entry points (pipeline plus one-off diagnostics).
- `experiments/` holds one directory per fit: `<name>/config.yaml`, `<name>/data/`,
  `<name>/results/`, `<name>/plots/`. A sweep is `<sweep_name>/` holding `rep00/`, `rep01/`,
  ... (each itself a full experiment directory) plus `<sweep_name>/agg/` and the sweep's
  `manifest_*` files. Only `config.yaml`, `manifest_*`, and `agg/` are tracked; `data/`,
  `results/`, and `plots/` are gitignored at any depth (see Do not).
- `configs/` holds standalone historical configs pinned to old tags (the pre-restructure
  `alpha`/`lam` generator schema); nothing new is written here under the current layout.
- `COSMIC_sig/` (the signature catalogue) and `tables/` (cross-cutting committed tables:
  `scaling_results.csv`, `agg/`, `agg40/`) live at the repo root. Scripts run from `scripts/`,
  so the configs' `../experiments` and `../COSMIC_sig` resolve to those.

## Running the pipeline

Run Python from `scripts/` (as `run_replicates.sbatch` does with `cd scripts`); the configs'
`../` paths depend on it. A single run is generate then infer then score:

```
cd scripts
python generate_data.py --config ../experiments/<name>/config.yaml   # -> ../experiments/<name>/data/
python run_inference.py --config ../experiments/<name>/config.yaml   # inference.model: fixed | denovo
python recovery_vs_truth.py --trace ../experiments/<name>/results/trace_raw.nc \
    --true-activities ../experiments/<name>/data/true_activities.csv \
    --newick ../experiments/<name>/data/newick_string.nwk \
    --true-signatures ../experiments/<name>/data/fixed_signatures.csv \
    --metrics tv hellinger cosine --outdir ../experiments/<name>/results/recovery
python nmf_baseline.py --counts ../experiments/<name>/data/mutation_count_matrix.csv \
    --true-activities ../experiments/<name>/data/true_activities.csv \
    --true-signatures ../experiments/<name>/data/fixed_signatures.csv \
    --metrics tv hellinger cosine --outdir ../experiments/<name>/results/nmf   # de novo only
```

Rules that matter:

- `run_inference.py` reads `inference.model` (`fixed` or `denovo`) and dispatches on it;
  configs written before the runner collapse that don't set it fall back to whether
  `inference.num_signatures` is present (denovo) or absent (fixed) -- the same signal the old
  two-script split encoded implicitly. De novo (`S` latent) always gets the post-hoc per-draw
  alignment to chain 0 (`trace_aligned.nc`), gated on the built model's `S_known`; without it
  the summary's `r_hat` and `ess` are meaningless, since chains merely order the signatures
  differently. `--true-signatures` is passed to scoring for de novo only (fixed-sig has no
  label switching, so component k already is true signature k).
- Generation writes to `<experiment_root>/<experiment_name>/data/`: `mutation_count_matrix.csv`,
  `newick_string.nwk`, `tree_edges.csv`, `fixed_signatures.csv`, `true_activities.csv`,
  `true_active_sets.csv`, `ground_truth_params.json` (plus `plots/` when `make_plots` is set).
  Inference writes to `<experiment_root>/<experiment_name>/results/`. `experiment_root` and
  `experiment_name` are shared top-level config keys, so both scripts always agree on which
  experiment directory they write into.
- Local GPU: `source enable_local_gpu.sh` (sets PyTensor to JAX mode).

## Config structure

Shared identity keys, plus two top-level blocks. `simulation` drives `generate_data.py`
(`TreeSwitchDriftGenerator`, see `simulator_spec.md`); `inference` drives the runners.

```
experiment_name, experiment_root (../experiments/), git_tag
simulation:
  seed, make_plots
  repertoire: {source, path, signatures}      # catalogue signatures, never synthesised
  switching: {enabled, branch_length_scaling,
              units: [{signatures, root_prob, lambda_on, lambda_off}]}
  levels: {enabled, walk, concentration, branch_length_scaling, concentration_floor,
           root_concentration, activation_pseudocount}
  forest: {n_trees, nodes_per_tree, depth, max_attempts,
           branch_lengths: {distribution, params|path}}
  burden: {mean, distribution, dispersion}
  counts: {model, kappa}
inference:
  model                        # fixed | denovo, required
  num_signatures               # K, denovo only
  data: {count_matrix, newick_string, tree_edges, fixed_signatures, true_activities,
         ground_truth_params}
  priors:
    sigma_prior, sigma_prior_parm   # walk-scale prior on sigma
    sigma_0                          # root-baseline std
    sigma_mu                         # de novo ILR only: forest-pooled usage-level std
    beta                             # de novo only: S_k ~ Dir(beta * 1_96)
  draws, tune, chains, cores, target_accept, max_treedepth
```

`experiment_root` and `experiment_name` are read directly by `generate_data.py` and
`run_inference.py` (there is no separate `simulation.results_dir`/`inference.results_dir`
any more, so the two scripts can never disagree on where an experiment's files live).
`experiment_name` may itself contain `/` for a sweep replicate (e.g. `corr_sweep/rep03`).
Fixed-sig configs use `sigma_0` and no `sigma_mu`/`beta`. De novo ILR configs use `sigma_0`
plus `sigma_mu` and `beta`.

## Sweeps and replicates

A replicate is one dataset (one `simulation.seed`) fit and scored on its own; averaging
replicates of a setting gives the error bars. A sweep is `experiments/<sweep_name>/`,
holding `rep00/`, `rep01/`, ... (each a full experiment directory) plus `agg/` once
aggregated.

- `make_replicate_configs.py` expands one base config into R replicate configs that differ
  only in `simulation.seed` and identity, writing each straight into
  `experiments/<sweep_name>/rep<NN>/config.yaml` (derived from the base config's own
  `experiment_root` and `experiment_name`; there is no separate `--outdir`, so a replicate's
  config can never disagree with where its own data and results land). `repertoire.path` is
  left untouched so every replicate loads the same frozen signature matrix; only the forest
  and activities vary. Run from `scripts/`, like every other pipeline script.
- `run_replicates.sbatch <manifest> [fixed|denovo] [scratch]` is one SLURM array task per
  replicate (generate, infer, score), moving traces to scratch afterwards.
- `aggregate_replicates.py` groups the per-replicate `recovery_*.csv` by setting into
  mean/sd/se/count, writing to `experiments/<sweep_name>/agg/`.

The correlation-overlap and forest-size sweeps that used to live under `configs/bases_*` and
`configs/corr_sweep*`/`configs/size_sweep` are deleted (recoverable from the `treehdp-v1`/
`switch-drift-v1` tags); no base config for a live sweep currently exists. `sweep_aggregate.py`,
which those sweeps used (a two-root `results/`/`data/` design that does not fit
`experiment_root`), is deleted with them; `aggregate_replicates.py`'s manifest flow carries
the sweep-aggregation job forward. The calibrated simulator's own sweep configs (Direction of
travel) will be the first to exercise this machinery for real.

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
  permutation for `chain_perms_to_true`; and shapes, seed-determinism, and per-stage RNG-stream
  isolation (ground truth invariant to `burden`/`kappa`/`counts.model`, topology invariant to
  switching rate) for `TreeSwitchDriftGenerator`.
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
- Commit an experiment's `data/`, `results/`, or `plots/` subdirectory, or any trace. Every
  experiment's `config.yaml`, a sweep's `manifest_*` and `agg/`, and `tables/` (the
  cross-cutting `scaling_results.csv`, `agg/`, `agg40/`) are the tracked exceptions.
- Add a feature without its tests, or mix a `ruff format` pass into a logic commit.
