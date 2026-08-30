# Tree-HDP simulator specification (v3)

The simulator generates a forest of tumour phylogenies whose per-node signature activities carry
two kinds of structure: which signatures are on or off at each node (the switch), and how the
levels of the on-signatures drift down the tree (the drift). Both are driven by branch length.

The design is general to cancer, not tied to one cohort. `hdp_simulator.py` knows how to build a
switch-plus-drift forest from a repertoire, a set of switching rates, and tree shapes; it does
not know what any specific cancer type or dataset is. Every biological specific is a value the
config supplies. The triple-negative breast repertoire validated on the public SECEDO cohort is
one instantiation (Section 13), not an assumption in the code.

The two invariants that keep the benchmark honest: signatures are drawn from a known repertoire
rather than synthesised, and the ground truth is not generated from the inference model.

## 1. What it produces

A forest with branch lengths. Each node `j` carries an active set `a_j` (binary over the
repertoire), activity levels `e_j` on the simplex with off-signatures at exactly zero, and counts
`x_j` over the 96 channels. Ground truth stored: the loaded signatures `S`, every `a_j`, every
`e_j`, every burden `M_j`, and the tree with branch lengths.

The on-disk contract is plain CSV, extending the current one so the scoring code reads a fixed set
of files: `mutation_count_matrix.csv`, `newick_string.nwk`, `tree_edges.csv` (with branch lengths),
`fixed_signatures.csv`, `true_activities.csv` (nodes by signatures, simplex, exact zeros for offs),
`ground_truth_params.json`, and the new `true_active_sets.csv` (nodes by signatures, binary int).
`fixed_signatures.csv` keeps its current name rather than becoming `signatures.csv`; renaming it
would ripple into the `inference.data.fixed_signatures` config key, `recovery_vs_truth.py`, and the
sbatch script, a cascade this rewrite doesn't need to couple itself to. `true_activities.csv`,
`true_active_sets.csv` and `fixed_signatures.csv` all use the real signature names (`SBS1`, `SBS2`,
...) as column headers, in the same order across all three, so the three files join cleanly by
column position or name; this replaces the old anonymous `Signature_0..K-1` convention, which only
ever existed to name synthesized signatures and has no reason to survive now that signatures are
always named. The active set is stored explicitly rather than derived from the activities by
nonzero, both to avoid float-zero ambiguity and because it is the primary target of the on/off
metric; `true_active_sets.csv` shares its node index with `true_activities.csv` (all nodes,
including zero-mutation ones), so the two join by node label directly. `ground_truth_params.json`
is nested to mirror this config schema (`repertoire`, `switching`, `levels`, `forest`, `burden`,
`counts`, each a sub-object), plus the resolved per-unit rates, the realised branch-length
normaliser (Section 4), and the generator seed.

In the truth an off signature is exactly zero, which the simplex-direct generator produces
naturally. The current softmax-walk model has no mass at zero and holds every signature at a small
floor, so on/off recovery is testing a real gap, not an accident.

## 2. The general mechanism

- **Repertoire.** The signature pool is loaded from a named catalogue; which catalogue and which
  signatures are config. Nothing is synthesised.
- **Switch (on/off).** The repertoire is partitioned into switch units. A unit is one signature,
  or several that switch together (co-regulation). Each unit's on/off state evolves along the
  tree as a two-state gain/loss process, with flip probability rising with branch length. Per-unit
  gain and loss rates and a root probability come from config.
- **Drift (levels).** The generator works on the simplex directly, so an off signature is exactly
  zero. Among the on-signatures, activity drifts down the tree as a mean-preserving Dirichlet walk
  whose spread scales with branch length. Scale and the scaling flag are config.
- **Counts.** Drawn Dirichlet-multinomial around `e_j S`, over-dispersion and depth from config.

The generator class is `TreeSwitchDriftGenerator`, named for the mechanism it implements, not
`TreeSignatureGenerator`; the old name described the previous cohort-baseline walk and no longer
fits. One seeded `np.random.Generator` threads through every draw in the generator -- topology,
on/off, the Dirichlet steps, burden, counts -- with no per-function or global RNG state, so one
seed reproduces one forest exactly, which both the unit tests and the across-seed averaging depend
on. The clock guard (below) makes an empty active set impossible under a valid config, so the
generator asserts non-emptiness at every node rather than treating it as an error path: a config
that somehow breaks the guard fails loudly at the node it breaks, instead of silently producing an
invalid simplex downstream.

## 3. Config schema (the interface)

This is the contract `hdp_simulator.py` reads. Nothing biological is hardcoded.

```yaml
simulation:
  seed: 0
  n_seeds: 20                      # sweep-level replicate count; not read by the generator itself
                                    # (Section 14) -- make_replicate_configs.py and the SLURM array
                                    # own replicate expansion, one seed per invocation, unchanged

  repertoire:
    source: cosmic                 # which reference catalogue; a generic CSV path works the same
                                    # way (mirrors the old signatures.source: load) -- cosmic is not
                                    # hardcoded as the only accepted value, but there is no registry
    path: ../COSMIC_sig/cosmic_signatures.csv
    signatures: [SBS1, SBS5, SBS2, SBS13, SBS3]   # the pool loaded from the catalogue (K)

  switching:
    enabled: true                  # false = no switching: every pool signature on at every node;
                                    # units below become optional and are ignored if present
    branch_length_scaling: true    # true: lambdas are rates (may exceed 1), flip probability rises
                                    # with branch length; false: lambdas are per-edge probabilities,
                                    # applied flat regardless of length, and must lie in [0, 1] --
                                    # validation branches on this flag (see Section 4)
    units:                         # every pool signature belongs to exactly one unit; required
                                    # only when switching.enabled is true
      - signatures: [SBS1]
        root_prob: 1.0             # on at the root
        lambda_on: 0.0
        lambda_off: 0.0            # never leaves -> always-on (clock)
      - signatures: [SBS5]
        root_prob: 1.0
        lambda_on: 0.0
        lambda_off: 0.0
      - signatures: [SBS2, SBS13]  # co-switching unit: gain/lose as one (APOBEC pair)
        root_prob: 0.0
        lambda_on: 0.4
        lambda_off: 0.1
      - signatures: [SBS3]
        root_prob: 0.5             # per-patient presence, set at the root
        lambda_on: 0.05
        lambda_off: 0.05

  levels:
    enabled: true                  # false = no drift: no Dirichlet noise between events; a
                                    # composition is not frozen through events, since carry-forward
                                    # and zero-renormalise still run on every switch (Section 4)
    walk: dirichlet                # mean-preserving Dirichlet walk on the simplex
    concentration: 50.0            # base concentration c, at the forest's median branch length
    branch_length_scaling: true    # false = constant concentration c on every edge, no median
                                    # normalisation and no floor (the old uniform behaviour)
    concentration_floor: 1.0       # floor on the effective per-edge concentration, so the
                                    # long-branch tail can't collapse activities to near one-hot
    root_concentration: 1.0        # Dirichlet concentration of the root activity draw
    activation_pseudocount: 0.02   # mean mass a signature enters with when it switches on

  forest:
    n_trees: 10
    nodes_per_tree: [3, 9]         # [min, max]; rejection-sampled against phylox output, Section 5
    depth: [2, 4]
    max_attempts: 200               # retry cap for the rejection sampler; raise this if a tight
                                     # nodes_per_tree/depth range exhausts it (Section 5)
    branch_lengths:
      distribution: lognormal      # or empirical; stored in mutation scale (Section 6)
      params: {mu: 0.0, sigma: 0.6}
      # path: ../data/empirical_branch_lengths.csv   # if distribution: empirical -- a single
      #   column of lengths, header optional, sampled i.i.d. with replacement

  burden:
    mean: 120                      # per-node mutation count, swept; negative-binomial mean
    distribution: negbin
    dispersion: 2.0

  counts:
    model: dirichlet_multinomial   # or multinomial (well-specified control)
    kappa: 50.0
```

Rules the loader enforces, only when `switching.enabled` is true (the `units` block is optional,
and ignored if present, when it is false): every signature in `repertoire.signatures` appears in
exactly one switching unit; `root_prob` in `[0, 1]`; and at least one always-on unit (`root_prob:
1`, `lambda_off: 0`) must exist. Rate bounds depend on `branch_length_scaling`: when it is true,
`lambda_on`/`lambda_off` are rates and only need to be non-negative (they may exceed 1); when it is
false they are used directly as per-edge probabilities and must lie in `[0, 1]`. The always-on unit
is the clock guard: it keeps every node's active set non-empty (so the simplex is always
well-defined), which then frees every other unit to carry `lambda_off > 0` and genuinely switch
off, so deactivation, half of what the on/off benchmark studies, is available rather than banned.
Irreversibility (`lambda_off: 0`) stays a per-unit choice for signatures where it is biologically
real, not a blanket assumption. A co-switching unit shares one state across its members, so
`[SBS2, SBS13]` are on together or off together, which is how the APOBEC pair is expressed rather
than as correlated independent rates. The clock guard makes an empty active set impossible under a
valid config; the generator asserts this at every node rather than handling it as an error path
(see Section 2).

The loader also enforces `levels.concentration_floor <= levels.concentration`, but only when
`levels.branch_length_scaling` is true: a floor above the base concentration would otherwise
silently collapse the branch-length-scaling axis into the constant-concentration control while the
config still claims scaling is on, exactly the kind of silent disagreement this design is meant to
avoid. When scaling is false the floor is inert -- constant `c` applies and there is nothing to
floor -- so a config with `concentration_floor > concentration` in that mode isn't lying about
anything and is not rejected. This mirrors the switching-rate bounds branching on their own scaling
flag (non-negative rates when scaling is on, `[0, 1]` probabilities when off): a knob is validated
only in the mode where it actually bites.

## 4. Branch length drives both axes

On an edge of length `L_e`, when `switching.branch_length_scaling` is true an off unit gains with
probability `1 - exp(-lambda_on * L_e)` and an on unit loses with `1 - exp(-lambda_off * L_e)` (a
two-state process whose hazard scales with branch length, `lambda` a rate); when it is false the
same roles are played by flat per-edge probabilities instead, `P(gain) = lambda_on`,
`P(loss) = lambda_off`, with no dependence on `L_e` at all, which is exactly why those rates must be
probabilities in `[0, 1]` in that mode and can be unbounded rates otherwise (Section 3).

Among the on-signatures, activity drifts by a mean-preserving Dirichlet step. Its concentration is
not `c / L_e` directly: stored branch lengths stay in mutation scale (Section 6), but the length
that drives drift is normalised by the forest's median branch length first, `L_e / L_median`, so `c`
means the concentration at a typical branch of that forest rather than an absolute per-mutation rate
that would otherwise depend on the arbitrary scale of the chosen branch-length distribution. The
per-edge concentration is therefore `c * L_median / L_e`, floored at `levels.concentration_floor` so
a long-branch tail cannot collapse the draw toward a one-hot activity: as `L_e -> infinity` this
concentration would otherwise fall toward zero, and a Dirichlet with concentration near zero
degenerates toward the simplex's vertices rather than spreading out, the opposite of realistic
drift. `L_median` is computed once per forest, over every edge in every tree of that forest, and
stored in `ground_truth_params.json` alongside the other realised parameters, since it is a property
of the sampled branch lengths, not a config value. A per-forest median is noisier at the smallest
forest-size sweep points (`n_trees` as low as 2, so the median is taken over very few edges) than at
larger `n_trees`; if such a point looks jumpy across seeds, this is the documented reason, not a
defect. It is still the right choice over a fixed cross-forest normaliser, which would break
one-seed-one-forest reproducibility, or a per-tree median, which would make sibling trees within a
forest non-comparable; `n_seeds` averaging is what absorbs the extra variance at small forest sizes.
The median is used rather than the mean because the branch-length distribution is deliberately
right-skewed (a few long branches, many short ones,
Section 5); the mean would be dragged upward by the long tail and would no longer describe a
"typical" branch. A longer-than-median branch means lower effective concentration, wider spread, and
more divergence, down to the floor; a shorter one means higher concentration and less divergence.
One quantity, the evolution on the edge, governs both whether a unit switches and how far the active
levels drift (subject to the floor), and the single global similarity parameter of the old simulator
is gone.

The level drift is a Dirichlet walk on the simplex, not Brownian motion. Brownian is the textbook
tree model for a continuous trait but it is unbounded and cannot sit on the simplex or produce a
true zero without moving to log-ratio (logit) coordinates, which is exactly the softmax floor the
model has and the truth must avoid. Working on the simplex directly, with a mean-preserving
Dirichlet step, keeps activities as proper proportions, admits exact zeros for off signatures, and
preserves the "a subclone's activity is an unbiased noisy copy of its parent's" belief. The one
thing given up is Brownian's clean "correlation equals shared path length" statement, which holds
only approximately under the bounded walk; for generating plausible ground truth that is fine and
the paper states it in a sentence. The activity is drawn on the simplex directly rather than
through a logit, so the root activity is a Dirichlet draw at `root_concentration` and no softmax
appears anywhere in the generator.

On every edge, regardless of whether that edge carries an activation, a deactivation, both, or
neither, the same rule builds the child's base composition and turns it into a Dirichlet draw:
restrict to `a_child`'s support, add `activation_pseudocount` for each newly-on signature (nothing
added for anything else), renormalise the result to sum to one, and draw the child activity as one
Dirichlet at the edge's concentration (above) around that renormalised base. This is not a rule that
only fires on activation: a deactivation-only edge, for instance, simply restricts to the shrinking
support and renormalises the survivors, with no pseudocount involved, before the same scaling
applies.

When a signature switches on along an edge, it cannot be resurrected by the Dirichlet step alone (a
zero component stays zero), so activation is the one case that injects mass explicitly, via
`activation_pseudocount` for each newly-on signature, ahead of the renormalisation above.
Renormalising first, rather than scaling the unnormalised augmented base directly, gives the same
child mean either way, but also keeps the *total* concentration a function of branch length only,
`c * L_median / L_e` (floored), never bumped up by how many signatures just activated; the two axes,
how far levels drift and which signatures are on, stay separable, which is a property the rest of
the design relies on. The new signature therefore enters at a small mean and immediately drifts with
the same branch-length-scaled spread as the rest, which is the "activates along the edge, starts
small, then drifts" behaviour resolved at the child node rather than tracked continuously within the
branch. When a signature switches off its component is set to zero and the remaining active mass
renormalises (the general rule above, with no pseudocount involved); the off signature keeps no
latent state and re-enters through the activation pseudo-count if it ever switches back on.

Setting `levels.branch_length_scaling: false` uses a constant concentration `c` on every edge (the
old uniform-divergence behaviour) as a well-specified control -- no median normalisation and no
floor apply in this mode, since there is no `L_e`-dependence left to floor -- so a poor result can be
attributed to on/off structure rather than to branch-length mismatch.

Each axis can also be switched off entirely, independently, which is the clean way to isolate
scenarios. `switching.enabled: false` makes every pool signature on at every node (a full,
unchanging active set), so only the drift is exercised; the `units` block becomes optional and is
ignored if given. `levels.enabled: false` removes the Dirichlet draw: no noise is added between
events, so a signature's level, once set, stays fixed until something about the active set changes.
This is not the same as freezing the whole composition through the tree: a switch event still runs
its carry-forward and zero-renormalise mechanics as usual, so a signature activating mid-tree still
enters at `activation_pseudocount` and the remaining actives still renormalise on a switch-off; only
the stochastic redraw of already-active signatures' relative levels is suppressed. These are
explicit flags rather than emergent from the per-unit rates (which could also express an always-on
repertoire, but not fragile-to-configure and not able to express constant levels at all), so a
scenario is stated in one place and cannot be got subtly wrong across many units.

## 5. Forest and tree shape

General to cancer: small and shallow. `nodes_per_tree`, `depth`, `n_trees`, and the branch-length
distribution are all config, with those defaults reflecting the cancer-wide norm rather than any
one cohort. A specific cohort's shape (for example SECEDO's, Section 13) is a config you write,
not a shape the generator assumes.

Topology can reuse the existing random-tree generator (`generate_network_random_tree_child_sequence`
from phylox), which is parameterised by leaf count, not by `nodes_per_tree` or `depth` directly. The
generator bridges this by rejection sampling: draw a candidate leaf count, build the tree, and
accept it only if its realised node count and depth both fall inside the configured `nodes_per_tree`
and `depth` ranges; otherwise resample and retry, up to `forest.max_attempts` attempts (default 200),
after which it raises an error naming both the `nodes_per_tree`/`depth` range and the attempt count
reached, so the message says which to loosen and that the cap itself can be raised, rather than
looping forever or silently returning a tree outside the stated range. A benchmark can't let
realised shape silently drift outside what the
config says it is, so rejection with a hard failure is the right behaviour here, not a bespoke
node-count-and-depth-controlled generator (more code for no gain at these sizes) and not a soft,
approximate target. Some ranges are geometrically impossible for a tree-child network (nine nodes
at depth two, say); the retry cap being exhausted and the config-naming error firing is the correct
outcome for those, not a bug to work around.

Branch lengths come from the config distribution (lognormal default, or empirical from the cohort)
rather than uniform integers, because branch length now drives both the switch and the drift, and
real subclone branch lengths are right-skewed, a few long branches and many short ones. A flat
uniform draw would put an unbiological length distribution at the centre of everything. Empirical
branch lengths are a single column of lengths (header optional), sampled i.i.d. with replacement;
preserving any length-depth correlation structure present in the empirical data is a later
refinement, not built now. Realistic topology shape (tumour trees tend to be unbalanced) is likewise
a later refinement; the branch-length distribution is the part that feeds the new machinery and is
worth getting right now.

## 6. Burden

Per-node burden `M_j` is config and is swept, since the low-count regime is where sharing across
the tree should decide on/off that a single sparse node cannot. Branch length and burden are both
in mutations but distinct: `L_e` is how different a node is from its parent, `M_j` is the
observed depth at the node. Keep them separate in code. The negative-binomial burden is
parameterised directly by its mean (`burden.mean`, matching the old `lam`) and `dispersion`; a
median parameterisation would need a numerical CDF inversion for no benefit here.

## 7. Counts

`x_j ~ DirichletMultinomial(M_j, kappa * e_j S)` with finite `kappa`, so the counts are not drawn
from the inference likelihood. `counts.model: multinomial` gives the well-specified control.

## 8. Both signature modes

Every experiment runs in both settings. Fixed: `S` given from the catalogue, the head-to-head with
PhySigs (which needs `S` known). De novo: `S` inferred jointly, the on/off capability PhySigs
cannot match and the more distinctive result.

## 9. Difficulty axes and controls

Primary sweeps: forest size (`n_trees`) and burden (`burden.mean`). Secondary: switch rate (scale
the unit `lambda`s up or down, how often signatures flip). Stress: signature overlap, interpolating
between two real catalogue signatures rather than synthesising a profile -- a small separate helper
for that one stress test, not a revival of the deleted random-signature synthesis. Controls run
alongside: constant-variance levels (isolates on/off from branch-length mismatch) and multinomial
counts (well-specified likelihood); by default these run as a single robustness spot check, not
across the full sweep, since the realistic settings carry the headline result. One axis at a time,
pool fixed within a sweep, `n_seeds` forests per point (a sweep-level quantity for the replicate
tooling, Section 14, not something the generator itself loops over).

## 10. Metrics

**Primary, on/off recovery.** For Tree-HDP, report the posterior activation probability
`P(signature k active at node j)` from the samples, scored against the true active set by
precision-recall and by calibration. For PhySigs, NMF, and the de novo extractors, threshold their
exposures to a point on/off call on the same axes. Measured in both modes. This is the selling
point the point-estimate baselines lack.

**Secondary, levels of the actives.** Conditional on the true active set, recover the on-signature
levels by `L_1` (range 0 to 2) or cosine restricted to the actives. This is the "how active are
the active signatures" question, and where the drift belief does its work.

**Signatures (de novo).** Cosine to the matched true signature, recovery rate at cosine >= 0.90.

State plainly that the current model reads on/off off a continuous posterior with a compositional
floor, so its on/off ceiling is a model property, not a threshold artefact, which motivates a
zero-mass switch model as the follow-up.

## 11. Baselines on on/off

PhySigs (fixed, native edge shifts, the honest hard comparator); the tree-free ablation of
Tree-HDP (isolates whether the tree prior improves on/off separation); SigProfilerExtractor,
MuSiCal, NMF (de novo, thresholded).

## 12. Pilot, the first experiment

Hand-build one switch-structured toy forest (a few nodes, a couple of signatures genuinely off at
some nodes, low burden) and fit the current model in both modes. Check whether on-node and off-node
activation posteriors separate at all, and whether the tree version separates them better than the
tree-free ablation in the low-burden regime. Poor absolute separation is expected; the green light
is that the tree helps. If it does not help even on easy data, the headline mechanism is missing,
and that must be known before building the benchmark. (If the pilot draws its signatures from the
triple-negative breast repertoire in Section 13 rather than an arbitrary subset already in the
catalogue, see that section's catalogue prerequisite first.)

## 13. Example config: triple-negative breast, validated on SECEDO

One instantiation, not a default in the code. The public SECEDO cohort is chosen because it is a
public single-cell breast dataset, not because the method is breast-specific.

Prerequisite: the current `COSMIC_sig/cosmic_signatures.csv` catalogue holds ten signatures but not
`SBS2`, `SBS13`, or `SBS3`, all three named in the repertoire below. Extending the catalogue (or
pointing `repertoire.path` at one that already has them) is a data task, not a generator change, and
blocks this example config and the pilot until done; nothing about building or unit-testing the
generator itself depends on it.

- **Repertoire and rates:** the switching block in Section 3 (clock SBS1/SBS5 always on; the APOBEC
  pair SBS2/SBS13 as one gain unit; SBS3 present per patient at the root). The `lambda` values there
  are the starting point to calibrate against APOBEC and HRD prevalence across subclones.
- **Forest shape:** `n_trees: 5`, three-or-so clones plus an ancestor per tree, matching the five
  SECEDO sections; branch-length distribution calibrated to SECEDO's per-branch SNV counts (its
  Figure 3 edge labels are additional SNVs per subclone, a branch-length proxy).
- **Burden:** anchored to SECEDO's per-subclone counts scaled by a single-cell recovery fraction,
  centred in the low-count regime.

A different cohort is a different config against the same generator, with no code change.

## 14. Implementation sketch

```
rng   = np.random.default_rng(cfg.seed)            # one generator, threaded through every draw below
S     = load_catalogue(cfg.repertoire.source, cfg.repertoire.path, cfg.repertoire.signatures)
units = parse_switch_units(cfg.switching)          # validates the partition of the pool; skipped,
                                                    # units optional, when switching.enabled is false
forest = sample_forest(cfg.forest, rng)            # topology + branch lengths, rejection-sampled
                                                    # against nodes_per_tree/depth (Section 5)
L_median = median(all edge lengths in forest)      # realised per forest, stored in ground_truth_params
for tree in forest:
    state  = draw_root_states(units, rng)              # root_prob per unit -> a_root
    a_root = expand_units_to_signatures(state)
    assert a_root.any()                                # clock guard, at the root too
    e      = draw_root_activity(a_root, cfg.levels.root_concentration, rng)  # Dirichlet on the simplex
    for (parent, child, L) in preorder_edges(tree):
        state   = evolve_units(state, L, units, cfg.switching, rng)  # gain/loss, ~ L or flat (Sec. 3/4)
        a_child = expand_units_to_signatures(state)
        assert a_child.any()                       # clock guard: never empty under a valid config
        base    = carry_forward(e, a_child, cfg.levels.activation_pseudocount)  # +mass for newly-on
        base    = base / base.sum()                # renormalise before scaling (Section 4)
        conc    = max(cfg.levels.concentration * L_median / L, cfg.levels.concentration_floor)
        e       = dirichlet_step(base, conc, cfg.levels, rng)   # offs stay 0
    for node in tree:
        M = draw_burden(cfg.burden, rng)
        x = draw_counts(M, e_node @ S, cfg.counts, rng)
save(S, active_sets, levels, tree_with_lengths, counts, ground_truth_params)  # one dataset, this seed
```

One dataset per invocation, seeded by `simulation.seed`. `n_seeds` is a sweep-level quantity: the
replicate averaging is driven externally by `make_replicate_configs.py` expanding one base config
into per-replicate configs and the SLURM array running one process per replicate, exactly as today,
because inference, not generation, is the expensive per-replicate step that has to run on the array.
An earlier per-seed loop in this sketch was illustrative only, not a mandate that one call emits N
datasets; `make_replicate_configs.py` and `run_replicates.sbatch` are unchanged by this rewrite.

Delivery is two phases, not one commit. Phase one adds `TreeSwitchDriftGenerator` in a new module,
`src/models/switch_drift_generator.py`, separate from `hdp_simulator.py`, alongside the untouched
old code, so the generator diff is purely additive and reviewable in isolation. This is a plain
working name, not a permanent second module: phase two moves its contents into `hdp_simulator.py`
and deletes both the interim module and the old class, so nobody should treat the interim path as
anything but transitional. The module stays dependency-free of `src/config.py`:
`TreeSwitchDriftGenerator` and its helper functions take plain dicts and dataclasses, never YAML or
a `get_prior`-style config object. Parsing a config file into these resolved parameters is
`generate_data.py`'s job, done at phase two; this module's job is turning already-resolved
parameters into data. This preserves the existing config-layer-versus-mechanics-layer split and
keeps the phase-one tests isolated from any config-loading machinery.

Phase one's tests construct `TreeSwitchDriftGenerator` directly against hand-built config dicts,
bypassing YAML and `generate_data.py` entirely; proving the config-file path end to end is phase
two's job, not phase one's. They land in `tests/test_simulator.py`, written now against the
phase-one module (`tests/README.md`'s planned entry for it updated to match); only its import line
changes, from `switch_drift_generator` to `hdp_simulator`, when phase two merges the module in, so
writing it now costs nothing later. They cover: shapes; byte-for-byte seed determinism (two
constructions from the same seed produce identical forests on every array); every loader validation
rule, including the branch_length_scaling-dependent rate bounds and the
`concentration_floor <= concentration` check; the renormalise-then-scale arithmetic on every edge
(activation, deactivation, both, and neither); the concentration floor at extreme `L`; the rejection
sampler's retry and its hard failure past `max_attempts`; mean-preservation of the Dirichlet step
conditional on the active set (the expected draw equals the renormalised base); that activation
preserves the continuing signatures' relative proportions (the on-signatures' pairwise ratios are
unchanged when a new one activates); and that off-signatures are exactly zero and agree with the
active set (activity is zero exactly where the active set is 0).

Phase one also lands, as its own separate commit with a message stating the behaviour change
explicitly, a fix to the current generator's redraw-on-every-call bug: `get_mutation_count_matrix()`
today draws fresh multinomial counts on every call, so two calls against the same generator return
different counts, a reproducibility bug. The new generator draws counts once, at construction, and
the accessor returns the stored values. This is kept separate from the generator-mechanics commits
so it is visible in history as a deliberate, trivially revertible behaviour change, not folded
silently into the rewrite.

Phase two is the cutover: rewrite `generate_data.py` against the new class, add a new smoke config
exercising switch-plus-drift, update the smoke test, and delete the old generator class
(`synthesize_signatures` included), the interim module, and `generate_walk_figure.py`, which renders
the retired cohort-baseline model and cannot be patched without misrepresenting the new mechanism --
all under the new tag this change needs. Both deletions are recoverable from git history under their
old tags; phase two is a genuine cutover, not a permanent alongside shim.

## 15. What's still open

Deferred to when the config is written (config values, not generator design):

- The unit gain/loss rates and `kappa`, together with the branch-length distribution's absolute
  scale (Section 3's example values are illustrative only). These set how hard the switch is, how
  over-dispersed the counts are, and what a "typical" branch means in mutation counts; calibrate
  them jointly against APOBEC and HRD subclonal prevalence, or fit to the example cohort, when
  writing the config.

Run choices for the benchmark, not the generator (settle before the on/off numbers go in the
paper):

- The threshold convention for turning a baseline's point exposures into on/off calls, a fixed
  cutoff versus a per-method calibrated one. This one is load-bearing, since the on/off comparison
  against PhySigs partly rests on it, so it needs a defensible answer.
- Whether a larger pool (adding further signatures) is tested as a stress case in this paper.
