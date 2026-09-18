# Plan: per-node signature on/off states in Tree-HDP

Implementation plan for Claude Code. Read `CLAUDE.md` first; every rule there applies
(branch not main, small commits, tests with every change, ruff, no trailers, no artefacts).
Read `simulator_spec.md` sections 2, 4 and 10 for the ground truth this model must recover.

## 0. The problem and the decision

The simulator (`TreeSwitchDriftGenerator`) gives every node `j` a binary active set `a_j`
over the `K` signatures. Signatures switch on and off along edges as a two-state gain/loss
process whose flip probability rises with branch length, and off signatures are exactly zero
in `e_j`. The current `TreeHDP` has no such state: every signature is on everywhere at a
softmax floor, so "is signature `k` active at node `j`" can only be read off a continuous
posterior with an arbitrary threshold. Headline claim C2 (calibrated per-edge switch
detection) has no direct evidence because the model has nothing to be calibrated about.

The obstacle is that `a_j` is discrete and the sampler is NUTS on numpyro, which needs a
smooth log density. The options are

1. relax the indicators (Concrete / Gumbel-softmax gates): approximate, temperature-tuned,
   biased activation probabilities, so C2's calibration claim would rest on an artefact;
2. Gibbs-within-NUTS on the indicators: not available on the numpyro path, and states and
   levels are strongly coupled, so it mixes badly;
3. marginalise `a` exactly and give the sampler the marginal likelihood.

Option 3 is the decision. Conditional on the continuous parameters, the states of one
signature along one tree form a two-state Markov chain, the trees are independent, and the
per-node emission depends on the joint state `a_j` through `e_j S`. Summing over all `a`
is Felsenstein pruning on the joint state space `{0,1}^K` with factorised transitions:
exact, smooth in every parameter, cost `O(N 2^K C)` per likelihood evaluation, and the
per-node activation probabilities come out of a forward-backward pass as Rao-Blackwellised
posterior quantities, which is exactly the calibrated `P(signature k active at node j)`
the spec asks for. This is the same construction as PyMC's marginalised HMMs and standard
phylogenetic likelihoods; nothing here is novel machinery.

The cost is the state space. `K = 5` gives 32 states (the breast repertoire, the paper's
main setting); `K = 10` gives 1024. That is roughly a 30x and 1000x multiplier on the old
likelihood. This is the main practical risk and is addressed below (batching, an
`always_on` option that removes the clock signatures from the state space, a hard cap).

Everything else stays as it is: the ILR/ZeroSumNormal walk, `mu_level`, fixed and de novo
modes, the runner, the scoring scripts, the simulator, `realdata/`.

## 1. The model

Notation as in the `TreeHDP` docstring. New per-node latent `a_j in {0,1}^K` (marginalised,
never a sampled variable). `eta_j` keeps its meaning as the walk state for all `K`
signatures; for an off signature it is a dormant level that keeps walking down the tree
under the prior and re-enters through the softmax if the signature switches back on.

```
sigma, mu_level, z_root, z_level, eta_j        as now (ILR walk, non-centred)

e_j(a)      = (a * exp(eta_j)) / sum_k a_k exp(eta_jk)        masked softmax
theta_j(a)  = e_j(a) @ S
x_j | a_j   ~ Multinomial(M_j, theta_j(a_j))                   with P(x_j | a_j = 0) = 0
                                                               whenever M_j > 0

per signature k, per tree, along the edges:
  a_root,k          ~ Bernoulli(pi_k)
  P(0 -> 1 | edge e) = 1 - exp(-lambda_on,k  * l_e)
  P(1 -> 0 | edge e) = 1 - exp(-lambda_off,k * l_e)
  l_e = L_e / L_median                                          (see 1.2)

lambda_on,k, lambda_off,k ~ LogNormal (config)                  shape (K,)
pi_k                      ~ Beta      (config)                  shape (K,)
```

The likelihood contribution is `sum_trees log P(x_tree | eta, S, lambda, pi)` computed by
pruning and entered as one `pm.Potential("switch_loglik", ...)`. Include the multinomial
normalising constant (`gammaln(M+1) - sum gammaln(x+1)`) so the Potential equals the true
log-likelihood; it is constant in every parameter and it makes the regression test in 4.2
an equality, not an equality up to a constant.

### 1.1 The all-off state

`e_j(0)` is undefined. The correct likelihood statement is that a node with `M_j > 0`
cannot have every signature off: `log P(x_j | a_j = 0) = -inf` at observed nodes. At
unobserved nodes (no counts, for example the real-data pseudo-normal root, or a node whose
row sums to zero) every state has log-emission 0, including all-off. No renormalisation of
the switch prior is needed; the data excludes the state. Keep the all-off state in the
`2^K` grid so the axis-wise transition contraction (2.2) stays uniform, and put the `-inf`
in as a constant, never as the result of an arithmetic operation (see 3.3).

### 1.2 Branch lengths

The model reads branch lengths from the Newick edge attributes (`phylox` round-trips
`:length`, and `nx.compose` in `_BaseTreeHDP` keeps the `length` edge attribute; verified).
Normalise by the forest median, `l_e = L_e / L_median`, computed over every edge of every
tree in the input, and store `L_median` on the model. The rates are then in units of "per
median branch", so one prior on `lambda` is sensible for a simulated forest with lognormal
lengths near 1 and for a real tree with lengths in hundreds of SNVs. The simulator does
not normalise the switching hazard (only the drift), so on simulated data the truth's
`lambda` and the model's differ by the factor `L_median`; record this in the docstring and
convert when comparing.

`inference.switching.branch_length_source: newick | unit`. `newick` (default) fails loudly
if any edge lacks a length. `unit` sets every `l_e = 1`; this is what the real-data SNV
tree needs today, since `build_snv_tree.py::digraph_to_newick` writes no lengths.

### 1.3 Reported quantities (Deterministics, all computed after sampling)

Per depth `d`, matching the existing `e_level_d` row order:

- `a_prob_level_d`, shape `(n_d, K)`: `P(a_jk = 1 | data, theta)` per draw. The posterior
  activation probability is its mean over draws. This is the C2 quantity.
- `e_level_d`, shape `(n_d, K)`: redefined as the state-mixed expected activity
  `sum_{a != 0} q_j(a) e_j(a) / sum_{a != 0} q_j(a)`. Rows sum to one. Every downstream
  consumer of `e_level_*` (`recovery_vs_truth.py`, `scaling_metrics.py`, `align_trace`)
  keeps working unchanged and now scores the Bayes activity estimate under the switch model.
  When switching is disabled it is `softmax(eta)` exactly as today.
- `eta_level_d` unchanged.

Do not store the full node marginal `q_j` over `2^K` states in the trace; at `K = 10` that
is `1024 x N x draws x chains` and it is never needed downstream (edge quantities come from
section 6).

## 2. New module `src/models/switch_pruning.py`

Pure PyTensor graph builders, no PyMC, no config. Takes tensors and NumPy index arrays,
returns tensors. This is the single implementation of the pruning maths; the post-hoc
tools (section 6) compile it with `pytensor.function` rather than reimplementing it in
NumPy, which keeps CLAUDE.md's one-definition rule and means the tests in 4.1 cover both
uses.

### 2.1 State grid

`state_grid(K) -> np.ndarray (2**K, K)` of 0/1 masks, row `s` having bit `k` equal to
`(s >> (K-1-k)) & 1`, so flat index `s` corresponds to position `(s_0, ..., s_{K-1})` in a
tensor of shape `(2,)*K` in C order. Flat index 0 is all-off. Fix this convention once and
test it (`state_grid(K).reshape((2,)*K + (K,))[idx] == idx`).

### 2.2 Per-depth batching (no scan)

Mirror the existing model: nodes are grouped by depth (`_get_nodes_by_depth`, same order
as today so `node_index_map` is unchanged). Precompute at build time, as NumPy:

- `parent_pos[d]`, int `(n_d,)`: index of each node's parent within depth `d-1`
- `length[d]`, float `(n_d,)`: normalised edge length to the parent
- `counts[d]`, int `(n_d, C)` and `observed[d]`, bool `(n_d,)`
- `tree_of_root`, for depth 0: which root belongs to which tree (roots are independent, so
  `logZ = sum over roots`)

Upward pass, `d = max_depth .. 0`:

```
log_em[d]       (n_d, 2^K)      emissions, section 2.3
log_beta[d]     = log_em[d] + acc[d]           acc[d] starts at zeros (n_d, 2^K)
log_msg[d]      (n_d, 2^K)      message to parent, section 2.4 (d > 0 only)
acc[d-1]        = inc_subtensor(acc[d-1][parent_pos[d]], log_msg[d])
logZ            = sum_r logsumexp_s( log_pi(s) + log_beta[0][r, s] )
```

`log_pi(s) = sum_k [s_k log pi_k + (1-s_k) log(1-pi_k)]`, shape `(2^K,)`, via the mask
matrix: `masks @ log(pi) + (1-masks) @ log1p(-pi)`.

Downward pass, `d = 0 .. max_depth` (Deterministics only, so cost is not on the sampler):

```
log_alpha[0]    = log_pi[None, :]
log_alpha[d]    = contract_parent_to_child(
                     (log_alpha[d-1] + log_beta[d-1] - log_msg[d])[parent_pos[d]],
                     logT[d])
log_q[d]        = log_alpha[d] + log_beta[d] - logZ_of_that_tree[node]
```

`log_msg[d]` is finite whenever `log_beta` has one finite entry, which is guaranteed, so the
subtraction cannot produce `-inf - (-inf)`. Assert this in the brute-force test with an
observed node (which has `log_beta(all-off) = -inf`).

The whole thing is a Python loop over depths at graph-build time; the graph size is
proportional to `max_depth * K`, not to `N`.

### 2.3 Emissions

For depth `d`, `eta` is `(n_d, K)`, `masks` is `(2^K, K)`:

```
w      = masks[None, :, :] * exp(eta[:, None, :] - max(eta, axis=-1)[:, None, None])
den    = sum(w, axis=-1)                                     (n_d, 2^K); zero only at all-off
e_all  = w / where(den > 0, den, 1)                          (n_d, 2^K, K); all-off row is 0
theta  = e_all @ S                                           (n_d, 2^K, C)
ll     = counts[:, None, :] * log(clip(theta, 1e-300)) summed over C, + constant
log_em = where(observed[:, None], ll, 0)
log_em = set_subtensor(log_em[:, 0], where(observed, -inf, 0))
```

The `-inf` enters only through `set_subtensor` of a constant, never through `log(0)` or a
`where` whose unselected branch carries a NaN gradient (JAX's `where` propagates NaN
gradients from the unselected branch). `clip` mirrors what `pm.Multinomial` does for
zero-probability channels in the current model; it is not a behaviour change. `e_all` is
reused for `e_level_d` (weighted by `exp(log_q)`), so compute it once.

Memory: `(n_d, 2^K, C)` doubles at `K = 10`, 100 nodes: about 80 MB per evaluation plus
autodiff intermediates. Fine on Euler and on a laptop for `K <= 8`. State it in the
docstring.

### 2.4 Factorised transitions and the axis-wise contraction

Per edge `logT` is `(K, 2, 2)` with `logT[k, i, j] = log P(s_k: i -> j)`:

```
p_gain = exp(log1mexp(-lambda_on  * l))    computed as  log_p_gain = log1mexp(-lambda_on * l)
p_loss likewise
logT[k, 0, 1] = log_p_gain      logT[k, 0, 0] = -lambda_on  * l
logT[k, 1, 0] = log_p_loss      logT[k, 1, 1] = -lambda_off * l
```

Use `pt.log1mexp` (JAX dispatch exists in PyTensor 2.31.7, verified), not `log(1 - exp())`,
so a small `lambda * l` neither underflows nor loses precision. Never build the
`2^K x 2^K` Kronecker transition (that is `4^K`, 1M entries per edge at `K = 10`).

`contract_child_to_parent(log_beta_child (n, 2^K), logT (n, K, 2, 2)) -> (n, 2^K)`:
reshape to `(n,) + (2,)*K`, then for `k = 0..K-1`: move axis `k+1` to position 1, view as
`(n, 2, R)`, compute `logsumexp(logT[:, k, :, :, None] + x[:, None, :, :], axis=2)` which
contracts the child index and leaves the parent index, move the axis back. Result axis `k`
now indexes the parent's `s_k`. `contract_parent_to_child` is the same with `logT`
transposed on its last two axes. Cost `O(K 2^K)` per edge; batched over the depth.

Test both against the explicit Kronecker product for `K <= 4`, and test that contracting
with an identity transition (lambda -> 0) is the identity.

## 3. Model integration (`src/models/hdp_inference.py`)

### 3.1 Constructor

`TreeHDP(newick_string, data_matrix, priors, fixed_signatures=None, num_signatures=None,
switching=None)`. `switching=None` or `{"enabled": False}` builds exactly today's model:
same variables, same likelihood (`pm.Multinomial` observed), same `node_index_map`. Every
existing config therefore runs unchanged. Keep `_BaseTreeHDP` as is.

### 3.2 `switching` dict (resolved, plain values; `run_inference.py` does the config work)

```
enabled: bool
branch_length_source: "newick" | "unit"
lambda_on_prior, lambda_on_prior_parm      get_prior style, dim=K   e.g. LogNorm {mu: -1.2, sigma: 1.0}
lambda_off_prior, lambda_off_prior_parm    same
pi_root_prior, pi_root_prior_parm          e.g. Beta {alpha: 1, beta: 1}
always_on: list[str]                       fixed mode only, section 7.1; default []
tree_coupled: bool                         default True, section 7.2
```

Validation in the constructor, loud errors: `K - len(always_on) <= 12` (state space cap;
error message states the `O(N 2^K C)` cost and names `always_on`); `always_on` names must
be in the fixed signature index and `always_on` must be empty in de novo mode; `newick`
source with a missing edge length; a `unit` source with lengths present is allowed but
logs the fact in the model docstring's `L_median = 1` convention.

### 3.3 Build order inside `_build_pymc_model`

Walk block unchanged (`sigma`, `mu_level`, `z_*`, `eta_level_d`). Then, if enabled:
`lambda_on`, `lambda_off`, `pi_root` via `get_prior(..., dim=K)`; the depth arrays from
2.2; `prune` -> `pm.Potential`; `backward` -> `a_prob_level_d`, `e_level_d`. If disabled:
`e_level_d = softmax(eta)` and the existing `pm.Multinomial`. `get_prior` builds
per-parameter lists of length `dim`, which is what `(K,)` needs; check it produces a
`(K,)` LogNormal, not a scalar, in the unit test.

Do not write `-inf` anywhere except section 2.3's constant. Do not use `pt.where` with a
branch that can be non-finite. Do not use `scan`.

### 3.4 Sanity requirements the tests must enforce

- `model.compile_logp()` and `model.compile_dlogp()` at the initial point and at 20 random
  points from the prior are finite, in the default backend and after `mode="JAX"`
  compilation (the sampler is numpyro; a graph that only works in C is a failure).
- With `enabled: False`, the Potential path is not built and the logp equals today's.
- With `enabled: True` and `lambda_on = lambda_off -> 0`, `pi -> 1` (pass these as
  fixed values through a test hook, or evaluate the pruning function directly), the
  marginal likelihood equals `pm.logp(Multinomial(n, softmax(eta) @ S), x)`: the switch
  model with everything forced on is the current model. This is the bridge test.

## 4. Tests to write (`tests/test_switch_pruning.py`, `tests/test_inference_switch.py`)

### 4.1 Brute force oracle

For `K in {1, 2, 3}` and a hand-built forest of two trees with 3 to 4 nodes and random
lengths, random `eta`, `lambda`, `pi`, random counts (one node unobserved): enumerate every
joint assignment of states to nodes (`(2^K)^N`, at most `8^4 = 4096`), sum
`prior(states) * prod emissions`, and compare `logZ` (rtol 1e-8), node marginals
`P(a_jk = 1)` (atol 1e-8), the expected activity `e_level` (atol 1e-8), and edge event
probabilities `P(a_pk = 0, a_jk = 1)` (used by section 6). Include a case where an
observed node's all-off exclusion matters (small `pi`, so the prior puts real mass on
all-off) and one with `pi = 1` for all `k` and `lambda = 0` (single state, must reduce to
the plain multinomial). This test is the correctness anchor; nothing downstream is
trusted until it passes.

### 4.2 Component tests

- `state_grid` convention (2.1).
- Axis-wise contraction equals Kronecker for `K <= 4`; identity transition is the
  identity; log-space stability with one `-inf` entry in `log_beta`.
- Emissions: `e_all` rows sum to one for `s != 0`, are zero at `s = 0`, agree with an
  explicit masked softmax; log-emission of an unobserved node is zero for every state.
- `log1mexp` path: `p_gain + exp(-lambda l) == 1` to 1e-12, and finite at `lambda l = 1e-12`.
- JAX compilation of `prune` and `backward` on a tiny input (a `pytensor.function` with
  `mode="JAX"`), values equal to the default backend.
- `TreeHDP` builds with switching on and off on the smoke data; variable names and shapes
  (`a_prob_level_d (n_d, K)`, `e_level_d` rows sum to one); logp and dlogp finite (3.4);
  bridge test (3.4); `always_on` reduces the state space and forces `a_prob = 1` on those
  signatures; every validation error fires.

### 4.3 Smoke and slow

- New `experiments/smoke_switch/config.yaml`: the smoke config plus an `inference.switching`
  block, `K = 3` (8 states), 50 draws, 1 chain, around a minute. Extend `test_smoke.py` (or
  add `test_smoke_switch.py`) to run generate, infer, `switch_states`, `switch_recovery` and
  assert files, shapes, finiteness, `a_prob in [0, 1]`, `e_level` rows sum to one.
- `@pytest.mark.slow`: an easy hand-built forest (high burden, long branches, one
  signature clearly off in one subtree) fit with 2 chains: node-level accuracy at 0.5
  >= 0.9, `r_hat` on `e_level`, `sigma`, `lambda`, `pi` < 1.05.

## 5. Runner and alignment (`scripts/run_inference.py`)

- Read `inference.switching` (absent means disabled), resolve it to the plain dict of 3.2,
  pass to `TreeHDP`. `_validate_inference_config` checks `always_on` only appears with
  `model: fixed`.
- Label switching. `align_trace` currently permutes `signatures`, `mu_level` and
  `e_level_*` only. With the switch model, `a_prob_level_*`, `lambda_on`, `lambda_off` and
  `pi_root` also carry a signature axis, and the post-hoc state sampler (section 6) needs a
  per-draw consistent `(eta, S, lambda, pi)`. Replace the hard-coded list with a registry
  on the model, `TreeHDP.signature_axis_vars() -> dict[name, axis]` (`signatures: 0`, all
  `(..., K)` variables including `eta_level_*`, `z_root_*`, `z_level_*`: axis -1), and
  permute every registered variable. Permuting a ZeroSumNormal draw keeps it zero-sum, so
  aligning `z_*` is harmless and makes `trace_aligned.nc` self-consistent per draw. Save
  `perms.npy` `(chains, draws, K)` next to `switching_table.csv`. Test: a fabricated
  two-chain trace where chain 1 is chain 0 with a known permutation aligns to identical
  arrays for every registered variable.
- `az.summary` is unchanged.

`scripts/scaling_metrics.py`: `a_prob_level_*` can be constant (exactly 0 or 1) across a
chain, which makes `r_hat` and `ess` NaN and `max`/`min` propagate NaN. Compute
`max_rhat` and `min_ess` over `e_level_*`, `sigma`, `lambda_*`, `pi_root` with `nanmax` /
`nanmin` and drop variables whose posterior variance is below 1e-12 before the diagnostic.
Keep `eta_level`, `z_level` excluded as before. Test with a fabricated constant variable.

## 6. Post-hoc state samples and on/off scoring

### 6.1 `src/analysis/switch_posterior.py`

- `compile_pruning(K, C, depth_arrays)`: one `pytensor.function` from `(eta per depth, S,
  lambda_on, lambda_off, pi)` to `(log_beta per depth, log_msg per depth, log_pi)`, built
  from `switch_pruning.py` (one implementation, section 2). Evaluated per draw in NumPy.
- `sample_states(...)`: forward-filter backward-sample per draw. Root state from
  `softmax(log_pi + log_beta[0])`, then top-down each node from
  `softmax(logT(s_parent -> .) + log_beta_child)`. Returns int8 `(chains, draws, N, K)`.
  With `--thin` to subsample draws. Cost `O(N 2^K C)` per draw; at `K = 5` negligible, at
  `K = 10` minutes for 8000 draws, acceptable and documented.
- From the samples, any joint functional: per-edge gain probability
  `P(a_pk = 0, a_jk = 1)`, loss probability, number of active signatures per node,
  co-activation. The node marginal from the samples must agree with the mean of
  `a_prob_level_*` (Rao-Blackwellised) to Monte Carlo error; assert that in the smoke test
  as a cross-check between the two code paths.

### 6.2 `scripts/switch_states.py`

Inputs `--trace` (the aligned trace for de novo, `trace.nc` for fixed), `--newick`,
`--counts`, `--fixed-signatures` (fixed mode), `--outdir`, `--thin`. Writes
`state_samples.npz` and `switch_edges.csv` (`tumour, parent, child, signature, p_gain,
p_loss, p_switch`). CSV only, no verdicts on stdout.

### 6.3 `scripts/switch_recovery.py`

Inputs the trace, `true_active_sets.csv`, `true_activities.csv`, `newick_string.nwk`,
`tree_edges.csv`, optional `switch_edges.csv` from 6.2, optional `--true-signatures` for
de novo (reuse `chain_perms_to_true` and a new shared helper
`node_variable_rows(post, prefix, newick, perms)` in `analysis.py` that generalises
`recovery_vs_truth._aligned_activities`; refactor that script to call the helper rather
than keeping two copies). Per chain, then best/mean/worst as `recovery_vs_truth.py` does.

Outputs:
- `switch_nodes.csv`: node, signature, `p_active` (per chain and mean), `true_active`,
  `true_level`.
- `switch_summary.csv`: AUROC, AUPRC, precision, recall, F1 and accuracy at 0.5, Brier,
  ECE (10 equal-width bins), overall and per signature; plus the same accuracy stratified
  by `true_level` bins (`0`, `(0, 0.05]`, `> 0.05`), because "on at a level the counts
  cannot see" is an identification limit, not a model error, and the stratified number
  separates the two.
- `switch_calibration.csv`: reliability bins (bin, n, mean predicted, observed frequency).
- `switch_edges_scored.csv` and `switch_edge_summary.csv` when `switch_edges.csv` is given:
  per-edge gain/loss AUROC, AUPRC, calibration. This is the C2 evidence.
- Levels of the actives (spec section 10, secondary): restrict `e_level` to the true
  active set, renormalise, `L1` and cosine against `true_activities`.

Use `sklearn.metrics` (pinned already) for AUROC, AUPRC, Brier; implement ECE and
reliability bins in `analysis.py` with unit tests on hand-built arrays (perfect, inverted,
constant predictions). Guard degenerate cases (a signature that is on everywhere in truth
has no AUROC; write NaN, not an exception).

`scripts/plot_switch_recovery.py`: reliability diagram and PR curve through
`figure_style`, `PALETTE` roles, nothing ad hoc.

## 7. Optional flags, each its own commit with its test, all default off

### 7.1 `always_on` (fixed mode)

Signatures listed here are forced to state 1: drop them from the state grid (`2^(K-m)`
states) and reinsert as a constant 1 column in the masks. Biologically the clock
signatures SBS1 and SBS5, exactly the simulator's clock guard. This is the cheap way to
run `K = 7` at 32 states. `a_prob_level_d[:, k] = 1` for those `k`. Not available in de
novo because signatures have no identity there.

### 7.2 `tree_coupled: false` (the tree-free switching ablation)

Replace every edge transition by the root prior, `logT[k, i, j] = log pi(j)` independent
of `i` and `l`: states are i.i.d. across nodes and the tree carries no information about
on/off. With the existing walk unchanged this isolates whether the Markov coupling along
the tree helps on/off recovery, which is the spec's pilot question (section 12) and the C1
style ablation for C2. One line in `log_transition` plus a test that `a_prob` at a node
is unchanged by its neighbours' counts under this flag.

### 7.3 `walk_branch_length_scaling` (in `priors`)

`eta_j = eta_parent + sigma * sqrt(l_e) * z_j`. Matches the simulator's branch-length-scaled
drift. Default `false` so nothing measured so far changes; turn on in a separate
experiment, one variable at a time. Uses the same `l_e` and `L_median` as the switch block.

## 8. Order of work and commits

Branch `feature/switch-states` off `refactor/unify-models`. Each numbered item is one or
two commits with tests passing and `ruff format . && ruff check --fix .` clean. Report to
Rado at the marked checkpoints before continuing.

1. Spike (throwaway, not committed): a 40-line script compiling `contract_child_to_parent`
   and a masked-softmax emission in `mode="JAX"` for `K = 3`, gradient finite. If any op
   fails to dispatch, stop and report; the design depends on it.
2. `src/models/switch_pruning.py` with `tests/test_switch_pruning.py` (4.1, 4.2 components).
   **Checkpoint 1**: brute-force oracle passing.
3. `TreeHDP` integration behind `switching` (3.x), `tests/test_inference_switch.py`
   including the bridge test, `experiments/smoke_switch/config.yaml`, smoke test extended.
   Default-off, so `pytest` must be fully green including the existing smoke test.
4. Runner: config plumbing, alignment registry, `perms.npy`, `scaling_metrics.py` NaN guard,
   tests. **Checkpoint 2**: run `smoke_switch` end to end in both modes (`fixed`, `denovo`),
   paste the `az.summary` rows for `lambda_*`, `pi_root`, one `a_prob_level_*`.
5. `switch_posterior.py`, `switch_states.py`, `switch_recovery.py`, `plot_switch_recovery.py`,
   the shared `node_variable_rows` helper, tests, smoke extended with the
   Rao-Blackwell-vs-samples cross-check.
6. Flags 7.1 to 7.3, one commit each.
7. Docs: `CLAUDE.md` (model description, config schema, new scripts, the new "do not"
   items below), `README.md` model equations, `tests/README.md`, `simulator_spec.md`
   section 10's sentence about the softmax floor now pointing at this model. Cut tag
   `treehdp-v2` on the merge commit; configs written from now on carry
   `git_tag: treehdp-v2`. **Checkpoint 3**: ready for the pilot.
8. Pilot (Rado runs, Claude prepares the config): the spec's section 12 forest, fixed and
   de novo, `tree_coupled` true and false, low burden. Green light is node-level and
   edge-level AUROC clearly above 0.5 with the tree beating the ablation. Only then design
   the burden and `n_trees` sweeps.

New "do not" lines for `CLAUDE.md`: do not compute `r_hat`/`ess` over `a_prob_level_*`
(it is a bounded Deterministic that can be constant); do not compare `lambda` from the
model to the simulator's without the `L_median` factor; do not run the switch model at
`K - len(always_on) > 10` on CPU without checking the cost estimate.

## 9. Anticipated problems and their answers

| Problem | Answer in this plan |
|---|---|
| Discrete states, gradient sampler | Exact marginalisation by pruning (section 0) |
| All-off state undefined | `-inf` emission at observed nodes, allowed at unobserved (1.1) |
| NaN gradients from `-inf` in `where` | `-inf` only as a `set_subtensor` constant; masked probabilities built without infinities (2.3) |
| Underflow in `1 - exp(-lambda l)` | `pt.log1mexp` (2.4) |
| `4^K` transition matrix | Axis-wise contraction, `O(K 2^K)`, tested against Kronecker (2.4) |
| Graph grows with `N` | Batch by depth with `inc_subtensor`, no scan (2.2) |
| Op without JAX dispatch | Spike first; JAX compile test in the suite (8.1, 4.2) |
| Branch-length scale differs across datasets | Normalise by forest median; rates per median branch (1.2) |
| Real-data Newick has no lengths | `branch_length_source: unit` (1.2) |
| Existing configs and results | `switching` absent = old model; bridge test proves it (3.1, 3.4) |
| New signature-axis variables and label switching | Registry-driven `align_trace`, `perms.npy` saved (5) |
| `r_hat` NaN on constant `a_prob` | Excluded from convergence stats, `nanmax` guard (5) |
| Trace bloat | Only `a_prob` and `e_level` stored, never `q` over `2^K` (1.3) |
| `2^K` cost at `K = 10` | Cap at 12 with a cost message; `always_on` for clock signatures; GPU for `K >= 8` (3.2, 7.1) |
| Memory of `(N, 2^K, C)` emissions | Documented, `~80 MB` at `K = 10`, fine on Euler (2.3) |
| "On at a tiny level" indistinguishable from off | Intrinsic; reported via calibration and level-stratified accuracy, not hidden (6.3) |
| `lambda` weakly identified on small forests | Weakly informative LogNormal per signature, shared across the forest; prior predictive check of switch frequency in the pilot (1, 8.8) |
| Edge probabilities need pairwise marginals | FFBS samples from the same compiled pruning, cross-checked against the Rao-Blackwellised node marginals (6.1) |
| Two implementations of the same maths | One PyTensor implementation, compiled for post-hoc use (2, 6.1) |
| Regression of the level metric | `e_level` redefined as the state-mixed Bayes activity; identical to today when disabled (1.3) |
| Node ordering drift | Reuse `_get_nodes_by_depth` order; `node_index_map` unchanged (2.2) |

## 10. Decisions for Rado (do not guess; ask at Checkpoint 1)

- Prior parameters for `lambda_on`, `lambda_off` (proposed `LogNorm(mu=-1.2, sigma=1.0)`
  in per-median-branch units, covering roughly 0.1 to 0.8 at one standard deviation, in
  line with the simulator's example rates) and `pi_root` (proposed `Beta(1, 1)`).
- Tag name (`treehdp-v2` proposed).
- Whether `e_level` keeps its name under the new definition (recommended: yes, it is the
  drop-in Bayes estimate) or the old softmax activity is also stored as `e_on_level_d`.
- State-space cap (12 proposed) and whether the `K = 10` stress sweep runs with
  `always_on: [SBS1, SBS5]`.
