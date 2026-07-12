"""
diagnose_camp_path.py

One pass over the straight line between the two camps that produces every number
the mixing-geometry figures need.

The on-path and orthogonal curvatures are measured by the same finite-difference
stencil with the same step, so they are directly comparable; their ratio is the
condition number of the local geometry. The adapted step size and tree depth are read
from sample_stats so the implied step (two over root of the stiffest curvature) can be
set beside the step NUTS actually used, and the path length divided by that step beside
two to the tree depth.

Outputs
  camp_path_profile.csv   per point along t in [0, 1]:
      t, s                        interpolation fraction and sampler-space arclength
      loglik, logp, logprior      exact model densities on the simplex path
      loglik_sampler, logp_sampler   same on the straight sampler-coordinate line
      dloglik_ds, d2loglik_ds2    derivative and curvature of the likelihood hill
      dlogp_ds,   d2logp_ds2      derivative and curvature of the posterior
      on_path_curv                second directional derivative of logp along the path
      orth_curv_min/mean/max      same along random orthogonal directions (walls)
      stiff_eig                   exact stiffest curvature at the point, the
                                  largest-magnitude Hessian eigenvalue by power
                                  iteration (signed, negative where concave); the
                                  random probes only lower-bound this
      rate_min                    smallest multinomial rate (e_t @ S_t) on the path
      finite                      logp/loglik finiteness flag
  camp_path_summary.csv   one row of scalars (arclength, hill height, logit-geometry
      inflation, on-path vs orthogonal curvature, condition number, adapted step,
      implied step, tree depth, steps to cross, max trajectory steps; and the exact
      counterparts orth_curv_stiffest_exact, condition_number_exact and
      implied_step_exact from the power-iteration Hessian eigenvalue).

Usage
    python scripts/diagnose_camp_path.py \\
        --trace ../results/<random-start run>/trace_raw.nc \\
        --counts ../data/<run>/mutation_count_matrix.csv \\
        --newick ../data/<run>/newick_string.nwk --num-signatures 10 \\
        --camp-a 0 1 2 5 6 --camp-b 3 4 7 \\
        --n-points 81 --n-orth 8 --outdir ../results/<run>/camp_path
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.analysis import (activities_draw, align, build_model,  # noqa: E402
                                    inverse_walk, split_camps)

def _forward_compilers(model):
    """Compile each free RV's constrained-to-unconstrained forward map once.

    The previous code called ``tr.forward(...).eval()`` inside _value_point, which
    recompiles the PyTensor graph on every draw. The best-draw scan calls that for
    every draw of every chain, so the recompilation, not the arithmetic, was the
    cost. Here each transform is compiled a single time and the compiled function is
    reused, so the script runs at full resolution in the default (FAST_RUN) mode
    without needing FAST_COMPILE or heavy thinning. RVs with no transform (the
    Gaussian eta and z) map to None and are copied through unchanged.

    Only the simplex transform on the signatures and the log transform on sigma are
    present in this model, and neither uses its distribution inputs, so the forward
    graph depends only on the value and compiles with a single argument.
    """
    fns = {}
    for rv in model.free_RVs:
        vv = model.rvs_to_values[rv]
        tr = model.rvs_to_transforms[rv]
        if tr is None:
            fns[vv.name] = None
            continue
        x = rv.type()                              # symbolic constrained value
        fwd = tr.forward(x, *rv.owner.inputs)
        fns[vv.name] = pytensor.function([x], fwd, on_unused_input="ignore")
    return fns


def _value_point(model, constrained, fwd_fns):
    """Constrained (natural) RV values -> the model's unconstrained value point,
    using the forward maps precompiled by _forward_compilers."""
    pt = {}
    for rv in model.free_RVs:
        vv = model.rvs_to_values[rv]
        x = np.asarray(constrained[rv.name], dtype=float)
        f = fwd_fns[vv.name]
        pt[vv.name] = x if f is None else np.asarray(f(x))
    return pt


def _extract_natural(post, obj, c, dr):
    """Activities of one draw (chain c, draw dr) as a {node: e_j} dict."""
    order, A = activities_draw(post, obj, c, dr)
    act = {n: A[i] for i, n in enumerate(order)}
    S = post["signatures"].isel(chain=c, draw=dr).values
    sig = float(post["sigma"].isel(chain=c, draw=dr).values)
    return act, S, sig


def _natural_to_freervs(obj, act, S, sig):
    """Map a {node: e_j} activity dict and signatures S back to the model free
    RVs (inverse walk on the activities, plus the signature block)."""
    out = inverse_walk(obj, act, sig)
    out["signatures"] = S
    out["sigma"] = sig
    return out


def _camp_best_draw(post, obj, model, logp_fn, chains, thin, fwd_fns):
    """Representative point for a camp, the highest-logp draw over the given
    chains (thinned), returned as its free-RV dict."""
    best = None
    for c in chains:
        for dr in range(0, post.sizes["draw"], thin):
            constrained = {rv.name: post[rv.name].isel(chain=c, draw=dr).values
                           for rv in model.free_RVs}
            lp = float(logp_fn(_value_point(model, constrained, fwd_fns)))
            if best is None or lp > best[2]:
                best = (c, dr, lp)
    return best


def _flat(pt, keys):
    """Flatten the named entries of a point dict into one 1-D vector."""
    return np.concatenate([np.asarray(pt[k], float).ravel() for k in keys])


def _unflat(vec, template, keys):
    """Inverse of _flat, split a flat vector back into a dict shaped like template."""
    out, i = {}, 0
    for k in keys:
        ref = np.asarray(template[k], float)
        out[k] = vec[i:i + ref.size].reshape(ref.shape)
        i += ref.size
    return out


def _curv(logp, v, u, eps, template, keys, base):
    """Second directional derivative of logp at v along unit vector u."""
    lp_p = float(logp(_unflat(v + eps * u, template, keys)))
    lp_m = float(logp(_unflat(v - eps * u, template, keys)))
    return (lp_p - 2.0 * base + lp_m) / (eps ** 2)


def _sample_stat(idata, name):
    """Named sample-stats array from idata as a float, or nan if absent."""
    if hasattr(idata, "sample_stats") and name in idata.sample_stats:
        return float(np.asarray(idata.sample_stats[name]).mean())
    return float("nan")


def _hvp_setup(model):
    """Compile an exact Hessian-vector product of the model logp once, in the
    value-variable coordinates (the same unconstrained space the curvature stencil
    uses). Returns (hvp, names) with names the value-variable order; calling
    hvp(*[point[n] for n in names], vec) returns the flat H @ vec. This is the
    Pearlmutter double-grad, so it is exact and needs no finite-difference step, and
    the logp carries the transform Jacobian to match model.compile_logp(). It lets
    power iteration find the genuine stiffest curvature, which a handful of random
    orthogonal probes only lower-bound."""
    value_vars = list(model.value_vars)
    names = [v.name for v in value_vars]
    logp = model.logp(jacobian=True)
    flat_grad = pt.concatenate([g.ravel() for g in pt.grad(logp, value_vars)])
    vec = pt.vector("hvp_vec")
    gdotv = pt.sum(flat_grad * vec)
    flat_hvp = pt.concatenate([h.ravel() for h in pt.grad(gdotv, value_vars)])
    hvp = pytensor.function(value_vars + [vec], flat_hvp, on_unused_input="ignore")
    return hvp, names


def _stiffest_eig(hvp, names, point, u, n_iter):
    """Largest-magnitude eigenvalue of the logp Hessian at `point` by power
    iteration warm-started from the unit vector `u`. Returns (lam, u). lam is signed,
    negative where logp is concave, and its magnitude is the stiffest curvature, the
    one that caps the leapfrog step. `point` is a value dict keyed by `names`; `u` and
    the returned vector are flat in the value-variable order of `names`."""
    args = [np.asarray(point[n], float) for n in names]
    lam = float("nan")
    for _ in range(max(1, n_iter)):
        Hu = np.asarray(hvp(*args, u), float)
        lam = float(u @ Hu)                        # Rayleigh quotient, u is unit
        nrm = float(np.linalg.norm(Hu))
        if not np.isfinite(nrm) or nrm < 1e-30:
            break
        u = Hu / nrm
    return lam, u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--counts", required=True)
    ap.add_argument("--newick", required=True)
    ap.add_argument("--num-signatures", type=int, required=True)
    ap.add_argument("--camp-a", type=int, nargs="*", default=None)
    ap.add_argument("--camp-b", type=int, nargs="*", default=None)
    ap.add_argument("--map-thin", type=int, default=5)
    ap.add_argument("--n-points", type=int, default=81)
    ap.add_argument("--n-orth", type=int, default=0,
                    help="random orthogonal probes per point for the lower-bound "
                         "wall estimate; 0 skips them (the exact stiffest comes from "
                         "power iteration)")
    ap.add_argument("--eps", type=float, default=1e-3,
                    help="finite-difference step in sampler-space units")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stiff-eig", dest="stiff_eig", action="store_true", default=True,
                    help="also estimate the true stiffest curvature by power "
                         "iteration on the exact Hessian, beside the random probes")
    ap.add_argument("--no-stiff-eig", dest="stiff_eig", action="store_false")
    ap.add_argument("--power-iters", type=int, default=12,
                    help="power-iteration steps per point, warm-started; the first "
                         "point uses more")
    ap.add_argument("--outdir", default="camp_path")
    a = ap.parse_args()

    obj, counts = build_model(a.newick, a.counts, a.num_signatures)
    model = obj.model
    idata = az.from_netcdf(a.trace)
    post = idata.posterior
    ca, cb = (a.camp_a, a.camp_b) if (a.camp_a and a.camp_b) else split_camps(post, obj)
    print(f"camp A {ca}\ncamp B {cb}")

    logp = model.compile_logp()
    lik = model.compile_logp(vars=model.observed_RVs, sum=True)
    pri = model.compile_logp(vars=model.free_RVs, sum=True)
    fwd_fns = _forward_compilers(model)            # compile transforms once

    cA, dA, lpA = _camp_best_draw(post, obj, model, logp, ca, a.map_thin, fwd_fns)
    cB, dB, lpB = _camp_best_draw(post, obj, model, logp, cb, a.map_thin, fwd_fns)
    print(f"camp A rep chain {cA} draw {dA} logp {lpA:.1f}; "
          f"camp B rep chain {cB} draw {dB} logp {lpB:.1f}")

    actA, S_A, sigA = _extract_natural(post, obj, cA, dA)
    actB, S_B, sigB = _extract_natural(post, obj, cB, dB)
    P, _ = align(S_B, S_A)                         # relabel B into A's frame
    actB = {n: actB[n][P] for n in actB}
    S_B = S_B[P, :]

    nodes = list(actA.keys())
    ptA = _value_point(model, _natural_to_freervs(obj, actA, S_A, sigA), fwd_fns)
    ptB = _value_point(model, _natural_to_freervs(obj, actB, S_B, sigB), fwd_fns)
    keys = sorted(ptA.keys())
    vA, vB = _flat(ptA, keys), _flat(ptB, keys)
    d = vB - vA
    L = float(np.linalg.norm(d))                   # sampler-space arclength A->B
    dunit = d / (L + 1e-30)
    print(f"sampler-space distance A->B = {L:.3f}")

    rng = np.random.default_rng(a.seed)
    hvp = names = u_pow = None
    if a.stiff_eig:
        try:
            hvp, names = _hvp_setup(model)
            dim = int(sum(np.asarray(ptA[n]).size for n in names))
            u_pow = rng.standard_normal(dim)
            u_pow /= (np.linalg.norm(u_pow) + 1e-30)
        except Exception as e:                     # noqa: BLE001
            print("stiffest-eigenvalue setup failed, skipping:", e)
            hvp = None
    ts = np.linspace(0.0, 1.0, a.n_points)
    rows = []
    for t in ts:
        # straight line in sampler coordinates
        v = vA + t * d
        pt_s = _unflat(v, ptA, keys)
        lp_s = float(logp(pt_s))
        ll_s = float(lik(pt_s))

        # matched-label simplex path
        act_t = {n: (1 - t) * actA[n] + t * actB[n] for n in nodes}
        act_t = {n: np.clip(x, 1e-9, None) / np.clip(x, 1e-9, None).sum()
                 for n, x in act_t.items()}
        S_t = (1 - t) * S_A + t * S_B
        sig_t = (1 - t) * sigA + t * sigB
        pt_n = _value_point(model, _natural_to_freervs(obj, act_t, S_t, sig_t), fwd_fns)
        lp_n, ll_n, pr_n = float(logp(pt_n)), float(lik(pt_n)), float(pri(pt_n))
        rate_min = min(float((act_t[n] @ S_t).min()) for n in nodes)

        # curvature at the sampler-line point: on-path vs orthogonal, same stencil
        on_path = _curv(logp, v, dunit, a.eps, ptA, keys, lp_s)
        if a.n_orth > 0:
            try:
                orth = []
                for _ in range(a.n_orth):
                    r = rng.standard_normal(d.shape)
                    r -= (r @ dunit) * dunit
                    r /= (np.linalg.norm(r) + 1e-30)
                    orth.append(_curv(logp, v, r, a.eps, ptA, keys, lp_s))
                o_min, o_mean, o_max = float(np.min(orth)), float(np.mean(orth)), float(np.max(orth))
            except Exception as e:                 # noqa: BLE001
                o_min = o_mean = o_max = float("nan")
                if t == ts[0]:
                    print("orthogonal-curvature probe failed:", e)
        else:
            o_min = o_mean = o_max = float("nan")

        # exact stiffest curvature at the same point, by power iteration on the
        # Hessian
        stiff_eig = float("nan")
        if hvp is not None:
            iters = max(a.power_iters, 30) if t == ts[0] else a.power_iters
            try:
                stiff_eig, u_pow = _stiffest_eig(hvp, names, pt_s, u_pow, iters)
            except Exception as e:                 # noqa: BLE001
                if t == ts[0]:
                    print("stiffest-eigenvalue power iteration failed:", e)
                hvp = None

        rows.append({"t": float(t), "s": float(t * L),
                     "loglik": ll_n, "logp": lp_n, "logprior": pr_n,
                     "loglik_sampler": ll_s, "logp_sampler": lp_s,
                     "on_path_curv": on_path,
                     "orth_curv_min": o_min, "orth_curv_mean": o_mean,
                     "orth_curv_max": o_max, "stiff_eig": stiff_eig,
                     "rate_min": rate_min,
                     "finite": bool(np.isfinite(lp_n) and np.isfinite(ll_n))})

    df = pd.DataFrame(rows)
    s = df.s.values
    df["dloglik_ds"] = np.gradient(df.loglik.values, s)
    df["d2loglik_ds2"] = np.gradient(df.dloglik_ds.values, s)
    df["dlogp_ds"] = np.gradient(df.logp.values, s)
    df["d2logp_ds2"] = np.gradient(df.dlogp_ds.values, s)

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "camp_path_profile.csv", index=False)

    # scalar summary
    i0, i1 = 0, len(df) - 1
    ll_end = min(df.loglik.iloc[i0], df.loglik.iloc[i1])
    inside = df[(df.t > 0.02) & (df.t < 0.98)]
    hill_height = float(df.loglik.max() - ll_end)
    ll_end_s = min(df.loglik_sampler.iloc[i0], df.loglik_sampler.iloc[i1])
    sampler_barrier = float(ll_end_s - inside.loglik_sampler.min()) if len(inside) else float("nan")
    simplex_barrier = float(ll_end - inside.loglik.min()) if len(inside) else float("nan")
    on_path_med = float(np.nanmedian(np.abs(df.on_path_curv)))
    stiff = (float(-np.nanmin(df.orth_curv_min))        # magnitude of stiffest wall
             if df.orth_curv_min.notna().any() else float("nan"))
    condition = stiff / on_path_med if on_path_med > 0 else float("nan")
    adapted_step = _sample_stat(idata, "step_size")
    tree_depth = _sample_stat(idata, "tree_depth")
    implied_step = 2.0 / np.sqrt(stiff) if stiff > 0 else float("nan")
    steps_to_cross = L / adapted_step if adapted_step > 0 else float("nan")
    max_traj = 2.0 ** tree_depth if np.isfinite(tree_depth) else float("nan")

    # exact stiffest curvature from power iteration
    if "stiff_eig" in df and df.stiff_eig.notna().any():
        idx = int(df.stiff_eig.abs().idxmax())
        stiff_exact_signed = float(df.stiff_eig.iloc[idx])
        stiff_exact = abs(stiff_exact_signed)
        condition_exact = stiff_exact / on_path_med if on_path_med > 0 else float("nan")
        implied_exact = 2.0 / np.sqrt(stiff_exact) if stiff_exact > 0 else float("nan")
    else:
        stiff_exact_signed = condition_exact = implied_exact = float("nan")

    summary = pd.DataFrame([{
        "arclength": L, "hill_height": hill_height,
        "simplex_barrier": simplex_barrier, "sampler_barrier": sampler_barrier,
        "logit_inflation": sampler_barrier - simplex_barrier,
        "on_path_curv_med": on_path_med, "orth_curv_stiffest": -stiff,
        "condition_number": condition, "adapted_step": adapted_step,
        "implied_step": implied_step, "tree_depth": tree_depth,
        "steps_to_cross": steps_to_cross, "max_trajectory_steps": max_traj,
        "orth_curv_stiffest_exact": stiff_exact_signed,
        "condition_number_exact": condition_exact,
        "implied_step_exact": implied_exact,
    }])
    summary.to_csv(out / "camp_path_summary.csv", index=False)

    print(f"hill height above nearer camp     : {hill_height:+.1f} nats")
    print(f"simplex vs sampler loglik barrier : {simplex_barrier:.1f} / {sampler_barrier:.1f} "
          f"(logit inflation {sampler_barrier - simplex_barrier:+.1f})")
    print(f"on-path |curv| (median)           : {on_path_med:.3g}")
    print(f"stiffest orthogonal curvature     : {-stiff:.3g}  (condition {condition:.0f})")
    print(f"stiffest exact curvature (power)  : {stiff_exact_signed:.3g}  "
          f"(condition {condition_exact:.0f}, implied step {implied_exact:.4g})")
    print(f"adapted step {adapted_step:.4g} vs implied 2/sqrt(stiff) {implied_step:.4g}")
    print(f"steps to cross {steps_to_cross:.0f} vs max trajectory 2^depth {max_traj:.0f}")
    print(f"non-finite points {(~df.finite).sum()} / {len(df)}; "
          f"min rate on path {df.rate_min.min():.3g}")
    print(f"wrote camp_path_profile.csv and camp_path_summary.csv to {out}")


if __name__ == "__main__":
    main()