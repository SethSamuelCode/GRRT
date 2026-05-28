# To fix: null-constraint (Hamiltonian) monitoring + re-projection

**Date:** 2026-05-28
**Branch:** `claude/nice-knuth-4uLAq`
**Status:** to fix — no spec yet. Pair with the RGB/spectral raymarch unification.
**Priority:** low-effort insurance; promote if photon-ring / shadow-edge artifacts appear.

---

## Problem

In the Hamiltonian formulation the super-Hamiltonian is

```
H = ½ gᵘᵛ p_μ p_ν
```

For a photon this must be **exactly 0** — the null / mass-shell condition. It is
both a constraint on initial data and a constant of motion (`dH/dλ = 0`
analytically under Hamilton's equations).

We currently conserve two of the three relevant invariants *exactly*:
`E = −p_t` and `L = p_φ`, because `dp_t = dp_φ = 0` in
`Kerr::compute_derivatives` (`src/kerr.cpp:210-213`). The null condition is the
third, and unlike E and L it is **not** pinned: it couples the evolving
components `p_r`, `p_θ` to the fixed `E, L` through the metric, and nothing in
the integrator forces them back onto the null shell.

## Why it drifts numerically

DP45 controls the **local truncation error of the trajectory** — the embedded
4th/5th-order difference in `RK4::step_kerr_rkdp45`
(`include/grrt/geodesic/rk4.h:149-177`). That is not the same as controlling
`H`. `H` is a nonlinear function of the state, so per-step error leaks into it
and accumulates.

- Ray that escapes after modest deflection: drift negligible, never noticed.
- Ray that **loops the hole near the unstable photon orbit** (high spin,
  edge-on disk, pixels near the shadow boundary): long, sensitive path, `H`
  wanders from 0. A non-null "photon" bends slightly wrong **and contaminates
  the redshift factor `p·u`** that drives `nu_emit` and Doppler beaming in the
  raymarch.

CLAUDE.md lists `H < 1e-10` as a validation invariant, but the only check that
exists is in the spectral raymarch, **under `#ifndef NDEBUG`**
(`src/geodesic_tracer.cpp:658-671`), and it only warns. `trace()` and
`trace_debug()` never look at it.

## Fix — part 1: monitor (do this first, zero behavior change)

Add a helper (the inverse metric is already available via `Kerr::g_upper`, and
computed analytically inside `compute_derivatives`):

```cpp
double null_hamiltonian(const Kerr& g, const GeodesicState& s) {
    Matrix4 gu = g.g_upper(s.position);
    double H = 0.0;
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b)
            H += gu.m[a][b] * s.momentum[a] * s.momentum[b];
    return 0.5 * H;
}
```

Report it dimensionless — divide by `E²` (or the largest `gᵘᵛ p_μ p_ν` term) so
the `1e-10` threshold is meaningful across radii. Track the max along each ray
as a diagnostic. This tells us *whether* drift is actually a problem in our
scenes before touching the trajectory.

## Fix — part 2: re-projection (safety net, only if monitoring shows drift)

Snap the momentum back onto the null cone. `E` and `L` are fixed, so the (t,φ)
block is fixed:

```
gᵗᵗp_t² + 2gᵗᵠp_t p_φ + gᵠᵠp_φ²  +  gʳʳp_r² + gᶿᶿp_θ²  =  0
   └────────── fixed (call it C) ──────────┘     └── adjustable ──┘
```

One equation, two unknowns (`p_r`, `p_θ`) — can't fix both uniquely. Keep their
*direction* in the r-θ plane and rescale the magnitude by a common factor:

```cpp
double num = -C;                                    // = -(t,φ block)
double den = gu.m[1][1]*pr*pr + gu.m[2][2]*pθ*pθ;   // current spatial part
double s = std::sqrt(num / den);                    // guard den > 0 and num/den > 0
pr *= s;  pθ *= s;
```

This preserves the photon's direction of travel in the (r,θ) subspace and only
corrects the magnitude the integrator let slip.

## Tradeoffs / wiring

- **Re-project sparingly** — every N steps, or only when `|H|` exceeds a
  threshold. Not every step: constant re-projection perturbs the adaptive step
  controller and masks the real cause (a too-large step).
- Cost: one `g_upper` call; negligible next to the step's 7 derivative evals.
- Heavy re-projection is a *symptom* of too-loose tolerance. Order of operations:
  monitor → tighten tolerance if needed → project as backstop.
- Heavyweight alternative (symplectic / constraint-preserving integrator) is a
  much bigger change; not worth it unless drift proves chronic.

## Pairs with the RGB/spectral unification

Once `trace` / `trace_debug` / `trace_spectral` and the two raymarch routines
share one core, put **both** the monitor and the optional re-projection in that
shared core. The spectral path's existing `#ifndef NDEBUG` H-check
(`src/geodesic_tracer.cpp:658-671`) is the seed; promoting it to the shared core
gives both color outputs the same guarantee instead of just the debug spectral
build. Treat #2 (unification) and this as one piece of work.

## Suggested first step

Add the monitor unconditionally, run a high-spin edge-on render, and only enable
re-projection if `max|H|` actually breaches `1e-10`.

## References

- Hamiltonian derivatives + exact E,L conservation: `src/kerr.cpp:129-216`
- DP45 step + error estimate: `include/grrt/geodesic/rk4.h:76-180`
- Existing debug-only H check: `src/geodesic_tracer.cpp:658-671`
- Inverse metric: `Kerr::g_upper` (`src/kerr.cpp:42-44`)
- Pipeline review (item 4): `/root/.claude/plans/can-you-review-the-eager-tide.md`
