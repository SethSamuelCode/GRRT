# Future optimization: Romberg trajectory reuse via interpolation

**Status:** noted, not implemented. Ranked low-priority.
**Origin:** brainstorm for `2026-05-01-cliff-aware-raymarch-design.md` on 2026-05-01.

## Idea

Inside a single Romberg evaluation in `romberg_step`, we today do **three** geodesic integrations:

1. One full step of length `ds` (used to compute `Δτ_full`).
2. Two half-steps of length `ds/2` (used to compute `Δτ_half`).

The expensive part is the geodesic stepping — RK4 in Kerr requires several metric evaluations per step. Doing 3× that work per accepted Romberg step roughly triples the geodesic cost.

The optimization: instead of three independent integrations, do **one** integration of the full step and ask the integrator for the trajectory at the midpoint by interpolation. Then compute the half-step `Δτ` estimate by sampling at this interpolated midpoint plus the endpoints we already have.

## Why it's not free

- The interpolated midpoint isn't *exactly* on the half-step's true geodesic — it's the position the full-step's RK4 trajectory passes through at parameter `λ + ds/2`. Dormand-Prince RK45 has a natural dense-output (CIDR) interpolant for this; standard fixed-step RK4 does not, but we can construct a quartic-Hermite interpolant from the start and end states and their derivatives.
- The error in `Δτ_half` thus picks up an additional small term from the interpolation. The Romberg comparison `|Δτ_full − Δτ_half|` then conflates "integration error in τ" with "interpolation error in geodesic trajectory."
- For the cliff problem we're solving, this distinction is small — the trajectory is smooth on the scale of `ds`. But it does mean **the error estimator is no longer purely about τ-integration**, so the bound on per-step τ error becomes "at most `tol` plus a small geodesic-interpolation contribution."

## Estimated gain

- Geodesic stepping is the dominant per-step cost (~80% of work).
- Reducing it from 3 substeps to 1 substep + interpolation could give a **~2.5× speedup of the helper's per-step cost.**
- Per-render speedup depends on whether the helper is the bottleneck. If raymarch dominates a render (typical for thick disks or low geodesic-tolerance settings), this could be **~1.5–2× whole-render speedup.**

## Why it was *not* taken

For the cliff-aware-raymarch spec, we chose to keep the error estimator clean and physical:
- The user-facing tolerance directly translates to per-step error in τ, full stop.
- No conflation with geodesic-trajectory interpolation accuracy.
- Makes the design and the convergence guarantees easy to reason about.

If profiling shows geodesic stepping inside Romberg is the bottleneck, we can revisit with a proper validation pass against the honest 3-step reference.

## Sketch of a validation plan (when revisited)

1. Render reference with the honest 3-step Romberg.
2. Implement trajectory-reuse path; render the same scene.
3. Compute per-pixel `|ΔL|` (linear-light, before tone mapping).
4. Quality gate: `max-pixel |ΔL| < 0.5%`. Verify especially at the photosphere edge where trajectory curvature is highest.
5. Test the dense-output interpolant choice — quartic Hermite vs. RK45's natural CIDR — and pick whichever passes the quality gate with lower cost.

## How to use this doc

Pick this back up if and only if profiling shows geodesic stepping inside `romberg_step` is the bottleneck. Otherwise leave it.
