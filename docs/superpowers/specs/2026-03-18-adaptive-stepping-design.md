# Adaptive RK4 Step Doubling

## Goal

Replace the fixed geodesic step size with adaptive step doubling that automatically adjusts step size based on local error estimates. Improves both accuracy (smaller steps near photon sphere) and performance (larger steps far from the hole).

## Current State

The tracer uses `dλ = 0.005 * r` — a fixed heuristic. This over-steps near the photon sphere (causing the speckle artifacts we fixed by halving the step) and under-steps far away (wasting computation on straight-line propagation).

## Algorithm: Step Doubling

For each integration step:

1. From current state `S` with step size `dλ`:
   - Take one full RK4 step of size `dλ` → `S_full`
   - Take two half-steps of size `dλ/2` → `S_half`
2. Compute error: `err = max_i |S_full[i] - S_half[i]| / (|S_half[i]| + ε)` across the spatial position and momentum components (indices 1-3 for `r, θ, φ` position and 0-3 for momentum), with `ε = 1e-10`. The `t` component (position index 0) is excluded because it grows monotonically without bound and its relative error is always tiny — this is fine since `t` doesn't affect ray geometry.
3. Decision:
   - If `err > tolerance`: **reject** step. Shrink `dλ *= 0.5`. Retry from `S` (do not advance).
   - If `err < tolerance × 0.01`: **accept** `S_half`. Grow `dλ *= 2.0` for next step.
   - Otherwise: **accept** `S_half`. Keep `dλ` unchanged.
4. Clamp `dλ` to `[dλ_min, dλ_max]` after any resize:
   - `dλ_min = 1e-6` (prevents infinitesimal steps near singularities)
   - `dλ_max = 5.0 * r` (prevents overshooting at large radii)
5. **Minimum step guard**: If `dλ == dλ_min` and the step is still rejected, **force-accept** `S_half` anyway. This prevents infinite rejection loops near the horizon. The step is inaccurate but the ray is about to hit the horizon termination check anyway.
6. Return the accepted state and the recommended `dλ` for the next iteration.

The two-half-steps result `S_half` is always used because it's more accurate than `S_full` by a constant factor (same order, smaller error coefficient). This is NOT Richardson extrapolation — both are 4th-order, but `S_half` has ~1/16th the leading error term.

**Initial step size**: `dλ_0 = 0.01 * r_observer` — a conservative starting guess that adapts quickly.

## Step Counter Semantics

The tracer's `max_steps` limit counts only **accepted** steps, not rejected attempts. This ensures `max_steps` bounds the actual integration distance, not the number of retry attempts. A separate `max_rejections = 100` per-step guard prevents pathological cases where no step size is small enough:

```cpp
for (int accepted = 0; accepted < max_steps_; ) {
    auto result = integrator_.adaptive_step(metric_, state, dlambda, tolerance_);
    if (result.accepted) {
        state = result.state;
        dlambda = result.next_dlambda;
        ++accepted;
        // ... check termination conditions ...
    }
    // adaptive_step handles retry internally, always returns accepted=true
    // (force-accepts at dλ_min)
}
```

Actually, simpler: `adaptive_step()` handles the retry loop internally (shrinking until accepted or hitting `dλ_min`), and always returns an accepted state. The tracer loop stays clean — every call to `adaptive_step()` advances the state.

## New Types

In `integrator.h`:
```cpp
struct AdaptiveResult {
    GeodesicState state;    // Accepted new state (always valid)
    double next_dlambda;    // Recommended step size for next iteration
};
```

## Modified Files

| File | Change |
|------|--------|
| `include/grrt/geodesic/integrator.h` | Add `AdaptiveResult` struct |
| `include/grrt/geodesic/rk4.h` | Add `AdaptiveResult adaptive_step(const Metric&, const GeodesicState&, double dlambda, double tolerance) const` |
| `src/rk4.cpp` | Implement step doubling with error control and retry loop |
| `include/grrt/geodesic/geodesic_tracer.h` | Add `tolerance_` member. New constructor: `GeodesicTracer(const Metric&, const Integrator&, double observer_r, int max_steps, double r_escape, double tolerance)` |
| `src/geodesic_tracer.cpp` | Use `adaptive_step()` in trace loop. Track `dλ` across iterations. Remove fixed `0.005 * r`. Initial `dλ = 0.01 * observer_r_`. |
| `src/api.cpp` | Pass `integrator_tolerance` from params to tracer (default `1e-8` when zero-initialized) |

## Parameters

- `integrator_tolerance`: exposed in `GRRTParams` (already exists in `types.h`). Default `1e-8`.
- `dλ_min = 1e-6`: hardcoded, prevents degenerate tiny steps
- `dλ_max = 5.0 * r`: hardcoded, scales with distance

## Validation

- Hamiltonian constraint `H = ½ g^μν p_μ p_ν` should stay below `~1e-10` throughout integration
- Schwarzschild shadow radius should match analytical prediction
- Kerr shadow shape should be unchanged
- Render should be faster (fewer total RK4 evaluations for far-field rays)
- No speckle artifacts near the photon sphere
