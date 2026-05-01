# Future optimization: τ-scaled raymarch tolerance

**Status:** noted, not implemented. Ranked low-priority.
**Origin:** brainstorm for `2026-05-01-cliff-aware-raymarch-design.md` on 2026-05-01.

## Idea

Inside `raymarch_volumetric`, the per-step τ tolerance is currently constant (`tol = 1e-2` from `--raymarch-tol`). Performance could improve by scaling it with current transmission:

```
eff_tol = tol / max(T_max, T_min_floor)
```

where `T_max` is the maximum across the active channels and `T_min_floor ≈ 1e-3` prevents runaway.

## Why it should help

Per-step error in observed intensity J is roughly `T · S · d(dτ)`. With constant τ-tolerance, this shrinks deep in the disk where T is small — meaning the fine-grained τ accuracy there is wasted (it's accurate to a fraction of an already-tiny number).

Scaled tolerance keeps the error in J roughly constant per step, so it lets the integrator take larger steps in the deep core (τ > 1) where T·tol underutilizes the tolerance budget.

## Estimated gain

~1.3×–2× faster overall, depending on disk geometry. Bigger savings for thin/dense disks where the core is a large fraction of the integration; less for puffy disks where atmosphere/cliff dominates.

## Why it was *not* taken

For the cliff-aware-raymarch spec we chose correctness-first. Constant tolerance has a clean theoretical bound (`total J error ≈ S·tol`); scaled tolerance's bound depends on number of steps, which depends on the disk profile. Three things would need empirical validation before adopting:

1. **Non-smooth regions:** gradient discontinuities and opacity edges may produce visible artifacts that constant tolerance avoids.
2. **Per-channel T mismatch:** in the spectral path, one channel can attenuate before another (especially near opacity edges) — picking which T to scale by is non-obvious.
3. **`T_min_floor` discontinuity:** the floor introduces a step where eff_tol stops growing, which itself could cause a band-like artifact at that depth.

None of these are showstoppers, but each demands a validation render against a constant-tolerance reference before shipping.

## Sketch of a validation plan (when revisited)

1. Render a representative scene with constant `tol = 1e-2` (reference).
2. Render the same scene with scaled tol and `T_min_floor ∈ {1e-2, 1e-3, 1e-4}`.
3. Compute per-pixel `|ΔL|` and `max-pixel ΔL` (linear-light, before tone mapping).
4. Quality gate: `max-pixel |ΔL| < 0.5%` and `mean |ΔL| < 0.05%`. If passed, accept.
5. Also test at an absorption edge with a spectral render (FITS), looking for edge-induced banding.

## How to use this doc

Pick this back up if and only if profiling shows the deep-core integration is the bottleneck *and* the user asks for more speed. Otherwise leave it.
