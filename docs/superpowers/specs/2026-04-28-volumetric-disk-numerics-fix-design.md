# Volumetric Disk Numerics Fix — Phase 1

**Date:** 2026-04-28
**Branch:** `fix/volumetric-ring`
**Related:** `2026-04-27-volumetric-disk-smoothing-design.md` (this spec is a follow-up that fixes test failures and refinement cap-binding exposed by that work)

## Summary

Three small, independent fixes to retire the three remaining test failures from the boundary-smoothing branch and (probably) eliminate refinement cap-binding. Iterative scope: land these changes, measure the residual Promptable-warning count and refinement convergence, then decide whether a follow-up adaptive-RK45 pass on `solve_column` is needed.

## Background

After the boundary-smoothing implementation (2026-04-27), three tests fail:

1. **`test_density_profile`** — at `r=10M, z=H, 3H` with `turbulence=0`, `density()` is non-monotonic in `z`.
2. **`test_density_smooth_across_zmax`** — at `r=6M, z=0.99·z_max`, `density()` is much larger than `1e-10·rho_mid`.
3. **`test_tau_midplane_near_target`** — integrated `kappa·rho·dz` at peak-flux radius is ~0 instead of `tau_mid=100`.

In addition, refinement caps at `n_r=4096, n_z=1024` on every disk construction, emitting 4–5 Promptable warnings that gate the CLI behind `--force` for every render.

Diagnostic work showed:

- **Failures 1 and 2** share a root cause: `interp_2d` (src/volumetric_disk.cpp:212) does per-column z-normalization (`z_abs / z_max_lut_[ri]`), then linearly blends the two columns radially. The per-column z-scaling correctly captures hydrostatic disks' self-similarity in `z`, but it amplifies any radial discontinuity in `z_max(r)`. There is exactly such a discontinuity at `i=1`: `validate_luts` reports `H jump 9.27 at i=1`. The discontinuity originates in `compute_plunging_region_decay`'s use of a Gaussian `taper(r)` with width `(r_isco - r_horizon) / 3`. With `r_min = r_horizon + 0.01M`, the first LUT bin sits at the steepest point of the taper's exponential tail (`taper(r_min) ~ 1e-4`), while the second bin is already in the bulk (`taper(r_1) ~ 0.4`). `H_lut_` and `z_max_lut_` inherit the discontinuity.
- **Failure 3** is a units mismatch in the test: `density()` returns geometric-scaled units (per its docstring), while the test integrates it as if it were CGS (g/cm³). The CGS accessor `density_cgs()` exists but isn't used.
- **Cap-binding** has two contributions. The H-jump at `i=1` causes `compare_columns` to report a large delta at that radius regardless of `n_r`, forcing radial refinement to keep doubling. Separately, the fixed-step RK4 in `solve_column` may oscillate near the photosphere boundary as ρ→0, contributing a second source of non-convergence. The first contribution is fixed by widening the taper (this spec); the second is left for a possible follow-up if measurement after Phase 1 still shows cap-binding.

## Goals

- All three failing tests pass.
- The `h_jump` Promptable warning from `validate_luts` no longer fires on healthy construction.
- Refinement caps are protective guardrails (not the operating point) for typical disks.

## Non-goals

- Adaptive-step ODE integration in `solve_column`. Deferred to Phase 2 if measurement shows it's needed.
- Replacing the per-column z-normalization in `interp_2d`. The current scheme is correct for smooth disks; we fix the data, not the interpolator.
- Changing the conceptual model of the plunging region. The Gaussian taper is still ad-hoc; this spec just adjusts its width.
- Investigating `normalize_density`'s peak-radius detection. If failure 3 doesn't pass after the units fix, that's a separate bug to investigate in a follow-up.

## Approach

Three independent changes:

1. **Widen the plunging-region taper** so its width spans the full plunging region instead of its inner third. The first LUT bin no longer sits in the steep tail; H_lut_ becomes smooth radially; interp_2d artifacts disappear.
2. **Expose the taper width as a tunable parameter** through `VolumetricParams`, the C API, and a CLI flag, with a default of `1.0` (full plunging region).
3. **Fix the tau test's units** by switching `density()` to `density_cgs()` and widening the clamps to match the opacity LUT's valid range.

## Components

### 1. Taper widening

In the `VolumetricDisk` constructor (`src/volumetric_disk.cpp`):

```cpp
// Before
taper_width_ = (r_isco_ - r_horizon_) / 3.0;

// After
taper_width_ = params_.plunging_taper_width_factor * (r_isco_ - r_horizon_);
```

The default `plunging_taper_width_factor = 1.0` makes the taper span the entire plunging region. Existing behavior corresponds to `factor = 1/3`, which is no longer the default — this is an intentional behavior change for volumetric renders.

### 2. Parameter plumbing

**`include/grrt/scene/volumetric_disk.h`** — add to `VolumetricParams`:

```cpp
double plunging_taper_width_factor = 1.0;  ///< Plunging-region taper width as fraction of (r_isco - r_horizon)
```

**`include/grrt/types.h`** — add to `GRRTParams` (C struct, no default initializer):

```c
double disk_plunging_taper_width_factor;  /* 0 = use VolumetricParams default (1.0) */
```

**`src/api.cpp`** — pass-through following the existing 0-default convention:

```cpp
if (params->disk_plunging_taper_width_factor > 0.0)
    vp.plunging_taper_width_factor = params->disk_plunging_taper_width_factor;
```

**`cli/main.cpp`** — flag parsing and help text:

```cpp
} else if (arg("--disk-plunging-taper-width")) {
    params.disk_plunging_taper_width_factor = std::stod(argv[++i]);
}
```

```cpp
std::println("  --disk-plunging-taper-width F  Plunging-region taper as fraction of (r_isco - r_horizon) (default: 1.0)");
```

Negative or zero values fall through to the default (consistent with the existing convention for other 0-default fields).

### 3. Tau test units fix

In `tests/test_volumetric.cpp::test_tau_midplane_near_target`:

- Replace `disk.density(r, 0.0, 0.0)` in the peak-radius scan with `disk.density_cgs(r, 0.0, 0.0)`.
- Replace both `disk.density(r, z_a, 0.0)` and `disk.density(r, z_b, 0.0)` in the integration loop with `density_cgs`.
- Change clamps from `[1e-30, 1e-3]` to `[1e-18, 1e-6]` to match the opacity LUT's valid range.

## Files modified

| File | Lines | Change |
|------|-------|--------|
| `include/grrt/scene/volumetric_disk.h` | +1 | New `VolumetricParams` field |
| `include/grrt/types.h` | +1 | New `GRRTParams` field |
| `src/volumetric_disk.cpp` | ~1 | Replace `/3.0` with parameterized factor |
| `src/api.cpp` | +2 | Pass-through with 0-default |
| `cli/main.cpp` | +3 | Flag parsing + help text |
| `tests/test_volumetric.cpp` | ~6 | Replace `density()` → `density_cgs()`, widen clamps |

Estimated total: ~15 lines changed across 6 files. One commit.

## Validation

After landing:

- `test_density_profile` passes — LUT is smooth radially, `interp_2d` no longer produces non-monotonic z behavior at r=10M.
- `test_density_smooth_across_zmax` passes — same root cause fixed.
- `test_tau_midplane_near_target` passes — units corrected. If it still fails, that's a real bug in `normalize_density`'s peak-radius detection; flag for a follow-up spec.
- `validate_luts`'s `h_jump` Promptable drops from "9.27 at i=1" to well below the 0.5 threshold, so the warning no longer fires on healthy construction.
- Refinement may converge at modest `n_r`/`n_z` (e.g., 512×128) without hitting caps. We measure to decide.

## Checkpoint decision point

After Phase 1 lands, run:

```bash
./build/Release/grrt-cli --metric kerr --spin 0.998 --observer-r 50 --observer-theta 80 \
    --disk-volumetric --mass-solar 10 --eddington-fraction 0.1 \
    --output post_phase1 --width 256 --height 256
```

Observe the construction log:

- **`Refinement done: n_r=X, n_z=Y`** — record values.
- **Promptable warning count** — if zero (no cap warnings, no h_jump), Phase 1 is sufficient and Phase 2 (RK45) is not needed.
- **If cap warnings remain**, examine the `compare_columns` deltas reported at the cap-bind point (may need to add a debug log temporarily). If the residual delta is dominated by oscillations in the last few z-bins of each column, RK45 is needed. If the delta is spread across z, the refinement metric itself may need tuning — out of this spec's scope.

The decision is binary and based on a single render's output. No instrumentation changes needed beyond what `validate_luts` and the existing refinement logs already produce.

## Error handling

All three changes are additive defaults. Existing callers see no behavior change unless `disk_plunging_taper_width_factor` is set explicitly.

The factor=1.0 default IS a behavior change for the volumetric path itself: existing renders will produce a wider taper. This is intentional. `visual_stellar.png` and similar previously-rendered volumetric disks will look slightly different (smoother near the inner edge, no flat plate-like inner cylinder). Documented as a known visual change in the commit message.

No new error paths. The factor is a multiplier; even pathological values (10×, 100×) just produce a very smooth, nearly-uniform H across the plunging region. The LUT remains well-defined.

## Testing

No new tests. The existing 3 failing tests are the validation gate. After the change:

- 3 currently-failing tests pass.
- The `h_jump` Promptable retires; the remaining 4 cap-related Promptables may also retire if refinement converges below cap.
- All other existing tests unchanged.

If `test_smoke_parameter_sweep` is re-enabled (currently commented out for runtime), all 7 cases should still pass and may run faster (since refinement converges earlier).

## Future work

- **Adaptive RK45 in `solve_column`** if Phase 1's measurement shows residual cap-binding from photosphere oscillations. Reuse the existing Dormand-Prince RK4(5) from `include/grrt/geodesic/rk4.h` (extracted into a generic adaptive-RK utility) or implement inline in `solve_column`.
- **`normalize_density` peak-radius reconciliation** if the tau test still fails after the units fix. The test's scan and `normalize_density`'s internal `peak_idx` may pick different radii.
- **Wider visual validation** — render a sweep of mass scales (1e-3 M_sun to 1e10 M_sun) at high resolution and inspect for residual artifacts before claiming the disk model is smooth.

## Migration notes

Existing volumetric renders will look different after this change (smoother near the inner edge). Users with reference images should re-render and update.

The C-API ABI is preserved: `disk_plunging_taper_width_factor` is a new appended field in `GRRTParams`. Callers that zero-initialize the struct (the documented convention) get the default behavior automatically.
