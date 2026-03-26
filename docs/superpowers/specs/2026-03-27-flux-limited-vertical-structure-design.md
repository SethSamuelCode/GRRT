# Flux-Limited Vertical Structure & Optical Depth Surface — Design Spec

**Supersedes:** Sections 1.1 (vertical structure), 1.6 (volume bounds), and 5.1 (entry detection) of `2026-03-19-volumetric-accretion-disk-design.md`. All other sections of the existing spec remain in effect.

## Problem

The volumetric disk has a hard density cutoff at z = 3H. The hydrostatic equilibrium solver includes radiation pressure in a way that makes the density profile nearly flat (99.9% of midplane value at the boundary). This creates a visible hard edge at the disk boundary — a blocky line with a shadow-like band.

## Goal

Replace the hard geometric boundary with a physically self-consistent vertical structure where:
1. Density falls off smoothly through the disk atmosphere
2. The visible "surface" emerges from optical depth, not an arbitrary z cutoff
3. The model handles all regimes (radiation-dominated, gas-dominated, optically thin) without special-case logic

## Approach: Flux-Limited Diffusion (Levermore-Pomraning)

### Why Not Two-Zone

A simpler approach would split the atmosphere at the photosphere (τ=1): full radiation pressure below, gas-only above. This works for common cases but has edge cases:
- Grazing rays see a density "kink" at the zone boundary
- Optically thin disks have no well-defined photosphere, requiring fallback logic
- Future features (winds, coronae) would need additional zones

### Flux Limiter Physics

The radiation pressure contribution is scaled by a flux limiter that smoothly transitions between the diffusion limit (optically thick) and free-streaming limit (optically thin).

**Radiation parameter:**
```
R = |∇E_rad| / (κ_R · ρ · E_rad)
```
where `E_rad = a_rad · T⁴` is radiation energy density, `κ_R` is the Rosseland mean opacity from the existing `opacity_luts_.lookup_kappa_ross(rho, T)` LUT.

**Levermore-Pomraning flux limiter:**
```
λ(R) = (2 + R) / (6 + 3R + R²)
```

**Eddington factor** (the actual radiation pressure coefficient):
```
f(R) = λ(R) + λ(R)² · R²
```

Limits:
- Optically thick (R → 0): f → 1/3, isotropic radiation pressure P_rad = E_rad/3
- Optically thin (R → ∞): f → 1, directed radiation momentum flux P_rad = E_rad

**Effective radiation pressure:**
```
P_rad = f(R) · E_rad
```

**Total pressure:**
```
P_total = P_gas + P_rad = ρkT/(μmp) + f(R) · a_rad · T⁴
```

**Hydrostatic equation** (in dP/dz form, preserving the temperature gradient):
```
dP_total/dz = -ρ · Ωz² · z
```

Expanding:
```
(kT/μmp) · dρ/dz + ρ · d(kT/μmp)/dz + d(f · E_rad)/dz = -ρ · Ωz² · z
```

Solving for dρ/dz:
```
dρ/dz = [-ρ · Ωz² · z - ρ · d(kT/μmp)/dz - d(f · E_rad)/dz] / (kT/μmp)
```

The `d(kT/μmp)/dz` and `d(f · E_rad)/dz` terms are computed from the previous iteration's T(z), μ(z), and ρ(z) profiles via central finite differences.

**Numerical safety:** When ρ is near zero (atmosphere edge), R can blow up. Guard: if `κ_R · ρ · E_rad < 1e-30`, set f = 1.0 directly (free-streaming limit). Clamp ρ to a floor of 1e-15 × ρ_midplane at each RK4 substep to prevent negative density.

## Changes

### 1. Extended Vertical Grid

**Current:** z ∈ [0, 3H], 64 uniform bins per radial column.

**New:** z ∈ [0, z_max(r)], 128 uniform bins per radial column. `z_max(r)` is determined dynamically: the solver marches upward and stops when density drops below 1e-10 of the midplane value. Capped at 20H to prevent runaway in extreme radiation-pressure-dominated cases.

**Initial z_max:** Starts at 3H (the old value) for the first iteration. Extended if the density hasn't fallen sufficiently by the grid edge.

A new 1D LUT `z_max_lut_[n_r]` stores the atmosphere extent per radius, alongside the existing `H_lut_`. Interpolated for arbitrary r the same way `H_lut_` is (via `interp_radial`).

The 2D LUT indexing changes from `z_abs / (3H) → [0, n_z-1]` to `z_abs / z_max(r) → [0, n_z-1]`.

**Grid spacing:** Uniform within each column. Non-uniform spacing could improve photosphere resolution but adds interpolation complexity. 128 uniform bins provides adequate resolution (~10-15 bins across the photosphere transition).

### 2. Modified Hydrostatic Solver

**Location:** `VolumetricDisk::compute_vertical_profiles()` in `src/volumetric_disk.cpp`

**Current algorithm per radial bin:**
1. Compute τ(z) from current ρ(z) (integrate inward from surface)
2. Compute T(z) from Eddington relation: T⁴ = (3/4) T_eff⁴ (τ + 2/3)
3. Integrate ρ(z) via forward Euler with c_eff² = P_gas/ρ + P_rad/ρ
4. Repeat 4 times

**New algorithm per radial bin:**
1. Compute τ(z) from current ρ(z) (integrate inward from surface)
2. Compute T(z) from Eddington relation (unchanged)
3. Compute μ(z) from opacity LUT: `opacity_luts_.lookup_mu(rho_cgs, T)`
4. Compute E_rad(z) = a_rad · T(z)⁴ at each z
5. Compute dE_rad/dz via finite differences on T(z) profile. Boundary treatment: at z=0 (midplane symmetry), dE_rad/dz = 0. At z=z_max (last bin), use one-sided backward difference. Interior bins use central differences.
6. Compute R(z) = |dE_rad/dz| / (κ_R(z) · ρ(z) · E_rad(z)) at each z, where κ_R is from `opacity_luts_.lookup_kappa_ross(rho_cgs, T)`. Guard: if denominator < 1e-30, set R = 1e30 (free-streaming).
7. Compute λ(z) = (2 + R) / (6 + 3R + R²) and f(z) = λ + λ²R²
8. Compute d(f · E_rad)/dz and d(kT/μmp)/dz via finite differences (same boundary treatment as step 5)
9. Integrate ρ(z) via RK4 using the full hydrostatic equation. Clamp ρ ≥ 1e-15 × ρ_midplane at each substep.
10. If density at the grid edge is still above 1e-10 × ρ_midplane, increase z_max by 1H (up to 20H cap), re-allocate the grid, restart the z-integration from the midplane with the new grid. This counts as one of the 8 outer iterations (the counter does not reset).
11. Repeat the outer loop up to 8 times total. Early exit if max |Δρ/ρ| < 0.001 between consecutive iterations.

**Convergence failure:** If the solver does not converge within 8 iterations, accept the last profile and emit a warning: `"[VolumetricDisk] WARNING: vertical profile did not converge at r_idx=%d (max delta=%.2e)"`. Do not abort — the profile is still usable, just not fully self-consistent.

**Integration method:** RK4 replaces forward Euler for stability over 128 bins.

### 3. Remove Hard Boundary

**`inside_volume(r, z)`:** Changes from `|z| < 3H` to `|z| < z_max(r)`, where `z_max(r)` is interpolated from `z_max_lut_` via `interp_radial`.

**`density(r, z, phi)`:** Guard changes from `inside_volume()` returning 0 to checking against `z_max(r)`. Since the LUT extends to where density is genuinely negligible (1e-10 of midplane), this is physically meaningful.

**`temperature(r, z)`:** Same change as density.

**`interp_2d(lut, r, z_abs)`:** The z normalization changes from `z_abs / (3H)` to `z_abs / z_max(r)`. The function needs access to `z_max_lut_` — either passed as a parameter or accessed through the class member.

### 4. Raymarcher Updates

**Location:** `GeodesicTracer::raymarch_volumetric()` in `src/geodesic_tracer.cpp`

**Entry detection:** The `near_disk` trigger changes from `|z| < 6H` to `|z| < z_max(r) + H`. One scale height of margin beyond the atmosphere. `z_max(r)` is obtained from a new public accessor `VolumetricDisk::z_max(r)` which interpolates `z_max_lut_`. This is tighter for thin outer regions and wider for puffy inner regions.

**Exit condition:** Changes from `been_inside && |z| > 6H` to `been_inside && |z| > z_max(r) + H`.

**Entry trigger in main trace loop:** Same change to the `near_disk` check in `GeodesicTracer::trace()`.

**Step size logic:** No changes needed. The existing adaptive step control (optical-depth-based `ds_tau` and geometric `ds_geo`) already handles varying density naturally. The coarse-to-fine transition outside the volume uses `|z|`-proportional steps, which still works with the new boundary.

### 5. Density Normalization Update

**Location:** `VolumetricDisk::normalize_density()` in `src/volumetric_disk.cpp`

The column integral changes from integrating over `[0, 3H]` to `[0, z_max(r)]` at the peak-flux radius. Since the atmosphere adds column density, `rho_scale_` will decrease slightly to maintain the target `tau_mid`.

Noise scale calculation (`noise_scale_ = 2 * H_at_peak`) unchanged — based on pressure scale height, not atmosphere extent.

**Note on per-radius accuracy:** The normalization uses only the peak-flux radius, so column optical depth at other radii may deviate from `tau_mid`. This was true before (with the 3H grid), but z_max variation is now larger (potentially 3H to 20H across radii). This is acceptable — `tau_mid` is a target at peak, not a global constraint. The radial variation in optical depth is physical (inner disk is hotter and more optically thick than outer disk).

### 6. CUDA Backend

Not updated. Left with existing hard-boundary behavior. CPU is faster for this double-precision workload (measured 8x faster on RTX 2080). CUDA can be synced later if a GPU with better FP64 throughput becomes available.

**Known divergence:** The CPU backend will produce visually different results from CUDA at the disk edges. The CUDA backend continues to use the 3H hard cutoff with 64 z-bins. This is a deliberate trade-off — the existing validation requirement "CUDA output must match CPU output within floating-point tolerance" is suspended for volumetric disk rendering. Specifically: any pixel where the CPU volumetric density at the ray's disk-crossing point differs from what the CUDA 3H-cutoff model would produce is exempt from the tolerance check. In practice, this means all volumetric disk pixels are exempt until the CUDA backend is synced.

**CUDA build compatibility:** The `n_z_` member default changes from 64 to 128 in the header, but the CUDA backend passes its own `vol_n_z` from `RenderParams` which is set during CUDA LUT upload. The CUDA path will continue to use 64 z-bins (the LUT size doesn't change). No shared header changes should break the CUDA build.

### 7. Mass & Eddington Fraction Parameter (Independent Feature)

**Motivation:** Currently `peak_temperature` is set manually. In reality, disk temperature depends on black hole mass and accretion rate:

```
T_peak ∝ (M / M_sun)^(-1/4) · (M_dot / M_dot_edd)^(1/4) · η(a)^(1/4)
```

where η(a) is the radiative efficiency, which depends on spin: η ≈ 0.057 for Schwarzschild (a=0), up to η ≈ 0.42 for maximal Kerr (a=1). The full formula:

```
T_peak = 5.0 × 10⁷ × η(a)^(1/4) × (M / M_sun)^(-1/4) × (M_dot / M_dot_edd)^(1/4)  [K]
```

The radiative efficiency is `η = 1 - E_isco` where `E_isco` is the specific energy at the ISCO, already computed by the metric code.

This means "large mass + large disk" looks identical to "small mass + small disk" because the code works in dimensionless M=1 units and temperature is independent of mass.

**Change:** Add two new optional CLI parameters:
- `--mass-solar <M>` — black hole mass in solar masses
- `--eddington-fraction <f>` — accretion rate as fraction of Eddington (0 to ~1)

When both are provided, `peak_temperature` is computed from the formula above using the spin-dependent efficiency. The existing `--disk-temperature` flag remains as a manual override (takes precedence if provided).

**Parameter structs:** Add `double mass_solar` and `double eddington_fraction` to `GRRTParams`. Default to 0 (meaning: use `disk_temperature` directly, current behavior).

**Resolution logic** (in CLI or API layer, before disk construction):
```
if (mass_solar > 0 && eddington_fraction > 0) {
    double eta = 1.0 - E_isco(spin);  // radiative efficiency from metric
    peak_temperature = 5e7 * pow(eta, 0.25) * pow(mass_solar, -0.25) * pow(eddington_fraction, 0.25);
}
```

This is independent of the vertical structure work and can be implemented and tested separately.

## Files Modified

| File | Change |
|------|--------|
| `src/volumetric_disk.cpp` | Hydrostatic solver, density normalization, inside_volume, interp_2d |
| `include/grrt/scene/volumetric_disk.h` | z_max_lut_, n_z_ default (128), z_max() accessor, inside_volume |
| `src/geodesic_tracer.cpp` | Entry detection, exit condition (z_max instead of 6H) |
| `include/grrt/types.h` | mass_solar, eddington_fraction fields |
| `cli/main.cpp` | New CLI flags |
| `src/api.cpp` | Temperature resolution logic |

## Files NOT Modified

| File | Reason |
|------|--------|
| `cuda/cuda_volumetric_disk.h` | CUDA left as-is (see Section 6) |
| `cuda/cuda_render.cu` | CUDA left as-is |
| `cuda/cuda_backend.cu` | CUDA left as-is |

## Performance Impact

**Startup (one-time):** Hydrostatic solver ~8x more work (RK4, 128 bins, 8 iterations). From ~100ms to ~800ms. Negligible vs render time.

**Per-ray:** Raymarcher steps through larger atmosphere volume. Typical views ~1.5-2x slower. Edge-on views up to 3x. High-inclination views of radiation-dominated inner disks (where z_max may reach 10-20H) could be slower still.

**Memory:** 2D LUTs double (64→128 z-bins). ~500KB → ~1MB. Plus z_max_lut_ (~4KB). Negligible.

## Validation

- Density at z_max(r) should be < 1e-10 of midplane at every radius
- Hydrostatic solver converges within 8 iterations (check convergence metric)
- Disk edge should be visually smooth at 1024×1024 — no hard lines or shadow bands
- Render of optically thin disk (low tau_mid) should have no artifacts
- Column optical depth at peak radius should still equal tau_mid after normalization
- Flux limiter limits: f → 1/3 at midplane (optically thick), f → 1 at atmosphere edge (optically thin)
- Mass/Eddington: T_peak for 10 M_sun at 10% Eddington with a=0 ≈ 16,000 K; with a=0.998 ≈ 26,000 K
