# Disk vertical-structure density normalization fix — design

**Date:** 2026-05-29
**Branch:** fix/volumetric-ring
**Status:** approved (brainstorming) — pending spec review → implementation plan
**Approach:** B (fix the existing grey radiative-equilibrium column solver). Approach A (full coupled-ODE BVP atmosphere with viscous energy generation) is documented as a follow-up in §9.

## 1. Problem

The volumetric disk renders with horizontal **banding** (edge-on) and **fireflies** (face-on). After the uniform-fine raymarch fix (committed), the banding metric went *up* (0.369 → 0.570) and the edge-on render showed coherent concentric bright/dark arcs.

Debug-pixel probing (env-gated `GRRT_RM_LOG`) of two adjacent rays crossing the disk at r≈8.3–8.5 found densities **18 orders of magnitude apart**: a "bright" ray sampled ρ ≈ 9.5e5 g/cm³ at the midplane; an adjacent "dark" ray sampled ρ ≈ 1.6e-12 just off the midplane (z ≈ 0.006 ≈ 0.1·z_max).

A LUT dump (`dump-disk-lut` tool, `disk_lut_dump.csv`) confirmed the cause: the **vertical density profile is collapsed to a delta-spike at z=0** at essentially every radius. `prof@0.1·z_max = prof@0.5·z_max = prof@top = 1.0e-18` (RHO_FLOOR) for **4056 of 4096** radial columns, while the midplane `ρ_norm[0] = 1.0`. The disk is a razor-thin density sheet, not a volume. The fine raymarch faithfully *resolves* that spike → near-binary hit/miss between adjacent rays → banding. (The earlier coarse stepper blurred it.)

So the banding is a **disk-construction bug**, not a raymarch bug. The raymarch fix is correct and stays.

## 2. Root cause (systematic-debugging, instrumented)

`density(r,z,φ) = ρ_mid(r) · ρ_norm(r,|z|) · ρ_scale · taper(r) · exp(turbulence)`. The radial midplane density `ρ_mid·ρ_scale` is smooth and correct (~1.46e6 at r=8.26, from the dump). The collapse is entirely in the **normalized vertical profile** `ρ_norm(r,|z|)`, produced by `VolumetricDisk::solve_column` (`src/volumetric_disk.cpp:597`).

`solve_column` integrates the vertical hydrostatic ODE (gas + flux-limited radiation pressure) outward from the midplane:

```
dρ/dz = [ −ρ·Ωz²·z  −  ρ·d(c_s²)/dz  −  d(f·E_rad)/dz / ρ_cgs_ref ] / c_s²      (volumetric_disk.cpp:739)
```

Env-gated term instrumentation (`GRRT_COL_LOG`) at r≈8.26, first integration step, shows:

```
[COL] z≈0  rho=1.0  grav≈-4e-5  gas≈+5e-32  rad≈-3.83  cs2g=4.75e-7  →  dρ/dz ≈ -8e6
```

The **radiation-pressure term (`rad ≈ −3.83`) dominates gravity (~−4e-5) by ~10⁵×** and is large-negative, so ρ crashes from 1.0 to RHO_FLOOR (1e-18) in a single step; the integrator then crawls at its step floor and the stored column is a spike.

**Why the radiation term is ~12 orders too large:** it is divided by `ρ_cgs_ref` (`volumetric_disk.cpp:635-636`), the assumed absolute density scale of the column:

```cpp
const double kappa_ref_total = std::max(kR_ref + kE_ref, 1.0);                       // line 634
const double rho_cgs_ref = std::clamp(tau_mid / (kappa_ref_total * 3.0*H),
                                      1e-18, 1e-6);                                   // line 635-636
```

`ρ_cgs_ref` is **clamped to a 1e-6 g/cm³ ceiling**, while the disk's true midplane density is ~1e6 — so `ρ_cgs_ref` is ~12 orders too small, inflating the radiation-pressure term ~1e12× → a spurious "super-Eddington" column that cannot support itself → collapse.

**Three coupled defects produce this:**

1. **`ρ_cgs_ref` clamp ceiling (1e-6) and κ floor (`max(...,1.0)`) are non-physical.** They pin the assumed density 12 orders below reality. The true value at T~3e6 K, ρ~1e6 makes the disk strongly **gas-pressure-dominated** (P_gas/P_rad ~ 1e9), so radiation pressure should be negligible here — not dominant. The κ floor (≥1.0) and the placeholder lookup density (`1e-10`, line 630-633) are likewise non-physical: the opacity floor of ionized gas is electron scattering κ_es ≈ 0.34 cm²/g, and κ should be evaluated at the column's actual density.

2. **The opacity LUT only covers ρ ∈ [1e-18, 1e-6] g/cm³** (`build_opacity_luts(1e-18, 1e-6, ...)`, `volumetric_disk.cpp:68`). The disk's real densities (~1e6) lie **above** the table's ceiling, so every opacity lookup at true density clamps to the table edge. This is *why* the 1e-6 clamps exist — they were sized to the table, not to the physics.

3. **The density normalization is self-referential and inconsistent.** `normalize_density` (`volumetric_disk.cpp:923`) computes the final `ρ_scale` from `τ_mid = κ·ρ_scale·peak_ρ·col_integral`, where `col_integral = ∫ρ_norm dz`. Because the profile is a collapsed spike, `col_integral` is tiny (~one bin), so `ρ_scale` is forced *huge* to still hit τ_mid → the fake 1.18e7 midplane density reported at construction. `solve_column` (which needs an absolute density to evaluate radiation pressure and opacity) and `normalize_density` (which sets the absolute density) use **two different, mutually inconsistent** density scales, both clamped to the same 1e-6 table ceiling.

These are coupled: (3) can't be solved without (1) and (2), and (1)'s collapse corrupts (3)'s `col_integral`.

## 3. Goal & constraints

- **Un-collapse the vertical profile**: `ρ_norm(r,z)` must be a smooth, physical, monotone-decreasing-in-|z| profile (gas-pressure Gaussian-like in the gas-dominated region), not a delta spike. Concretely: for a default disk, the dumped `prof@0.1·z_max` and `prof@0.5·z_max` must be O(0.1–1), not RHO_FLOOR, across the orbiting region.
- **Most-physically-derived** (user requirement): no non-physical pins. Opacity floored at electron scattering; density/opacity/τ normalization solved self-consistently; radiation pressure evaluated at the true local density.
- **Banding eliminated**: the edge-on render shows no concentric banding; the banding metric drops below the 0.25 threshold (target ≤ ~0.21, the pre-regression baseline). The metric is a flawed proxy (it penalizes faithfully-resolved structure) but a large drop + clean visual is the real bar.
- **Preserve the committed raymarch fixes** (uniform fine stepper, side-impact exit, mid_state).
- **Construction time** stays within a few minutes (current order of magnitude).

## 4. Design — self-consistent density normalization (Approach B)

The fix makes the column solver and the global normalization agree on **one** physically-derived absolute density scale, and gives the opacity table the range to represent it. The existing grey radiative-equilibrium column model (τ(z), Eddington T(z), flux-limited radiation pressure, gas+radiation hydrostatic ODE) is **kept** — only its density/opacity inputs are corrected.

### 4.1 Extend the opacity LUT to the disk's real density range
`build_opacity_luts(rho_min, rho_max, ...)` (`volumetric_disk.cpp:68`): raise `rho_max` from `1e-6` to a value above the disk's true midplane densities (≥ 1e8 g/cm³; choose from the expected ρ_mid scale so the table brackets it with margin). Keep `rho_min = 1e-18` (photosphere floor). The table is log-spaced (`opacity.cpp:264-266`), so widening the range increases coverage at fixed `n_rho=100` — verify resolution is still adequate (bump `n_rho` if a resolution check shows it is coarse across the wider range).

### 4.2 Self-consistent per-column reference density
Replace the clamped `ρ_cgs_ref` (`volumetric_disk.cpp:634-636`) with a self-consistent solve, per radial column:

```
ρ_cgs_ref(r) solves:   τ_mid = C · κ(ρ_cgs_ref, T_mid) · ρ_cgs_ref · H_cm(r)
```

- `κ` = Rosseland + electron-scattering opacity, evaluated at `(ρ_cgs_ref, T_mid)` — **no `max(κ,1.0)` floor**; the physical floor (κ_es) is already in the opacity table. Evaluate at the real density, not the `1e-10` placeholder.
- `C` = vertical column shape factor (≈1.0–1.25 for a Gaussian; the integral ∫ρ_norm dz / (ρ_mid·H)). Use a fixed physical constant or the measured `col_integral`; pick one and document it.
- `H_cm` = scale height in cm. The geometric→cm length conversion uses the black hole mass scale (GM/c²). Confirm the conversion source in the codebase; if absolute cm units are not otherwise needed, an equivalent is to make the whole column solve dimensionless and let the global `ρ_scale` (§4.3) carry the absolute calibration — **but the radiation-pressure term must then be evaluated at the consistent absolute density, not a clamped reference.**
- Solve by fixed-point iteration (κ depends on ρ_cgs_ref): ~3–5 iterations, seed from the gas-dominated estimate. **No clamp to a non-physical ceiling**; the only bounds are the (now wide) opacity-table range.

With `ρ_cgs_ref ~ 1e6` (true scale), the radiation-pressure term drops ~12 orders → subdominant to gravity in this gas-dominated disk → the hydrostatic ODE produces a normal gas-pressure profile of width H ≈ c_s/Ωz. In hotter inner regions where radiation genuinely matters, the term is now correctly scaled and supports (thickens) the disk rather than collapsing it.

### 4.3 Reconcile `normalize_density` with the column solver
`normalize_density` (`volumetric_disk.cpp:923-985`) must use the **same** self-consistent density scale and the **same** un-clamped opacity. Specifically: drop its `[1e-18,1e-6]` lookup clamps (lines 962, 965, 980); with the un-collapsed profile, `col_integral` becomes physical (≈ ρ_mid·H, not a single bin), so `ρ_scale` self-heals to a physical value instead of being inflated to compensate for the spike. After the fix, `ρ_scale·peak_ρ` (reported at construction) and `ρ_cgs_ref(peak)` must agree to within the iteration tolerance — this agreement is a regression check (§6).

### 4.4 Verify the radiation hydrostatic term is correctly formulated
With `ρ_cgs_ref` corrected, confirm via `GRRT_COL_LOG` that at r≈8.26 the `rad` term is now ≪ `grav` (gas-dominated) and the column holds. Separately confirm that in a deliberately radiation-dominated test column (high T_eff), the `rad` term is *positive-supporting* (thickens) rather than collapsing — i.e. the sign/formulation of `d(f·E)/dz / ρ` in the hydrostatic balance is physically correct, not merely rescaled. If the term is mis-signed or mis-formulated (independent of the ρ_cgs_ref scale), fix the formulation. (Instrumentation already in place.)

## 5. Components & boundaries

| File | Change |
|---|---|
| `src/volumetric_disk.cpp` | `build_opacity_luts` range (§4.1); `solve_column` ρ_cgs_ref self-consistent solve, drop κ floor + clamps (§4.2, §4.4); `normalize_density` drop clamps, reconcile scale (§4.3). Remove the `GRRT_COL_LOG` instrumentation once the fix is verified (or leave it env-gated — decide in the plan). |
| `include/grrt/scene/volumetric_disk.h` | Only if a helper signature changes (e.g. a `reference_density(r)` accessor); otherwise unchanged. |
| `tools/dump_disk_lut.cpp` | Keep (diagnostic; already committed-worthy). Used for verification. |
| `tests/test_volumetric.cpp` | New unit tests: vertical profile is non-collapsed and monotone (§6); ρ_scale ↔ ρ_cgs_ref agreement. Banding regression already present. |

The raymarch (`geodesic_tracer.cpp`) is **not** touched (its uniform-fine fix is correct and committed; the env-gated `GRRT_RM_LOG` probe there is parked separately).

## 6. Testing & verification

**Unit (TDD), `tests/test_volumetric.cpp`:**
- **Profile not collapsed:** for a default disk at a mid-orbit radius (e.g. r=8), assert `ρ_norm(r, 0.1·z_max) > 1e-3` and `ρ_norm(r, 0.5·z_max) > 1e-6` (i.e. a real profile, not RHO_FLOOR). This test FAILS on the current code and PASSES after the fix.
- **Monotone decreasing in |z|:** `ρ_norm(r,z)` is non-increasing from midplane to z_max (no spike-then-floor).
- **Normalization agreement:** `ρ_scale·peak_ρ` ≈ `ρ_cgs_ref(peak_radius)` within the solver tolerance.
- **τ_mid recovered:** the vertical optical depth `∫κρ dz` at the peak-flux radius equals `params_.tau_mid` within tolerance (this is the defining normalization; likely tightens the pre-existing `test_tau_midplane_near_target`, which currently fails).

**Integration:**
- **LUT dump:** `dump-disk-lut` → grep `disk_lut_dump.csv`: the count of collapsed columns (`prof@0.1zmax < 0.01`) drops from 4056/4096 to ~0; `rho_mid_cgs` stays smooth.
- **Banding regression** (`test_no_horizontal_bands`): metric drops below 0.25 (record the value; the metric is a flawed proxy — pair it with the visual check, and surface before/after to the user rather than silently recalibrating).
- **Edge-on visual render** (`--observer-theta 80 --fov 30 --background black --disk-turbulence 0.4 --samples 30`): no concentric banding; smooth disk with resolved turbulence.
- **Construction warnings:** the silent-collapse is gone; note any remaining `did not converge` / `H jump` warnings (the r_idx≈32 inner-edge warning is a separate, pre-existing issue — out of scope unless trivially related).

## 7. Edge cases

- **Inner disk (r < r_isco, plunging region):** H is frozen at H(r_isco); ensure the reference-density solve and opacity range behave there (T_eff and ρ differ). The r_idx≈32 non-convergence warning lives here — note it; fix only if it falls out of this work.
- **Radiation-dominated regime:** at high T_eff the corrected term must support, not collapse (§4.4). Covered by the deliberate-radiation test column.
- **Opacity-table edge:** after widening, confirm true densities sit *inside* the table (not at the new ceiling). If ρ_mid can exceed the chosen `rho_max`, raise it further or derive it from the expected peak density.
- **Iteration non-convergence:** the ρ_cgs_ref fixed-point must have a bounded iteration count and a sane fallback (e.g. last iterate) — no infinite loop, no silent floor.

## 8. Success criteria (definition of done)

1. Vertical profile non-collapsed and monotone (unit tests pass).
2. Collapsed-column count in the LUT dump ≈ 0.
3. Banding metric < 0.25 **and** edge-on render visually band-free with resolved turbulence.
4. ρ_scale ↔ ρ_cgs_ref agreement; τ_mid recovered at peak radius.
5. Committed raymarch fixes untouched and still correct (debug-pixel still emits; bottom disk renders).
6. Construction time within a few minutes; no silent density floors.

## 9. Follow-up: Approach A (documented, out of scope here)

Once B lands and the density/opacity/normalization are self-consistent, the **most physically accurate** model is a full coupled-ODE atmosphere solved as a two-point boundary value problem (Hubeny 1990 / Shakura-Sunyaev vertical structure), adding the physics B's grey model omits:

- **Viscous energy generation:** `dF/dz = (3/2)·α·Ω·P` — heat generated throughout the column (not assumed from T_eff), coupling thermal structure to the α-stress.
- **Radiative diffusion for T(z):** `F = −(c/3κρ)·d(aT⁴)/dz` solved for the temperature profile (replaces the grey Eddington `T⁴∝(τ+2/3)`).
- **Two-point BCs:** midplane `F=0, dT/dz=0`; surface `τ=2/3, T=T_eff, F=σT_eff⁴`.
- **Self-consistent EOS + opacity** in the relaxation loop.

This is a from-scratch BVP/relaxation solver (~5 coupled ODEs × 4096 radial bins, convergence-tuned). B's corrected density normalization and widened opacity table are **prerequisites** for A, and A is independently testable on top of B. Deferred as a separate spec → plan.

## 10. Out of scope
- The spectral raymarch (`raymarch_volumetric_spectral`).
- Approach A (§9).
- The inner-edge (r_idx≈32) convergence warning, unless it resolves as a side effect.
- Frequency-dependent (non-grey) radiative transfer.
