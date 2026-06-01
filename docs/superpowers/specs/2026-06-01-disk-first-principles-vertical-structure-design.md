# Disk first-principles vertical structure (Approach A) — design

**Date:** 2026-06-01
**Branch:** fix/volumetric-ring
**Status:** approved (brainstorming) — pending spec review → implementation plan
**Supersedes:** `2026-05-29-disk-vertical-structure-density-fix-design.md` (Approach B, the patch). This is Approach A — the from-first-principles redesign the user chose instead of B: "redesign the whole thing on first principles so there is no magic numbers, no clamps, just physics."

---

## 1. Problem & motivation

The volumetric disk renders with horizontal **banding** (edge-on) and **fireflies** (face-on). Approach B traced the proximate cause to a collapsed vertical density profile held together by a `1e-6 g/cm³` clamp. But B only makes the existing fiction *self-consistent*; it does not remove the fiction. Deeper investigation found the **root cause**:

> **The disk has no physical length scale.** `mass_solar` enters `api.cpp:97`, is used once to derive `T_peak`, then **discarded**. `VolumetricDisk` is constructed with geometric `mass = 1.0` and a temperature — it never learns the physical mass, so it cannot compute `r_g = GM/c²` and has **no cm length scale**. Consequently the optical-depth integral `∫κρ dz` multiplies cgs opacity by *geometric* `dz`, silently treating 1 geometric unit as 1 cm. Every absolute density is therefore convention-dependent, and every clamp downstream (`ρ_cgs_ref` ceiling, `κ≥1` floor, placeholder lookup densities, `tau_mid` back-calibration) exists to prop up that one missing number.

`G_cgs`, `M_sun`, `c_cgs` are already in `constants.h` — the length scale is one multiply away; it is simply never wired in.

Approach A introduces the length scale and rebuilds the vertical structure as an honest coupled-ODE atmosphere on top of it. With a real `r_g`, densities, temperatures, opacities, and optical depths are all the same physics — **no physics-distorting pins needed.**

**Note on "no clamps":** the target is **zero physics fudges** (the `1e-6` ceiling, the `κ≥1` floor, the placeholder densities, the geometric-as-cm assumption, the `tau_mid` knob). Legitimate *numerical* guards that any BVP solver keeps (opacity-table edges, iteration caps, divide-by-zero floors, line-search positivity) are retained — they do not distort the physics.

This redesign also **retires two known issues** (`known-issues-2026-05-02.md`):
- **Item 2** — `test_tau_midplane_near_target` fails 4× (τ=403 vs 100). With a real length scale, τ is unambiguous; the "mismatch of conventions" disappears.
- **Item 3** — `nested_refine`/`compare_columns` caps at `n_z=1024` with `delta≈4.2` (never converges), the `r_idx=32` non-convergence, the H-jump. This refinement apparatus is a band-aid for an unconverged solver aliasing on a pathological photosphere cliff. The Newton solver on a τ-grid dissolves it.

## 2. Goal & constraints

- **Physically derived, no magic numbers:** the disk is anchored on physical `(M, a, Ṁ/f_Edd)`; `r_g`, `T_eff(r)`, and the full vertical structure (`ρ(r,z), T(r,z), Σ(r), τ_mid(r), H(r)`) are derived. `tau_mid` and `T_peak` become emergent outputs, not input knobs.
- **Most terms physically derived (user requirement):** full coupled atmosphere BVP — hydrostatic equilibrium + distributed viscous heating + grey radiative diffusion for `T(z)` + gas+radiation EOS + ionization (Saha). No assumed Eddington `T(τ)` closure.
- **Banding eliminated:** edge-on render band-free with resolved turbulence; the banding metric drops below the 0.25 threshold (the metric is a flawed proxy — pair with the visual check and surface before/after to the user).
- **Preserve the committed raymarch fixes** (uniform fine stepper, side-impact exit, `mid_state`).
- **Construction time** within a few minutes (same order as today).
- **CPU path is the deliverable.** CUDA is knowingly left stale and ported in a separate task (§11).

## 3. Scope

**In scope:** the length scale; the physical accretion anchor; the radial `F(r)/T_eff(r)` made absolute; the per-column Newton-Raphson BVP replacing `solve_column`; the τ→uniform-z resample with **log-density** encoding; retirement of `normalize_density`/`nested_refine`/`compare_columns`; the CPU consumer read-path change for log-density; tests.

**Out of scope (future, separate specs):**
- **CUDA backend port** — the log-density encoding and absolute `ρ_mid` (`rho_scale_≡1`) must be carried into `cuda/cuda_vol_host_data.cpp` and the device interpolation in a *separate task*. CUDA is currently behind; the `CUDA == CPU` invariant is suspended until then.
- **Full non-grey atmosphere** (frequency-resolved structure). Grey Rosseland is sufficient for a continuum-opacity renderer (§7 rationale).
- **Spectral line opacity** (bound-bound transitions, NLTE, the relativistic Fe Kα line). The opacity model is continuum-only (ff + bf + H⁻ + e-scattering); adding lines is its own subsystem.
- **Self-gravitating outer disk** (Toomre `Q→1`, fragmentation). Irrelevant to the rendered inner disk where `Q≫1`.
- **The spectral raymarch** (`raymarch_volumetric_spectral`) read-path itself — it consumes the same LUT and benefits automatically; only the shared CPU `interp_2d` log change touches it.

## 4. Architecture & data flow

```
INPUTS:  M [M_sun]   a [spin]   f_Edd (or Ṁ) [accretion rate]            ← anchor
   │
   ▼
LENGTH SCALE:   r_g = G·(M·M_sun) / c²   [cm]                            ← NEW (keystone)
   │            every geometric length (H, z, dz) × r_g → cm;
   │            geometric Ω, Ω_z × (c/r_g) → 1/s   (time unit r_g/c)
   ▼
RADIAL STRUCTURE  (per r):
   • Ω(r), Ω_z(r), r_isco, r_horizon        ← KEEP  (Kerr mechanics)
   • F(r)  Novikov–Thorne flux, scaled by Ṁ ← KEEP shape, make ABSOLUTE
   • T_eff(r) = (F(r)/σ_SB)^¼               ← absolute surface temperature
   │
   ▼
COLUMN BVP  (per r ≥ r_isco) — Newton-Raphson on a τ_R-grid:            ← NEW (replaces solve_column)
   unknowns/node: [ρ, T, F, z];   P, κ_R algebraic in (ρ,T)
   5 relations: hydrostatic + viscous heating + radiative diffusion
                + EOS + dτ/dz, grey κ_R, EOS μ(ρ,T)
   BCs: surface τ_R=2/3 → T=T_eff, F=σT_eff⁴ ;  midplane z=0 → F=0, dT/dz=0
   OUTPUTS (emergent): ρ_mid(r), Σ(r), τ_mid(r), z_max(r), profile shape
   │
   ▼
RESAMPLE  τ-grid → uniform-z ;  store LOG ρ_norm(r,z), T(r,z), ρ_mid_cgs(r), z_max(r)
   │                                                                     ← LUT layout KEPT, encoding LOG
   ▼
CONSUMERS:  raymarch RGB + spectral (CPU)                               ← read path: log-interp
            CUDA host-data + device                                     ← DEFERRED (separate task)
```

**Plunging region `r < r_isco`:** free-falling matter is not in hydrostatic/thermal equilibrium, so the BVP is meaningless there. **Keep** the existing BPT72 mass-conservation taper + frozen `H(r_isco)` (`volumetric_disk.cpp:184-206`, `compute_plunging_region_decay`). The BVP runs only for `r ≥ r_isco`.

**Kept / replaced / new / retired:**

| | Items |
|---|---|
| **KEEP** | Kerr orbital mechanics (`omega_orb`, `omega_z_sq`, ISCO/horizon); Novikov–Thorne `F(r)` *shape* (`build_flux_lut`); BPT72 plunging taper; opacity *physics* (`solve_saha`, ff/bf/H⁻/e-scattering); uniform-z LUT storage layout; raymarch; turbulence/noise overlay |
| **REPLACE** | `solve_column` → Newton BVP; "proportional" `Σ`/`ρ_mid` → absolute emergent `ρ_mid(r)`; `F(r)` normalization → absolute via `Ṁ`; linear-density encoding → log-density |
| **NEW** | length scale `r_g` + cgs unit conversions; accretion-rate plumbing; opacity-derivative supplier (`∂κ_R/∂ρ, ∂κ_R/∂T`) |
| **RETIRE** | `normalize_density` + `rho_scale_` (≡1); `nested_refine` + `compare_columns` (item-3 band-aid); `tau_mid` as an input knob; every `[1e-18, 1e-6]` density clamp |

## 5. Physical anchor & length scale

```
M_phys = mass_solar · M_sun                       [g]
r_g    = G_cgs · M_phys / c_cgs²                   [cm]      (10 M_⊙ → 1.48e6 cm ≈ 14.8 km)
```
Unit conversions into the cgs column ODEs:
- length: `z_cm = z_geom · r_g`, `dz_cm = dz_geom · r_g`
- frequency: `Ω_cgs = Ω_geom · c_cgs / r_g` (geometric time unit = `r_g/c`); same for `Ω_z`.

**Accretion anchor (input form: `f_Edd` primary, locked §2 decision):**
```
η      = 1 − E_isco                                (radiative efficiency; E_isco already computed)
L_Edd  = 4π G M_phys m_p c / σ_T                   [erg/s]
Ṁ      = f_Edd · L_Edd / (η c²)                    [g/s]    (or mdot_override if supplied)
```
The Novikov–Thorne flux becomes absolute:
```
F(r) = Ṁ · f_NT(r, a) / r_g²   (× constants)       [erg/cm²/s]   — f_NT from build_flux_lut shape
T_eff(r) = (F(r) / σ_SB)^¼                          [K]
```
**Optional `T_peak` override:** if `peak_temperature` (constructor arg) is set, scale `F(r)` so its peak gives that `T_peak` (back-compat; `Ṁ` then becomes diagnostic). **Default `mass_solar`** (~10 M_⊙) when unset, so `r_g` is always defined even on the legacy `T_peak`-only construction path.

## 6. Radial inputs & emergent outputs

The radial layer provides the column with exactly one thing: the absolute `F(r)` / `T_eff(r)`. **Σ, ρ_mid, H, τ_mid are emergent outputs of the column BVP, not inputs.** The usual α-disk circularity (Σ needs ν needs H needs structure needs Σ) is broken because `F(r)` is fixed by `(M,a,Ṁ)` and the relativistic potential independent of vertical structure; the column's flux BCs (`F=0` midplane, `F=σT_eff⁴` surface) automatically enforce the vertically-integrated dissipation law `σT_eff⁴ = (3/2)αΩ∫P dz`.

## 7. The column BVP (per r ≥ r_isco)

**Unknowns** per τ-grid node: `ρ, T, F, z`. `P` and `κ_R` are algebraic in `(ρ,T)`.

**Relations (all cgs):**
```
(1) Hydrostatic:         dP/dz = −ρ · Ω_z²(r) · z                       [vertical tidal gravity]
(2) Viscous heating:     dF/dz = (3/2) · α · Ω(r) · P                   [Shakura–Sunyaev dissipation]
(3) Radiative diffusion: dT/dz = −3 κ_R(ρ,T) ρ F / (4 a c T³)          [grey, Rosseland κ_R]
(4) EOS (algebraic):     P     = ρ k_B T / (μ(ρ,T) m_p) + (1/3) a T⁴   [gas + radiation]
(5) Optical depth:       dτ/dz = −κ_R(ρ,T) · ρ                          [defines τ_R grid; surface τ=0]
```

**Boundary conditions (two-point):**
- **Surface** (photosphere `τ_R = 2/3`): `T = T_eff(r)`, `F = σ_SB T_eff⁴`
- **Midplane** (`z = 0`): `F = 0`, `dT/dz = 0`   (reflection symmetry)

**Opacity:** grey Rosseland mean `κ_R(ρ,T)` from the existing `kappa_ross_lut` (`opacity.cpp:285-299`), widened in density range (§10). Frequency-dependence lives only in the *rendering* transfer (`lookup_kappa_abs(ν,ρ,T)`), not here — rationale: the Rosseland mean is the exact frequency-average for total energy transport, and the opacity model is continuum-only so there is no line-blanketing to motivate non-grey.

**Self-gravity neglected** (Toomre `Q = Ω c_s/(πGΣ) ≫ 1` in the rendered region); the only vertical gravity is the BH tidal term `Ω_z² z`. A `Q(r)` diagnostic warns if `Q` approaches ~1.

## 8. Newton-Raphson solver

- **State vector** `U` = `[ρ, T, F, z]` × `N` nodes (`N ~ 100–300`) plus the global unknown `τ_mid` (total optical depth, an output). Free boundary handled by the **normalized coordinate** `ξ = τ/τ_mid ∈ [0,1]` (fixed domain; `τ_mid` an extra unknown).
- **Residual** `𝓕(U) = 0`: relations (1)(2)(3)(5) discretized between adjacent nodes (trapezoidal), EOS (4) applied pointwise, plus the 4 boundary conditions.
- **Jacobian** `J = ∂𝓕/∂U`: **block-tridiagonal**, `4×4` blocks (each node couples only to neighbors) + the `τ_mid` border row/column. Solve `J ΔU = −𝓕` by **block-Thomas**, `O(N·4³)`.
- **Opacity derivatives** `∂κ_R/∂ρ, ∂κ_R/∂T` enter the (3)(5) blocks, supplied via a **swappable supplier interface** (default: finite-difference / analytic-gradient of the bilinear-interpolated `kappa_ross_lut`; escape hatch: bicubic-smooth supplier, one-file swap).
- **Robustness:**
  - **Damped step / line search:** `U ← U + λΔU`, `λ≤1` chosen so `ρ,T>0` and the residual norm decreases (absorbs cell-boundary roughness of the finite-diff supplier).
  - **Analytic initial guess:** seed from the closed-form Shakura–Sunyaev midplane (algebraic α-disk relations) draped into a polytrope-like profile.
  - **Convergence:** `max |ΔU/U| < tol` (e.g. `1e-8`); iteration cap (e.g. 50).
  - **Honest fallback:** on non-convergence, fall back to the analytic α-disk profile for that column and **emit a Promptable warning naming the radius** — never a silent floor.
- **Parallelism:** each column is an independent BVP → OpenMP across the ~4096 radial columns (existing pattern).

## 9. LUT storage, resample & log-density encoding

- The solve yields `ρ(τ_i), T(τ_i), z(τ_i)` (non-uniform in z; τ=0 at surface where z=z_max, τ=τ_mid at midplane z=0).
- **Resample** onto uniform-z `z_j ∈ [0, z_max]`, interpolating in **log-density** (Gaussian → parabola in log → captured cleanly by few nodes).
- **Store** (layout the raymarch already reads):
  - `rho_profile_lut_[r,z]` = **log** of normalized profile `log(ρ(z)/ρ_mid(r))` (midplane → 0).
  - `rho_mid_lut_[r]` = **absolute cgs** midplane density `ρ_mid(r)` (emergent).
  - `T_profile_lut_[r,z]` = `T(z)` (**linear**; T varies only by a factor of a few).
  - `z_max_lut_[r]`.
- **`rho_scale_` retired** (≡1); `normalize_density` deleted. Everything absolute from the start.
- **Read path (CPU):** `density_cgs(r,z) = exp(interp_2d(rho_profile_log, r,z)) · rho_mid_lut_[r]`. `interp_2d` for density becomes a **log-interp** (matching the `lookup_kappa_abs` precedent at `opacity.cpp:327-335`); temperature interp stays linear.
- **Turbulence overlay** composes additively in log: `ρ_final = exp(logρ_interp + σ·fBm) · ρ_mid` — unchanged behavior, cleaner composition.
- **σ_s / β regime detection** (`compute_sigma_s_phys`) now reads real `ρ_mid, T_mid` → correct gas/radiation `β` with no special handling.

## 10. Opacity LUT changes (input table)

- **Range:** widen the density axis from `[1e-18, 1e-6]` to `[1e-18, 1e9] g/cm³` (`volumetric_disk.cpp:68`) so the table represents the disk's true densities; **resolution** `n_rho 100→220` (`opacity.cpp:244`) to keep ~8 bins/decade across the wider range. (`log_interp` already clamps to table edges — `opacity.cpp:221` — so widening + removing the external `[1e-18,1e-6]` clamps is safe.)
- **Derivatives:** the supplier (§8) computes `∂κ_R/∂ρ, ∂κ_R/∂T` from this table.
- The Rosseland-mean *physics* (50-point integral, `opacity.cpp:285-299`) is unchanged; widening `n_rho` only modestly increases the one-time table build.

## 11. Interface changes

- **C API struct: unchanged.** `mass_solar`, `eddington_fraction` already exist (`api.cpp:82`); stop discarding them — thread into `VolumetricParams`.
- **`VolumetricParams`** gains: `mass_solar` (default ~10), `eddington_fraction` (default ~0.1), optional `mdot_override`; the existing `peak_temperature` constructor arg becomes the optional `T_peak` override.
- **CLI:** unchanged (`--mass-solar`, `--eddington-fraction` flags already exist).
- **Construction pipeline:** `build_opacity_luts (wide+derivs)` → `r_g` → `compute_radial_structure (absolute F,T_eff)` → `compute_vertical_profiles (Newton BVP, OUTPUTS ρ_mid/Σ/τ_mid/z_max)` → resample/store → `compute_sigma_s_phys` → `validate (+Toomre Q)`. **Deleted:** `normalize_density`, `nested_refine`, `compare_columns`.
- **CUDA (DEFERRED — separate task):** `cuda/cuda_vol_host_data.cpp` + device interpolation must adopt the log-density encoding and absolute `ρ_mid`. Until that task lands, CUDA output is knowingly stale and the `CUDA == CPU` invariant is suspended. Tracked as out-of-scope here.
- **Tests** with value assertions against old outputs are updated to the new physical results (expected for a structure redesign).

## 12. Testing & verification

**Unit — column physics invariants (per converged column):**
- Hydrostatic residual `|dP/dz + ρΩ_z²z|` below tol.
- Energy conservation `∫(3/2)αΩP dz ≈ σT_eff⁴` within tol.
- Flux BCs `F(midplane)=0`, `F(surface)=σT_eff⁴`; `τ(photosphere)=2/3`.
- Profile monotone non-increasing in `|z|`, midplane-normalized; `τ_mid, Σ` positive/finite.

**Unit — analytic limiting cases:**
- Gas-dominated isothermal limit → Gaussian of width `H=c_s/Ω_z` (compare closed form).
- Radiation-dominated limit (hot column) → column *thickens* (the radiation term supports, not collapses — this settles, by construction, the old §4.4 sign concern from Approach B).

**Unit — length scale & τ:**
- `r_g=GM/c²` correct (10 M_⊙ → 1.48e6 cm).
- Vertical optical depth dimensionally consistent → known-issue **item 2** (`test_tau_midplane_near_target`) becomes well-defined and passes.

**Unit — solver:** Newton converges within the cap across representative radii; a deliberately hard column triggers the *honest fallback warning* (not a silent floor). Opacity-derivative supplier matches reference; swap interface works.

**Integration:**
- **LUT dump** (`dump-disk-lut`): profile **not collapsed**, smooth, monotone; `Σ(r), τ_mid(r), ρ_mid(r)` smooth in radius; **Toomre Q ≫ 1**.
- **Banding** (`test_no_horizontal_bands`): metric below 0.25 (record before/after; pair with visual; surface to user, do not silently recalibrate).
- **Edge-on visual render** (`--observer-theta 80 --fov 30 --background black --disk-turbulence 0.4 --samples 30`): band-free, resolved turbulence.
- **Tile == full-frame** (CLAUDE.md invariant). **CUDA == CPU is suspended** (§11).

**Regression & budget:** committed raymarch fixes still pass; construction time within a few minutes.

## 13. Edge cases

- **Plunging region (r < r_isco):** no BVP; BPT72 taper + frozen `H(r_isco)`. Ensure the resample/encoding handle the tapered columns.
- **Inner edge / r_isco boundary:** the column at `r ≈ r_isco` is the hottest, most radiation-influenced — verify Newton converges there (this is where item-3's `r_idx≈32` warning lived; it should retire).
- **Newton non-convergence:** bounded iterations + analytic-profile fallback + named warning. No infinite loop, no silent floor.
- **Opacity-table edge:** with the widened range, confirm true densities sit inside the table; raise `rho_max` if a column's `ρ_mid` approaches the ceiling.
- **Toomre Q → 1:** warn (thin-disk assumption breaking); does not occur for normal inner-disk renders.
- **Optically thin outer atmosphere (τ < 2/3):** the BVP solves to the photosphere; the envelope `z_max` extends modestly above where ρ is negligible.

## 14. Success criteria (definition of done)

1. Length scale `r_g` wired in; all column quantities in honest cgs; τ dimensionally consistent.
2. Column BVP converges across the orbiting region; physics invariants (§12) pass; honest fallback on the rare non-convergent column.
3. Vertical profile non-collapsed, smooth, monotone; `Σ(r), τ_mid(r), ρ_mid(r)` emergent and smooth.
4. Banding metric < 0.25 **and** edge-on render visually band-free with resolved turbulence.
5. `normalize_density`, `nested_refine`, `compare_columns`, and all `[1e-18,1e-6]` clamps deleted; `tau_mid`/`T_peak` are outputs/overrides, not required knobs.
6. Known-issues **item 2** retired (τ well-defined, test passes); **item 3** retired (no refinement-cap/non-convergence warnings in the orbiting region).
7. Committed raymarch fixes untouched and still correct.
8. Construction time within a few minutes; no silent density floors. CUDA deferral documented.

## 15. Suggested implementation phasing (for the plan)

The pieces are sequential dependencies, best built and tested in order (one spec, phased plan):

1. **Length scale + physical anchor** — `r_g`, `Ṁ` from `f_Edd`, absolute `F(r)/T_eff(r)`, `VolumetricParams` plumbing. Testable: `r_g` value, absolute `T_eff(r)`.
2. **Opacity table widening + derivative supplier** — range/resolution, the swappable `∂κ_R` interface. Testable: table coverage, derivative accuracy.
3. **Column BVP Newton solver** — equations, BCs, block-tridiagonal Newton, seed, damping, fallback. Testable: physics invariants, analytic limits, convergence (against a single column first).
4. **LUT resample + log-density encoding + read-path** — store/resample, `interp_2d` log change, retire `normalize_density`/`rho_scale_`. Testable: profile not collapsed, density round-trip, banding.
5. **Retire `nested_refine`/`compare_columns`; wire `compute_sigma_s_phys` to real density; validation (Toomre Q).** Testable: warnings gone, β correct.
6. **Integration sweep** — LUT dump, banding, edge-on render, surface before/after to user.

## 16. Out of scope / future follow-ups

- **CUDA port** of the log-density encoding + absolute density (separate task; CUDA is behind).
- **Non-grey atmosphere** (frequency-resolved structure; ~50–500× cost; no benefit for continuum opacity).
- **Spectral line opacity** (bound-bound, NLTE, relativistic Fe Kα) — a new opacity subsystem; would reopen the non-grey question.
- **Self-gravitating outer disk** (Toomre `Q→1`, fragmentation).
- **LUT on-disk caching** (known-issues item 8) — orthogonal, still useful given construction cost.

## 17. References

- Shakura & Sunyaev 1973 (α-disk; vertical structure).
- Novikov & Thorne 1973 (relativistic disk flux `F(r)`).
- Hubeny 1990 (vertical structure of accretion-disk atmospheres; the coupled BVP).
- Bardeen, Press & Teukolsky 1972 (ISCO, `E_isco`, plunging geodesics — already used).
- Henyey et al. 1964 (relaxation method for stellar structure).
- Constants: `include/grrt/math/constants.h` (`G_cgs`, `M_sun`, `c_cgs`, `σ_SB`, `a_rad`, `σ_T`, `k_B`, `m_p`).
