# Volumetric Disk Boundary Smoothing and Physical Noise Composition

**Date**: 2026-04-27
**Status**: Draft (post-brainstorm, awaiting review)
**Branch**: `fix/volumetric-ring`

## Summary

The volumetric accretion disk currently has four hard cliffs at its model boundaries (top photosphere, outer radial edge, inside-ISCO transition, noise-carved hole edges) that produce visible flat-plane and vertical-extrusion artifacts in renders. This spec replaces those cliffs with physically-derived smooth tapers: an extended hydrostatic-equilibrium ODE that puts the Gaussian atmosphere tail directly in the LUT, a smoothstep radial taper at `r_outer`, continuous `H(r)` decay through the plunging region, and **log-normal multiplicative noise composition** — the canonical MHD-turbulence density PDF — replacing the additive-clip form. As a structural by-product, LUT sizing (`n_r`, `n_z`) becomes data-driven via fully-nested Richardson refinement weighted by an optical-depth contribution function, so the model adapts cleanly across mass scales from micro to supermassive.

## Problem

Visible artifacts in current volumetric renders fall into one underlying class — model boundaries that the soft-taper code does not actually soften:

1. **Photosphere ceiling** at `z ≈ z_max(r)`. The vertical-profile ODE in `compute_vertical_profiles()` (`src/volumetric_disk.cpp`) iteratively extends `z_max` until `ρ(z_max) ≤ 1e-10` of midplane — i.e., until density is *already* essentially zero. The "soft-edge zone" in `density()` (`zm ≤ |z| < zm + 1.5·H`) then samples the LUT at `zm * 0.9999` and multiplies by a Gaussian, but this Gaussian is tapering from 1e-10 to 0. The visible cliff lives at `z ≈ zm − 1·H` where the LUT itself drops sharply, not in the cosmetic edge zone above.
2. **Outer radial wall** at `r = r_outer`. `density()` and `inside_volume()` return 0 for `r > r_outer_` with no taper. The disk literally ends in a cylindrical wall.
3. **Inside-ISCO cylinder**. For `r < r_isco` the model freezes `H = H_isco` and uses `ρ_mid_lut_[isco_idx]` for `ρ_mid`. Combined with the existing Gaussian `taper(r)` multiplier in `density()`, the plunging region is a tapered cylinder of constant scale height — geometrically a flat plate.
4. **Noise-hole edges**. `density()` clamps the noise multiplier with `max(0, base · (1 + turb·n))`, producing a hard `factor = 0` cliff wherever the additive perturbation drives density below zero. With `noise_scale_` fixed globally at `2·H(r_peak)` and `H(r) ≠ H(r_peak)` elsewhere, the noise's vertical correlation length is wrong almost everywhere — features in the inner disk become *vertically extruded cylinders* rather than 3D blobs.

These four cliffs share a common mechanism: a threshold predicate (`if z >= zm`, `if r > r_outer`, `if r < r_isco`, `max(0, factor)`) toggling between physics and exact zero, with no smooth interpolation between them. Fixes share a common form: replace each predicate with a smoothing kernel whose width is set by the local physical scale.

## Goals

- Eliminate the four visible cliffs while preserving the existing Eddington/flux-limited vertical-structure physics inside the disk.
- Replace the additive-clip noise with the canonical MHD log-normal density PDF, with amplitude `σ_s` derived from the existing `α` viscosity parameter.
- Make LUT resolution (`n_r`, `n_z`) data-driven, so the model adapts to disk regimes from micro-BH to SMBH without per-mass tuning.
- Surface construction-phase warnings to the user via a clear severity-graded prompt, so a long-running render never proceeds silently from a compromised disk.

## Non-goals

- **CUDA backend updates.** `cuda_volumetric_disk.h` and `cuda_vol_host_data.cpp` are out of scope; CUDA will catch up in a follow-up spec. The `--validate` CLI flag is disabled for the volumetric disk path during the gap.
- **Power-spectrum-baked 3D noise LUT.** Anisotropic Goldreich-Sridhar spectra and FFT-synthesized noise fields are the natural follow-up after this lands. The log-normal composition committed to here means that swap is a single-method change in `density()`.
- **Physical derivation of outer-edge truncation and inside-ISCO H decay.** Both are visual smoothing in this spec. Future specs may replace them with derived models (viscous timescale / self-gravity for outer taper; Bardeen-Press-Teukolsky streamline integration for plunging-region thickness).
- **Opacity LUT bound extension** for extreme mass scales (SMBHs below `1e-15 g/cm³`, micro-BHs above `1e8 K`). Today's opacity LUTs are clamped within `[1e-18, 1e-6] × [3e3, 1e8]`; widening that range is independent and deferred.
- **Log-spaced radial bins.** For SMBH disks with very large `r_outer` (e.g., `1000·M`), uniform radial spacing under-resolves the inner disk even at the `n_r` cap. Switching `interp_radial`/`interp_2d` to log-spacing is its own follow-up spec.

## Approach

The fix splits along the existing construction-vs-accessor seam in `VolumetricDisk`:

- **Layer 1 — Smooth volumetric envelope** (LUT/ODE edits during construction). The `H_lut_`, `rho_mid_lut_`, `rho_profile_lut_`, `T_profile_lut_`, and `z_max_lut_` arrays themselves come to encode smooth fades to zero at all envelope boundaries. The accessors mostly read these LUTs unchanged.
- **Layer 2 — Physical noise composition** (accessor edits during sampling). `density()` switches to log-normal multiplicative noise with `H(r)`-relative correlation length.

After Layer 1, the meaning of `z_max_lut_[ri]` changes from "height past which density falls below `1e-10` of midplane" (numerical bookkeeping) to "height past which density is *truly negligible* and the LUT itself encodes the smooth tail to that point" (physical definition). Code that reads `z_max_at(r)` (raymarcher exit, `inside_volume`, step-clamping) needs its margin constants reviewed once but no semantic change.

Two semantic changes to `VolumetricParams` (acceptable per project owner — solo user, no script-compatibility concern):

- `turbulence` becomes a dimensionless boost on physically-derived `σ_s` (default `1.0` = pure physical; was an additive amplitude in `[0, ~1]`).
- `noise_scale` becomes a multiplier on `c_corr · H(r)` (default `0.0` = auto, uses `c_corr = 0.5`; was an absolute geometric length in `M`).

CLI flag names (`--disk-turbulence`, `--disk-noise-scale`) remain identical; their effects are different.

## Architecture

### Layer 1 — Smooth volumetric envelope

All edits live in `src/volumetric_disk.cpp`, construction phase.

- **1a. Photosphere ceiling.** Lower the `z_max` ODE convergence floor from `1e-10` to `1e-15`; raise `Z_MAX_CAP` from `20·H` to `30·H`; remove the `z_lookup = zm * 0.9999` clamp and the `edge_factor` Gaussian zone in `density()`. After this, the LUT itself extends into the truly-negligible regime, so cosmetic post-hoc tapers are redundant and confusing.
- **1b. Outer radial wall.** Apply a smoothstep multiplicative taper to `rho_mid_lut_` over `[r_outer − Δr_out, r_outer]` after `compute_radial_structure()` runs. `Δr_out` defaults to `2 · H(r_outer)` (configurable). `H_lut_` is *not* radially tapered — physical truncation reduces density, not thickness.
- **1c. Inside-ISCO continuous decay.** Replace the frozen-`H_isco` segment with `H(r) = H_isco · taper(r)^p` where `taper(r)` is the existing Gaussian inner taper and `p` (default `0.5`) is exposed as `params_.plunging_h_decay_exponent`. `ρ_mid_lut_` for `r < r_isco` keeps the existing `ρ_mid_isco · taper(r)` form.

### Layer 2 — Physical noise composition

All edits in `src/volumetric_disk.cpp`, accessor phase plus a one-shot precompute in the constructor.

- **2a. Log-normal noise composition.** Replace `max(0, base · (1 + turb · n))` with `base · exp(σ_s · turb · n)` in `density()`. `σ_s` is precomputed in the constructor from `params_.alpha` and the local pressure regime (see *Algorithms*).
- **2b. H(r)-relative noise scale.** In `density()`, the noise position scale becomes `L = c_corr · H(r)`, where `c_corr = params_.noise_correlation_length_factor` (default `0.5`). The old `noise_scale_` member is removed. `params_.noise_scale > 0` overrides `c_corr` directly.

### LUT sizing — data-driven Richardson refinement

`n_r` and `n_z` are no longer fixed constants. They are determined at construction by **fully-nested Richardson refinement** as a fixed-point iteration in `(n_r, n_z)`, comparing successive resolutions via an **optical-depth-contribution-function-weighted soft-max norm** evaluated as a **max envelope over the output frequency range**. Refinement stops when the inter-resolution delta falls below `params_.target_lut_eps` (default `1e-3`) or when caps `params_.max_n_r` / `params_.max_n_z` bind. The user can force specific resolutions by setting `params_.bins_per_h > 0` or `params_.bins_per_gradient > 0`, bypassing refinement.

### Error handling — graded warnings with user prompt

The library accumulates structured construction warnings (`Info` / `Warning` / `Promptable` / `Severe`) in the `VolumetricDisk` instance. The CLI inspects these via a new C API surface and, when any `Promptable` or `Severe` warning is present, halts to ask the user whether to proceed. Default behavior in non-interactive sessions is to abort (safer than rendering a compromised disk in a walk-away batch run). New `--force` and `--strict` flags override.

## Components

### `VolumetricParams` (final)

`include/grrt/scene/volumetric_disk.h`. Existing struct, semantics changes documented in comments:

```cpp
struct VolumetricParams {
    // --- Physical (unchanged) ---
    double alpha          = 0.1;     ///< Shakura-Sunyaev viscosity. Drives both
                                     ///< viscosity AND log-normal noise amplitude.
    uint32_t seed         = 42;      ///< Noise seed.
    double tau_mid        = 100.0;   ///< Midplane optical depth at peak-flux radius.
    double opacity_nu_min = 1e14;    ///< Opacity LUT frequency lower bound (Hz).
    double opacity_nu_max = 1e16;    ///< Opacity LUT frequency upper bound (Hz).
    int noise_octaves     = 2;       ///< fBm octave count.

    // --- Noise composition (CHANGED semantics) ---
    double turbulence     = 1.0;     ///< Dimensionless boost on physically-derived
                                     ///< σ_s. 1.0 = pure physical (default).
                                     ///< 0.0 = axisymmetric (no noise).
                                     ///< WAS: additive amplitude.
    double noise_scale    = 0.0;     ///< Multiplier on c_corr·H(r) for noise
                                     ///< correlation length. 0 = auto (uses c_corr).
                                     ///< WAS: absolute length in M.

    // --- Noise physics (NEW — data-derived defaults) ---
    double noise_compressive_b              = 0.0;   ///< 0 = derive from peak-flux β.
    double noise_correlation_length_factor  = 0.5;   ///< c_corr; eddy length / H(r).

    // --- Smooth volumetric envelope (NEW) ---
    double outer_taper_width        = 0.0;   ///< 0 = auto = 2·H(r_outer); units M.
    double plunging_h_decay_exponent = 0.5;  ///< H(r<r_isco) = H_isco · taper(r)^p.

    // --- LUT sizing (NEW — data-driven with manual override) ---
    int bins_per_h         = 0;          ///< 0 = auto via Richardson refinement.
    int bins_per_gradient  = 0;          ///< 0 = auto via Richardson refinement.
    double target_lut_eps  = 1e-3;       ///< Refinement tolerance (relative).
    int min_n_r            = 256;        ///< Radial bin floor.
    int min_n_z            = 64;         ///< Vertical bin floor.
    int max_n_r            = 4096;       ///< Radial bin cap.
    int max_n_z            = 1024;       ///< Vertical bin cap.
    int refine_num_frequencies = 8;      ///< Frequency samples for max-envelope
                                         ///< contribution function during refinement.
};
```

### `VolumetricDisk` private members

```cpp
class GRRT_EXPORT VolumetricDisk {
    // ... public unchanged ...
private:
    // Existing members unchanged: mass_, spin_, r_outer_, peak_temperature_,
    // r_isco_, r_horizon_, r_min_, taper_width_, params_, noise_, E_isco_,
    // L_isco_, opacity_luts_, rho_scale_.

    // CHANGED — sized at construction by refinement (was fixed defaults)
    int n_r_;
    int n_z_;

    // Existing LUT storage unchanged.
    std::vector<double> H_lut_, rho_mid_lut_, T_eff_lut_;
    std::vector<double> rho_profile_lut_, T_profile_lut_, z_max_lut_;

    // NEW — physical noise amplitude precomputed in ctor
    double sigma_s_phys_ = 0.0;

    // NEW — outer-taper resolved width (cached after auto-resolution)
    double outer_taper_width_ = 0.0;

    // REMOVED: double noise_scale_;

    // NEW — construction warnings
    std::vector<ConstructionWarning> warnings_;

    // Construction helpers (existing, signatures changed)
    void compute_radial_structure();
    void compute_vertical_profiles();
    void normalize_density();

    // NEW helpers
    struct ColumnSolution {
        double z_max;
        std::vector<double> rho_z;   // size n_z, normalized so rho_z[0] = 1
        std::vector<double> T_z;     // size n_z, in Kelvin
    };
    ColumnSolution solve_column(double r, double H, double T_eff,
                                double rho_mid_proportional, int n_z) const;
    int refine_n_z_globally();
    int refine_n_r();
    std::pair<int, int> nested_refine();
    void apply_outer_radial_taper();
    void compute_plunging_region_decay();
    void compute_sigma_s_phys();
    bool validate_luts();
    void emit(WarningSeverity, std::string code, std::string message);

    static double smoothstep(double edge0, double edge1, double x);

    // Existing accessors unchanged
    double interp_radial(const std::vector<double>& lut, double r) const;
    double interp_2d(const std::vector<double>& lut, double r, double z_abs) const;
};
```

`enum class WarningSeverity { Info, Warning, Promptable, Severe };`
`struct ConstructionWarning { WarningSeverity severity; std::string code; std::string message; };`
`const std::vector<ConstructionWarning>& warnings() const`
`int promptable_count() const`

### `VolumetricDisk` public method changes (signatures unchanged; bodies modified)

| Method | Change |
|---|---|
| `density(r, z, phi)` | Remove `z_lookup * 0.9999` clamp and `edge_factor` Gaussian. Replace `max(0, base·(1 + turb·n))` with `base · exp(σ_s · turb · n)` (with arg clamped to `[-50, 50]` for safety). Use per-call `c_corr · H(r)` for noise position scale. |
| `density_cgs(r, z, phi)` | Unchanged — still calls `density()`. |
| `temperature(r, z)` | Remove the `z_lookup = std::min(z_abs, zm * 0.9999)` clamp. Direct `interp_2d(T_profile_lut_, r, z_abs)`; LUT now extends to surface temperature smoothly. |
| `inside_volume(r, z)` | Tighten margin from `+ 1.5·H` to `+ 0.5·H`. Extend r-range past `r_outer_` by `0.5·outer_taper_width_`. |
| `scale_height`, `z_max_at`, `taper`, velocities, frequencies | Unchanged. |

### `GeodesicTracer` margin updates

`src/geodesic_tracer.cpp`. Three sites in `trace()`, mirrored in `trace_debug()` and `trace_spectral()`:

| Site (line ref. for current code) | Currently | After |
|---|---|---|
| Step-size clamp zone (`:79`) | `\|z\| < zm + 6·H` | `\|z\| < zm + 3·H` |
| Outer-loop near-disk trigger (`:132–133`) | `\|z\| < zm + 2·H` | `\|z\| < zm + 1·H` |
| Raymarcher's "left the disk" exit (`:255`) | `\|z\| > zm + 3·H` | `\|z\| > zm + 1.5·H` |
| Outer-loop r-range check (`:135`) | `r_new <= vol_disk_->r_max()` | `r_new <= vol_disk_->r_max() + 0.5·outer_taper_width_` |

### C API additions

`include/grrt/api.h`:

```c
#define GRRT_SEV_INFO       0
#define GRRT_SEV_WARNING    1
#define GRRT_SEV_PROMPTABLE 2
#define GRRT_SEV_SEVERE     3

int          grrt_warning_count(const GRRTContext* ctx);
int          grrt_warning_severity(const GRRTContext* ctx, int i);
const char*  grrt_warning_message(const GRRTContext* ctx, int i);
int          grrt_promptable_warning_count(const GRRTContext* ctx);
```

Pointers returned by `grrt_warning_message` are valid until `grrt_destroy(ctx)`.

### CLI flags

`cli/main.cpp`:

- `--force` — skip the safety prompt; render even if construction reports `Promptable` or `Severe` warnings. For batch / scripted use.
- `--strict` — skip the prompt and abort on any `Promptable` or `Severe` warning. Mutually exclusive with `--force`.

The existing `--validate` flag is disabled when `disk_volumetric=1`; prints an explanatory stderr message and falls through to non-validate rendering.

## Algorithms

### Construction order

```
VolumetricDisk(mass, spin, r_outer, peak_temperature, params):
    set r_isco, r_horizon, r_min, taper_width, E_isco, L_isco
    build opacity LUTs
    outer_taper_width_ ← params.outer_taper_width > 0
                          ? params.outer_taper_width
                          : 2 · H_estimate(r_outer)

    (n_r_, n_z_) ← nested_refine()           // builds all LUTs as side effect

    compute_plunging_region_decay()           // continuous H(r), rho_mid(r) for r<r_isco
    apply_outer_radial_taper()                // smoothstep on rho_mid_lut_

    normalize_density()                       // unchanged math; sets rho_scale_
    compute_sigma_s_phys()                    // β-driven b, then σ_s = b·√(ln(1+α))

    validate_luts()                           // appends warnings on failure
```

### `solve_column` — extracted ODE solver per column

```
solve_column(r, H, T_eff, rho_mid_prop, n_z) -> ColumnSolution

  Z_MAX_CAP        = 30.0       // (was 20.0)
  CONV_FLOOR       = 1e-15      // (was 1e-10)
  RHO_FLOOR        = 1e-18      // numerical floor in RK4 step
  MAX_OUTER_ITERS  = 8

  z_max ← 3·H

  for outer_iter in 0..MAX_OUTER_ITERS:
    dz ← z_max / (n_z − 1)

    Pass 1 (existing): τ(z) backward integration from z_max to 0
    Pass 2 (existing): T(z) from Eddington T⁴ = ¾ T_eff⁴(τ + 2/3)
    Pass 3 (existing): f(z) Levermore-Pomraning flux limiter
    Pass 4 (existing): ρ(z) RK4 integration outward, with rho_floor

    if rho_z[n_z−1] > CONV_FLOOR and z_max < Z_MAX_CAP·H:
      z_max ← min(z_max + H, Z_MAX_CAP·H)
      continue

    if max relative change vs. prev iter < 1e-3: break

  if any rho_z[i] non-finite or T_z[i] < 0:
    emit(Promptable, "ode_divergence", "solve_column at r=%g produced non-finite values")
    return zeroed solution

  return { z_max, rho_z, T_z }
```

Mechanics identical to today's inline body in `compute_vertical_profiles()`; parameterized by `n_z` for refinement.

### `compare_columns` — optical-depth-weighted, max-envelope

```
compare_columns(lo, hi) -> double

  // (1) Sample frequencies log-spaced across the output range
  N_freq ← params.refine_num_frequencies
  log_min ← log10(params.opacity_nu_min)
  log_max ← log10(params.opacity_nu_max)
  for k in 0..N_freq−1:
    ν_samples[k] ← 10^(log_min + (log_max − log_min) · k / (N_freq − 1))

  dz_lo ← lo.z_max / (lo.n_z − 1)
  C_max[zi] ← 0  for all zi

  for each ν in ν_samples:
    // (2) Per-ν dτ stack
    for zi in 0..lo.n_z−1:
      κ_abs ← opacity_luts.lookup_kappa_abs(ν, lo.ρ_z[zi], lo.T_z[zi])
      κ_es  ← opacity_luts.lookup_kappa_es(lo.ρ_z[zi], lo.T_z[zi])
      dτ_local[zi] ← (κ_abs + κ_es) · lo.ρ_z[zi] · dz_lo

    τ[lo.n_z−1] ← 0
    for zi from lo.n_z−2 down to 0:
      τ[zi] ← τ[zi+1] + 0.5·(dτ_local[zi] + dτ_local[zi+1])

    // (3) Per-ν contribution function, normalized
    for zi: C_ν[zi] ← dτ_local[zi] · exp(−τ[zi])
    Z_ν ← Σ C_ν[zi]
    if Z_ν > 0:
      for zi: C_ν[zi] /= Z_ν

    // (4) Max envelope across frequencies
    for zi: C_max[zi] ← max(C_max[zi], C_ν[zi])

  // (5) Renormalize envelope to a probability mass = 1
  Z ← Σ C_max[zi]
  if Z > 0:
    for zi: w[zi] ← C_max[zi] / Z
  else:
    for zi: w[zi] ← 1 / lo.n_z

  // (6) Soft-weighted-max in normalized z
  zmax_delta ← |lo.z_max − hi.z_max| / lo.z_max
  max_weighted ← 0
  for zi in 0..lo.n_z−1:
    z_norm   ← zi / (lo.n_z − 1)
    ρ_hi_at  ← linear_interp(hi.ρ_z, z_norm · (hi.n_z − 1))
    δ        ← |lo.ρ_z[zi] − ρ_hi_at| / max(lo.ρ_z[zi], 1e-12)
    max_weighted ← max(max_weighted, δ · sqrt(w[zi]))

  return max(zmax_delta, max_weighted)
```

Why soft-weighted-max (`δ · √w`) rather than weighted-mean (`Σ w·δ`): the `√` softens the weight so a single-bin outlier in the optically-thin tail can't dominate, while still preserving the "any single bin's outlier matters" property. Pure weighted-max would obliterate atmosphere errors entirely; pure mean averages out photospheric outliers.

### `refine_n_z_globally`

```
refine_n_z_globally() -> int

  n_z ← params.min_n_z
  cols_lo ← [solve_column(r_i, H_i, T_eff_i, rho_mid_i, n_z) for i in 0..n_r_−1]

  while true:
    n_z_hi ← min(2·n_z, params.max_n_z)
    cols_hi ← [solve_column(r_i, ..., n_z_hi) for i in 0..n_r_−1]

    max_delta ← 0
    for i in 0..n_r_−1:
      max_delta ← max(max_delta, compare_columns(cols_lo[i], cols_hi[i]))

    if max_delta < params.target_lut_eps:
      store_columns_to_2d_lut(cols_hi, n_z_hi)
      return n_z_hi

    if n_z_hi >= params.max_n_z:
      severity ← (max_delta >= 2·params.target_lut_eps) ? Promptable : Warning
      emit(severity, "n_z_cap", "n_z capped at %d with delta=%g > %g")
      store_columns_to_2d_lut(cols_hi, n_z_hi)
      return n_z_hi

    cols_lo ← move(cols_hi)
    n_z ← n_z_hi
```

### `refine_n_r`

Same shape; comparison is a per-column-pair metric over `H_lut_`, `rho_mid_lut_`, `T_eff_lut_` in normalized r. Same severity classification on cap.

### `nested_refine`

```
nested_refine() -> (int, int)

  MAX_NESTED_ITERS = 5
  n_r ← params.min_n_r
  n_z ← params.min_n_z

  for iter in 0..MAX_NESTED_ITERS:
    n_z_new ← refine_n_z_globally(at radii implied by current n_r)
    n_r_new ← refine_n_r(with n_z fixed at n_z_new)
    if n_r_new == n_r and n_z_new == n_z:
      return (n_r, n_z)
    n_r, n_z ← n_r_new, n_z_new

  emit(Promptable, "nested_refine_no_fixed_point",
       "nested refinement did not reach fixed point in 5 iterations")
  return (n_r, n_z)
```

If `params.bins_per_gradient > 0` or `params.bins_per_h > 0`, the corresponding refinement is skipped and the manual value used directly.

### `compute_plunging_region_decay`

```
compute_plunging_region_decay()

  isco_idx ← first i where r_i >= r_isco
  if isco_idx < 0: return

  H_isco       ← H_lut_[isco_idx]
  rho_mid_isco ← rho_mid_lut_[isco_idx]
  p            ← params.plunging_h_decay_exponent

  for i in 0..isco_idx−1:
    r ← r_min_ + (r_outer_ − r_min_)·i / (n_r_ − 1)
    t ← taper(r)                          // existing Gaussian
    H_lut_[i]       ← H_isco · t^p
    rho_mid_lut_[i] ← rho_mid_isco · t
```

### `apply_outer_radial_taper`

```
apply_outer_radial_taper()

  width ← params.outer_taper_width > 0 ? params.outer_taper_width : 2·H_lut_.back()
  if width > (r_outer_ − r_min_) − 0.1·r_outer_:
    emit(Warning, "outer_taper_clamped", "outer_taper_width clamped to fit disk extent")
    width ← (r_outer_ − r_min_) − 0.1·r_outer_

  outer_taper_width_ ← width
  r_taper_start ← r_outer_ − width

  for i in 0..n_r_−1:
    r ← r_min_ + (r_outer_ − r_min_)·i / (n_r_ − 1)
    if r < r_taper_start: continue
    factor ← 1.0 − smoothstep(r_taper_start, r_outer_, r)
    rho_mid_lut_[i] *= factor
```

### `compute_sigma_s_phys`

```
compute_sigma_s_phys()

  if params.noise_compressive_b > 0:
    b ← params.noise_compressive_b
  else:
    peak_idx ← argmax_i(rho_mid_lut_[i] for r_i >= r_isco_)
    T_eff_peak  ← T_eff_lut_[peak_idx]
    T_mid       ← (0.75 · T_eff_peak⁴ · (params.tau_mid + 2/3))^(1/4)
    rho_mid_cgs ← rho_scale_ · rho_mid_lut_[peak_idx]
    rho_mid_cgs ← clamp(rho_mid_cgs, 1e-18, 1e-6)

    mu ← opacity_luts.lookup_mu(rho_mid_cgs, clamp(T_mid, 3000, 1e8))
    if mu <= 0 or !finite(mu): mu ← 0.6

    P_gas ← rho_mid_cgs · k_B · T_mid / (mu · m_p)
    P_rad ← (a_rad / 3) · T_mid⁴
    beta  ← P_gas / (P_gas + P_rad)
    if !finite(beta):
      emit(Info, "beta_fallback", "pressure regime detection failed; using b=0.5")
      b ← 0.5
    else:
      b ← 0.35 + 0.35·(1 − beta)        // 0.35 (gas) → 0.70 (radiation)

  sigma_s_phys_ ← b · sqrt(log1p(params.alpha))

  if sigma_s_phys_ < 0.05 or sigma_s_phys_ > 1.5:
    emit(Info, "sigma_s_atypical", "σ_s_phys=%g outside typical [0.05, 1.5]")

  log "[VolumetricDisk] σ_s_phys = %.4f (b = %.3f, β = %.3f)"
```

### `density()` final form

```
density(r, z, phi) -> double

  if r <= r_horizon_ or r > r_outer_ + 0.5·outer_taper_width_: return 0.0
  z_abs ← |z|
  if z_abs >= z_max_at(r): return 0.0

  rho_mid  ← interp_radial(rho_mid_lut_, r)        // outer taper baked in
  rho_norm ← interp_2d(rho_profile_lut_, r, z_abs)
  base     ← rho_mid · rho_norm · rho_scale_ · taper(r)   // taper(r) = inside-ISCO

  H_local ← scale_height(r)
  c_corr  ← params.noise_correlation_length_factor
  L       ← params.noise_scale > 0 ? params.noise_scale · H_local : c_corr · H_local

  nx ← r·cos(phi) / L
  ny ← r·sin(phi) / L
  nz ← z / L
  n  ← noise_.evaluate_fbm(nx, ny, nz, params.noise_octaves)

  arg ← sigma_s_phys_ · params.turbulence · n
  arg ← clamp(arg, −50.0, 50.0)            // defensive against pathological turbulence
  return base · exp(arg)
```

### `validate_luts`

```
validate_luts() -> bool
  ok ← true
  for i in 0..n_r_−1:
    if any of H_lut_[i], rho_mid_lut_[i], T_eff_lut_[i], z_max_lut_[i] non-finite or negative:
      emit(Severe, "validate_radial", "non-finite or negative cell at r-bin i=%d")
      ok ← false
    for zi in 0..n_z_−1:
      if any of rho_profile_lut_, T_profile_lut_ non-finite or negative:
        emit(Severe, "validate_2d", "non-finite or negative cell at (i=%d, zi=%d)")
        ok ← false

  for i in 1..n_r_−1:
    jump ← |H_lut_[i] − H_lut_[i−1]| / max(H_lut_[i−1], 1e-30)
    if jump > 0.5:
      emit(Promptable, "h_jump", "H jump %.2f at i=%d, smoothness violated")

  // monotonicity in outer-taper zone
  in [r_taper_start, r_outer_]: assert rho_mid_lut_ non-increasing
  if violation: emit(Warning, "outer_taper_non_monotone", ...)

  return ok
```

### CLI prompt logic

`cli/main.cpp` after `grrt_create`:

```cpp
const int prompt_count = grrt_promptable_warning_count(ctx);
if (prompt_count > 0) {
    std::println(stderr, "");
    std::println(stderr, "================================================================");
    std::println(stderr, "Volumetric disk construction completed with {} warning(s):", prompt_count);
    for (int i = 0; i < grrt_warning_count(ctx); ++i) {
        const int sev = grrt_warning_severity(ctx, i);
        if (sev >= GRRT_SEV_PROMPTABLE) {
            std::println(stderr, "  [{}] {}", severity_name(sev), grrt_warning_message(ctx, i));
        }
    }
    std::println(stderr, "================================================================");

    if (force_flag) {
        std::println(stderr, "[grrt-cli] --force specified, continuing.");
    } else if (strict_flag) {
        std::println(stderr, "[grrt-cli] --strict specified, aborting.");
        grrt_destroy(ctx); return 1;
    } else if (!stdin_is_tty()) {
        std::println(stderr,
            "[grrt-cli] Non-interactive session and no --force; aborting"
            " to avoid producing a compromised render.");
        grrt_destroy(ctx); return 1;
    } else {
        std::print(stderr, "Render may be compromised. Proceed anyway? [y/N]: ");
        std::fflush(stderr);
        std::string line;
        std::getline(std::cin, line);
        if (line.empty() || (line[0] != 'y' && line[0] != 'Y')) {
            std::println(stderr, "[grrt-cli] Aborted by user.");
            grrt_destroy(ctx); return 1;
        }
    }
}
```

`stdin_is_tty()` wraps `isatty(fileno(stdin))` (POSIX) / `_isatty(_fileno(stdin))` (Windows).

## Error handling

### Severity matrix

| Trigger | Severity |
|---|---|
| `σ_s_phys` outside `[0.05, 1.5]` | `Info` |
| `b` fallback to `0.5` (β detection failed) | `Info` |
| `outer_taper_width` clamped to fit disk | `Warning` |
| `n_z` cap, `delta < 2·target_lut_eps` | `Warning` |
| `n_r` cap, `delta < 2·target_lut_eps` | `Warning` |
| Outer-taper monotonicity violation | `Warning` |
| `n_z` cap, `delta ≥ 2·target_lut_eps` | **`Promptable`** |
| `n_r` cap, `delta ≥ 2·target_lut_eps` | **`Promptable`** |
| Nested refinement no fixed point | **`Promptable`** |
| `H_lut_` jump > 50% between adjacent radii | **`Promptable`** |
| `solve_column` returned non-finite | **`Promptable`** |
| `validate_luts` non-finite / negative cell | **`Severe`** |

### Default behavior matrix

| Severity | Interactive (TTY) | Non-interactive | `--force` | `--strict` |
|---|---|---|---|---|
| None / `Info` / `Warning` | render | render | render | render |
| `Promptable` | **prompt** | **abort** | render | abort |
| `Severe` | **prompt** | **abort** | render | abort |

### Runtime defenses

`density()` clamps the argument to `exp` to `[-50, 50]` (silently, per-pixel). The construction-time `Info` log on `σ_s_phys` value is the user-facing signal if pathological turbulence settings are in play.

`temperature()` and `inside_volume()` have no new failure modes.

`GeodesicTracer`'s existing per-step Hamiltonian-constraint check (`geodesic_tracer.cpp:228-232` debug-build) becomes slightly more important with tighter margins; otherwise unchanged.

## Testing

### Unit tests (extend `tests/test_volumetric.cpp`)

| Test | Assertion |
|---|---|
| `density_is_strictly_positive_inside_volume` | 1000 random `(r, z, φ)` inside the volume; `density() > 0`. |
| `density_is_zero_outside_volume` | `r > r_outer + 0.5·outer_taper_width` or `\|z\| > z_max(r)` → `density() == 0`. |
| `density_outer_taper_smoothstep` | Sample `density(r, 0, 0)` along radial line in `[r_outer−Δr_out, r_outer]`; fit to `1−smoothstep`; residual < 1%. |
| `H_lut_continuous_across_isco` | Max relative jump in `H_lut_[]` < 5%. |
| `rho_mid_lut_continuous_across_isco` | Same for `rho_mid_lut_[]`. |
| `sigma_s_phys_in_expected_range` | Stellar-mass + 10% Eddington → `0.1 < σ_s_phys < 0.3`. |
| `noise_anisotropy_off` | Spectral content along vertical and radial axes consistent (no extruded-cylinder signature). |
| `data_driven_b_matches_override` | Auto-derived `b` reproduced by forcing `noise_compressive_b` to that value. |

### Refinement tests

| Test | Assertion |
|---|---|
| `refinement_converges_typical_case` | Default params; no cap warning fires. |
| `refinement_warns_at_n_z_cap` | `max_n_z = 32`, `target_lut_eps = 1e-6`; `Promptable` warning emitted. |
| `manual_n_z_skips_refinement` | `bins_per_h = 16`; final `n_z_` deterministic from `max_z_max_over_H · 16`. |
| `nested_refinement_terminates` | Pathological case; iter count ≤ 5. |
| `single_frequency_refinement_equals_old_metric` | `refine_num_frequencies = 1` reduces to single-frequency form. |
| `max_envelope_resolves_widest_photosphere` | Wide ν-range; refinement chose `n_z` ≥ both single-frequency choices. |

### Cross-component invariants

| Invariant | Test |
|---|---|
| Hamiltonian constraint preserved | Existing `H_check` doesn't fire for 1024-pixel high-spin Kerr render. |
| `inside_volume(r, z) ⟺ density(r, z, 0) > 0` | Random points; predicate matches density. |
| τ at midplane near peak ≈ `tau_mid` | Within 5% of `params.tau_mid` after construction. |
| Mass conservation under noise | `mean_φ density(r, z, φ) ≈ rho_smooth(r, z) · exp(σ_s²/2)`. |

### Smoke parameter sweep

```cpp
struct SmokeCase { double mass_solar, eddington_fr, spin, alpha, turbulence; };
const SmokeCase cases[] = {
    {  1.0,  0.01, 0.0,    0.01, 0.0  },
    { 10.0,  0.10, 0.998,  0.10, 1.0  },
    { 1e3,   0.30, 0.5,    0.05, 1.5  },
    { 1e6,   0.10, 0.998,  0.10, 1.0  },
    { 1e9,   0.01, 0.0,    0.30, 0.5  },
    { 0.001, 0.99, 0.99,   0.10, 2.0  },     // micro-BH near Eddington
    { 1e10,  0.001, 0.0,   0.01, 0.0  },     // very SMBH, gas-dominated
};
for each case:
    construct VolumetricDisk;
    assert no Severe warning emitted;
    assert validate_luts() == true;
    assert sigma_s_phys_ > 0 and finite;
```

### Warnings / prompt tests

| Test | Assertion |
|---|---|
| `warnings_collected_into_list` | Bad params → `warnings_.size() > 0` with correct severities. |
| `severe_warning_aborts_in_strict_mode` | Mock CLI; `--strict` returns 1 on Severe. |
| `force_flag_overrides_prompt` | `--force` proceeds despite Promptable. |
| `warning_count_via_c_api_matches_internal` | `grrt_warning_count` matches `disk.warnings().size()`. |

### Manual visual validation (in implementation plan, not automated)

Render checklist of canonical scenes for human inspection:

1. **Stellar-mass Kerr** (M=10, a=0.998, λ_Edd=0.1, observer θ=80°). Expect smooth photosphere ceiling, smooth outer fade, smooth ISCO transition, fluffy noise without vertical-edge holes.
2. **Schwarzschild non-spinning** (M=1, a=0). Sanity case.
3. **High-turbulence** (`turbulence=2.0`). Exercises log-normal extreme behavior.
4. **Mass scan** — stellar-mass + AGN at fixed inclination, identical CLI except `--mass-solar`.
5. **Spectral FITS, ν ∈ `[1e10, 1e16]`**. Verifies max-envelope refinement at frequency-range extremes.

### What is *not* tested

- Image-comparison regression (reference images bit-rot).
- CPU/CUDA equivalence (out of scope; gap covered by `--validate` disablement).
- Performance regression (per-pixel cost change below microbenchmark noise).
- Long-term `σ_s` drift (no dynamic α evolution).

## Future work (explicitly out of scope; tracked here)

1. **CUDA backend update.** Mirror Layer 1 + Layer 2 in `cuda/cuda_volumetric_disk.h` and `cuda/cuda_vol_host_data.cpp`. Restore `--validate` for the volumetric path. **Spec to follow immediately after this lands.**
2. **Power-spectrum-baked 3D noise LUT.** FFT-synthesized noise field with anisotropic Goldreich-Sridhar spectrum (`P(k_⊥, k_∥)`), in cylindrical `(r, φ, z)` coordinates. Uses pocketfft (header-only, BSD). The log-normal composition committed to here makes this a single-method swap in `density()`.
3. **Physical derivation of outer truncation.** Replace the smoothstep `outer_taper_width` with a model derived from viscous timescale, self-gravity, or external boundary conditions.
4. **Physical derivation of inside-ISCO H decay.** Replace the `H_isco · taper(r)^p` form with column-structure integration along Bardeen-Press-Teukolsky plunging streamlines.
5. **Opacity LUT bound extension** for extreme mass scales (densities below `1e-15 g/cm³` for SMBHs, temperatures above `1e8 K` for micro-BHs).
6. **Log-spaced radial bins.** For SMBH disks with very large `r_outer`, switch `interp_radial`/`interp_2d` to log-spacing.

## Related specs

- `2026-03-19-volumetric-accretion-disk-design.md` — original volumetric disk design.
- `2026-03-26-disk-noise-improvements-design.md` — earlier noise iteration this spec evolves.
- `2026-03-27-flux-limited-vertical-structure-design.md` — Eddington atmosphere ODE that Layer 1a extends.
- `2026-04-05-spectral-output-design.md` — spectral FITS output that drives the max-envelope refinement metric.

## Migration / compatibility notes

- `params_.turbulence` and `params_.noise_scale` change semantics. CLI flag names unchanged. Old saved invocations produce different (but smooth and physical) output. Solo user, no script-compatibility concern.
- `--validate` returns immediately for `--disk-volumetric` runs, with explanatory stderr message, until the CUDA follow-up spec lands.
- New default refinement adds ~30 s to disk construction. Negligible against multi-hour render budgets; can be bypassed by setting `bins_per_h` and `bins_per_gradient` directly.

## Implementation surface area estimate

- `src/volumetric_disk.cpp`: ~+200 lines (refinement, taper, plunging decay, σ_s computation, validation, warning emit, removed cosmetic taper code).
- `include/grrt/scene/volumetric_disk.h`: ~+50 lines (warning types, new params fields, helper signatures).
- `src/geodesic_tracer.cpp`: ~−10 lines (margin constants tightened).
- `include/grrt/api.h`: ~+10 lines (warning C API).
- `src/api.cpp`: ~+30 lines (warning C API implementation).
- `cli/main.cpp`: ~+50 lines (prompt logic, `--force`/`--strict` flags, `stdin_is_tty`, `--validate` disablement).
- `tests/test_volumetric.cpp`: ~+150 lines (unit, invariant, smoke, warning tests).

Total ~480 lines added, ~30 lines removed. Single-PR scope.
