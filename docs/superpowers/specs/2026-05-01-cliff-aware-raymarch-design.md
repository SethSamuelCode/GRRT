# Cliff-aware raymarch via Romberg-controlled step

**Date:** 2026-05-01
**Branch:** `fix/volumetric-ring`
**Author:** brainstormed with Claude

## Summary

Replace the current fixed-clamp adaptive step in `raymarch_volumetric` with a Romberg-style adaptive step controller that bounds per-step optical-depth error to a user-tunable tolerance. This eliminates the horizontal banding artifact caused by undersampling the photosphere cliff in the volumetric disk's vertical density profile, and makes integration accuracy a principled, user-controlled quantity rather than a hidden tuning constant.

The work also factors out a reusable `romberg_step` helper that batches multiple frequency channels through a single shared geodesic step. The RGB raymarcher adopts it now; the spectral raymarcher can adopt it later with no further design work.

## Background

### The bug

Rendering with `--disk-volumetric --samples 30` produces visible horizontal bands across the disk image. Investigation (debugging session 2026-05-01) showed:

- The volumetric disk's vertical extent reaches `z_max(r) ≈ 30·H(r)`, dominated by an essentially-empty atmosphere above a sharp **photosphere cliff** that's only ~1H thick.
- The current raymarch's adaptive step uses `dτ ≈ 0.05` per step, but `ds` is hard-clamped to `[H/64, H]`. In the atmosphere `κρ` is ~0, so `ds` gets pinned at `H_max = H` — meaning the entire 30H column is traversed in ~30 samples.
- Adjacent rays therefore sample the cliff at different *phases*: one ray's sample lands inside the cliff and accumulates large `Δτ`; the next ray's sample lands above and accumulates almost nothing. Image-y rows acquire structured intensity differences → bands.

A diagnostic visualization is at `cliff_undersampling_viz.html`.

### Why the existing fix (`H_max → H/8`) is unsatisfying

Tightening the clamp from `H` to `H/8` makes the bands disappear (verified empirically), at ~1.5–2× render-time cost. But this is an arbitrary constant divorced from physics: there's no convergence guarantee, no way for the user to ask for higher fidelity, and the constant will need re-tuning when disk parameters change. A correctness-first design must replace the tuning constant with a quantity the integrator itself controls.

## Goals

- **G1.** Eliminate the horizontal-banding artifact at default settings, verifiable by automated test.
- **G2.** Expose render fidelity as a user-tunable tolerance (`--raymarch-tol`) with a physically meaningful unit (per-step error in optical depth τ).
- **G3.** Bound the per-step τ-integration error for *every* control channel the caller passes, not just one. The bound is local truncation error per step, capped by the user's tolerance.
- **G4.** Provide the helper as a reusable building block that the spectral raymarcher can adopt later without redesign.

## Non-goals

- Changing the spectral raymarcher (`raymarch_volumetric_spectral`) in this work. The helper is shaped to support it, but wiring is a separate PR.
- Fixing the `n_z` / `n_r` LUT cap-binding warnings (already tracked as Phase 3 future work; orthogonal to this design).
- Implementing optimizations identified during brainstorming (τ-scaled tolerance, Romberg trajectory reuse). Both noted in `docs/superpowers/optimizations/`.
- CUDA-side parity. Phase 2 will mirror the design separately.

## Design

### Architecture overview

One new helper function, modifications to one existing function (`raymarch_volumetric`), one new CLI flag, and three test files.

### The `romberg_step` helper

```cpp
// MAX_CH bounds the per-step channel array; covers RGB (3) and modest spectral
// outputs without heap allocation. Spectral callers wanting more bins must
// either raise this constant or split their bins across multiple helper calls.
constexpr int MAX_CH = 32;

struct RombergStep {
    GeodesicState end_state;            // state at end of accepted step
    std::array<double, MAX_CH> dtau;    // per-channel Δτ from the half-step pass
    double max_err;                     // max over channels of |Δτ_full − Δτ_half|
    double ds_taken;                    // = ds_proposed (helper does not shrink)
    int n_channels;                     // count of valid entries in dtau[]
};

RombergStep romberg_step(
    GeodesicState state,
    double ds_proposed,
    std::span<const double> control_channels,
    const VolumetricDisk& disk,
    const KerrMetric& metric,
    const RK4& integrator
);
```

**Internal behaviour:**

1. Take **one** geodesic step of length `ds_proposed`. Sample `(ρ, T_local)` at start and end. For each control channel, look up `κ_abs(ν, ρ, T) + κ_es(ρ, T)` and compute `Δτ_full[ch]` by trapezoidal rule.
2. Take **two** geodesic substeps of length `ds_proposed/2`. Sample at start, midpoint, end. Compute `Δτ_half[ch]` by composite trapezoid.
3. Return `Δτ_half[]` (more accurate by one Romberg order) and `max_err = max_ch |Δτ_full[ch] − Δτ_half[ch]|`.

**Why the caller passes `control_channels`:** the helper does **one** geodesic step path and **one** density/temperature lookup at each sample point. The per-channel work is only the cheap `κ` LUT lookup. Batching N control channels through this single pass amortizes the geodesic and density costs across all of them. RGB passes 3 channels; spectral can pass any subset of its bins it wants bounded.

**Why we report `max_err` across channels:** a step is accepted only when **all** control channels meet the tolerance. This guarantees per-step τ error ≤ `tol` for every wavelength the caller cares about — no monotonicity-in-ν assumption, no edge handling, no special cases.

### Caller — `raymarch_volumetric` (RGB)

The radiative transfer arithmetic (`J += T·S·(1-exp(-dτ))`, `T *= exp(-dτ)`) and all geometric exit conditions (horizon, escape, opaque, leaving disk volume) are unchanged. Only the step-control machinery changes.

```cpp
void GeodesicTracer::raymarch_volumetric(GeodesicState& state, Vec3& color,
                                          double J_rgb[3], double T_rgb[3]) const {
    constexpr std::array<double, 3> nu_obs = {
        c_cgs / 450e-7, c_cgs / 550e-7, c_cgs / 650e-7
    };

    double J[3] = {J_rgb[0], J_rgb[1], J_rgb[2]};
    double T[3] = {T_rgb[0], T_rgb[1], T_rgb[2]};
    double ut_obs = 1.0 / std::sqrt(1.0 - 2.0 / observer_r_);

    double r = state.position[1];
    const double z_start = r * std::cos(state.position[2]);
    const double H_start = vol_disk_->scale_height(r);
    double ds_proposed = vol_disk_->inside_volume(r, z_start)
                       ? H_start / 16.0
                       : std::min(std::abs(z_start) / 8.0, H_start * 2.0);

    int step_count = 0;
    constexpr int MAX_STEPS = 4096;

    while (step_count < MAX_STEPS) {
        if (r < vol_disk_->r_horizon())                                    break;
        if (r > vol_disk_->r_max())                                        break;
        if (T[0] < 1e-6 && T[1] < 1e-6 && T[2] < 1e-6)                     break;

        auto rs = romberg_step(state, ds_proposed,
                               std::span{nu_obs.data(), 3},
                               *vol_disk_, metric_, integrator_);

        if (rs.max_err > raymarch_tol_) {
            ds_proposed *= 0.5;
            const double ds_floor = vol_disk_->scale_height(r) / 256.0;
            if (ds_proposed < ds_floor) ds_proposed = ds_floor;
            continue;
        }
        step_count++;

        // Per-channel radiative transfer using rs.dtau[ch] — same arithmetic
        // as today, just with dτ supplied by the helper instead of computed inline.
        const double r_step = rs.end_state.position[1];
        const double theta_step = rs.end_state.position[2];
        const double phi_step = rs.end_state.position[3];
        const double z_step = r_step * std::cos(theta_step);
        const double rho_cgs = vol_disk_->density_cgs(r_step, z_step, phi_step);
        const double T_local = vol_disk_->temperature(r_step, std::abs(z_step));
        if (rho_cgs > 0.0 && T_local > 0.0) {
            // ... compute redshift g, source S, ε per channel ...
            for (int ch = 0; ch < 3; ++ch) {
                const double dtau = rs.dtau[ch];
                const double exp_dtau = std::exp(-dtau);
                J[ch] += T[ch] * S[ch] * (1.0 - exp_dtau);
                T[ch] *= exp_dtau;
            }
        }

        state = rs.end_state;
        r = state.position[1];

        if (rs.max_err < raymarch_tol_ / 8.0) {
            ds_proposed = std::min(ds_proposed * 2.0,
                                    vol_disk_->scale_height(r));
        }
    }

    J_rgb[0] = J[0]; J_rgb[1] = J[1]; J_rgb[2] = J[2];
    T_rgb[0] = T[0]; T_rgb[1] = T[1]; T_rgb[2] = T[2];
}
```

**Deleted:** `DTAU_TARGET = 0.05` constant; `ds_tau`/`ds_geo`/`clamp[H/64, H]` block at the bottom of the loop.

**Step-size growth:** when an accepted step's error is well under tolerance (`< tol/8`), the proposed step doubles for the next iteration, capped at `1·H` (same upper bound as the prior implementation — preserved for safety). This keeps the loop responsive to changing local conditions: it shrinks fast at the cliff, grows fast in the atmosphere.

**Safety floor:** `ds_floor = H/256` prevents pathological infinite-shrink loops if `tol` is set unreasonably tight or the LUT itself contains a discontinuity. If shrinking hits the floor, the step is accepted at floor size — implicitly accepting that the step's error will exceed `tol`. (This is correct behaviour: the LUT, not the integrator, is at fault.)

### Tolerance — what it bounds

`tol = 1e-2` (default) bounds **per-step local truncation error in optical depth τ** for every control channel.

Translation into observable error in transmission `T = exp(−τ)`:

| Current τ | T | dT for dτ = tol = 0.01 |
|---|---|---|
| 0 | 1.00 | 1.0% |
| 1 (photosphere) | 0.37 | 0.37% |
| 3 | 0.05 | 0.05% |
| 10 | 4.5×10⁻⁵ | 4.5×10⁻⁷ |

The tolerance is *worst* at the surface (where it should be) and exponentially tighter deeper in the disk (where transmission is already tiny and error doesn't propagate to the image).

Total integrated error in image intensity J across the whole ray is bounded approximately by `S · tol`, independent of how many steps the integrator takes — a clean global bound.

### CLI flag

`--raymarch-tol T` with default `1e-2`. Suggested values:

- `1e-3` — very high fidelity; ~3× cost vs default.
- `1e-2` — **default**. Smooth, no visible bands, ~2× cost vs current `H_max=H` behaviour.
- `1e-1` — preview quality; faint bands may return; ~30% faster than default.
- `1e0` — essentially equivalent to today's `H_max = H` behaviour; bands return; baseline cost.

### Wavelength independence — what we do *not* assume

Photons of all wavelengths follow the *same* geodesic (vacuum and our non-dispersive medium). Per-channel quantities differ only in `κ(ν, ρ, T)`. The helper exploits this by computing the path *once* and per-channel τ contributions cheaply.

We do **not** assume `κ(ν)` is monotonic across the control channels. We do **not** require the caller to pre-compute "worst-case ν." We bound error for every channel passed in, full stop.

### Spectral raymarcher (future, not in this work)

`raymarch_volumetric_spectral` adopts the helper later by passing its own frequency bins as `control_channels`. No design changes. The cost scales linearly with the number of bins passed (κ-LUT lookups dominate the per-channel cost). A "smart spectral caller" might pass only `(ν_min, ν_max)` plus opacity-edge frequencies; a "safe" one passes every bin. Both correct.

## Validation & testing

### 6a — Unit tests for `romberg_step` (new file `tests/test_romberg_step.cpp`)

Synthetic disk with analytic `ρ(z)` (Gaussian core × tanh cliff) and analytic `κ(ν, ρ, T)`. Tests:

- Smooth-region order test: `max_err` falls as `O(ds⁴)` when `ds` is halved (Romberg's expected order).
- Cliff resolution test: starting with large `ds` across a cliff, the helper's reject/halve loop converges to a step that produces accepted `Δτ` matching the analytic answer to within `tol`.
- Tolerance scaling test: `tol = 1e-1` produces visibly larger accepted steps than `tol = 1e-3` on the same path.
- Multi-channel correctness: with three control channels having different `κ(ν)`, each channel's `Δτ` matches analytic and `max_err` reflects the channel with the largest local error.
- Synthetic edge test: a control-channel pair whose `κ(ν)` is non-monotonic across the channels still produces correct `Δτ` for every channel.

### 6b — Tolerance convergence test (extends `tests/test_volumetric.cpp`)

Render a 64×64 image at three tolerances. Linear-light comparison before tone mapping:

- `def` at `1e-2` vs `ref` at `1e-3`: `max-pixel |def − ref| < 1%`.
- `loose` at `1e-1` vs `ref`: `max-pixel |loose − ref| < 5%` AND `loose` runs at least 2× faster than `ref`.

Validates the user-facing knob does what its docs claim.

### 6c — Banding regression test (extends `tests/test_volumetric.cpp`)

Render the user's bug scenario (`--samples 30 --disk-volumetric` defaults). For each row of pixels that intersects the disk, compute the standard deviation of intensity along that row. The bands manifest as anomalously high σ on rows that cross the disk surface (where the cliff position varies relative to the discrete sample grid).

Concrete pass criterion: along the horizontal centerline (image-y = height/2 ± 5 rows), the linear-light intensity standard deviation σ along each scanline must be **less than 30% of the row's mean intensity**. The original buggy build produced σ/mean ≈ 0.7–0.9 on these rows; the H/8 debug build measured ≈ 0.15. The new Romberg-based build is expected to fall comfortably below the threshold.

This is calibrated empirically — captured during this work, recorded in the test file as a comment alongside the threshold constant.

### Out of scope for testing in this work

- Spectral path validation (spectral path not modified).
- CUDA backend parity (Phase 2).
- LUT cliff cleanliness (Phase 3 future work — has its own validation gate).

## Error handling & edge cases

- **`ds_proposed` shrinks below floor `H/256`:** accept the step at floor size; `rs.max_err` may exceed `tol`. Real-world cause is almost always a discontinuity in the LUT itself; emit a debug-only warning to stderr (gated on `NDEBUG`).
- **Geodesic step lands outside the disk volume:** existing exit logic in the caller handles this — unchanged.
- **`control_channels` is empty:** `romberg_step` returns `max_err = 0` and `n_channels = 0`. The caller should not pass empty arrays; tests assert this is not a hot path.
- **Negative or zero `tol`:** the CLI parser rejects `tol ≤ 0` at startup with an error message ("--raymarch-tol must be > 0"). As a defensive second layer, the helper treats `tol ≤ 0` internally as `1e-12` to prevent division-by-zero or infinite-shrink loops if a programmatic caller (Python binding, tests) passes a bad value.
- **`raymarch_tol_` not initialized in legacy contexts:** default-initialize to `1e-2` in `GeodesicTracer`'s default constructor and member initializer list.

## File changes

**New:**
- `src/romberg_step.cpp`
- `include/grrt/render/romberg_step.h`
- `tests/test_romberg_step.cpp`

**Modified:**
- `src/geodesic_tracer.cpp` — `raymarch_volumetric` rewired; `DTAU_TARGET` and old step-control block deleted.
- `include/grrt/render/geodesic_tracer.h` — add `double raymarch_tol_` member.
- `include/grrt/api.h` — add `double raymarch_tol` field to `GRRTContextParams`, default `1e-2`.
- `src/api.cpp` — pass `params->raymarch_tol` through.
- `cli/main.cpp` — `--raymarch-tol` flag; help-text update.
- `tests/test_volumetric.cpp` — tolerance convergence + banding regression tests.

**Reverted:**
- The current `H/8` debug edit at the bottom of `raymarch_volumetric` (made obsolete by the helper).

## Future work

- **`raymarch_volumetric_spectral` adoption.** One PR: add `--spectral-raymarch-tol` (or share `--raymarch-tol`), call the helper with the user's frequency bins. No design needed.
- **τ-scaled tolerance** for per-render speedup. See `docs/superpowers/optimizations/2026-05-01-tau-scaled-raymarch-tolerance.md`.
- **Romberg trajectory reuse** for per-step speedup. See `docs/superpowers/optimizations/2026-05-01-romberg-trajectory-reuse.md`.
- **LUT cliff cleanup** (existing Phase 3 future work). Romberg integrates the LUT we hand it correctly; if the LUT itself is discontinuous, that's the LUT's bug, not Romberg's. Cleaner LUTs let Romberg take larger steps and reach lower per-render cost.

## References

- **Romberg integration / Richardson extrapolation:** Stoer & Bulirsch, *Introduction to Numerical Analysis* (3rd ed.), §3.4.
- **Adaptive integration with embedded error estimate:** Press et al., *Numerical Recipes in C++* (3rd ed.), §17.2 (adaptive Runge-Kutta) — the same principle applied to a quadrature.
- **Radiative transfer formalism (J / T accumulation):** Mihalas & Mihalas, *Foundations of Radiation Hydrodynamics*, §6.2.
- **Optical depth and the photosphere:** Rybicki & Lightman, *Radiative Processes in Astrophysics*, §1.4.
- **Existing project context:** `cliff_undersampling_viz.html` (visualization of the bug); `docs/superpowers/specs/2026-04-29-bpt72-taper-and-dprk45.md` (prior numerics work that established the disk's vertical structure).

## Migration notes

This work has **no API breaking changes**. Existing callers of the C API and CLI continue to work; they get the new default `tol = 1e-2` automatically and benefit from the bug fix without code changes.

The CUDA path (Phase 2) is not yet implemented; mirroring this design there is part of Phase 2's separate planning.
