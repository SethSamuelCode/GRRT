# Uniform fine raymarch through the disk envelope — design

**Date:** 2026-05-29
**Branch:** fix/volumetric-ring
**Status:** approved (brainstorming) — pending spec review → implementation plan

## 1. Problem

`raymarch_volumetric` renders the volumetric disk with visible **horizontal banding** (edge-on scenes) and **fireflies** (face-on). The banding regression test (`test_volumetric.cpp::test_no_horizontal_bands`, 256×256 spp=30, observer_θ=80°, fov=90°, turbulence=0.4) measures **0.369**, versus a recorded clean baseline of **0.211** and a "buggy" baseline of **0.281** — i.e. worse than the bug it replaced. Threshold is 0.25.

## 2. Root cause (systematic-debugging, 2026-05-29)

The banding is **systematic, not Monte-Carlo variance** — raising spp 30→120 did *not* reduce it (it rose, dominated by a metric row-count confound; the decisive point is it never trended toward the ~0.18 a variance artifact predicts).

Isolation experiment (banding metric at fixed spp=30):

| Config | metric |
|---|---|
| Pre-Task-3 (end-S, H growth cap) | 0.211 (recorded) |
| **EXP1**: midpoint-S, **no z-gate**, H cap | **0.254** |
| **Current**: midpoint-S + z-gate + H/4 cap | **0.369** |

So Task 3's **z-dependent step-size machinery** (`step_needs_z_refinement` gate + the H/4 growth cap) contributes the larger part (+0.115); midpoint-S contributes a smaller residual (+0.043).

**Mechanism.** Emission per step is a single-point (midpoint) quadrature `J += T·S(mid)·(1−e^−dτ)`. The disk emission is sharply peaked at z=0 (base width ~H) and modulated by turbulence with correlation length `L ≈ 0.5·H`. With fine *adaptive* steps the quadrature **resolves and then aliases** that peak against the step grid, and the **binary** refine/don't-refine and inside/outside-envelope decisions inject step-pattern discontinuities. Adjacent image rows (slightly different z-trajectories) fall on opposite sides of these thresholds → row-to-row emission jumps = banding. It is deterministic in the ray geometry, hence spp-immune.

**Turbulence is not in the LUT** (`volumetric_disk.cpp:284-309`): density = smooth base (2D r×|z| LUT) × `exp(σ·turb·fBm(r·cosφ, r·sinφ, z))`, a continuous 3D field with correlation `L=0.5H`. Baking it into a LUT is infeasible (3D at that fidelity ≈ 10¹⁰ cells) and would be ≤ accurate and orthogonal to the banding. So the field representation is correct; the fix is in **along-ray sampling**.

## 3. Goal & constraints

- **Eliminate banding/fireflies** → banding metric < 0.25 (target near or below the 0.211 baseline).
- **Resolve turbulence faithfully** (user decision): no smoothing of the noise; sample finely enough to integrate the real structure, so clumps/holes render correctly.
- **Preserve the transversal-transit emission fix** (bottom disk renders; debug-pixel (150,180) keeps emitting).
- **Performance: quality-first** (user accepts 2–5× slower per affected ray).
- Keep prior fixes intact: the side-impact `raymarch_exits_outer` exit and `RombergStep::mid_state`.

## 4. Design — two-regime deterministic stepper

`raymarch_volumetric` switches stepping mode on whether the photon is inside the disk's vertical envelope `z_max(r)` (`= z_max_at(r)`):

### 4.1 Outside the envelope (`|z| > z_max(r)`) — coarse, adaptive
Density is exactly 0 here, so no emission. Keep the existing coarse adaptive stepping (Romberg step + error-based growth) to reach the disk. **Revert the current envelope-aware growth cap (H/4 inside / H outside) to a plain `H(r)` cap** — the inside is now handled by fixed fine stepping, so the only growth cap that remains, applied outside the envelope, is `H`. The direction-aware outer-radius exit (`raymarch_exits_outer`) is unchanged.

**Delivery to the disk may take one or several coarse steps**, depending on the handoff distance. The orchestrator clamps its step near the disk, so `prev` is typically ~0.25·H outside the envelope → a single (snapped) coarse step reaches `z_max`. The side-impact case (ray entering at `r > r_max` and marching inward through ρ=0 space) or a grazing approach takes several. Both are correct: density is zero in this region, so the coarse step count has **no effect on the image** — only the snap (fine sampling begins exactly at `z_max`) and the uniform fine grid matter. We deliberately do **not** leap to the crossing in one large step: the geodesic is curved and must be integrated, and capping coarse steps at `H(r)` keeps RK4 accurate (`H ≪ r`, the curvature scale). The snap bisection provides the precise landing on the boundary.

### 4.2 Boundary snapping (entry — REQUIRED)
When a proposed coarse step would cross from outside to inside the envelope (the sign of `|z| − z_max(r)` changes across the step), **shorten the step to land on `|z| = z_max(r)`** before switching to fine mode. This is required, not cosmetic: a full coarse step (~`z_max`-sized) that overshoots into the disk skips the upper dense region (at z≈1.3H density is still ~40% of peak), and *how far* it overshoots depends on the coarse step phase → placement-dependent skipped emission → banding returns.

Implementation: bisection on the step length `ds ∈ [0, ds_coarse]` (~12 iterations of `step_kerr` from the current state), using the test `|z(ds)| ≤ z_max(r(ds))`, converging to the boundary. Take that clamped step; the next iteration begins fine mode anchored at the boundary.

### 4.3 Inside the envelope (`|z| ≤ z_max(r)`) — fixed uniform fine steps
Use a **deterministic, uniform** step:

```
ds_fine(r) = min( H(r), L(r) ) / k        // k = FINE_SAMPLES_PER_CORR, default 4
```

where `L(r)` is the turbulence correlation length (§5) and `k` is the quality/faithfulness knob. With the default `L = 0.5H`, this is `ds_fine ≈ H/8` → ~8 samples across the base peak and ~4 across each turbulence correlation length.

At each fine step (reusing existing machinery):
- `RombergStep rs = romberg_step(state, ds_fine, ...)` — supplies per-channel `dtau[]` and `mid_state`.
- Sample S, ρ, T, redshift at `rs.mid_state` and accumulate exactly as today (midpoint-S; unchanged code): `J[ch] += T[ch]·S·(1−e^−dτ); T[ch] ·= e^−dτ`.
- Advance `state = rs.end_state`.
- **No error-based rejection, no z-gate, no growth cap** inside the envelope — the step is fixed and uniform. (`ds_fine` is recomputed each step from local `H,L`; it varies smoothly with r, so no discontinuity → no banding.)

This uniformity is what removes the banding: evenly spaced sample points, no placement-dependent decisions, midpoint quadrature converges smoothly → adjacent rays get near-identical accurate integrals.

### 4.4 Boundary snapping (exit — symmetric)
When a fine step would cross below `−z_max(r)` (leaving the envelope), shorten it to land on `|z| = z_max(r)`, then resume coarse mode. This is the forgiving boundary (ρ→0 there, so it skips no emission); clamped for symmetry and cleanliness.

### 4.5 What is removed vs kept

| | |
|---|---|
| **Remove** | `step_needs_z_refinement` (the z-gate) and its unit tests; the H/4 envelope-aware growth cap |
| **Replace** | the in-disk step-size *policy* → fixed uniform fine + boundary snapping |
| **Keep** | `RombergStep::mid_state` + midpoint-S accumulation; `raymarch_exits_outer` side-impact exit; the dtau/J/T radiative transfer; the disk model & `VolumetricDiskSampler`; `MAX_STEPS = 16384` |

The net effect on `raymarch_volumetric` is a **simplification** of the in-disk logic.

## 5. New accessor: turbulence correlation length

`raymarch_volumetric` needs `L(r)`. Add to `VolumetricDisk`, mirroring the computation already inside `density()`:

```cpp
// include/grrt/scene/volumetric_disk.h
/// Turbulence correlation length L(r) [M] — the spatial scale of the fractal
/// density noise. ds_fine is sized against this so the raymarch resolves it.
double noise_correlation_length(double r) const;

// src/volumetric_disk.cpp
double VolumetricDisk::noise_correlation_length(double r) const {
    const double H = scale_height(r);
    const double c = (params_.noise_correlation_length_factor > 0.0)
                   ? params_.noise_correlation_length_factor : 0.5;
    return (params_.noise_scale > 0.0) ? params_.noise_scale * H : c * H;
}
```

If turbulence is off, `L` is still well-defined (the formula is independent of `turbulence`), so `ds_fine` is sized correctly regardless.

## 6. File-by-file changes

| File | Change |
|---|---|
| `include/grrt/scene/volumetric_disk.h` | Declare `noise_correlation_length(double r)`. |
| `src/volumetric_disk.cpp` | Define it (§5). |
| `include/grrt/geodesic/raymarch_step_control.h` | Remove `step_needs_z_refinement`; **keep** `raymarch_exits_outer`. |
| `src/geodesic_tracer.cpp` | Restructure `raymarch_volumetric`: two-regime stepper (§4.1–4.4); remove the z-gate call + the H/4 cap. Keep the `raymarch_step_control.h` include (`raymarch_exits_outer` is still used). |
| `tests/test_raymarch_step_control.cpp` | Remove the `step_needs_z_refinement` cases; keep the `raymarch_exits_outer` cases. |
| `tests/test_volumetric.cpp` | Banding test: record the Approach-A metric in the calibration-history comment; confirm it passes 0.25 (recalibration is a judgment step — see §8). |

`raymarch_volumetric` is a private method needing a disk + sampler, so it is **not** unit-tested in isolation — verified by integration (§8), matching how Task 3/4 were validated.

## 7. Edge cases

- **Multiple crossings / photon ring.** A ray may enter/exit the envelope several times (lensed transversal transits, the photon ring). The regime logic is per-iteration on the current `|z|` vs `z_max(r)`, so each crossing is snapped and finely sampled independently. `MAX_STEPS=16384` bounds total work.
- **Very thin disk (small H).** `ds_fine` shrinks with `H`, so step count per transit stays ~constant (envelope is ~6H tall → ~`6k/min(1,L/H)` steps ≈ ~48 at default). For pathologically thin disks the `MAX_STEPS` backstop applies (Phase-1 instrumentation showed maxsteps≈0 today; finer stepping raises it but stays well under the cap — verify in §8).
- **Turbulence pushed hard (sharp holes).** Higher-frequency fBm octaves live below `L`. Default `k=4` (ds≈L/4) resolves the dominant octaves; if the user pushes turbulence to very sharp structure, raise `k` (lower `ds_fine`). `k` is a single constant (see §9).
- **Snap bisection failure.** If bisection does not bracket (e.g., a step that grazes tangent to `z_max`), fall back to taking `ds_fine` from the current point (no overshoot possible since we're at/near the boundary). Bound iterations (~12).
- **Boundary at `r > r_max`.** The side-impact path lets inward rays march through `r>r_max` (ρ=0 there, coarse). The envelope test still applies once `r ≤ r_max`.

## 8. Testing & verification

**Unit (TDD):**
- `test_volumetric.cpp`: assert `noise_correlation_length(r)` equals `c_corr·scale_height(r)` for a default disk (and `noise_scale·H` when `noise_scale>0`).
- `test_raymarch_step_control.cpp`: `raymarch_exits_outer` cases remain green; `step_needs_z_refinement` cases removed.
- Existing suites stay green: `test-romberg-step`, `test-disk-step-entry`, `test-spectral`.

**Integration:**
- **Banding regression** (`test_no_horizontal_bands`): metric must drop below 0.25 (target ≤ ~0.21). Record the new value in the calibration-history comment. **JUDGMENT GATE:** if the new value is unexpectedly high (>0.25) despite a visually clean render, surface the before/after to the user before any threshold change — do not loosen the threshold silently.
- **Debug-pixel (150,180)**: the transversal-transit RAYMARCH still returns non-zero emission (bottom disk preserved).
- **Turbulent visual render** (`--disk-volumetric --samples 100 --fov 30`): clumps/holes still render; no horizontal stripes. Left on disk for user inspection (executor can read the PNG).
- **MAX_STEPS check**: confirm rays do not routinely hit 16384 (no silent optical-depth truncation) via the existing/temporary exit-reason instrumentation pattern.

**Performance:** report wall-clock for the spp=100 render (expected slower; quality-first — acceptable).

## 9. Tuning knob

`k = FINE_SAMPLES_PER_CORR` is a single `constexpr int` in `raymarch_volumetric`, default **4**. Lower = faster/coarser; higher = finer (sharper turbulence/holes). Exposing it as a CLI/`VolumetricParams` field is intentionally **out of scope** here (YAGNI); it can be promoted to a parameter in a follow-up if interactive control proves necessary.

## 10. Out of scope

- The **spectral** raymarch (`raymarch_volumetric_spectral`) — a separate fixed-step integrator; its own banding/quality work is a future task (the planned `romberg_step` unification).
- Variance-reduction / importance sampling beyond uniform fine stepping.
- Exposing `k` as a user parameter (§9).
