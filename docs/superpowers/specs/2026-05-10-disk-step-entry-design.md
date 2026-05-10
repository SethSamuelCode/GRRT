# Segment-aware disk-entry detection

**Date:** 2026-05-10
**Branch:** `fix/volumetric-ring`
**Status:** spec — pending implementation plan

## 1. Problem

`--disk-volumetric` renders at narrow FOV (e.g. `--fov 30`) exhibit sharp,
wedge-shaped black voids cutting into what should be solid disk material.

Reproduction:
```
./grrt-cli.exe --disk-volumetric --samples 100 --width 256 --height 256 --output t --force --fov 30
```

Confirmed by user as pre-existing the Owen-scrambled Sobol switch (`f058c8c`)
and likely also the cliff-aware-raymarch work. Documented in
`docs/superpowers/known-issues-2026-05-02.md` as item 1 (top priority).

## 2. Root cause

The geodesic-tracer's existing entry-detection predicate is **endpoint-only**.
At three call sites in `src/geodesic_tracer.cpp` (lines 192–202, 441–449,
585–595), each integration step decides whether to raymarch via:

```cpp
should_raymarch = crossed_midplane || inside_now || near_disk;
```

`crossed_midplane` flips on `z_prev * z_new < 0`; `inside_now` queries
`is_in_volume(r_new, z_new)`; `near_disk` checks `|z| < z_max(r) + H` at
either endpoint within an r-cushion.

This predicate samples the geodesic only at step *endpoints*. At narrow FOV,
rays cluster near the optical axis and refract through the photon ring along
strongly curved trajectories. A single integration step can:

1. Start with `z_prev > z_max(r) + H` (above the disk, predicate false),
2. Curve through the disk volume mid-step (`|z(λ)| < z_max(r)` for some
   `λ` interior to the step),
3. End with `z_new > z_max(r) + H` (still above, predicate still false),

without changing midplane sign and without either endpoint satisfying
`inside_now` or `near_disk`. The raymarch never fires; the pixel renders
black; the user sees a wedge.

A second contributing failure mode: the LUT cap-binding warnings (issue 3 in
`known-issues-2026-05-02.md`) document `H` jumps of ~97% between adjacent
r-bins, which translate to similarly steep cliffs in `z_max(r)`. A grazing
ray crossing such a cliff in `(r, z)` space sees a near-vertical wall;
endpoint sampling can land on either side without the predicate firing.

Both failure modes share a structural cause: **endpoint predicates cannot
detect events that occur strictly between endpoints when the trajectory is
curved relative to the step size**. Geodesic curvature in `(r, z)` near the
photon ring is high enough per step to violate the implicit "segments are
nearly straight" assumption of endpoint-only sampling.

## 3. Goals and non-goals

**Goals.**
- Eliminate wedge artifacts in narrow-FOV `--disk-volumetric` renders.
- Robust to both step-too-coarse and LUT-cliff failure modes.
- Deduplicate the three-way duplicated entry-detection code in
  `geodesic_tracer.cpp`.
- Preserve existing behavior on the no-bug path (deep-space rays, clean
  disk-crossings) byte-for-byte where possible.
- Mass-regime robust: works correctly for stellar-mass through supermassive
  black holes via adaptive resolution.

**Non-goals.**
- Not fixing the LUT cap-binding warnings themselves (issue 3, separate
  spec).
- Not fixing the tau midplane test factor-of-4 (issue 2, separate spec).
- Not adding quantitative wedge-regression test that depends on disk-model
  parameters (would break under disk tuning; user preference).
- Not optimizing for performance until correctness is verified
  (correctness-first; tuning sweeps deferred to post-validation work, §8).

## 4. Architecture

A new free-function helper, `check_disk_step_entry`, replaces the inlined
endpoint-only predicate at all three call sites. Internally the helper
applies a three-tier gate:

```
Tier A: existing endpoint predicate (5 flops)         → return immediately if true
Tier B: segment-bound test (15-20 flops)              → if false, return false
Tier C: recursive substep + recheck Tier A            → up to adaptive depth_limit
```

If Tier C surfaces an interior detection, the helper returns
`should_raymarch = true` along with a `refined_endpoint`: the substep state
where Tier A first fired. The caller's main loop resumes from
`refined_endpoint` rather than the original `new_state`, so the next step
does not re-cross the same disk segment.

If Tier C exhausts depth without firing Tier A, conservative policy applies
(see §6.1): return `should_raymarch = true` with `refined_endpoint` set to
the original `new_state`. The raymarch loop's own `is_in_volume` check
short-circuits if no actual entry exists.

## 5. Components

### 5.1 Public API

`include/grrt/geodesic/disk_step_entry.h`:

```cpp
namespace grrt {

struct DiskStepEntryResult {
    bool should_raymarch;
    GeodesicState refined_endpoint;   // valid only when should_raymarch == true
};

struct DiskStepEntryOptions {
    int    depth_limit_floor = 4;     // minimum subdivisions
    int    depth_limit_cap   = 10;    // hard ceiling (1024x refinement)
    double curvature_pad     = 0.5;   // chord-length multiplier; see §5.4
};

DiskStepEntryResult check_disk_step_entry(
    const GeodesicState& prev_state,
    const GeodesicState& new_state,
    const VolumetricDisk& disk,
    const GeodesicTracer& tracer,
    const DiskStepEntryOptions& opts = {});

} // namespace grrt
```

The `tracer` reference lets the helper substep using the same integrator the
main loop uses — no duplicated integration code, identical numerical
behavior. The helper is pure (no globals, no statics), reentrant, and
OpenMP-safe.

The main loop's contract: when `should_raymarch == true`, the caller:
1. Runs `raymarch_volumetric` from `(prev_state, refined_endpoint)`.
2. Treats `refined_endpoint` (not the original `new_state`) as the basis
   for the next integrator step.

### 5.2 Endpoint-predicate extraction

The duplicated `crossed_midplane || inside_now || near_disk` block at three
sites is factored into a single private function in
`src/disk_step_entry.cpp` (anonymous namespace):

```cpp
bool endpoint_predicate(const GeodesicState& prev,
                        const GeodesicState& curr,
                        const VolumetricDisk& disk);
```

Initial implementation is byte-for-byte equivalent to the inline blocks at
`geodesic_tracer.cpp` lines 192–202, 441–449, 585–595. The three call sites
are then replaced with calls to `check_disk_step_entry`, which delegates to
`endpoint_predicate` for Tier A.

### 5.3 Segment-bound test (Tier B)

```cpp
bool segment_could_intersect_disk(
    const GeodesicState& prev,
    const GeodesicState& curr,
    const VolumetricDisk& disk,
    double curvature_pad);
```

Constructs a conservative `(r, |z|)` bounding rectangle for the segment and
tests intersection with the disk's volume envelope. Three rejection paths:

1. Segment's r-range disjoint from `[r_horizon, r_outer + outer_taper_width]`
   → reject fast.
2. Segment's `|z|`-range entirely above `max(z_max(r) + 0.5*H(r))` over the
   segment's r-range → reject fast.
3. Otherwise → return true (subdivide).

The `max(z_max(r) + 0.5*H(r))` is computed by sampling the LUT at endpoints
plus midpoint of the segment's r-range (3 LUT queries). Conservative because
LUTs are smooth-ish across r at the resolution the bound requires.

### 5.4 Curvature pad — velocity-aware

Both endpoints can be above the disk top while the trajectory dips below.
The conservative `|z|`-min for the segment uses both a chord-based and a
velocity-based estimate.

Notation: `z_prev`, `z_curr` are the Cartesian z-coordinates derived from
`GeodesicState` (`r * cos(θ)`); `vz_prev`, `vz_curr` are the corresponding
contravariant z-velocities `dz/dλ`, derived from the contravariant
position-derivatives `(dr/dλ, dθ/dλ)` returned by
`RK4::derivatives_kerr(metric, state)` via the same chain rule the
renderer's existing step clamp uses (`src/geodesic_tracer.cpp:147-152`):
`dz/dλ = cos(θ) · dr/dλ − r · sin(θ) · dθ/dλ`.
`dλ_full` is the proper-time duration of the full integrator step
(`λ_curr − λ_prev`).

`GeodesicState::momentum` stores covariant `p_μ`; using it directly in the
chain rule would conflate energy with velocity (units mismatch by a factor
of inverse-metric-component, ~Σ for Kerr). Tier B remains conservative
either way, but using contravariant velocities gives a tight, physically
meaningful bound.

```cpp
const double dz_chord = std::abs(z_prev - z_curr);
const double dz_swing = 0.5 * std::abs(vz_prev - vz_curr) * dλ_full;
const double pad      = std::max(opts.curvature_pad * dz_chord, dz_swing);
const double abs_z_min = std::max(0.0,
                                  std::min(std::abs(z_prev), std::abs(z_curr)) - pad);
```

The `dz_swing` term protects against trajectories where the chord is small
but `|dz/dλ|` is large — i.e., a ray that enters and re-exits the disk top
within one step. For such rays the chord-based pad alone underestimates the
trajectory's `|z|` excursion.

If the segment crosses midplane (`z_prev * z_curr < 0`), `abs_z_min` is
forced to 0 (trajectory definitely passes through midplane).

`curvature_pad = 0.5` is a tunable default. Over-padding produces extra
subdivisions (cost) but never misses entries; under-padding risks the
original bug class. Start safe.

### 5.5 Subdivision recursion (Tier C) — adaptive depth

Internal to the helper:

```cpp
DiskStepEntryResult subdivide(prev, curr, depth_remaining, dλ_remaining):
    if (depth_remaining == 0)
        return { true, curr };   // conservative, see §6.1

    // Substep: re-integrate from prev with dλ/2 using tracer
    GeodesicState mid = tracer.step(prev, dλ_remaining * 0.5);

    // Tier A on each half
    if (endpoint_predicate(prev, mid, disk))
        return { true, mid };
    if (endpoint_predicate(mid, curr, disk))
        return { true, curr };

    // Tier B on each half — recurse only on halves that might still contain entry
    if (segment_could_intersect_disk(prev, mid, disk, curvature_pad))
        if (auto r = subdivide(prev, mid, depth_remaining - 1, dλ_remaining * 0.5);
            r.should_raymarch)
            return r;

    if (segment_could_intersect_disk(mid, curr, disk, curvature_pad))
        return subdivide(mid, curr, depth_remaining - 1, dλ_remaining * 0.5);

    return { false, {} };
```

The initial `depth_remaining` is computed adaptively at the helper's top
level from the segment's local geometry. `dλ_full` here is the same
proper-time duration introduced in §5.4:

```cpp
const double H_min = std::min(disk.scale_height(r_prev),
                              disk.scale_height(r_curr));
const int needed = static_cast<int>(std::ceil(std::log2(dλ_full / H_min)));
const int depth_limit = std::clamp(needed,
                                   opts.depth_limit_floor,
                                   opts.depth_limit_cap);
```

This guarantees the smallest substep is on the order of `H_min` whenever the
floor-and-cap permit. Mass-regime behavior:

| BH regime | Typical `H/M` | Adaptive depth |
|---|---|---|
| Stellar (10 M_sun) | 0.1 – 1 | 4 (floor) |
| Intermediate (10⁴ M_sun) | 0.05 – 0.5 | 4 – 5 |
| Supermassive AGN (10⁸ M_sun) | 0.01 – 0.1 | 7 – 8 |
| Pathological (clamp floor 0.001) | 0.001 | 10 (cap), then conservative |

## 6. Error handling and edge cases

### 6.1 Subdivision depth exhaustion — conservative policy

If the recursion exhausts `depth_limit` while Tier B kept passing, return
`should_raymarch = true` with `refined_endpoint = deepest_substep_mid`. The
raymarch path then handles "no actual entry" via its own `is_in_volume`
check, which short-circuits in ~1µs.

This is the **conservative** policy: never miss entry, slight over-cost.
The optimistic alternative (return false on exhaustion) is rejected because
it risks the original bug class for sub-`H` photosphere grazes.

The adaptive `depth_limit` from §5.5 means exhaustion is rare except in
pathological mass regimes (clamp-floor disks).

### 6.2 Substep integration produces a bad state (NaN, energy drift)

Treat as Tier A firing: return
`{should_raymarch = true, refined_endpoint = prev_state}`. Existing
raymarch defensive handling for bad states applies. Zero new code path for
NaN handling.

### 6.3 Degenerate inputs

- Empty disk r-range or zero-sized `z_max_lut` → return
  `should_raymarch = false` immediately.
- `dλ` between `prev_state` and `new_state` is zero or negative → run Tier A
  only, skip Tier C.
- `prev_state` inside horizon or beyond escape boundary → run Tier A only.
  Substepping past these boundaries is undefined.

### 6.4 Both endpoints already inside disk volume

Tier A (`inside_now`) fires; no subdivision needed. The new code path is
**strictly additive** on the no-bug case — existing behavior preserved.

### 6.5 Thread safety and reproducibility

- Helper is pure: takes references, returns by value, no globals.
  OpenMP-safe by construction.
- Substep integration via `tracer.step(state, dλ)` is deterministic for
  identical inputs. Helper preserves Sobol's deterministic-render guarantee.
- Sobol sub-pixel sampling is upstream of the helper; no interaction.

### 6.6 False-positive subdivision risk

Over-tuned `curvature_pad` causes wasteful subdivision. Mitigations:

- Default 0.5 plus the momentum-aware `dz_swing` term provides correctness
  margin without aggressive over-padding.
- Diagnostic counter `tracer.substep_invocation_count_` (atomic, OpenMP-safe)
  bumped each time Tier C recurses. After fix lands, render the repro and
  report `(invocations / total_steps)`. If `>> 1%`, flag for tuning
  follow-up.
- Both `curvature_pad` and `depth_limit_cap` are tunable via
  `DiskStepEntryOptions` for production scenes if needed.

## 7. Testing

### 7.1 Unit tests for helper components

New file `tests/test_disk_step_entry.cpp`. Pure helper logic with synthetic
states and a minimal `VolumetricDisk` — no rendering, no LUT construction —
so tests are stable across disk-model parameter tuning.

| Test | Verifies |
|---|---|
| `test_endpoint_predicate_equivalence` | Extracted free function produces identical results to original three-site inline logic across a battery of `(prev, curr)` pairs. Lock-step extraction. |
| `test_segment_bound_rejects_far_above` | Segment with both endpoints at `\|z\| = 50M` rejects fast. |
| `test_segment_bound_passes_when_dipping` | Synthetic step with large `\|p_z\|` such that `dz_swing > min(\|z_prev\|, \|z_curr\|) - z_max`; helper returns subdivide. |
| `test_subdivide_finds_interior_entry` | Construct `prev,curr` where Tier A fails on full step but fires on half-step. Helper returns `{should_raymarch=true, refined_endpoint=mid}`. |
| `test_subdivide_depth_limit_respected` | Pathological: Tier B always passes, Tier A never fires. Helper terminates at depth_limit, returns conservative `should_raymarch=true`. |
| `test_adaptive_depth_supermassive` | `dλ_full / H_min` ratio of ~100 produces `depth_limit ≈ 7`. |

### 7.2 Integration smoke tests (trace-loop equivalence)

Append to existing tracer test or add `tests/test_geodesic_tracer.cpp`:

- Deep-space ray (no disk encounter) → output byte-identical to pre-fix.
- Clean disk-crossing ray → output byte-identical to pre-fix.
- Known wedge-triggering ray (constructed from failing repro) → post-fix
  produces non-zero raymarch entry; pre-fix produces zero.

### 7.3 Visual smoke test (success criterion)

Build, then run:

```
./grrt-cli.exe --disk-volumetric --samples 100 --width 256 --height 256 --output t --force --fov 30
```

Pass: no wedge-shaped voids visible by eye. Same FOV / spp / dimensions as
the original failure repro.

For confidence, also render a wider-FOV scene to confirm no new artifacts:

```
./grrt-cli.exe --disk-volumetric --samples 100 --width 256 --height 256 --output t_wide --force --fov 90
```

### 7.4 Diagnostic counter (instrumentation, not a gate)

`tracer.substep_invocation_count_` reports Tier C invocation rate post-fix.
`<< 1%` of total steps = well-tuned. `>> 1%` = follow-up tuning.

### 7.5 No banding regression

Existing `test_no_horizontal_bands` in `tests/test_volumetric.cpp` should
pass *without recalibration*. The new helper changes entry detection, not
raymarch sampling — banding is downstream of entry. If banding moves more
than measurement noise (relative > 0.25), Tier C is shifting raymarch start
positions non-trivially; investigate before merge.

## 8. Implementation scope

In scope:
- New `include/grrt/geodesic/disk_step_entry.h` and
  `src/disk_step_entry.cpp`.
- Refactor of three call sites in `src/geodesic_tracer.cpp` to use the
  helper.
- New test file `tests/test_disk_step_entry.cpp`.
- New `GeodesicTracer::step(state, dλ)` public method exposing single-step
  integration to the helper, if no equivalent exists. Plan task 1 audits
  the existing tracer API before adding this; if a private equivalent
  exists, promote it rather than duplicate.
- Atomic `substep_invocation_count_` instrumentation on `GeodesicTracer`.
- CMakeLists update for the new source / test file.

Out of scope:
- LUT smoothing (issue 3).
- Tau midplane test fix (issue 2).
- Tuning sweeps for `curvature_pad` or `depth_limit_cap` beyond defaults.
  Defaults are correctness-safe; tuning is post-validation work.
- Any change to raymarch_volumetric internals.

## 9. References

- Known-issues catalog: `docs/superpowers/known-issues-2026-05-02.md` (item 1)
- Prior cliff-aware work: `docs/superpowers/specs/2026-04-29-bpt72-taper-and-dprk45.md`
- Sobol sub-pixel sampling: `docs/superpowers/specs/2026-05-02-low-discrepancy-sub-pixel-sampling-design.md`
- Existing entry-detection sites: `src/geodesic_tracer.cpp` lines 192–202, 441–449, 585–595
- Disk envelope definition: `src/volumetric_disk.cpp::is_in_volume` (line 211),
  `scale_height` and `z_max_at` (lines 276, 280)
