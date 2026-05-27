# Raymarch transit sampling — midpoint source + z-resolution control

**Date:** 2026-05-15
**Branch:** `fix/volumetric-ring`
**Status:** spec — pending implementation plan

## 1. Problem

After the disk-step-entry helper (Tier A/B/C) was wired into all three trace
paths, narrow-FOV `--disk-volumetric` renders show large black regions where
disk material should be visible — most prominently the bottom half of the
disk, where light from the far side is bent around the black hole and crosses
the midplane transversally on its way back to the observer.

A no-op-raymarch detection fix (already committed, `8b56852`) eliminated an
earlier infinite-loop symptom, but exposed the deeper issue: `raymarch_volumetric`
runs for these segments and returns essentially zero color.

Reproduction:
```
./grrt-cli.exe --disk-volumetric --samples 100 --width 256 --height 256 --output t --force --fov 30
```

Debug-pixel at (150, 180), fov=30, shows two ENTRY events whose RAYMARCH exit
color is ~1e-19 (effectively zero), then the ray ESCAPES. The disk crossing is
real but produces no emission.

## 2. Root cause

`raymarch_volumetric` integrates radiative transfer along the geodesic, one
Romberg-controlled step at a time. Per step it accumulates:

```
J[ch] += T[ch] * S * (1 - exp(-dtau[ch]));
T[ch] *= exp(-dtau[ch]);
```

`dtau` (optical depth over the step) is correctly integrated by `romberg_step`'s
composite half-step pass — it samples the integrand `κρ|p·u|` at start, midpoint,
and end. But the **source function `S`** (and the density, temperature, and
redshift factor it depends on) is sampled only at the step's **end state**.

This is a "constant source over the step" approximation, accurate only when the
step is small relative to how fast `S` varies. For a transversal disk transit,
`S` goes `0 → peak → 0` across one disk thickness. Two failure modes compound:

1. **Coarse steps.** The step controller grows `ds` toward `H*2` in empty space.
   When the ray reaches the thin disk, a single Romberg step's path can span the
   entire disk thickness, with both endpoints outside the disk envelope.
   `density_cgs(end) == 0` → `S == 0` → zero emission, even though the path
   traversed dense disk material.

2. **End-only sampling.** Even at moderate step sizes, the last step before
   exiting the disk has its endpoint outside the envelope (`S = 0`), so
   end-sampling systematically under-counts the exit boundary.

The original (pre-helper) inline predicate masked this: it fired raymarch only
when an endpoint was within `H` of `z_max`, so raymarch always started near the
disk and its initial `ds = inside ? H/16 : min(|z|/8, H*2)` picked fine
resolution. The Tier B/C helper validly fires raymarch for *transversal* transits
where the ray starts far from the disk and shoots across it — a mode the step
controller and source sampling were never designed for.

## 3. Goals and non-goals

**Goals.**
- Capture emission correctly for transversal disk transits (far-side light bent
  around the BH), so the bottom half of the disk renders.
- Keep along-disk (shallow graze) rendering correct and efficient.
- Handle rays that cross the disk multiple times (photon-ring orbits) within a
  single raymarch call.
- Preserve behavior on the no-bug path (existing near-disk entries) as closely
  as possible.

**Non-goals.**
- Not restructuring the trace/raymarch boundary into an explicit "deliver then
  walk" two-phase architecture (considered as Approach C; deferred as a possible
  future cleanup if we later want the main integrator to never overshoot the
  disk).
- Not changing the disk model, the helper (`check_disk_step_entry`), or the
  trace-loop call sites.
- Not adding pixel-color regression tests (disk-value-dependent; break on every
  disk-model tweak — per established project preference).

## 4. Architecture

Two coordinated changes inside `raymarch_volumetric`, plus one supporting field
on `RombergStep`:

1. **Midpoint source sampling.** Sample `S` (and density, temperature, redshift
   `g`) at the step's *midpoint* rather than its *end*. The midpoint is the
   junction of `romberg_step`'s two half-steps — already computed, surfaced via a
   new `RombergStep::mid_state` field. Second-order accurate; catches the
   emission peak even when both step endpoints sit at zero density.

2. **z-resolution step control.** After each Romberg step, if the step's vertical
   excursion `|Δz|` exceeds `H/4` *and* the step's signed z-range overlaps the
   disk's vertical extent, reject the step and halve `ds` — reusing the existing
   `max_err` shrink-and-retry loop. Also cap step-growth at `H/4` while the step
   overlaps the disk, to avoid grow-then-reject thrashing inside the disk.

`romberg_step` remains a pure geodesic + optical-depth integrator that now also
reports its midpoint. `raymarch_volumetric` owns radiative-transfer accumulation
and step control. The already-committed no-op-raymarch detection in the trace
loop is unchanged.

The two changes are complementary: the z-cap ensures steps are small enough that
the midpoint is representative of the step; midpoint-S ensures we sample where
the emission is rather than at the (often-zero) endpoint.

## 5. Components

### 5.1 `RombergStep::mid_state`

`include/grrt/geodesic/romberg_step.h` — add a field:

```cpp
struct RombergStep {
    GeodesicState end_state;   ///< Geodesic state at end of accepted half-step path.
    GeodesicState mid_state;   ///< Geodesic state at the step midpoint (junction of the two half-steps).
    std::array<double, MAX_ROMBERG_CHANNELS> dtau;
    double max_err;
    double ds_taken;
    int n_channels;
};
```

### 5.2 `romberg_step` populates `mid_state`

`src/romberg_step.cpp` — set `out.mid_state = mid;` (the `mid` already computed at
the half-step pass, currently line 71). In the `n_channels <= 0` early-return
path, also set `mid_state`: compute the half-step (`integrator.step_kerr(metric,
start_state, 0.5 * ds_proposed)`) and assign it. Zero new integration on the hot
path — `mid` already exists.

### 5.3 Midpoint source sampling in `raymarch_volumetric`

`src/geodesic_tracer.cpp` — in the source-function block (currently lines
~299-336), replace the end-state sampling with midpoint sampling:

```cpp
const GeodesicState& mid = rs.mid_state;          // was: rs.end_state
const double r_mid     = mid.position[1];
const double theta_mid = mid.position[2];
const double phi_mid   = mid.position[3];
const double z_mid     = r_mid * std::cos(theta_mid);

const double rho_cgs = vol_disk_->density_cgs(r_mid, z_mid, phi_mid);
const double T_local = vol_disk_->temperature(r_mid, std::abs(z_mid));
if (rho_cgs > 0.0 && T_local > 0.0) {
    double ut_emit = 0.0, ur_emit = 0.0, uphi_emit = 0.0;
    if (r_mid >= vol_disk_->r_isco()) {
        vol_disk_->circular_velocity(r_mid, ut_emit, uphi_emit);
    } else {
        vol_disk_->plunging_velocity(r_mid, theta_mid, ut_emit, ur_emit, uphi_emit);
    }
    const double p_dot_u_emit = mid.momentum[0] * ut_emit
                               + mid.momentum[1] * ur_emit
                               + mid.momentum[3] * uphi_emit;
    const double p_dot_u_obs  = mid.momentum[0] * ut_obs;
    const double g_factor     = p_dot_u_emit / p_dot_u_obs;
    // ... per-channel S, dtau, J/T accumulation unchanged ...
}
```

`dtau` is unchanged — it remains the correctly-integrated optical depth over the
whole step. Only the source function's sampling point moves to the midpoint.

### 5.4 z-resolution gate (testable free function)

`src/geodesic_tracer.cpp` (anonymous namespace) — extract the rejection predicate
so its logic is named and unit-testable:

```cpp
/// Returns true if a raymarch step from Cartesian z0 to z1 should be refined
/// for z-resolution. The step needs refinement when its signed z-interval
/// [min(z0,z1), max(z0,z1)] overlaps the disk's vertical extent [-env, +env]
/// AND its vertical excursion |z1 - z0| exceeds quarter_H. The signed-interval
/// test (not endpoint membership) is what catches transversal transits, where
/// both endpoints lie outside the envelope but the path crosses z = 0.
bool step_needs_z_refinement(double z0, double z1, double quarter_H, double env) {
    const double dz = std::abs(z1 - z0);
    const bool crosses = (std::min(z0, z1) < env) && (std::max(z0, z1) > -env);
    return crosses && dz > quarter_H;
}
```

### 5.5 z-resolution control in `raymarch_volumetric`

After the existing `max_err` rejection, before `step_count++`:

```cpp
const double z0 = state.position[1]       * std::cos(state.position[2]);
const double z1 = rs.end_state.position[1] * std::cos(rs.end_state.position[2]);
const double r_for_H = rs.mid_state.position[1];
const double H_z     = vol_disk_->scale_height(r_for_H);
const double env_z   = vol_disk_->z_max_at(r_for_H) + H_z;
const double ds_floor_z = H_z / 256.0;
if (step_needs_z_refinement(z0, z1, 0.25 * H_z, env_z)
    && ds_proposed > ds_floor_z) {
    ds_proposed = std::max(ds_proposed * 0.5, ds_floor_z);
    continue;
}
```

And cap step-growth while overlapping the disk (replaces the current
`ds_proposed = std::min(ds_proposed * 2.0, H_now)`):

```cpp
if (rs.max_err < raymarch_tol_ / 8.0) {
    const double z_now  = r * std::cos(state.position[2]);
    const double H_now  = vol_disk_->scale_height(r);
    const double env_now = vol_disk_->z_max_at(r) + H_now;
    // Cap growth at H/4 while the ray is inside the disk envelope, else H.
    // (Point test, not the interval test used for rejection: a ray that is
    // currently inside the envelope keeps fine steps; the single boundary
    // step on entry is caught by the rejection gate above.)
    const double grow_cap = (std::abs(z_now) < env_now) ? (0.25 * H_now) : H_now;
    ds_proposed = std::min(ds_proposed * 2.0, grow_cap);
}
```

### 5.6 raymarch MAX_STEPS headroom

Bump `MAX_STEPS` in `raymarch_volumetric` from 4096 to 16384, so very thin disks
(H near the 0.001M clamp floor) have headroom for the finer transit stepping.
Each step is cheaper-and-finer than before; the increase is graceful-degradation
insurance, not an expected hot path.

## 6. Error handling and edge cases

### 6.1 Shrink-loop floor
If `|Δz| > H/4` persists at `ds_floor = H/256` (LUT discontinuity), the step is
accepted anyway via the `ds_proposed > ds_floor` guard — same as the existing
`max_err` floor. The loop cannot spin forever.

### 6.2 Very thin disk
`H/4` becomes tiny; transit needs many fine steps. If raymarch's `MAX_STEPS`
(now 16384) is reached mid-transit, emission accumulated so far is kept,
`state = rs.end_state`, trace loop resumes — graceful degradation, no loop.

### 6.3 `mid_state` in the empty-channel path
`romberg_step`'s `n_channels <= 0` path must set a valid `mid_state` (§5.2). Not
hit in production (raymarch always passes 3 channels) but prevents a latent trap
for tests/future callers.

### 6.4 Shallow tangential grazes
Ray skimming the disk top (`z ≈ env`, small `Δz`): `crosses_disk_z = true` but
`Δz < H/4` → no rejection → coarse steps retained. Correct — along a shallow
graze, density varies slowly with arc length. The `Δz`-based gate distinguishes
"transiting across" (fine) from "skimming along" (coarse OK).

### 6.5 Interaction with no-op-raymarch detection
The committed no-op check (`entry.position unchanged after raymarch → don't
revert state`) stays. With this fix, transit rays genuinely advance `entry`, so
the no-op path no longer fires for them — it remains a guard for the true-no-op
case (inside disk r-cylinder but never crossing the envelope). No conflict.

### 6.6 Performance
Correctness-over-speed by design. Old code took ~3 coarse steps per transit and
produced zero (fast but wrong); new code takes ~20 growth-capped fine steps and
produces correct emission. Disk-heavy renders get slower but render content that
was previously black. Profiling-driven optimization can follow if needed.

## 7. Testing

### 7.1 Unit tests (stable, no disk-model dependency)

- **`mid_state` population** (`tests/test_romberg_step.cpp`): assert
  `rs.mid_state` equals `integrator.step_kerr(metric, start, 0.5*ds)` for a
  synthetic sampler. Assert the empty-channel path sets a valid `mid_state`.
- **`step_needs_z_refinement` logic** (new test, e.g. in
  `tests/test_disk_step_entry.cpp` or a dedicated file): exercise the
  signed-interval cases:
  | Case | z0 | z1 | quarter_H | env | Expected | Why |
  |---|---|---|---|---|---|---|
  | transversal transit | −0.27 | +0.12 | 0.005 | 0.076 | true | crosses env, dz≫H/4 |
  | entirely below disk | −0.30 | −0.20 | 0.005 | 0.076 | false | no env overlap (max=−0.20 < −env) |
  | skim far above disk | +0.50 | +0.49 | 0.005 | 0.076 | false | no env overlap (min=+0.49 > +env) |
  | already-fine step near top | +0.072 | +0.070 | 0.005 | 0.076 | false | overlaps env but dz=0.002 < H/4 |
  | step at disk top, coarse | +0.08 | +0.07 | 0.005 | 0.076 | true | overlaps env (0.07<env) and dz=0.01 > H/4 |
  | inside disk, coarse | −0.03 | +0.03 | 0.005 | 0.076 | true | crosses env, dz=0.06 > H/4 |

### 7.2 Integration verification (in the plan, not committed tests)
- Debug-pixel (150,180) fov=30: RAYMARCH exit color meaningfully non-zero
  (was ~1e-19). Trace terminates ESCAPED/HORIZON, not MAX_STEPS.

### 7.3 Visual smoke + banding (success criterion)
- Wedge repro render `--disk-volumetric --samples 100 --fov 30`: bottom half of
  the disk renders (not black). User inspects.
- `test_no_horizontal_bands`: re-run. If the metric shifts beyond noise but the
  render is visually correct, recalibrate the threshold with a documented
  before/after measurement (the protocol used when Sobol moved it 0.183→0.211),
  and update the test comment with the new baseline.

## 8. Implementation scope

In scope:
- `include/grrt/geodesic/romberg_step.h` — `mid_state` field.
- `src/romberg_step.cpp` — populate `mid_state` (both paths).
- `src/geodesic_tracer.cpp` — `step_needs_z_refinement` helper; midpoint-S
  sampling; z-resolution rejection; growth cap; `MAX_STEPS` bump.
- `tests/test_romberg_step.cpp` — `mid_state` assertions.
- Unit test for `step_needs_z_refinement`.

Out of scope:
- Approach C (deliver/walk phase split).
- Disk model, helper, trace-loop call sites.
- Pixel-color regression tests.

## 9. References

- Disk-step-entry design: `docs/superpowers/specs/2026-05-10-disk-step-entry-design.md`
- No-op-raymarch fix: commit `8b56852`
- raymarch_volumetric: `src/geodesic_tracer.cpp:246-351`
- romberg_step: `src/romberg_step.cpp`, midpoint at line 71
- Known issues: `docs/superpowers/known-issues-2026-05-02.md` (item 1, wedge artifacts)
