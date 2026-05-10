# Known issues — fix/volumetric-ring as of 2026-05-02

This document captures the state of the branch after the cliff-aware-raymarch and low-discrepancy-sub-pixel-sampling work landed. Use it to prioritize follow-up work after a context reset.

---

## Active bugs (need investigation / fix before merge to main)

### 1. Sharp wedge artifacts at disk edges *(pre-existed Sobol; persists)*

**Symptom:** Sharp, geometric, wedge-shaped *black* voids cutting into what should be solid disk material. Visible especially at narrower FOV.

**Reproduction:**
```
./grrt-cli.exe --disk-volumetric --samples 100 --width 256 --height 256 --output t --force --fov 30
```

**Confirmed by user as pre-existing** (predates the Sobol switch in commit `f058c8c`). Likely also predates Romberg fix — needs verification.

**Suspected cause area:** geodesic-step / disk-volume entry detection at near-tangential disk encounters. The step clamp limits `dz/dλ`-based step size when within 3H of disk surface, but the precondition `r ∈ [r_horizon * 0.9, r_max * 1.5]` may have a hole. Or the tangential pass detection in `trace()` (the `crossed_midplane || inside_now || near_disk` triple) may miss certain geometries.

**Status:** No spec, no fix planned yet. Top priority next session.

---

### 2. `test_tau_midplane_near_target` fails by factor of 4 *(pre-existing)*

**Symptom:** τ measured at midplane = 403, target = 100.

**Documented in:** `docs/superpowers/specs/2026-04-29-bpt72-taper-and-dprk45.md` (under "Implementation Results" section, "Phase 4 future work").

**Suspected cause:** mismatch between the test's integration convention and `normalize_density`'s internal protocol. Possible fixes:
- Match test integration to internal convention.
- Add public accessors `peak_flux_radius()` and `column_optical_depth(int ri)` so the test can directly inspect what `normalize_density` does.

---

### 3. LUT cap-binding warnings *(pre-existing)*

**Symptoms (PROMPTABLE warnings during disk construction):**
```
[PROMPTABLE n_r_cap]: n_r capped at 4096 with delta=8.13e-01 > 1.00e-03
[PROMPTABLE n_z_cap]: n_z capped at 1024 with delta=4.20e+00 > 1.00e-03
[PROMPTABLE h_jump]: H jump 0.97 at i=4095, smoothness violated
[VolumetricDisk] WARNING: vertical profile did not converge at r_idx=32
```

**Documented as Phase 3 future work** in cliff-aware-raymarch spec.

**Fix path:** change `compare_columns` metric in `volumetric_disk.cpp` from point-density comparison to integrated optical depth (`∫κρ dz`). The current metric is overly sensitive to LUT z-grid storage aliasing of the photosphere cliff; an integrated-tau metric is naturally smooth across cliff position jitter.

---

## Cosmetic issues

### 4. Commit `0186290` has a garbled message body

The Romberg-switch commit message body has TUI/spinner characters interleaved. The code itself is correct. Can be amended pre-PR with `git commit --amend -F <clean-message-file>` (one of our earlier sessions tried `--amend` and got the same garbled text — the user's terminal may need to write the message via a clean editor or file).

### 5. `test_tolerance_convergence` doesn't strongly stress the tolerance knob

At the test scene (64² × spp=4 × no turbulence), `max|def-ref|` is only ~1e-6 (essentially below float precision). The test passes trivially regardless of `--raymarch-tol`. Not a bug — the assertion still holds — but it doesn't *prove* the tolerance is doing what the docs claim. Tightening would require a scene that exercises the cliff more (smaller disk, edge-on view, or higher spp).

---

## Documented future work (specs exist)

### 6. τ-scaled raymarch tolerance optimization
Path: `docs/superpowers/optimizations/2026-05-01-tau-scaled-raymarch-tolerance.md`
Estimated benefit: 1.3–2× raymarch speedup.

### 7. Romberg trajectory reuse optimization
Path: `docs/superpowers/optimizations/2026-05-01-romberg-trajectory-reuse.md`
Estimated benefit: per-step speedup in raymarch (concrete factor TBD).

### 8. LUT export / on-disk caching *(user-requested)*
Disk construction takes ~1 minute. Caching to a file (binary format TBD — perhaps HDF5 or a custom format) keyed on disk parameters would let users iterate on rendering without rebuilding the LUT each run. **No spec yet.**

### 9. CUDA backend port (Phase 2)
Would 30–100× speed up renders. Sobol sampler is already designed to be `__device__ __host__` portable.

### 10. Spectral path Sobol adoption
The current Sobol work landed in RGB only. The `raymarch_volumetric_spectral` path could adopt `sobol_sample_2d` with no further design work — one PR.

---

## Verified behaviors (NOT bugs — documented to prevent re-investigation)

### 11. "More transparent disk" with Sobol
**Measured comparison** (256² × spp=30, identical params except sampler):

| Metric | Stratified | Sobol+Owen | Δ |
|---|---|---|---|
| Total luminance energy | 38.40 | 39.18 | +2.0% |
| Disk pixels (L>1e-4) | 2716 | 3000 | +10.5% |
| Mean disk luminance | 0.01414 | 0.01306 | −7.6% |
| Max pixel luminance | 0.16929 | 0.12949 | −23.5% |

Sobol redistributes the same total energy over more pixels with lower peak hot-pixel brightness. Eye reads as "more transparent" but it's perceptually correct. At higher spp (200+), both samplers converge to the same image.

### 12. Speckle pattern at low spp
Monte Carlo variance, ~1/√N convergence. Inherent to volumetric raytracing. Mitigated by spp=200+ for production renders, eventual CUDA port for interactive previews.

### 13. Banding-regression threshold = 0.25
Calibration measurements:
- Romberg + Stratified (pre-Sobol): rel = 0.183
- Romberg + Sobol+Owen (current):   rel = 0.211
- Buggy build (H_max=H):            rel = 0.281

Threshold sits with 16% headroom over Sobol baseline and clearly fails the buggy regime. Comment in `tests/test_volumetric.cpp::test_no_horizontal_bands` documents this.

---

## Recommended priority for next session

1. **Investigate the wedge artifacts** *(item 1 — top priority, geometry/edge-detection bug)*
2. Either:
   - Address the LUT cap-binding warnings (item 3, Phase 3) — would also retire convergence warnings
   - Or address the tau test failure (item 2, Phase 4) — smaller scope, single test fix
3. **Optional polish:** amend commit `0186290`'s message (item 4)
4. **Wire spectral Sobol** (item 10) — small, mechanical PR

The current branch is otherwise green: cliff-aware-raymarch and Sobol+Owen sub-pixel sampling are both committed, all unit tests pass except the one pre-existing tau test, and renders show no banding artifacts (modulo item 1's wedges at narrow FOV).
