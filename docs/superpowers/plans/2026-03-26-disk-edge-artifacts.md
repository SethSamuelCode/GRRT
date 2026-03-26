# Fix Volumetric Disk Edge Artifacts — Implementation Plan

> **For agentic workers:** Execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate visual artifacts at the boundary of the volumetric accretion disk.

**Architecture:** Three root causes need fixing: (1) step size discontinuity at entry, (2) hard density cutoff at z=3H, (3) overly aggressive 6H entry trigger. Fixes are applied to both CPU (`src/geodesic_tracer.cpp`, `src/volumetric_disk.cpp`) and CUDA (`cuda/cuda_render.cu`, `cuda/cuda_volumetric_disk.h`) backends.

**Root Causes:**
- Rays entering the volume from the 3H–6H zone start with coarse steps that overshoot into the dense midplane, causing brightness spikes
- Density/temperature LUTs have a hard cutoff at z=3H (returns exactly 0), creating a visible discontinuity
- The 6H entry trigger causes raymarching to start far from the actual volume, wasting steps on empty space

---

### Task 1: Smooth the density/temperature cutoff at z=3H

Currently `vol_interp_2d` and `interp_2d` return exactly 0.0 when `z_abs >= 3*H`. This hard edge is visible. Apply a smooth exponential taper from 3H outward.

**Files:**
- Modify: `cuda/cuda_volumetric_disk.h` — `vol_interp_2d` (around line 128)
- Modify: `src/volumetric_disk.cpp` — `interp_2d` (around line 171)

- [ ] **Step 1: Update CUDA `vol_interp_2d`**

Find the early-return at the top of `vol_interp_2d`:
```cpp
const double z_max = 3.0 * H;
if (z_abs >= z_max) return 0.0;
```

Replace with a smooth taper for z beyond 3H (up to ~4H):
```cpp
const double z_max = 3.0 * H;
if (z_abs >= 4.0 * H) return 0.0;  // hard cutoff at 4H
if (z_abs >= z_max) {
    // Smooth exponential taper from 3H to 4H
    const double dz = (z_abs - z_max) / H;
    const double taper = exp(-4.0 * dz * dz);
    // Use the value at z=3H (last LUT entry) as the base
    // Fall through to interpolation with z_abs clamped to z_max - epsilon
    // then multiply by taper
}
```

More precisely — restructure so that for z_abs in [3H, 4H], we evaluate the LUT at z=3H (the boundary value) and multiply by `exp(-4 * ((z-3H)/H)^2)`. For z_abs < 3H, no change.

Implementation:
```cpp
__device__ inline double vol_interp_2d(const double* lut, double r, double z_abs,
                                        const RenderParams& params) {
    // Radial index (unchanged)
    const double r_frac = ...;  // existing code

    const double H = vol_scale_height(r, params);
    const double z_max = 3.0 * H;

    if (z_abs >= 4.0 * H) return 0.0;

    // Taper factor: 1.0 inside 3H, smooth falloff from 3H to 4H
    double taper = 1.0;
    double z_lookup = z_abs;
    if (z_abs >= z_max) {
        const double dz = (z_abs - z_max) / H;
        taper = exp(-4.0 * dz * dz);
        z_lookup = z_max - 1e-10;  // clamp to last valid LUT entry
    }

    // ... existing bilinear interpolation using z_lookup ...

    return result * taper;
}
```

- [ ] **Step 2: Apply same change to CPU `interp_2d` in `src/volumetric_disk.cpp`**

Same logic but using `std::exp` and `std::abs`.

- [ ] **Step 3: Update `vol_inside` / `inside_volume` to use 4H instead of 3H**

In `cuda/cuda_volumetric_disk.h` `vol_inside()`:
```cpp
return fabs(z) < 4.0 * H;  // was 3.0 * H
```

In `src/volumetric_disk.cpp` `inside_volume()`:
```cpp
return std::abs(z) < 4.0 * H;  // was 3.0 * H
```

This ensures the volume boundary matches the new taper range.

- [ ] **Step 4: Commit**
```
feat: Smooth density/temperature cutoff at disk boundary with exponential taper
```

---

### Task 2: Fix step size discontinuity at volume entry

The coarse-to-fine step transition is abrupt. When a ray approaches from outside, `ds` can be `2*H` (coarse), then jump to `H/16` (fine) the moment it enters. This causes sampling artifacts.

**Files:**
- Modify: `cuda/cuda_volumetric_disk.h` — `vol_raymarch` (lines ~444-490)
- Modify: `src/geodesic_tracer.cpp` — `raymarch_volumetric` (lines ~155-213)

- [ ] **Step 1: Smooth the coarse-to-fine transition in CUDA `vol_raymarch`**

Replace the binary coarse/fine step logic with a smooth ramp based on distance from the volume boundary:

When outside the volume (the `!vol_inside` branch around line 471), instead of the current sharp transition:
```cpp
if (!been_inside) {
    ds = fmin(fabs(z) / 8.0, H * 2.0);
    ds = fmax(ds, H / 64.0);
} else {
    ds = fmax(H / 4.0, H / 64.0);
    if (ds > H) ds = H;
}
```

Use a smooth ramp that depends on `|z|/H`:
```cpp
// Distance from midplane in units of scale height
const double zh = fabs(z) / H;
if (zh > 4.0) {
    // Far from volume: coarse steps proportional to distance
    ds = fmin(fabs(z) / 8.0, H * 2.0);
    ds = fmax(ds, H / 64.0);
} else {
    // Near or inside volume: smoothly ramp from coarse to fine
    // At zh=4: ds ~ H/2, at zh=3: ds ~ H/8, at zh<2: ds ~ H/16
    const double t = fmax(0.0, (zh - 2.0) / 2.0);  // 0 at zh<=2, 1 at zh=4
    const double ds_fine = H / 16.0;
    const double ds_coarse = H / 2.0;
    ds = ds_fine + t * (ds_coarse - ds_fine);
    ds = fmax(ds, H / 64.0);
}
```

This eliminates the sudden jump from `2*H` to `H/16`.

- [ ] **Step 2: Apply same change to CPU `raymarch_volumetric`**

Same smooth ramp logic in `src/geodesic_tracer.cpp`.

- [ ] **Step 3: Commit**
```
fix: Smooth coarse-to-fine step transition at volumetric disk boundary
```

---

### Task 3: Tighten the entry trigger from 6H to 4H

The render kernel triggers raymarching when a ray is within 6H of the midplane. Since the volume now extends to 4H (from Task 1), the trigger should be tightened to match, reducing wasted steps in empty space.

**Files:**
- Modify: `cuda/cuda_render.cu` — entry detection (lines ~135-138)
- Modify: `src/geodesic_tracer.cpp` — entry detection (lines ~84-87)

- [ ] **Step 1: Update CUDA entry detection**

Change the `near_disk` check from `6.0 * H` to `5.0 * H` (one scale height of margin beyond the 4H volume):
```cpp
const bool near_disk = (fabs(z_new) < 5.0 * H_new
                     || fabs(z_prev) < 5.0 * H_prev)
                    && r_new >= d_params.disk_r_horizon
                    && r_new <= d_params.disk_r_outer;
```

Also update the raymarcher exit condition in `vol_raymarch` to match:
```cpp
if (been_inside && fabs(z) > 5.0 * H) break;  // was 6.0 * H
```

- [ ] **Step 2: Update CPU entry detection**

Same change in `src/geodesic_tracer.cpp`:
```cpp
const bool near_disk = (std::abs(z_new) < 5.0 * H_new
                     || std::abs(z_prev) < 5.0 * vol_disk_->scale_height(r_prev))
                    && r_new >= vol_disk_->r_horizon()
                    && r_new <= vol_disk_->r_max();
```

And the exit condition:
```cpp
if (been_inside && std::abs(z) > 5.0 * H) break;
```

- [ ] **Step 3: Commit**
```
fix: Tighten volumetric disk entry trigger from 6H to 5H to match volume extent
```

---

### Task 4: Build and visual regression test

- [ ] **Step 1: Build**
```bash
cmake --build build --config Release 2>&1
```

- [ ] **Step 2: Render before/after comparison**

Render at 512x512 with default volumetric settings and compare to the pre-fix output. The disk boundary should appear smoother with no hard edges or brightness discontinuities.

```bash
./build/Release/grrt-cli --disk-volumetric --backend cuda --width 512 --height 512 --output edge_fix_test
```

- [ ] **Step 3: Test with high turbulence (noise holes)**

```bash
./build/Release/grrt-cli --disk-volumetric --backend cuda --width 512 --height 512 --disk-turbulence 1.5 --disk-noise-scale 5.0 --disk-noise-octaves 4 --output edge_fix_holes
```

Verify holes are still visible and the boundary is smooth.

- [ ] **Step 4: Commit any fixups**

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Smooth density taper at z=3H→4H | `cuda_volumetric_disk.h`, `volumetric_disk.cpp` |
| 2 | Smooth step size transition | `cuda_volumetric_disk.h`, `geodesic_tracer.cpp` |
| 3 | Tighten entry trigger 6H→5H | `cuda_render.cu`, `geodesic_tracer.cpp` |
| 4 | Build + visual regression | — |
