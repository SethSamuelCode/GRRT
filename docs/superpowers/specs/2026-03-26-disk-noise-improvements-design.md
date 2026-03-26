# Volumetric Disk Noise Improvements

**Date:** 2026-03-26
**Status:** Approved

## Problem

The volumetric disk's turbulent noise does not produce visible holes/gaps in the density field, even at high turbulence values. Three issues:

1. **`noise_scale` is hardcoded** to `2 * H` (twice the scale height at peak flux radius), producing features too small for visible structure.
2. **No density floor** — when `turbulence * |noise| > 1`, density goes negative, which is unphysical and causes incorrect radiative transfer.
3. **Only 2 octaves of noise** — insufficient detail for realistic turbulent structure. The octave count is hardcoded and not user-tunable.

## Changes

### 1. Expose `noise_scale` as a user parameter

- Add `double disk_noise_scale` to `GRRTParams` (C API struct). Default `0.0` means auto (current `2*H` behavior).
- Add `double noise_scale` to `VolumetricParams`. Default `0.0` means auto.
- In `VolumetricDisk::normalize_density()`: if `params_.noise_scale > 0`, use it directly; otherwise fall back to `2 * H_lut_[peak_idx]`.
- Thread through CLI (`--disk-noise-scale`), CUDA `RenderParams.disk_noise_scale`, and `VolDiskHostData.noise_scale`.
- Larger values produce bigger noise features (lower spatial frequency), making holes visible. Suggested starting point: `5.0`–`10.0` (geometric units, i.e. multiples of M).

### 2. Clamp density to zero

- In CPU `VolumetricDisk::density()`: return `std::max(0.0, base * (1.0 + turbulence * n))`.
- In CUDA `vol_density_cgs()`: return `fmax(0.0, base * (1.0 + turbulence * n))`.
- This ensures density is non-negative when noise troughs exceed the base density, producing clean holes instead of unphysical negative values.

### 3. Configurable noise octaves

- Add `int disk_noise_octaves` to `GRRTParams`. Default `2` (current behavior).
- Add `int noise_octaves` to `VolumetricParams`. Default `2`.
- **CPU:** Add `SimplexNoise3D::evaluate_fbm(double x, double y, double z, int octaves)` using standard fBm (lacunarity=2, persistence=0.5):
  ```
  result = 0; amplitude = 1.0; frequency = 1.0;
  for (i = 0; i < octaves; i++) {
      result += amplitude * evaluate(x*frequency, y*frequency, z*frequency);
      amplitude *= 0.5;
      frequency *= 2.0;
  }
  return result;
  ```
- **CUDA:** Add `cuda_simplex_noise_fbm(double x, double y, double z, int octaves)` with identical logic. Octave count passed via `RenderParams.disk_noise_octaves`.
- Replace calls to `evaluate_turbulent` / `cuda_simplex_noise_turbulent` with the new fBm functions.
- Note: This changes lacunarity from 3.0 to 2.0 (standard fBm). The old 3x jump skipped detail; 2x gives denser structure at each frequency band.

## Files Touched

| File | Change |
|------|--------|
| `include/grrt/types.h` | Add `disk_noise_scale`, `disk_noise_octaves` to `GRRTParams` |
| `include/grrt/scene/volumetric_disk.h` | Add `noise_scale`, `noise_octaves` to `VolumetricParams` |
| `include/grrt/math/noise.h` | Add `evaluate_fbm(x, y, z, octaves)` declaration |
| `src/noise.cpp` | Implement `evaluate_fbm`; old `evaluate_turbulent` becomes unused |
| `src/volumetric_disk.cpp` | Clamp density >= 0; use user `noise_scale`; call `evaluate_fbm` with `noise_octaves` |
| `src/api.cpp` | Thread `disk_noise_scale`, `disk_noise_octaves` into `VolumetricParams` |
| `cli/main.cpp` | Add `--disk-noise-scale` and `--disk-noise-octaves` flags |
| `cuda/cuda_types.h` | Add `disk_noise_octaves` to `RenderParams` |
| `cuda/cuda_noise.h` | Add `cuda_simplex_noise_fbm` device function |
| `cuda/cuda_volumetric_disk.h` | Clamp density >= 0; use fbm with octaves from params |
| `cuda/cuda_backend.cu` | Thread `disk_noise_scale`, `disk_noise_octaves` |
| `cuda/cuda_vol_host_data.h` | Add `noise_octaves` field |
| `cuda/cuda_vol_host_data.cpp` | Thread `noise_octaves` |

## Defaults and Backward Compatibility

All new parameters default to values that reproduce current behavior:
- `disk_noise_scale = 0.0` (auto = `2*H`, same as before)
- `disk_noise_octaves = 2` (same octave count as before)

The only behavioral change at defaults is the density clamp, which fixes a bug (negative density).
