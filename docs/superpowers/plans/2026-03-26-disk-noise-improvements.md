# Volumetric Disk Noise Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make turbulent density holes visible in the volumetric disk by exposing noise_scale and noise_octaves as user parameters and clamping density to zero.

**Architecture:** Three independent changes threaded through the same parameter pipeline (GRRTParams -> VolumetricParams/RenderParams -> density function). CPU and CUDA backends must stay in sync.

**Tech Stack:** C++23 (CPU), CUDA (GPU), CMake build system

**Spec:** `docs/superpowers/specs/2026-03-26-disk-noise-improvements-design.md`

---

### Task 1: Add new fields to parameter structs

**Files:**
- Modify: `include/grrt/types.h:43-46` (GRRTParams — add after `disk_seed`)
- Modify: `include/grrt/scene/volumetric_disk.h:13-18` (VolumetricParams — add fields)
- Modify: `cuda/cuda_types.h:79-81` (RenderParams — add after `disk_noise_scale`)
- Modify: `cuda/cuda_vol_host_data.h:21` (VolDiskHostData — add fields)

- [ ] **Step 1: Add to GRRTParams**

In `include/grrt/types.h`, after line 46 (`int disk_seed;`), add:

```c
double disk_noise_scale;        /* Noise feature size in M (0 = auto, default 0) */
int disk_noise_octaves;         /* fBm octave count (default 2) */
```

- [ ] **Step 2: Add to VolumetricParams**

In `include/grrt/scene/volumetric_disk.h`, inside `VolumetricParams` after `uint32_t seed`, add:

```cpp
double noise_scale  = 0.0;     ///< Noise feature size (0 = auto = 2*H at peak)
int noise_octaves   = 2;       ///< fBm octave count
```

- [ ] **Step 3: Add to CUDA RenderParams**

In `cuda/cuda_types.h`, after `double disk_noise_scale;`, add:

```cpp
int disk_noise_octaves;
```

- [ ] **Step 4: Add to VolDiskHostData**

In `cuda/cuda_vol_host_data.h`, after `double noise_scale;`, add:

```cpp
int noise_octaves;
```

- [ ] **Step 5: Commit**

```bash
git add include/grrt/types.h include/grrt/scene/volumetric_disk.h cuda/cuda_types.h cuda/cuda_vol_host_data.h
git commit -m "feat: Add noise_scale and noise_octaves fields to parameter structs"
```

---

### Task 2: Add `evaluate_fbm` to CPU noise

**Files:**
- Modify: `include/grrt/math/noise.h:30-36` (add declaration)
- Modify: `src/noise.cpp:98-103` (add implementation)

- [ ] **Step 1: Add declaration**

In `include/grrt/math/noise.h`, after the `evaluate_turbulent` declaration (line 36), add:

```cpp
/// Evaluate fractional Brownian motion noise at (x, y, z).
/// Standard fBm: lacunarity=2, persistence=0.5.
/// @param octaves Number of noise layers (1 = base only, 2 = matches old turbulent).
/// @return Value in approximately [-(2 - 2^(1-octaves)), +(2 - 2^(1-octaves))].
double evaluate_fbm(double x, double y, double z, int octaves) const;
```

- [ ] **Step 2: Add implementation**

In `src/noise.cpp`, after the `evaluate_turbulent` function (line 103), add:

```cpp
double SimplexNoise3D::evaluate_fbm(double x, double y, double z, int octaves) const {
    double result = 0.0;
    double amplitude = 1.0;
    double frequency = 1.0;
    for (int i = 0; i < octaves; ++i) {
        result += amplitude * evaluate(x * frequency, y * frequency, z * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return result;
}
```

- [ ] **Step 3: Commit**

```bash
git add include/grrt/math/noise.h src/noise.cpp
git commit -m "feat: Add evaluate_fbm (configurable octave fBm) to SimplexNoise3D"
```

---

### Task 3: Add `cuda_simplex_noise_fbm` to CUDA noise

**Files:**
- Modify: `cuda/cuda_noise.h:163-166` (add device function after `cuda_simplex_noise_turbulent`)

- [ ] **Step 1: Add device function**

In `cuda/cuda_noise.h`, after `cuda_simplex_noise_turbulent` (line 166), add:

```cpp
/// @brief Evaluate fBm simplex noise at (x, y, z) with configurable octaves.
///
/// Standard fBm: lacunarity=2, persistence=0.5.
/// @param octaves Number of noise layers.
/// @return Value in approximately [-(2 - 2^(1-octaves)), +(2 - 2^(1-octaves))].
__device__ inline double cuda_simplex_noise_fbm(double x, double y, double z, int octaves) {
    double result = 0.0;
    double amplitude = 1.0;
    double frequency = 1.0;
    for (int i = 0; i < octaves; ++i) {
        result += amplitude * cuda_simplex_noise_3d(x * frequency, y * frequency, z * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return result;
}
```

- [ ] **Step 2: Commit**

```bash
git add cuda/cuda_noise.h
git commit -m "feat: Add cuda_simplex_noise_fbm device function"
```

---

### Task 4: Update CPU density function — use fbm, clamp, honor noise_scale

**Files:**
- Modify: `src/volumetric_disk.cpp:204-218` (density function)
- Modify: `src/volumetric_disk.cpp:609-631` (normalize_density — honor user noise_scale)

- [ ] **Step 1: Update normalize_density to honor user noise_scale**

In `src/volumetric_disk.cpp`, replace line 630:

```cpp
noise_scale_ = 2.0 * H_lut_[peak_idx];
```

with:

```cpp
noise_scale_ = (params_.noise_scale > 0.0) ? params_.noise_scale : 2.0 * H_lut_[peak_idx];
```

- [ ] **Step 2: Update density() to use fbm and clamp**

In `src/volumetric_disk.cpp`, replace lines 217-218:

```cpp
    const double n = noise_.evaluate_turbulent(nx, ny, nz);
    return base * (1.0 + params_.turbulence * n);
```

with:

```cpp
    const double n = noise_.evaluate_fbm(nx, ny, nz, params_.noise_octaves);
    return std::max(0.0, base * (1.0 + params_.turbulence * n));
```

- [ ] **Step 3: Commit**

```bash
git add src/volumetric_disk.cpp
git commit -m "feat: Use fBm noise with configurable octaves, clamp density >= 0"
```

---

### Task 5: Update CUDA density function — use fbm, clamp, pass octaves

**Files:**
- Modify: `cuda/cuda_volumetric_disk.h:318-323` (vol_density_cgs)

- [ ] **Step 1: Update vol_density_cgs**

In `cuda/cuda_volumetric_disk.h`, replace lines 322-323:

```cpp
    const double n = cuda_simplex_noise_turbulent(nx, ny, nz);
    return base * (1.0 + params.disk_turbulence * n);
```

with:

```cpp
    const double n = cuda_simplex_noise_fbm(nx, ny, nz, params.disk_noise_octaves);
    return fmax(0.0, base * (1.0 + params.disk_turbulence * n));
```

- [ ] **Step 2: Commit**

```bash
git add cuda/cuda_volumetric_disk.h
git commit -m "feat: CUDA density uses fBm with configurable octaves, clamp >= 0"
```

---

### Task 6: Thread new params through CLI

**Files:**
- Modify: `cli/main.cpp:28-29` (help text)
- Modify: `cli/main.cpp:65-66` (defaults)
- Modify: `cli/main.cpp:116-119` (arg parsing)

- [ ] **Step 1: Add help text**

In `cli/main.cpp`, after the `--disk-seed` help line (line 29), add:

```cpp
    std::println("  --disk-noise-scale S  Noise feature size in M, 0=auto (default: 0)");
    std::println("  --disk-noise-octaves N  fBm octave count (default: 2)");
```

- [ ] **Step 2: Add defaults**

In `cli/main.cpp`, after `params.disk_seed = 42;` (line 66), add:

```cpp
    params.disk_noise_scale = 0.0;
    params.disk_noise_octaves = 2;
```

- [ ] **Step 3: Add arg parsing**

In `cli/main.cpp`, after the `--disk-seed` parsing block (line 119), add:

```cpp
        } else if (arg("--disk-noise-scale")) {
            if (auto v = next()) params.disk_noise_scale = std::atof(v);
        } else if (arg("--disk-noise-octaves")) {
            if (auto v = next()) params.disk_noise_octaves = std::atoi(v);
```

- [ ] **Step 4: Commit**

```bash
git add cli/main.cpp
git commit -m "feat: Add --disk-noise-scale and --disk-noise-octaves CLI flags"
```

---

### Task 7: Thread new params through CPU API

**Files:**
- Modify: `src/api.cpp:79-82` (VolumetricParams construction)

- [ ] **Step 1: Add param forwarding**

In `src/api.cpp`, after `vp.seed = static_cast<uint32_t>(params->disk_seed);` (line 82), add:

```cpp
            vp.noise_scale = params->disk_noise_scale;
            vp.noise_octaves = params->disk_noise_octaves;
```

- [ ] **Step 2: Commit**

```bash
git add src/api.cpp
git commit -m "feat: Thread noise_scale and noise_octaves through CPU API"
```

---

### Task 8: Thread new params through CUDA backend

**Files:**
- Modify: `cuda/cuda_vol_host_data.cpp:29` (extract noise_octaves)
- Modify: `cuda/cuda_backend.cu:126-128` (fill RenderParams)

- [ ] **Step 1: Update VolDiskHostData extraction**

In `cuda/cuda_vol_host_data.cpp`, after `data.noise_scale = disk.noise_scale();` (line 29), add:

```cpp
    data.noise_octaves = vp.noise_octaves;
```

Note: `noise_octaves` comes from `VolumetricParams vp`, not from the disk object, since the disk doesn't store it (it's only used at render time). So we also need to pass `noise_octaves` into `build_vol_disk_host_data`. Update the function signature in both `.h` and `.cpp`:

In `cuda/cuda_vol_host_data.h`, update the function signature to add `int noise_octaves` parameter:

```cpp
VolDiskHostData build_vol_disk_host_data(double mass, double spin,
                                          double r_outer, double peak_temperature,
                                          double alpha, double turbulence,
                                          double noise_scale,
                                          int noise_octaves,
                                          unsigned int seed);
```

In `cuda/cuda_vol_host_data.cpp`, update the function signature to match, add `vp.noise_scale = noise_scale;` after `vp.turbulence = turbulence;`, and set `data.noise_octaves = noise_octaves;` after `data.noise_scale = disk.noise_scale();`.

- [ ] **Step 2: Update CUDA backend to fill RenderParams**

In `cuda/cuda_backend.cu`, after `rp.disk_noise_scale = vd.noise_scale;` (line 127), add:

```cpp
        rp.disk_noise_octaves = vd.noise_octaves;
```

Also update the `build_vol_disk_host_data` call (~line 116) to pass the new parameters:

```cpp
        double vol_noise_scale = params->disk_noise_scale;  // 0.0 = auto
        int vol_noise_octaves = params->disk_noise_octaves > 0 ? params->disk_noise_octaves : 2;

        VolDiskHostData vd = build_vol_disk_host_data(mass, spin_a, disk_outer, disk_temp,
                                                       vol_alpha, vol_turb,
                                                       vol_noise_scale, vol_noise_octaves,
                                                       vol_seed);
```

- [ ] **Step 3: Commit**

```bash
git add cuda/cuda_vol_host_data.h cuda/cuda_vol_host_data.cpp cuda/cuda_backend.cu
git commit -m "feat: Thread noise_scale and noise_octaves through CUDA backend"
```

---

### Task 9: Build and smoke test

- [ ] **Step 1: Build**

```bash
cmake --build build --config Release 2>&1
```

Expected: clean build, no errors.

- [ ] **Step 2: Smoke test — default params (backward compat)**

```bash
./build/Release/grrt-cli --disk-volumetric --backend cuda --width 256 --height 256 --output test_default
```

Expected: renders without crash, produces `test_default.png`.

- [ ] **Step 3: Smoke test — high turbulence with large noise scale**

```bash
./build/Release/grrt-cli --disk-volumetric --backend cuda --width 256 --height 256 --disk-turbulence 1.5 --disk-noise-scale 5.0 --disk-noise-octaves 4 --output test_holes
```

Expected: renders without crash, `test_holes.png` should show visible holes/gaps in the disk.

- [ ] **Step 4: Commit (if any fixups needed)**
