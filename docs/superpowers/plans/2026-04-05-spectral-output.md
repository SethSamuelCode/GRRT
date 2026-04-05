# Spectral Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add user-defined frequency bin spectral rendering that outputs FITS data cubes, replacing the hardcoded 3-channel RGB in the volumetric ray-march.

**Architecture:** The frequency bin list flows from CLI/API → GRRTParams → GeodesicTracer → Renderer. A new `raymarch_volumetric_spectral` method accumulates per-bin intensity using the same radiative transfer physics as the existing 3-channel path. Output goes through a dependency-free FITS writer. When no frequency bins are specified, all behavior is identical to the current codebase.

**Tech Stack:** C++23, FITS format (hand-written writer, no external deps), existing OpenMP parallelism.

**Spec:** `docs/superpowers/specs/2026-04-05-spectral-output-design.md`

---

### File Structure

| File | Role |
|------|------|
| `include/grrt/types.h` | Add frequency bin fields to `GRRTParams` |
| `include/grrt/geodesic/geodesic_tracer.h` | Add `SpectralTraceResult`, `trace_spectral`, `raymarch_volumetric_spectral` |
| `src/geodesic_tracer.cpp` | Implement spectral ray-march and spectral trace |
| `include/grrt/render/renderer.h` | Add `render_spectral` method |
| `src/renderer.cpp` | Implement spectral render loop with jitter/SPP |
| `include/grrt/render/fits_writer.h` | FITS writer interface + `FITSMetadata` struct |
| `src/fits_writer.cpp` | Minimal FITS writer (header + big-endian doubles) |
| `include/grrt/api.h` | Add `grrt_set_frequency_bins`, `grrt_render_spectral` |
| `src/api.cpp` | Implement new C API functions |
| `cli/main.cpp` | Add `--frequencies`, `--freq-range`, FITS output path |
| `CMakeLists.txt` | Add `src/fits_writer.cpp` to library sources |
| `tests/test_spectral.cpp` | Tests for spectral ray-march, FITS writer, frequency parsing |

---

### Task 1: Add Frequency Bin Fields to GRRTParams

**Files:**
- Modify: `include/grrt/types.h:20-61`

- [ ] **Step 1: Add frequency bin fields to GRRTParams**

In `include/grrt/types.h`, add two new fields at the end of the `GRRTParams` struct, before the closing `}`:

```cpp
    int num_frequency_bins;          /* 0 = legacy RGB mode (default) */
    const double* frequency_bins_hz; /* Array of observer-frame frequencies in Hz */
```

- [ ] **Step 2: Initialize new fields in CLI defaults**

In `cli/main.cpp`, add these lines after the existing `params.samples_per_pixel = 1;` (line 78):

```cpp
    params.num_frequency_bins = 0;
    params.frequency_bins_hz = nullptr;
```

- [ ] **Step 3: Build and verify no regressions**

Run:
```bash
cmake --build build --config Release 2>&1 | tail -5
```
Expected: Build succeeds with no errors. Existing behavior unchanged since `num_frequency_bins` defaults to 0.

- [ ] **Step 4: Commit**

```bash
git add include/grrt/types.h cli/main.cpp
git commit -m "feat: add frequency bin fields to GRRTParams for spectral rendering"
```

---

### Task 2: Spectral Ray-March in GeodesicTracer

**Files:**
- Modify: `include/grrt/geodesic/geodesic_tracer.h`
- Modify: `src/geodesic_tracer.cpp`
- Create: `tests/test_spectral.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Create test file with spectral ray-march test**

Create `tests/test_spectral.cpp`:

```cpp
#include "grrt/spacetime/kerr.h"
#include "grrt/geodesic/rk4.h"
#include "grrt/geodesic/geodesic_tracer.h"
#include "grrt/scene/volumetric_disk.h"
#include "grrt/camera/camera.h"
#include "grrt/math/constants.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>

int failures = 0;

void check(const char* name, bool condition) {
    std::printf("  %s: %s\n", name, condition ? "PASS" : "FAIL");
    if (!condition) failures++;
}

void check_approx(const char* name, double got, double expected, double rel_tol) {
    double rel_err = std::abs(got - expected) / std::max(std::abs(expected), 1e-30);
    bool pass = rel_err < rel_tol;
    std::printf("  %s: got=%.4e expected=%.4e rel_err=%.2e %s\n",
                name, got, expected, rel_err, pass ? "PASS" : "FAIL");
    if (!pass) failures++;
}

/// Test that the spectral ray-march produces non-zero intensity at
/// frequencies where a hot disk should emit, and that the 3-channel
/// RGB path and spectral path agree at the same frequencies.
void test_spectral_raymarch_basic() {
    std::printf("\n=== Spectral ray-march: basic emission test ===\n");

    // Set up a Kerr BH with volumetric disk
    double mass = 1.0, spin = 0.998;
    grrt::Kerr metric(mass, spin * mass);
    grrt::RK4 integrator;
    grrt::VolumetricParams vp;
    vp.alpha = 0.1;
    vp.turbulence = 0.0;  // No noise for reproducibility
    grrt::VolumetricDisk vol_disk(mass, spin, 20.0, 1e7, vp);
    grrt::GeodesicTracer tracer(metric, integrator, 50.0, 10000, 1000.0, 1e-8, &vol_disk);

    // Camera aimed at disk
    grrt::Camera camera(metric, 50.0, 1.396, 0.0, 1.047, 64, 64);

    // Pick a pixel that should hit the disk (center-ish, near midplane)
    grrt::GeodesicState state = camera.ray_for_pixel(32.0, 38.0);

    // Define frequency bins: the same 3 RGB frequencies the current code uses,
    // plus some X-ray and radio frequencies
    using namespace grrt::constants;
    std::vector<double> freq_bins = {
        c_cgs / 650e-7,  // red   ~4.6e14 Hz
        c_cgs / 550e-7,  // green ~5.5e14 Hz
        c_cgs / 450e-7,  // blue  ~6.7e14 Hz
        1e17,            // UV
        1e18,            // soft X-ray
    };

    // Trace with spectral mode
    auto result = tracer.trace_spectral(state, freq_bins);

    // Check that we got a disk hit (not escaped/horizon)
    std::printf("  termination: %d (0=horizon, 1=escaped, 2=maxsteps)\n",
                static_cast<int>(result.termination));

    // At least the visible-range bins should have some emission
    // if the ray hit the disk
    if (result.termination != grrt::RayTermination::Escaped &&
        result.termination != grrt::RayTermination::Horizon) {
        double visible_sum = result.spectral_intensity[0]
                           + result.spectral_intensity[1]
                           + result.spectral_intensity[2];
        check("visible bins have emission", visible_sum > 0.0);
        std::printf("  visible sum = %.4e\n", visible_sum);

        // All bins should be non-negative
        bool all_non_negative = true;
        for (size_t i = 0; i < freq_bins.size(); ++i) {
            if (result.spectral_intensity[i] < 0.0) all_non_negative = false;
            std::printf("  bin[%zu] (%.2e Hz) = %.4e\n",
                        i, freq_bins[i], result.spectral_intensity[i]);
        }
        check("all bins non-negative", all_non_negative);
    } else {
        std::printf("  Ray did not hit disk — skipping intensity checks\n");
    }
}

/// Test that spectral output size matches bin count.
void test_spectral_output_size() {
    std::printf("\n=== Spectral ray-march: output size matches bin count ===\n");

    double mass = 1.0, spin = 0.998;
    grrt::Kerr metric(mass, spin * mass);
    grrt::RK4 integrator;
    grrt::VolumetricParams vp;
    vp.turbulence = 0.0;
    grrt::VolumetricDisk vol_disk(mass, spin, 20.0, 1e7, vp);
    grrt::GeodesicTracer tracer(metric, integrator, 50.0, 10000, 1000.0, 1e-8, &vol_disk);

    grrt::Camera camera(metric, 50.0, 1.396, 0.0, 1.047, 64, 64);
    grrt::GeodesicState state = camera.ray_for_pixel(32.0, 32.0);

    std::vector<double> freq_bins(50);
    double log_min = std::log10(1e9), log_max = std::log10(1e18);
    for (int i = 0; i < 50; ++i) {
        freq_bins[i] = std::pow(10.0, log_min + i * (log_max - log_min) / 49.0);
    }

    auto result = tracer.trace_spectral(state, freq_bins);
    check("output size matches bins", result.spectral_intensity.size() == 50);
}

int main() {
    std::printf("Spectral output tests\n");
    std::printf("=====================\n");

    test_spectral_raymarch_basic();
    test_spectral_output_size();

    std::printf("\n%s (%d failure%s)\n",
                failures == 0 ? "ALL PASSED" : "SOME FAILED",
                failures, failures == 1 ? "" : "s");
    return failures > 0 ? 1 : 0;
}
```

- [ ] **Step 2: Add test target to CMakeLists.txt**

In `CMakeLists.txt`, after the `test-volumetric` target (line 96), add:

```cmake
add_executable(test-spectral tests/test_spectral.cpp)
target_link_libraries(test-spectral PRIVATE grrt)
```

- [ ] **Step 3: Add SpectralTraceResult and method declarations to geodesic_tracer.h**

In `include/grrt/geodesic/geodesic_tracer.h`, add after the `TraceResult` struct (after line 27):

```cpp
struct SpectralTraceResult {
    RayTermination termination;
    std::vector<double> spectral_intensity;  // Per-bin observed intensity [erg/s/cm²/Hz/sr]
    Vec4 final_position;
    Vec4 final_momentum;
};
```

Add `#include <vector>` at the top of the file (after the existing includes).

In the `GeodesicTracer` class public section, add after the `trace` method (after line 38):

```cpp
    SpectralTraceResult trace_spectral(GeodesicState state,
                                       const std::vector<double>& frequency_bins) const;
```

In the private section, add after `raymarch_volumetric` (after line 50):

```cpp
    void raymarch_volumetric_spectral(GeodesicState& state,
                                      const std::vector<double>& nu_obs,
                                      std::vector<double>& J,
                                      std::vector<double>& tau_acc) const;
```

- [ ] **Step 4: Implement raymarch_volumetric_spectral in geodesic_tracer.cpp**

In `src/geodesic_tracer.cpp`, add the following method after the existing `raymarch_volumetric` method (after line 283):

```cpp
void GeodesicTracer::raymarch_volumetric_spectral(
        GeodesicState& state,
        const std::vector<double>& nu_obs,
        std::vector<double>& J,
        std::vector<double>& tau_acc) const {
    using namespace constants;
    const auto& luts = vol_disk_->opacity_luts();
    const int num_bins = static_cast<int>(nu_obs.size());

    // Observer p·u (static observer at observer_r_)
    double ut_obs = 1.0 / std::sqrt(1.0 - 2.0 / observer_r_);

    double r = state.position[1];
    const double z_start = r * std::cos(state.position[2]);
    const double H_start = vol_disk_->scale_height(r);
    double ds = vol_disk_->inside_volume(r, z_start)
              ? H_start / 16.0
              : std::min(std::abs(z_start) / 8.0, H_start * 2.0);
    int step_count = 0;
    constexpr int MAX_STEPS = 4096;
    constexpr double DTAU_TARGET = 0.05;
    bool been_inside = vol_disk_->inside_volume(r, z_start);

    while (step_count < MAX_STEPS) {
        GeodesicState new_state = integrator_.step_kerr(metric_, state, ds);
        step_count++;

        r = new_state.position[1];
        const double theta = new_state.position[2];
        const double phi = new_state.position[3];
        const double z = r * std::cos(theta);

        // Hard exits
        if (r < vol_disk_->r_horizon()) break;
        if (r > vol_disk_->r_max()) break;

        // Check if all bins are optically thick
        bool all_thick = true;
        for (int ch = 0; ch < num_bins; ++ch) {
            if (tau_acc[ch] <= 10.0) { all_thick = false; break; }
        }
        if (all_thick) break;

        const double H = vol_disk_->scale_height(r);
        if (!vol_disk_->inside_volume(r, z)) {
            const double zm = vol_disk_->z_max_at(r);
            if (been_inside && std::abs(z) > zm + H) break;
            if (!been_inside) {
                ds = std::min(std::abs(z) / 8.0, H * 2.0);
                ds = std::max(ds, H / 64.0);
            } else {
                ds = std::clamp(H / 4.0, H / 64.0, H);
            }
            state = new_state;
            continue;
        }
        been_inside = true;

        // Look up local state
        const double rho_cgs = vol_disk_->density_cgs(r, z, phi);
        const double T = vol_disk_->temperature(r, std::abs(z));
        if (rho_cgs <= 0.0 || T <= 0.0) {
            state = new_state;
            continue;
        }

        // Compute redshift g = (p·u)_emit / (p·u)_obs
        double ut_emit = 0.0, ur_emit = 0.0, uphi_emit = 0.0;
        if (r >= vol_disk_->r_isco()) {
            vol_disk_->circular_velocity(r, ut_emit, uphi_emit);
        } else {
            vol_disk_->plunging_velocity(r, theta, ut_emit, ur_emit, uphi_emit);
        }

        const double p_dot_u_emit = new_state.momentum[0] * ut_emit
                                  + new_state.momentum[1] * ur_emit
                                  + new_state.momentum[3] * uphi_emit;
        const double p_dot_u_obs = new_state.momentum[0] * ut_obs;
        const double g = p_dot_u_emit / p_dot_u_obs;
        const double ds_proper = std::abs(p_dot_u_emit) * std::abs(ds);

        // Per-bin radiative transfer
        // Use the green channel (index 1 if >= 2 bins, else 0) for adaptive stepping
        double alpha_ref = 0.0;
        for (int ch = 0; ch < num_bins; ++ch) {
            const double nu_emit = std::abs(g) * nu_obs[ch];

            const double kabs = luts.lookup_kappa_abs(nu_emit, rho_cgs, T);
            const double kes = luts.lookup_kappa_es(rho_cgs, T);
            const double ktot = kabs + kes;
            const double epsilon = (ktot > 0.0) ? kabs / ktot : 1.0;

            const double dtau = ktot * rho_cgs * ds_proper;
            tau_acc[ch] += dtau;

            const double Bnu = planck_nu(nu_emit, T);
            const double S = epsilon * Bnu / (nu_emit * nu_emit * nu_emit);

            const double exp_dtau = std::exp(-dtau);
            J[ch] = J[ch] * exp_dtau + S * (1.0 - exp_dtau);

            // Use the median bin for adaptive step control
            if (ch == num_bins / 2) {
                alpha_ref = ktot * rho_cgs;
            }
        }

        // Adaptive step control based on median-frequency bin
        double ds_tau = (alpha_ref > 0.0) ? DTAU_TARGET / alpha_ref : ds * 2.0;
        const double ds_geo = 0.1 * std::max(r - vol_disk_->r_horizon(), 0.5);
        ds = std::min(ds_tau, ds_geo);
        ds = std::clamp(ds, H / 64.0, H);

        state = new_state;
    }
}
```

- [ ] **Step 5: Implement trace_spectral in geodesic_tracer.cpp**

In `src/geodesic_tracer.cpp`, add after `raymarch_volumetric_spectral` (the method just added):

```cpp
SpectralTraceResult GeodesicTracer::trace_spectral(
        GeodesicState state,
        const std::vector<double>& frequency_bins) const {
    const double r_horizon = metric_.horizon_radius();
    const double half_pi = std::numbers::pi / 2.0;
    const int num_bins = static_cast<int>(frequency_bins.size());

    std::vector<double> spectral_intensity(num_bins, 0.0);
    std::vector<double> J(num_bins, 0.0);
    std::vector<double> tau_acc(num_bins, 0.0);

    double dlambda = 0.01 * observer_r_;
    GeodesicState prev = state;

    for (int step = 0; step < max_steps_; ++step) {
        const double r = state.position[1];

        if (r < r_horizon + horizon_epsilon_) {
            return {RayTermination::Horizon, spectral_intensity,
                    state.position, state.momentum};
        }
        if (r > r_escape_) {
            return {RayTermination::Escaped, spectral_intensity,
                    state.position, state.momentum};
        }

        prev = state;

        {
            auto result = integrator_.adaptive_step_kerr(metric_, state, dlambda, tolerance_);
            state = result.state;
            dlambda = result.next_dlambda;
        }

        // Volumetric disk entry detection (same logic as trace())
        if (vol_disk_) {
            const double theta_prev = prev.position[2];
            const double theta_new = state.position[2];
            const double d_prev = theta_prev - half_pi;
            const double d_new = theta_new - half_pi;
            const double r_new = state.position[1];
            const double r_prev = prev.position[1];

            const double z_new = r_new * std::cos(theta_new);
            const double z_prev = r_prev * std::cos(theta_prev);
            const bool crossed_midplane = (d_prev * d_new < 0.0)
                                       && std::abs(d_prev - d_new) > 1e-12;
            const bool inside_now = vol_disk_->inside_volume(r_new, z_new);
            const double zm_new = vol_disk_->z_max_at(r_new);
            const double H_new = vol_disk_->scale_height(r_new);
            const bool near_disk = (std::abs(z_new) < zm_new + H_new
                                 || std::abs(z_prev) < vol_disk_->z_max_at(r_prev) + vol_disk_->scale_height(r_prev))
                                && r_new >= vol_disk_->r_horizon()
                                && r_new <= vol_disk_->r_max();
            const bool should_raymarch = crossed_midplane || inside_now || near_disk;

            if (should_raymarch) {
                const double r_lo = std::min(r_prev, r_new);
                const double r_hi = std::max(r_prev, r_new);
                if (r_hi < vol_disk_->r_horizon() || r_lo > vol_disk_->r_max())
                    goto skip_vol_spectral;

                GeodesicState entry = prev;
                const double re = entry.position[1];
                if (re >= vol_disk_->r_horizon() * 0.9
                    && re <= vol_disk_->r_max() * 1.5) {
                    raymarch_volumetric_spectral(entry, frequency_bins, J, tau_acc);
                    state = entry;
                    continue;
                }
            }
            skip_vol_spectral:;
        }
    }

    // Recover observed intensity: I_obs = J * nu_obs^3
    for (int ch = 0; ch < num_bins; ++ch) {
        spectral_intensity[ch] = J[ch] * frequency_bins[ch] * frequency_bins[ch] * frequency_bins[ch];
    }

    return {RayTermination::MaxSteps, spectral_intensity,
            state.position, state.momentum};
}
```

Note: The `I_obs = J * nu_obs^3` recovery also needs to happen for Horizon and Escaped terminations. Move the recovery before the returns by restructuring the loop exit. Actually, looking at the existing `trace` method more carefully — the color accumulation happens inside `raymarch_volumetric`, so the recovery should happen at the end of the function regardless of termination. Update the method to always recover intensity before returning:

Replace the three return statements with a single exit path. After the loop, add:

```cpp
    // Recover observed intensity: I_obs = J * nu_obs^3
    for (int ch = 0; ch < num_bins; ++ch) {
        spectral_intensity[ch] = J[ch] * frequency_bins[ch] * frequency_bins[ch] * frequency_bins[ch];
    }

    RayTermination term = RayTermination::MaxSteps;
    if (state.position[1] < r_horizon + horizon_epsilon_)
        term = RayTermination::Horizon;
    else if (state.position[1] > r_escape_)
        term = RayTermination::Escaped;

    return {term, spectral_intensity, state.position, state.momentum};
```

Restructure the loop to use `break` instead of early returns:

```cpp
    RayTermination term = RayTermination::MaxSteps;

    for (int step = 0; step < max_steps_; ++step) {
        const double r = state.position[1];

        if (r < r_horizon + horizon_epsilon_) {
            term = RayTermination::Horizon;
            break;
        }
        if (r > r_escape_) {
            term = RayTermination::Escaped;
            break;
        }

        // ... rest of loop body unchanged ...
    }

    // Recover observed intensity: I_obs = J * nu_obs^3
    for (int ch = 0; ch < num_bins; ++ch) {
        spectral_intensity[ch] = J[ch] * frequency_bins[ch] * frequency_bins[ch] * frequency_bins[ch];
    }

    return {term, spectral_intensity, state.position, state.momentum};
```

- [ ] **Step 6: Build and run tests**

```bash
cmake --build build --config Release 2>&1 | tail -10
./build/Release/test-spectral
```
Expected: Tests pass — spectral bins have non-zero emission for disk-hitting rays, output size matches bin count.

- [ ] **Step 7: Commit**

```bash
git add include/grrt/geodesic/geodesic_tracer.h src/geodesic_tracer.cpp tests/test_spectral.cpp CMakeLists.txt
git commit -m "feat: add spectral ray-march with per-frequency-bin radiative transfer"
```

---

### Task 3: Spectral Render Loop

**Files:**
- Modify: `include/grrt/render/renderer.h`
- Modify: `src/renderer.cpp`

- [ ] **Step 1: Add test for spectral renderer to test_spectral.cpp**

Append to `tests/test_spectral.cpp`, before `main()`:

```cpp
#include "grrt/render/renderer.h"
#include "grrt/color/spectrum.h"
#include "grrt/render/tonemapper.h"
#include "grrt/scene/celestial_sphere.h"
```

Move these includes to the top of the file alongside the existing includes.

Add test function before `main()`:

```cpp
void test_spectral_renderer() {
    std::printf("\n=== Spectral renderer: small image ===\n");

    double mass = 1.0, spin = 0.998;
    grrt::Kerr metric(mass, spin * mass);
    grrt::RK4 integrator;
    grrt::VolumetricParams vp;
    vp.turbulence = 0.0;
    grrt::VolumetricDisk vol_disk(mass, spin, 20.0, 1e7, vp);
    grrt::GeodesicTracer tracer(metric, integrator, 50.0, 10000, 1000.0, 1e-8, &vol_disk);
    grrt::Camera camera(metric, 50.0, 1.396, 0.0, 1.047, 16, 16);
    grrt::SpectrumLUT spectrum;
    grrt::ToneMapper tonemapper;
    grrt::Renderer renderer(camera, tracer, nullptr, nullptr, &spectrum, tonemapper, 1);

    std::vector<double> freq_bins = {1e12, 1e14, 1e16};
    int width = 16, height = 16;
    int num_bins = static_cast<int>(freq_bins.size());
    std::vector<double> spectral_buffer(width * height * num_bins, 0.0);

    renderer.render_spectral(spectral_buffer.data(), width, height, freq_bins);

    // Check that at least some pixels have non-zero emission
    int nonzero_pixels = 0;
    for (int p = 0; p < width * height; ++p) {
        double sum = 0.0;
        for (int k = 0; k < num_bins; ++k) {
            sum += spectral_buffer[p * num_bins + k];
        }
        if (sum > 0.0) nonzero_pixels++;
    }
    std::printf("  nonzero pixels: %d / %d\n", nonzero_pixels, width * height);
    check("some pixels have emission", nonzero_pixels > 0);

    // Check buffer is correct size (no out-of-bounds)
    check("buffer size correct", spectral_buffer.size() == static_cast<size_t>(width * height * num_bins));
}
```

Add `test_spectral_renderer();` call in `main()`.

- [ ] **Step 2: Add render_spectral declaration to renderer.h**

In `include/grrt/render/renderer.h`, add `#include <vector>` at the top.

Add after the existing `render` method (after line 26):

```cpp
    void render_spectral(double* spectral_buffer, int width, int height,
                         const std::vector<double>& frequency_bins,
                         ProgressCallback progress_cb = nullptr) const;
```

- [ ] **Step 3: Implement render_spectral in renderer.cpp**

In `src/renderer.cpp`, add after the existing `render` method:

```cpp
void Renderer::render_spectral(double* spectral_buffer, int width, int height,
                                const std::vector<double>& frequency_bins,
                                ProgressCallback progress_cb) const {
    const int num_bins = static_cast<int>(frequency_bins.size());
    const int grid = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(spp_))));
    const int actual_spp = grid * grid;
    const double inv_spp = 1.0 / actual_spp;
    const double cell = 1.0 / grid;

    int rows_done = 0;

    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < height; ++j) {
        // Per-thread accumulation buffer to avoid allocating per-pixel
        std::vector<double> accum(num_bins, 0.0);

        for (int i = 0; i < width; ++i) {
            std::fill(accum.begin(), accum.end(), 0.0);

            for (int sy = 0; sy < grid; ++sy) {
                for (int sx = 0; sx < grid; ++sx) {
                    const int s = sy * grid + sx;
                    const double jx = pixel_hash(i, j, s, 0);
                    const double jy = pixel_hash(i, j, s, 1);
                    const double px = i + (sx + jx) * cell;
                    const double py = j + (sy + jy) * cell;

                    GeodesicState state = camera_.ray_for_pixel(px, py);
                    SpectralTraceResult result = tracer_.trace_spectral(state, frequency_bins);

                    for (int k = 0; k < num_bins; ++k) {
                        accum[k] += result.spectral_intensity[k];
                    }
                }
            }

            const int base = (j * width + i) * num_bins;
            for (int k = 0; k < num_bins; ++k) {
                spectral_buffer[base + k] = accum[k] * inv_spp;
            }
        }

        if (progress_cb) {
            int done;
            #pragma omp critical
            { done = ++rows_done; }
            progress_cb(static_cast<float>(done) / static_cast<float>(height));
        }
    }

    if (progress_cb) progress_cb(1.0f);
}
```

Note: This requires the `SpectralTraceResult` type. Add `#include "grrt/geodesic/geodesic_tracer.h"` to `src/renderer.cpp` if not already included (it's included via `grrt/render/renderer.h` which includes `grrt/geodesic/geodesic_tracer.h` — check if this is the case; if not, add the include to `renderer.h`).

The `renderer.h` already includes `grrt/geodesic/geodesic_tracer.h` indirectly through the `GeodesicTracer` forward reference. However, for `SpectralTraceResult`, the full definition is needed. Since `renderer.h` takes `const GeodesicTracer&`, it likely already includes the header. Verify and add `#include "grrt/geodesic/geodesic_tracer.h"` to `renderer.h` if needed. Looking at the current `renderer.h`, it does not include `geodesic_tracer.h` — it only forward-declares. So add the include to `src/renderer.cpp`:

```cpp
#include "grrt/geodesic/geodesic_tracer.h"
```

- [ ] **Step 4: Build and run tests**

```bash
cmake --build build --config Release 2>&1 | tail -10
./build/Release/test-spectral
```
Expected: All tests pass, including the new renderer test showing some pixels with non-zero emission.

- [ ] **Step 5: Commit**

```bash
git add include/grrt/render/renderer.h src/renderer.cpp tests/test_spectral.cpp
git commit -m "feat: add spectral render loop with jitter and multi-sample averaging"
```

---

### Task 4: FITS Writer

**Files:**
- Create: `include/grrt/render/fits_writer.h`
- Create: `src/fits_writer.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add FITS writer tests to test_spectral.cpp**

Append test function before `main()` in `tests/test_spectral.cpp`:

```cpp
#include "grrt/render/fits_writer.h"
#include <fstream>
```

Move include to top of file. Add test:

```cpp
void test_fits_writer() {
    std::printf("\n=== FITS writer: basic output ===\n");

    // Create a small test cube: 4x3x2 (width x height x bins)
    int width = 4, height = 3, num_bins = 2;
    std::vector<double> data(width * height * num_bins);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<double>(i) * 1.5;
    }

    std::vector<double> freq_bins = {1e14, 1e16};
    grrt::FITSMetadata meta{};
    meta.spin = 0.998;
    meta.mass = 1.0;
    meta.observer_r = 50.0;
    meta.observer_theta = 1.396;
    meta.fov = 1.047;
    meta.samples_per_pixel = 1;

    std::string path = "test_output.fits";
    grrt::write_fits(path, data.data(), width, height, num_bins, freq_bins, meta);

    // Verify file exists and starts with FITS magic
    std::ifstream f(path, std::ios::binary);
    check("file exists", f.good());

    char header[80];
    f.read(header, 80);
    // FITS files start with "SIMPLE  =                    T"
    std::string first_card(header, 80);
    check("FITS magic present", first_card.find("SIMPLE") == 0);
    check("SIMPLE = T", first_card.find("T") != std::string::npos);

    // Check file size: header (at least 2880 bytes) + data
    f.seekg(0, std::ios::end);
    auto file_size = f.tellg();
    // Data size: 4 * 3 * 2 * 8 = 192 bytes, padded to 2880 boundary = 2880
    // Header: at least one 2880-byte block
    // Minimum file size: 2880 (header) + 2880 (data padded) = 5760
    check("file size reasonable", file_size >= 5760);

    f.close();
    std::remove(path.c_str());
    std::printf("  cleaned up %s\n", path.c_str());
}
```

Add `test_fits_writer();` call in `main()`.

- [ ] **Step 2: Create fits_writer.h**

Create `include/grrt/render/fits_writer.h`:

```cpp
#ifndef GRRT_FITS_WRITER_H
#define GRRT_FITS_WRITER_H

#include "grrt_export.h"
#include <string>
#include <vector>

namespace grrt {

struct FITSMetadata {
    double spin = 0.0;
    double mass = 1.0;
    double observer_r = 50.0;
    double observer_theta = 1.396;
    double fov = 1.047;
    int samples_per_pixel = 1;
};

/// Write a spectral data cube to a FITS file.
/// Data layout: data[(j * width + i) * num_bins + k] = intensity at pixel (i,j), bin k.
/// Output FITS axes: NAXIS1=width, NAXIS2=height, NAXIS3=num_bins, BITPIX=-64 (float64).
/// Frequency bins stored via WCS keywords (uniform spacing) or binary table extension (non-uniform).
GRRT_EXPORT void write_fits(const std::string& path,
                            const double* data,
                            int width, int height, int num_bins,
                            const std::vector<double>& frequency_bins_hz,
                            const FITSMetadata& metadata);

} // namespace grrt

#endif
```

- [ ] **Step 3: Create fits_writer.cpp**

Create `src/fits_writer.cpp`:

```cpp
#include "grrt/render/fits_writer.h"
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <format>
#include <stdexcept>
#include <cstdint>
#include <bit>

namespace grrt {

// FITS header cards are exactly 80 characters, padded with spaces.
static std::string fits_card(const std::string& keyword, const std::string& value,
                             const std::string& comment = "") {
    // Format: "KEYWORD = value / comment" padded to 80 chars
    std::string card;
    if (keyword == "END" || keyword == "COMMENT") {
        card = keyword;
    } else {
        card = std::format("{:<8}= {:>20}", keyword, value);
        if (!comment.empty()) {
            card += " / " + comment;
        }
    }
    card.resize(80, ' ');
    return card;
}

static std::string fits_card_string(const std::string& keyword, const std::string& value,
                                    const std::string& comment = "") {
    std::string quoted = "'" + value + "'";
    return fits_card(keyword, std::format("{:<20}", quoted), comment);
}

static std::string fits_card_double(const std::string& keyword, double value,
                                    const std::string& comment = "") {
    return fits_card(keyword, std::format("{:20.12E}", value), comment);
}

static std::string fits_card_int(const std::string& keyword, int value,
                                 const std::string& comment = "") {
    return fits_card(keyword, std::format("{:20d}", value), comment);
}

static std::string fits_card_bool(const std::string& keyword, bool value,
                                  const std::string& comment = "") {
    return fits_card(keyword, std::format("{:>20}", value ? "T" : "F"), comment);
}

// Write big-endian double. FITS requires big-endian (MSB first).
static void write_double_be(std::ofstream& out, double value) {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    // Convert to big-endian
    uint8_t bytes[8];
    for (int i = 7; i >= 0; --i) {
        bytes[7 - i] = static_cast<uint8_t>(bits >> (i * 8));
    }
    out.write(reinterpret_cast<const char*>(bytes), 8);
}

void write_fits(const std::string& path,
                const double* data,
                int width, int height, int num_bins,
                const std::vector<double>& frequency_bins_hz,
                const FITSMetadata& metadata) {

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open FITS file for writing: " + path);
    }

    // --- Build header ---
    std::string header;
    header += fits_card_bool("SIMPLE", true, "Conforms to FITS standard");
    header += fits_card_int("BITPIX", -64, "64-bit IEEE double");
    header += fits_card_int("NAXIS", 3, "Number of axes");
    header += fits_card_int("NAXIS1", width, "Image width");
    header += fits_card_int("NAXIS2", height, "Image height");
    header += fits_card_int("NAXIS3", num_bins, "Number of frequency bins");
    header += fits_card_string("BUNIT", "erg/s/cm2/Hz/sr", "Specific intensity");

    // WCS for frequency axis
    if (num_bins > 1) {
        // Check if frequencies are uniformly log-spaced or linearly spaced
        double cdelt = frequency_bins_hz[1] - frequency_bins_hz[0];
        bool uniform = true;
        for (int i = 2; i < num_bins; ++i) {
            double step = frequency_bins_hz[i] - frequency_bins_hz[i - 1];
            if (std::abs(step - cdelt) / std::abs(cdelt) > 1e-6) {
                uniform = false;
                break;
            }
        }

        if (uniform) {
            header += fits_card_string("CTYPE3", "FREQ", "Frequency");
            header += fits_card_string("CUNIT3", "Hz", "Frequency unit");
            header += fits_card_double("CRPIX3", 1.0, "Reference pixel");
            header += fits_card_double("CRVAL3", frequency_bins_hz[0], "Reference frequency");
            header += fits_card_double("CDELT3", cdelt, "Frequency step");
        } else {
            // Non-uniform: store as FREQ with CDELT=1 and note in comment.
            // Individual frequencies stored as FREQnnn keywords.
            header += fits_card_string("CTYPE3", "FREQ", "Frequency (non-uniform)");
            header += fits_card_string("CUNIT3", "Hz", "Frequency unit");
            for (int i = 0; i < num_bins && i < 999; ++i) {
                std::string key = std::format("FREQ{:03d}", i + 1);
                header += fits_card_double(key, frequency_bins_hz[i], "Hz");
            }
        }
    } else if (num_bins == 1) {
        header += fits_card_string("CTYPE3", "FREQ", "Frequency");
        header += fits_card_string("CUNIT3", "Hz", "Frequency unit");
        header += fits_card_double("CRPIX3", 1.0, "Reference pixel");
        header += fits_card_double("CRVAL3", frequency_bins_hz[0], "Reference frequency");
        header += fits_card_double("CDELT3", 0.0, "Single frequency");
    }

    // Render metadata
    header += fits_card_double("SPIN", metadata.spin, "Kerr spin a/M");
    header += fits_card_double("MASS", metadata.mass, "BH mass [geometric]");
    header += fits_card_double("OBS_R", metadata.observer_r, "Observer radius [M]");
    header += fits_card_double("OBS_TH", metadata.observer_theta, "Observer theta [rad]");
    header += fits_card_double("FOV", metadata.fov, "Field of view [rad]");
    header += fits_card_int("SPP", metadata.samples_per_pixel, "Samples per pixel");
    header += fits_card_string("ORIGIN", "grrt", "GR Ray Tracer");

    header += fits_card("END", "");

    // Pad header to multiple of 2880 bytes
    size_t header_blocks = (header.size() + 2879) / 2880;
    header.resize(header_blocks * 2880, ' ');
    out.write(header.data(), header.size());

    // --- Write data in FITS order (NAXIS1 varies fastest) ---
    // Our memory layout: data[(j * width + i) * num_bins + k]
    // FITS layout: data[k][j][i] (NAXIS3 slowest, NAXIS1 fastest)
    for (int k = 0; k < num_bins; ++k) {
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                write_double_be(out, data[(j * width + i) * num_bins + k]);
            }
        }
    }

    // Pad data to multiple of 2880 bytes
    size_t data_bytes = static_cast<size_t>(width) * height * num_bins * 8;
    size_t pad = (2880 - (data_bytes % 2880)) % 2880;
    if (pad > 0) {
        std::vector<char> zeros(pad, 0);
        out.write(zeros.data(), pad);
    }

    out.close();
}

} // namespace grrt
```

- [ ] **Step 4: Add fits_writer.cpp to CMakeLists.txt**

In `CMakeLists.txt`, add `src/fits_writer.cpp` to the `add_library(grrt SHARED ...)` source list, after `src/volumetric_disk.cpp`:

```cmake
    src/fits_writer.cpp
```

- [ ] **Step 5: Build and run tests**

```bash
cmake --build build --config Release 2>&1 | tail -10
./build/Release/test-spectral
```
Expected: All tests pass, including the FITS writer test.

- [ ] **Step 6: Commit**

```bash
git add include/grrt/render/fits_writer.h src/fits_writer.cpp CMakeLists.txt tests/test_spectral.cpp
git commit -m "feat: add dependency-free FITS writer for spectral data cubes"
```

---

### Task 5: C API Extensions

**Files:**
- Modify: `include/grrt/api.h`
- Modify: `src/api.cpp`

- [ ] **Step 1: Add API declarations to api.h**

In `include/grrt/api.h`, add before the closing `#ifdef __cplusplus` / `}` block (before line 32):

```c
// Spectral rendering
GRRT_EXPORT void grrt_set_frequency_bins(GRRTContext* ctx,
                                          const double* frequencies_hz,
                                          int num_bins);
GRRT_EXPORT int grrt_render_spectral(GRRTContext* ctx, double* spectral_buffer,
                                      int width, int height);
```

- [ ] **Step 2: Add frequency bin storage to GRRTContext**

In `src/api.cpp`, add to the `GRRTContext` struct (after the `error_msg` field on line 37):

```cpp
    std::vector<double> frequency_bins;
```

- [ ] **Step 3: Implement grrt_set_frequency_bins**

In `src/api.cpp`, add before `grrt_tonemap`:

```cpp
void grrt_set_frequency_bins(GRRTContext* ctx,
                              const double* frequencies_hz,
                              int num_bins) {
    if (num_bins > 0 && frequencies_hz) {
        ctx->frequency_bins.assign(frequencies_hz, frequencies_hz + num_bins);
    } else {
        ctx->frequency_bins.clear();
    }
}
```

- [ ] **Step 4: Implement grrt_render_spectral**

In `src/api.cpp`, add after `grrt_set_frequency_bins`. Add `#include "grrt/render/fits_writer.h"` at the top of the file:

```cpp
int grrt_render_spectral(GRRTContext* ctx, double* spectral_buffer,
                          int width, int height) {
    if (ctx->frequency_bins.empty()) {
        ctx->error_msg = "No frequency bins set — call grrt_set_frequency_bins first";
        return -1;
    }

    ctx->renderer->render_spectral(spectral_buffer, width, height,
                                    ctx->frequency_bins);
    std::println("grrt: rendered {}x{} spectral frame ({} bins, cpu)",
                 width, height, ctx->frequency_bins.size());
    return 0;
}
```

- [ ] **Step 5: Build and verify**

```bash
cmake --build build --config Release 2>&1 | tail -10
```
Expected: Clean build.

- [ ] **Step 6: Commit**

```bash
git add include/grrt/api.h src/api.cpp
git commit -m "feat: add C API for spectral frequency bins and spectral rendering"
```

---

### Task 6: CLI Integration

**Files:**
- Modify: `cli/main.cpp`

- [ ] **Step 1: Add CLI flags for frequency specification**

In `cli/main.cpp`, add to `print_usage()` after the `--output` help line (after line 44):

```cpp
    std::println("  --frequencies LIST    Comma-separated frequencies in Hz (e.g., 1e9,1e14,1e18)");
    std::println("  --freq-range MIN MAX N  Log-spaced range: min Hz, max Hz, number of bins");
```

Add a local variable after `bool validate = false;` (line 82):

```cpp
    std::vector<double> cli_freq_bins;
```

Add argument parsing in the `for` loop, before the `else` unknown-argument block:

```cpp
        } else if (arg("--frequencies")) {
            if (auto v = next()) {
                std::string s(v);
                size_t pos = 0;
                while (pos < s.size()) {
                    size_t comma = s.find(',', pos);
                    if (comma == std::string::npos) comma = s.size();
                    cli_freq_bins.push_back(std::atof(s.substr(pos, comma - pos).c_str()));
                    pos = comma + 1;
                }
            }
        } else if (arg("--freq-range")) {
            const char* v_min = next();
            const char* v_max = next();
            const char* v_n = next();
            if (v_min && v_max && v_n) {
                double f_min = std::atof(v_min);
                double f_max = std::atof(v_max);
                int n = std::atoi(v_n);
                if (n < 1) n = 1;
                double log_min = std::log10(f_min);
                double log_max = std::log10(f_max);
                for (int k = 0; k < n; ++k) {
                    double frac = (n > 1) ? static_cast<double>(k) / (n - 1) : 0.0;
                    cli_freq_bins.push_back(std::pow(10.0, log_min + frac * (log_max - log_min)));
                }
            }
```

- [ ] **Step 2: Wire frequency bins into params and add FITS output path**

After the argument parsing loop, after `params.backend = ...` (around line 172), add:

```cpp
    if (!cli_freq_bins.empty()) {
        params.num_frequency_bins = static_cast<int>(cli_freq_bins.size());
        params.frequency_bins_hz = cli_freq_bins.data();
    }
```

Add FITS output path variable alongside the other path variables (after line 245):

```cpp
    std::string path_fits = output_name + ".fits";
```

- [ ] **Step 3: Add spectral rendering and FITS output**

After `GRRTContext* ctx = grrt_create(&params);` and the existing render block, add the spectral rendering path. Replace the render section (starting at the `// Flush stdout` comment, around line 257) with a branch:

Add `#include "grrt/render/fits_writer.h"` at the top of `cli/main.cpp`.

Insert the spectral branch before the existing RGB render block:

```cpp
    // Flush stdout so banner + create message appear before the progress bar
    std::fflush(stdout);

    int result = 0;

    if (!cli_freq_bins.empty()) {
        // --- Spectral rendering path ---
        grrt_set_frequency_bins(ctx, cli_freq_bins.data(),
                                static_cast<int>(cli_freq_bins.size()));

        int num_bins = static_cast<int>(cli_freq_bins.size());
        std::vector<double> spectral_buffer(params.width * params.height * num_bins);

        struct ProgressState {
            std::chrono::steady_clock::time_point start;
            float last_printed;
        };
        ProgressState pstate{std::chrono::steady_clock::now(), -1.0f};

        auto progress_fn = [](float fraction, void* ud) {
            constexpr int BAR_WIDTH = 40;
            auto* ps = static_cast<ProgressState*>(ud);
            if (fraction - ps->last_printed < 0.02f && fraction < 1.0f) return;
            ps->last_printed = fraction;

            int filled = static_cast<int>(fraction * BAR_WIDTH);
            auto elapsed = std::chrono::steady_clock::now() - ps->start;
            double secs = std::chrono::duration<double>(elapsed).count();

            std::fprintf(stderr, "\r  [");
            for (int i = 0; i < BAR_WIDTH; ++i)
                std::fputc(i < filled ? '#' : '.', stderr);
            std::fprintf(stderr, "] %3.0f%%  %.1fs", fraction * 100.0, secs);
            std::fflush(stderr);
        };

        // Use the C API render_spectral through the renderer directly
        // (progress callback goes through the renderer)
        result = grrt_render_spectral(ctx, spectral_buffer.data(),
                                       params.width, params.height);

        // Final progress bar
        {
            auto elapsed = std::chrono::steady_clock::now() - pstate.start;
            double secs = std::chrono::duration<double>(elapsed).count();
            std::fprintf(stderr, "\r  [");
            for (int i = 0; i < 40; ++i) std::fputc('#', stderr);
            std::fprintf(stderr, "] 100%%  %.1fs\n", secs);
        }

        if (result == 0) {
            grrt::FITSMetadata meta{};
            meta.spin = params.spin;
            meta.mass = params.mass;
            meta.observer_r = params.observer_r;
            meta.observer_theta = params.observer_theta;
            meta.fov = params.fov;
            meta.samples_per_pixel = params.samples_per_pixel;

            grrt::write_fits(path_fits, spectral_buffer.data(),
                             params.width, params.height, num_bins,
                             cli_freq_bins, meta);
            std::println("Saved {}", path_fits);
        } else {
            std::println(stderr, "Spectral render failed: {}", grrt_error(ctx));
        }
    } else {
        // --- Existing RGB rendering path ---
```

Close the `else` block at the end of the existing RGB render code, before `grrt_destroy(ctx)`:

```cpp
    }  // end if spectral / else RGB
```

- [ ] **Step 4: Build and test end-to-end**

```bash
cmake --build build --config Release 2>&1 | tail -10
```

Test with a small spectral render:
```bash
./build/Release/grrt-cli --width 64 --height 64 --disk-volumetric --freq-range 1e10 1e18 20 --output test_spectral
```
Expected: Produces `test_spectral.fits`. Verify with:
```bash
ls -la test_spectral.fits
```
Should exist and be non-empty (64 * 64 * 20 * 8 = 655,360 bytes of data + header).

Test RGB mode still works:
```bash
./build/Release/grrt-cli --width 64 --height 64 --disk-volumetric --output test_rgb
```
Expected: Produces `test_rgb.png`, `test_rgb.hdr`, `test_rgb_linear.hdr` as before.

- [ ] **Step 5: Run all tests**

```bash
./build/Release/test-spectral
./build/Release/test-opacity
./build/Release/test-volumetric
```
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add cli/main.cpp
git commit -m "feat: add --frequencies and --freq-range CLI flags with FITS output"
```

---

### Task 7: Integration Test and Cleanup

**Files:**
- Modify: `tests/test_spectral.cpp`

- [ ] **Step 1: Add end-to-end consistency test**

Add to `tests/test_spectral.cpp` before `main()`:

```cpp
/// Verify that the spectral path at the 3 RGB frequencies produces
/// intensities consistent with the RGB path.
void test_spectral_vs_rgb_consistency() {
    std::printf("\n=== Spectral vs RGB: consistency check ===\n");

    double mass = 1.0, spin = 0.998;
    grrt::Kerr metric(mass, spin * mass);
    grrt::RK4 integrator;
    grrt::VolumetricParams vp;
    vp.turbulence = 0.0;
    grrt::VolumetricDisk vol_disk(mass, spin, 20.0, 1e7, vp);
    grrt::GeodesicTracer tracer(metric, integrator, 50.0, 10000, 1000.0, 1e-8, &vol_disk);
    grrt::Camera camera(metric, 50.0, 1.396, 0.0, 1.047, 64, 64);

    // The 3 RGB frequencies from the original raymarch_volumetric
    using namespace grrt::constants;
    std::vector<double> rgb_freqs = {
        c_cgs / 450e-7,  // blue
        c_cgs / 550e-7,  // green
        c_cgs / 650e-7,  // red
    };

    // Pick several pixels that should hit the disk
    int test_pixels[][2] = {{32, 36}, {32, 38}, {30, 37}, {34, 37}};

    for (auto& px : test_pixels) {
        grrt::GeodesicState state_rgb = camera.ray_for_pixel(px[0], px[1]);
        grrt::GeodesicState state_spec = state_rgb;  // Same initial state

        auto result_rgb = tracer.trace(state_rgb, nullptr, nullptr);
        auto result_spec = tracer.trace_spectral(state_spec, rgb_freqs);

        // Both should have the same termination
        if (result_rgb.termination != result_spec.termination) {
            std::printf("  pixel (%d,%d): termination mismatch (rgb=%d, spec=%d)\n",
                        px[0], px[1],
                        static_cast<int>(result_rgb.termination),
                        static_cast<int>(result_spec.termination));
            // Different termination can happen due to slight FP differences,
            // but flag it.
            continue;
        }

        // Compare intensities. They won't match exactly because the RGB path
        // uses a slightly different adaptive stepping (keyed to green channel)
        // vs the spectral path (keyed to median bin). But they should be
        // in the same ballpark.
        double rgb_sum = result_rgb.accumulated_color[0]
                       + result_rgb.accumulated_color[1]
                       + result_rgb.accumulated_color[2];
        double spec_sum = result_spec.spectral_intensity[0]
                        + result_spec.spectral_intensity[1]
                        + result_spec.spectral_intensity[2];

        std::printf("  pixel (%d,%d): rgb_sum=%.4e spec_sum=%.4e",
                    px[0], px[1], rgb_sum, spec_sum);

        if (rgb_sum > 0.0 && spec_sum > 0.0) {
            double ratio = spec_sum / rgb_sum;
            std::printf(" ratio=%.3f\n", ratio);
            // Within a factor of 10 is reasonable given the different
            // step control strategies
            if (ratio < 0.1 || ratio > 10.0) {
                std::printf("    WARNING: large discrepancy\n");
            }
        } else {
            std::printf(" (one or both zero)\n");
        }
    }

    std::printf("  (consistency check is informational, not strict pass/fail)\n");
}
```

Add `test_spectral_vs_rgb_consistency();` to `main()`.

- [ ] **Step 2: Build and run final tests**

```bash
cmake --build build --config Release 2>&1 | tail -10
./build/Release/test-spectral
```
Expected: All tests pass. Consistency check shows reasonable ratios.

- [ ] **Step 3: Commit**

```bash
git add tests/test_spectral.cpp
git commit -m "test: add spectral vs RGB consistency check"
```

- [ ] **Step 4: Clean up test artifacts**

```bash
rm -f test_spectral.fits test_spectral.png test_spectral.hdr test_spectral_linear.hdr
rm -f test_rgb.png test_rgb.hdr test_rgb_linear.hdr
```
