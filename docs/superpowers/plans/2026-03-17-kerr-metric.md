# Kerr Metric Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Kerr (spinning) black hole metric with `a = 0.998M`, producing a D-shaped shadow with frame-dragging effects, and update the accretion disk for Kerr circular orbits.

**Architecture:** Add Kerr metric implementing existing `Metric` interface. Add `isco_radius()` to the interface. Add full 4×4 matrix inverse to `Matrix4`. Update `AccretionDisk` to accept spin parameter for Kerr orbit formulas. Wire metric selection in API based on `metric_type` param.

**Tech Stack:** C++23, CMake, MSVC 2022

**Spec:** `docs/superpowers/specs/2026-03-17-kerr-metric-design.md`

**Build command:** `build.bat`

**Run command:** `build/Release/grrt-cli.exe` (produces `output.png`)

---

## Chunk 1: Matrix4 Inverse + Metric Interface

### Task 1: Add full 4×4 inverse to Matrix4

**Files:**
- Modify: `include/grrt/math/matrix4.h`

- [ ] **Step 1: Add `inverse()` method**

Add this method to the `Matrix4` struct, after `inverse_diagonal()`. This uses the block structure of Boyer-Lindquist metrics: `(t,φ)` is a 2×2 block, `(r,θ)` are diagonal. This is more efficient and numerically stable than full cofactor expansion, and works for both Schwarzschild (where `g_tφ = 0`) and Kerr.

```cpp
    // General inverse exploiting block-diagonal structure of BL metrics:
    // (t,φ) form a 2×2 block; r and θ are diagonal.
    // Also works for fully diagonal matrices.
    Matrix4 inverse() const {
        Matrix4 inv;

        // (t,φ) 2×2 block inverse
        double det_tf = m[0][0] * m[3][3] - m[0][3] * m[3][0];
        inv.m[0][0] = m[3][3] / det_tf;
        inv.m[3][3] = m[0][0] / det_tf;
        inv.m[0][3] = -m[0][3] / det_tf;
        inv.m[3][0] = -m[3][0] / det_tf;

        // Diagonal entries
        inv.m[1][1] = 1.0 / m[1][1];
        inv.m[2][2] = 1.0 / m[2][2];

        return inv;
    }
```

- [ ] **Step 2: Build to verify**

Run: `build.bat`
Expected: Compiles (method not yet called).

### Task 2: Add `isco_radius()` to Metric interface

**Files:**
- Modify: `include/grrt/spacetime/metric.h`
- Modify: `include/grrt/spacetime/schwarzschild.h`
- Modify: `src/schwarzschild.cpp`

- [ ] **Step 1: Update `metric.h`**

Add to the `Metric` class, after `horizon_radius()`:

```cpp
    // Innermost stable circular orbit radius
    virtual double isco_radius() const = 0;
```

- [ ] **Step 2: Update `schwarzschild.h`**

Add declaration after `horizon_radius()`:

```cpp
    double isco_radius() const override;
```

- [ ] **Step 3: Update `src/schwarzschild.cpp`**

Add at end of the namespace, before the closing brace:

```cpp
double Schwarzschild::isco_radius() const {
    return 6.0 * mass_;
}
```

- [ ] **Step 4: Build to verify**

Run: `build.bat`
Expected: Compiles and links.

- [ ] **Step 5: Commit**

```bash
git add include/grrt/math/matrix4.h include/grrt/spacetime/metric.h include/grrt/spacetime/schwarzschild.h src/schwarzschild.cpp
git commit -m "feat: add Matrix4 block inverse and isco_radius() to Metric interface"
```

---

## Chunk 2: Kerr Metric

### Task 3: Kerr Metric Implementation

**Files:**
- Create: `include/grrt/spacetime/kerr.h`
- Create: `src/kerr.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Create `kerr.h`**

```cpp
#ifndef GRRT_KERR_H
#define GRRT_KERR_H

#include "grrt/spacetime/metric.h"

namespace grrt {

class Kerr : public Metric {
public:
    Kerr(double mass, double spin);

    Matrix4 g_lower(const Vec4& x) const override;
    Matrix4 g_upper(const Vec4& x) const override;
    double horizon_radius() const override;
    double isco_radius() const override;

    double mass() const { return mass_; }
    double spin() const { return spin_; }

private:
    double mass_;  // M
    double spin_;  // a, with |a| < M

    // Auxiliary quantities
    double sigma(double r, double theta) const;
    double delta(double r) const;
};

} // namespace grrt

#endif
```

- [ ] **Step 2: Create `src/kerr.cpp`**

```cpp
#include "grrt/spacetime/kerr.h"
#include <cmath>
#include <algorithm>

namespace grrt {

Kerr::Kerr(double mass, double spin) : mass_(mass), spin_(spin) {}

double Kerr::sigma(double r, double theta) const {
    double a = spin_;
    double cos_t = std::cos(theta);
    return r * r + a * a * cos_t * cos_t;
}

double Kerr::delta(double r) const {
    return r * r - 2.0 * mass_ * r + spin_ * spin_;
}

Matrix4 Kerr::g_lower(const Vec4& x) const {
    double r = x[1];
    double theta = x[2];
    double sin_t = std::sin(theta);
    double sin_t_safe = std::max(std::abs(sin_t), 1e-10);
    if (sin_t < 0.0) sin_t_safe = -sin_t_safe;
    double sin2 = sin_t_safe * sin_t_safe;

    double M = mass_;
    double a = spin_;
    double S = std::max(sigma(r, theta), 1e-20);
    double D = delta(r);

    Matrix4 g;
    g.m[0][0] = -(1.0 - 2.0 * M * r / S);                          // g_tt
    g.m[0][3] = -2.0 * M * a * r * sin2 / S;                        // g_tφ
    g.m[3][0] = g.m[0][3];                                            // g_φt = g_tφ
    g.m[1][1] = S / D;                                                // g_rr
    g.m[2][2] = S;                                                     // g_θθ
    g.m[3][3] = (r * r + a * a + 2.0 * M * a * a * r * sin2 / S) * sin2; // g_φφ

    return g;
}

Matrix4 Kerr::g_upper(const Vec4& x) const {
    return g_lower(x).inverse();
}

double Kerr::horizon_radius() const {
    double M = mass_;
    double a = spin_;
    return M + std::sqrt(M * M - a * a);
}

double Kerr::isco_radius() const {
    double M = mass_;
    double a = spin_;
    double a_star = a / M;  // Dimensionless spin

    // Prograde ISCO (Bardeen, Press & Teukolsky 1972)
    double Z1 = 1.0 + std::cbrt(1.0 - a_star * a_star)
                     * (std::cbrt(1.0 + a_star) + std::cbrt(1.0 - a_star));
    double Z2 = std::sqrt(3.0 * a_star * a_star + Z1 * Z1);

    return M * (3.0 + Z2 - std::sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)));
}

} // namespace grrt
```

- [ ] **Step 3: Add `src/kerr.cpp` to CMakeLists.txt**

Add `src/kerr.cpp` to the `add_library(grrt SHARED ...)` block.

- [ ] **Step 4: Build to verify**

Run: `build.bat`
Expected: Compiles and links.

- [ ] **Step 5: Commit**

```bash
git add include/grrt/spacetime/kerr.h src/kerr.cpp CMakeLists.txt
git commit -m "feat: add Kerr metric with Boyer-Lindquist coordinates"
```

---

## Chunk 3: Update AccretionDisk for Kerr

### Task 4: Make AccretionDisk spin-aware

**Files:**
- Modify: `include/grrt/scene/accretion_disk.h`
- Modify: `src/accretion_disk.cpp`

- [ ] **Step 1: Update `accretion_disk.h`**

Replace entire file:

```cpp
#ifndef GRRT_ACCRETION_DISK_H
#define GRRT_ACCRETION_DISK_H

#include "grrt/math/vec3.h"
#include "grrt/math/vec4.h"
#include "grrt/color/spectrum.h"
#include <vector>

namespace grrt {

class AccretionDisk {
public:
    AccretionDisk(double mass, double spin, double r_isco,
                  double r_outer, double peak_temperature, int flux_lut_size = 500);

    double r_inner() const { return r_inner_; }
    double r_outer() const { return r_outer_; }

    double temperature(double r) const;

    Vec3 emission(double r_cross, const Vec4& p_cross,
                  double observer_r, const SpectrumLUT& spectrum) const;

private:
    double mass_;
    double spin_;     // a (Kerr spin parameter, 0 for Schwarzschild)
    double r_inner_;  // ISCO
    double r_outer_;
    double peak_temperature_;

    std::vector<double> flux_lut_;
    double flux_max_;
    double flux_r_min_;
    double flux_r_max_;
    int flux_lut_size_;

    // Kerr-aware circular orbit quantities
    // ω = √(M/r³), used as building block
    double omega_kepler(double r) const;
    double Omega(double r) const;    // Ω = ω / (1 + a·ω)
    double E_circ(double r) const;   // Specific energy
    double L_circ(double r) const;   // Specific angular momentum

    void build_flux_lut();
    double flux(double r) const;
    double redshift(double r_cross, const Vec4& p_cross, double observer_r) const;
};

} // namespace grrt

#endif
```

- [ ] **Step 2: Update `src/accretion_disk.cpp`**

Replace entire file:

```cpp
#include "grrt/scene/accretion_disk.h"
#include <cmath>
#include <algorithm>
#include <numbers>

namespace grrt {

AccretionDisk::AccretionDisk(double mass, double spin, double r_isco,
                             double r_outer, double peak_temperature, int flux_lut_size)
    : mass_(mass), spin_(spin),
      r_inner_(r_isco),
      r_outer_(r_outer),
      peak_temperature_(peak_temperature),
      flux_lut_size_(flux_lut_size) {
    build_flux_lut();
}

double AccretionDisk::omega_kepler(double r) const {
    return std::sqrt(mass_ / (r * r * r));
}

double AccretionDisk::Omega(double r) const {
    double w = omega_kepler(r);
    return w / (1.0 + spin_ * w);
}

double AccretionDisk::E_circ(double r) const {
    double M = mass_;
    double a = spin_;
    double w = omega_kepler(r);
    double aw = a * w;  // a·√(M/r³)
    return (1.0 - 2.0 * M / r + aw) / std::sqrt(1.0 - 3.0 * M / r + 2.0 * aw);
}

double AccretionDisk::L_circ(double r) const {
    double M = mass_;
    double a = spin_;
    double w = omega_kepler(r);
    double aw = a * w;
    return std::sqrt(M * r) * (1.0 - 2.0 * aw + a * a / (r * r))
           / std::sqrt(1.0 - 3.0 * M / r + 2.0 * aw);
}

void AccretionDisk::build_flux_lut() {
    flux_r_min_ = r_inner_;
    flux_r_max_ = r_outer_;
    flux_lut_.resize(flux_lut_size_);

    double E_isco = E_circ(r_inner_);
    double L_isco = L_circ(r_inner_);

    flux_max_ = 0.0;
    double I_cumulative = 0.0;
    constexpr double fd_eps = 1e-6;
    double prev_integrand = 0.0;

    for (int i = 0; i < flux_lut_size_; ++i) {
        double r = r_inner_ + (r_outer_ - r_inner_) * i / (flux_lut_size_ - 1);

        if (i == 0) {
            flux_lut_[i] = 0.0;
            continue;
        }

        double E_prime = (E_circ(r + fd_eps) - E_circ(r - fd_eps)) / (2.0 * fd_eps);
        double L_prime = (L_circ(r + fd_eps) - L_circ(r - fd_eps)) / (2.0 * fd_eps);

        double integrand = (E_circ(r) - E_isco) * L_prime - (L_circ(r) - L_isco) * E_prime;

        double dr = (r_outer_ - r_inner_) / (flux_lut_size_ - 1);
        I_cumulative += 0.5 * (prev_integrand + integrand) * dr;
        prev_integrand = integrand;

        double Om = Omega(r);
        double E_r = E_circ(r);
        double L_r = L_circ(r);
        double dOmega_dr = (Omega(r + fd_eps) - Omega(r - fd_eps)) / (2.0 * fd_eps);

        double denominator = E_r - Om * L_r;
        if (std::abs(denominator) < 1e-20) {
            flux_lut_[i] = 0.0;
            continue;
        }

        double F = (3.0 * mass_ / (8.0 * std::numbers::pi * r * r * r))
                   * (1.0 / denominator) * (-dOmega_dr) * I_cumulative;

        flux_lut_[i] = std::max(F, 0.0);
        if (flux_lut_[i] > flux_max_) {
            flux_max_ = flux_lut_[i];
        }
    }
}

double AccretionDisk::flux(double r) const {
    if (r <= r_inner_ || r >= r_outer_ || flux_max_ <= 0.0) return 0.0;

    double frac = (r - flux_r_min_) / (flux_r_max_ - flux_r_min_) * (flux_lut_size_ - 1);
    int idx = static_cast<int>(frac);
    double t = frac - idx;

    if (idx >= flux_lut_size_ - 1) return flux_lut_[flux_lut_size_ - 1];
    return flux_lut_[idx] * (1.0 - t) + flux_lut_[idx + 1] * t;
}

double AccretionDisk::temperature(double r) const {
    double F = flux(r);
    if (F <= 0.0 || flux_max_ <= 0.0) return 0.0;
    return peak_temperature_ * std::pow(F / flux_max_, 0.25);
}

double AccretionDisk::redshift(double r_cross, const Vec4& p, double observer_r) const {
    double M = mass_;
    double a = spin_;

    // Observer: static at r_obs
    double u_t_obs = 1.0 / std::sqrt(1.0 - 2.0 * M / observer_r);
    double pu_obs = p[0] * u_t_obs;

    // Emitter: circular orbit at r_cross
    // u^t = 1 / √(1 - 3M/r + 2a·ω)
    double w = omega_kepler(r_cross);
    double aw = a * w;
    double u_t_emit = 1.0 / std::sqrt(1.0 - 3.0 * M / r_cross + 2.0 * aw);
    double u_phi_emit = Omega(r_cross) * u_t_emit;
    double pu_emit = p[0] * u_t_emit + p[3] * u_phi_emit;

    if (std::abs(pu_obs) < 1e-30) return 1.0;
    return pu_emit / pu_obs;
}

Vec3 AccretionDisk::emission(double r_cross, const Vec4& p_cross,
                             double observer_r, const SpectrumLUT& spectrum) const {
    double T = temperature(r_cross);
    if (T <= 0.0) return {};

    double g = redshift(r_cross, p_cross, observer_r);

    double T_obs = g * T;
    if (T_obs < 100.0) return {};

    Vec3 color = spectrum.temperature_to_color(T_obs);
    double g3 = g * g * g;
    return color * g3;
}

} // namespace grrt
```

- [ ] **Step 3: Build to verify**

This will fail because `api.cpp` still uses the old `AccretionDisk` constructor. That's expected — next task fixes it.

- [ ] **Step 4: Commit**

```bash
git add include/grrt/scene/accretion_disk.h src/accretion_disk.cpp
git commit -m "feat: make AccretionDisk spin-aware with Kerr orbit formulas"
```

---

## Chunk 4: Wire into API + CLI

### Task 5: Update API for Kerr metric selection

**Files:**
- Modify: `src/api.cpp`
- Modify: `cli/main.cpp`

- [ ] **Step 1: Update `src/api.cpp`**

Add Kerr include at top (after schwarzschild include):
```cpp
#include "grrt/spacetime/kerr.h"
```

Replace the metric construction block (lines 37-38):
```cpp
    ctx->metric = std::make_unique<grrt::Schwarzschild>(mass);
```

With:
```cpp
    double spin_a = 0.0;
    if (params->metric_type == GRRT_METRIC_KERR) {
        double spin_param = params->spin > 0.0 ? params->spin : 0.998;
        spin_a = spin_param * mass;  // a = spin × M
        ctx->metric = std::make_unique<grrt::Kerr>(mass, spin_a);
    } else {
        ctx->metric = std::make_unique<grrt::Schwarzschild>(mass);
    }
```

Replace the disk construction block (lines 51-56):
```cpp
    if (params->disk_enabled) {
        double disk_temp = params->disk_temperature > 0.0 ? params->disk_temperature : 1e7;
        double disk_outer = params->disk_outer > 0.0 ? params->disk_outer : 20.0;
        ctx->disk = std::make_unique<grrt::AccretionDisk>(
            mass, params->disk_inner, disk_outer, disk_temp);
    }
```

With:
```cpp
    if (params->disk_enabled) {
        double disk_temp = params->disk_temperature > 0.0 ? params->disk_temperature : 1e7;
        double disk_outer = params->disk_outer > 0.0 ? params->disk_outer : 20.0;
        double isco = ctx->metric->isco_radius();
        double r_inner = params->disk_inner > 0.0 ? params->disk_inner : isco;
        ctx->disk = std::make_unique<grrt::AccretionDisk>(
            mass, spin_a, r_inner, disk_outer, disk_temp);
    }
```

Update the log message (line 69) to show metric type and spin:
```cpp
    const char* metric_name = params->metric_type == GRRT_METRIC_KERR ? "kerr" : "schwarzschild";
    std::println("grrt: created context ({}x{}, {}, M={}, a={}, r_obs={}, disk={}, stars={})",
                 params->width, params->height, metric_name, mass, spin_a, observer_r,
                 params->disk_enabled ? "on" : "off",
                 params->background_type == GRRT_BG_STARS ? "on" : "off");
```

- [ ] **Step 2: Update `cli/main.cpp`**

Change metric type back to Kerr and set spin:
```cpp
    params.metric_type = GRRT_METRIC_KERR;
```

The spin is already in `GRRTParams` from the hello-world setup (`params.spin = 0.998`), so no change needed there.

- [ ] **Step 3: Build**

Run: `build.bat`
Expected: Compiles and links with no errors.

- [ ] **Step 4: Run and validate**

Run: `build/Release/grrt-cli.exe`
Expected:
- `output.png` shows a **D-shaped shadow** (flattened on one side)
- The disk's inner edge should be much closer to the shadow than before (ISCO ≈ 1.24M vs 6M)
- Doppler beaming should be more pronounced
- Stars should show asymmetric lensing patterns

- [ ] **Step 5: Commit**

```bash
git add src/api.cpp cli/main.cpp
git commit -m "feat: wire Kerr metric into API with spin parameter selection"
```

### Task 6: Debug and Tune (if needed)

- [ ] **Step 1: Verify ISCO**

Print `ctx->metric->isco_radius()` in `grrt_create()`. For `a = 0.998, M = 1` it should be approximately `1.24`.

- [ ] **Step 2: If shadow looks circular (not D-shaped)**

The Kerr metric `g_upper()` may have an issue. Print `g_lower` and `g_upper` at a test point and verify `g_lower × g_upper ≈ identity`. Use `(t=0, r=10, θ=π/2, φ=0)`.

- [ ] **Step 3: If disk looks wrong or crashes**

Check that `E_circ()` and `L_circ()` return reasonable values at `r = r_isco`. For Kerr `a = 0.998`, `E_circ(r_isco)` should be less than 1 (bound orbit).

- [ ] **Step 4: If render is very slow**

Near-extremal Kerr has the horizon very close to `r = M`. The step size `0.005 * r` may need to be smaller. Try `0.002 * r` if rays near the horizon misbehave.

- [ ] **Step 5: Commit any fixes**

```bash
git add -u
git commit -m "fix: tune Kerr metric rendering"
```
