# Accretion Disk + Color Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a glowing accretion disk with Page-Thorne temperature profile, blackbody colors, gravitational redshift, Doppler beaming, star field background, and tone mapping.

**Architecture:** Bottom-up: Vec3 → SpectrumLUT → ToneMapper → AccretionDisk → CelestialSphere → modify tracer for disk intersection → modify renderer for color → wire into API.

**Tech Stack:** C++23, CMake, MSVC 2022, OpenMP

**Spec:** `docs/superpowers/specs/2026-03-17-accretion-disk-design.md`

**Build command:** `build.bat` or `"/c/Program Files/CMake/bin/cmake.exe" -B build -G "Visual Studio 17 2022" -S . && "/c/Program Files/CMake/bin/cmake.exe" --build build --config Release`

**Run command:** `build/Release/grrt-cli.exe` (produces `output.png`)

---

## Chunk 1: Color Infrastructure (Vec3, Spectrum LUT, Tone Mapper)

### Task 1: Vec3

**Files:**
- Create: `include/grrt/math/vec3.h`

- [ ] **Step 1: Create `vec3.h`**

```cpp
#ifndef GRRT_VEC3_H
#define GRRT_VEC3_H

#include <cmath>

namespace grrt {

struct Vec3 {
    double data[3]{};

    double& operator[](int i) { return data[i]; }
    const double& operator[](int i) const { return data[i]; }

    double r() const { return data[0]; }
    double g() const { return data[1]; }
    double b() const { return data[2]; }

    Vec3 operator+(const Vec3& o) const {
        return {{data[0]+o[0], data[1]+o[1], data[2]+o[2]}};
    }

    Vec3 operator-(const Vec3& o) const {
        return {{data[0]-o[0], data[1]-o[1], data[2]-o[2]}};
    }

    Vec3 operator*(double s) const {
        return {{data[0]*s, data[1]*s, data[2]*s}};
    }

    // Component-wise multiply (color modulation)
    Vec3 operator*(const Vec3& o) const {
        return {{data[0]*o[0], data[1]*o[1], data[2]*o[2]}};
    }

    Vec3& operator+=(const Vec3& o) {
        data[0] += o[0]; data[1] += o[1]; data[2] += o[2];
        return *this;
    }

    double max_component() const {
        return std::fmax(data[0], std::fmax(data[1], data[2]));
    }
};

inline Vec3 operator*(double s, const Vec3& v) {
    return v * s;
}

} // namespace grrt

#endif
```

- [ ] **Step 2: Build to verify**

Run: `build.bat`
Expected: Compiles (header not yet used).

### Task 2: Spectrum LUT

**Files:**
- Create: `include/grrt/color/spectrum.h`
- Create: `src/spectrum.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Create `spectrum.h`**

```cpp
#ifndef GRRT_SPECTRUM_H
#define GRRT_SPECTRUM_H

#include "grrt/math/vec3.h"
#include <vector>

namespace grrt {

class SpectrumLUT {
public:
    // Build the LUT at construction time
    SpectrumLUT(double t_min = 1000.0, double t_max = 100000.0, int num_entries = 1000);

    // Look up color (chromaticity) and relative luminosity for a given temperature
    // Returns color scaled by luminosity — ready to multiply by g³
    Vec3 temperature_to_color(double temperature) const;

private:
    double t_min_;
    double t_max_;
    int num_entries_;
    std::vector<Vec3> color_lut_;       // Normalized chromaticity
    std::vector<double> luminosity_lut_; // Relative brightness (σT⁴ normalized)

    // Planck spectral radiance B(λ, T) in wavelength space
    static double planck(double wavelength_m, double temperature);

    // Compute XYZ -> linear sRGB for a blackbody at temperature T
    Vec3 blackbody_to_rgb(double temperature) const;
};

} // namespace grrt

#endif
```

- [ ] **Step 2: Create `src/spectrum.cpp`**

This file contains the CIE color matching function data and the LUT construction.

```cpp
#include "grrt/color/spectrum.h"
#include <cmath>
#include <algorithm>
#include <numbers>

namespace grrt {

// Physical constants
static constexpr double h_planck = 6.62607015e-34;  // J·s
static constexpr double c_light = 2.99792458e8;     // m/s
static constexpr double k_boltz = 1.380649e-23;     // J/K
static constexpr double sigma_sb = 5.670374419e-8;  // W/(m²·K⁴)

// CIE 1931 2° observer color matching functions
// 81 entries from 380nm to 780nm in 5nm steps
// Each row: {x_bar, y_bar, z_bar}
static constexpr double cie_data[][3] = {
    {0.0014,0.0000,0.0065}, {0.0022,0.0001,0.0105}, {0.0042,0.0001,0.0201},
    {0.0076,0.0002,0.0362}, {0.0143,0.0004,0.0679}, {0.0232,0.0006,0.1102},
    {0.0435,0.0012,0.2074}, {0.0776,0.0022,0.3713}, {0.1344,0.0040,0.6456},
    {0.2148,0.0073,1.0391}, {0.2839,0.0116,1.3856}, {0.3285,0.0168,1.6230},
    {0.3483,0.0230,1.7471}, {0.3481,0.0298,1.7826}, {0.3362,0.0380,1.7721},
    {0.3187,0.0480,1.7441}, {0.2908,0.0600,1.6692}, {0.2511,0.0739,1.5281},
    {0.1954,0.0910,1.2876}, {0.1421,0.1126,1.0419}, {0.0956,0.1390,0.8130},
    {0.0580,0.1693,0.6162}, {0.0320,0.2080,0.4652}, {0.0147,0.2586,0.3533},
    {0.0049,0.3230,0.2720}, {0.0024,0.4073,0.2123}, {0.0093,0.5030,0.1582},
    {0.0291,0.6082,0.1117}, {0.0633,0.7100,0.0782}, {0.1096,0.7932,0.0573},
    {0.1655,0.8620,0.0422}, {0.2257,0.9149,0.0298}, {0.2904,0.9540,0.0203},
    {0.3597,0.9803,0.0134}, {0.4334,0.9950,0.0087}, {0.5121,1.0002,0.0057},
    {0.5945,0.9950,0.0039}, {0.6784,0.9786,0.0027}, {0.7621,0.9520,0.0021},
    {0.8425,0.9154,0.0018}, {0.9163,0.8700,0.0017}, {0.9786,0.8163,0.0014},
    {1.0263,0.7570,0.0011}, {1.0567,0.6949,0.0010}, {1.0622,0.6310,0.0008},
    {1.0456,0.5668,0.0006}, {1.0026,0.5030,0.0003}, {0.9384,0.4412,0.0002},
    {0.8544,0.3810,0.0002}, {0.7514,0.3210,0.0001}, {0.6424,0.2650,0.0000},
    {0.5419,0.2170,0.0000}, {0.4479,0.1750,0.0000}, {0.3608,0.1382,0.0000},
    {0.2835,0.1070,0.0000}, {0.2187,0.0816,0.0000}, {0.1649,0.0610,0.0000},
    {0.1212,0.0446,0.0000}, {0.0874,0.0320,0.0000}, {0.0636,0.0232,0.0000},
    {0.0468,0.0170,0.0000}, {0.0329,0.0119,0.0000}, {0.0227,0.0082,0.0000},
    {0.0158,0.0057,0.0000}, {0.0114,0.0041,0.0000}, {0.0081,0.0029,0.0000},
    {0.0058,0.0021,0.0000}, {0.0041,0.0015,0.0000}, {0.0029,0.0010,0.0000},
    {0.0020,0.0007,0.0000}, {0.0014,0.0005,0.0000}, {0.0010,0.0004,0.0000},
    {0.0007,0.0002,0.0000}, {0.0005,0.0002,0.0000}, {0.0003,0.0001,0.0000},
    {0.0002,0.0001,0.0000}, {0.0002,0.0001,0.0000}, {0.0001,0.0000,0.0000},
    {0.0001,0.0000,0.0000}, {0.0001,0.0000,0.0000}, {0.0000,0.0000,0.0000},
};
static constexpr int cie_count = 81;
static constexpr double cie_lambda_min = 380e-9; // 380 nm in meters
static constexpr double cie_lambda_step = 5e-9;  // 5 nm in meters

double SpectrumLUT::planck(double lambda, double T) {
    double x = h_planck * c_light / (lambda * k_boltz * T);
    if (x > 500.0) return 0.0; // Prevent overflow in exp
    return (2.0 * h_planck * c_light * c_light) /
           (lambda * lambda * lambda * lambda * lambda * (std::exp(x) - 1.0));
}

Vec3 SpectrumLUT::blackbody_to_rgb(double T) const {
    // Integrate Planck × CIE color matching functions → XYZ
    double X = 0.0, Y = 0.0, Z = 0.0;
    for (int i = 0; i < cie_count; ++i) {
        double lambda = cie_lambda_min + i * cie_lambda_step;
        double B = planck(lambda, T);
        X += B * cie_data[i][0] * cie_lambda_step;
        Y += B * cie_data[i][1] * cie_lambda_step;
        Z += B * cie_data[i][2] * cie_lambda_step;
    }

    // XYZ → linear sRGB (IEC 61966-2-1)
    double R =  3.2406 * X - 1.5372 * Y - 0.4986 * Z;
    double G = -0.9689 * X + 1.8758 * Y + 0.0415 * Z;
    double B_val =  0.0557 * X - 0.2040 * Y + 1.0570 * Z;

    // Clamp negatives (out of gamut)
    R = std::max(R, 0.0);
    G = std::max(G, 0.0);
    B_val = std::max(B_val, 0.0);

    return {{R, G, B_val}};
}

SpectrumLUT::SpectrumLUT(double t_min, double t_max, int num_entries)
    : t_min_(t_min), t_max_(t_max), num_entries_(num_entries),
      color_lut_(num_entries), luminosity_lut_(num_entries) {

    // Find max luminosity for normalization
    double lum_max = sigma_sb * t_max * t_max * t_max * t_max;

    for (int i = 0; i < num_entries; ++i) {
        double t = t_min + (t_max - t_min) * i / (num_entries - 1);

        // Chromaticity (normalized so max component = 1)
        Vec3 rgb = blackbody_to_rgb(t);
        double max_c = rgb.max_component();
        if (max_c > 0.0) {
            rgb = rgb * (1.0 / max_c);
        }
        color_lut_[i] = rgb;

        // Relative luminosity via Stefan-Boltzmann
        luminosity_lut_[i] = (sigma_sb * t * t * t * t) / lum_max;
    }
}

Vec3 SpectrumLUT::temperature_to_color(double temperature) const {
    // Clamp to LUT range
    temperature = std::clamp(temperature, t_min_, t_max_);

    // Find position in LUT
    double frac = (temperature - t_min_) / (t_max_ - t_min_) * (num_entries_ - 1);
    int idx = static_cast<int>(frac);
    double t = frac - idx;

    if (idx >= num_entries_ - 1) {
        return color_lut_[num_entries_ - 1] * luminosity_lut_[num_entries_ - 1];
    }

    // Linear interpolation
    Vec3 color = color_lut_[idx] * (1.0 - t) + color_lut_[idx + 1] * t;
    double lum = luminosity_lut_[idx] * (1.0 - t) + luminosity_lut_[idx + 1] * t;

    return color * lum;
}

} // namespace grrt
```

- [ ] **Step 3: Add `src/spectrum.cpp` to CMakeLists.txt**

Add `src/spectrum.cpp` to the `add_library(grrt SHARED ...)` block.

- [ ] **Step 4: Build to verify**

Run: `build.bat`
Expected: Compiles and links.

- [ ] **Step 5: Commit**

```bash
git add include/grrt/math/vec3.h include/grrt/color/spectrum.h src/spectrum.cpp CMakeLists.txt
git commit -m "feat: add Vec3, blackbody spectrum LUT with CIE color matching"
```

### Task 3: Tone Mapper

**Files:**
- Create: `include/grrt/render/tonemapper.h`
- Create: `src/tonemapper.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Create `tonemapper.h`**

```cpp
#ifndef GRRT_TONEMAPPER_H
#define GRRT_TONEMAPPER_H

#include "grrt/math/vec3.h"

namespace grrt {

class ToneMapper {
public:
    // Apply Reinhard tone mapping + sRGB gamma
    Vec3 apply(const Vec3& hdr) const;
};

} // namespace grrt

#endif
```

- [ ] **Step 2: Create `src/tonemapper.cpp`**

```cpp
#include "grrt/render/tonemapper.h"
#include <cmath>

namespace grrt {

Vec3 ToneMapper::apply(const Vec3& hdr) const {
    // Per-channel Reinhard: c / (1 + c)
    auto reinhard = [](double c) { return c / (1.0 + c); };

    // sRGB gamma (simplified): v^(1/2.2)
    auto gamma = [](double v) { return std::pow(v, 1.0 / 2.2); };

    return {{
        gamma(reinhard(hdr[0])),
        gamma(reinhard(hdr[1])),
        gamma(reinhard(hdr[2]))
    }};
}

} // namespace grrt
```

- [ ] **Step 3: Add `src/tonemapper.cpp` to CMakeLists.txt**

- [ ] **Step 4: Build to verify**

- [ ] **Step 5: Commit**

```bash
git add include/grrt/render/tonemapper.h src/tonemapper.cpp CMakeLists.txt
git commit -m "feat: add Reinhard tone mapper with sRGB gamma"
```

---

## Chunk 2: Accretion Disk

### Task 4: Accretion Disk

**Files:**
- Create: `include/grrt/scene/accretion_disk.h`
- Create: `src/accretion_disk.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Create `accretion_disk.h`**

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
    AccretionDisk(double mass, double r_inner, double r_outer,
                  double peak_temperature, int flux_lut_size = 500);

    double r_inner() const { return r_inner_; }
    double r_outer() const { return r_outer_; }

    // Temperature at radius r (Page-Thorne profile)
    double temperature(double r) const;

    // Compute disk emission color for a ray hitting the disk at r_cross
    // with covariant momentum p_cross. Includes redshift and intensity scaling.
    // observer_r is the static observer's radial coordinate.
    Vec3 emission(double r_cross, const Vec4& p_cross,
                  double observer_r, const SpectrumLUT& spectrum) const;

private:
    double mass_;
    double r_inner_;    // ISCO
    double r_outer_;
    double peak_temperature_;

    // Precomputed flux lookup table
    std::vector<double> flux_lut_;
    double flux_max_;
    double flux_r_min_;
    double flux_r_max_;
    int flux_lut_size_;

    // Specific energy and angular momentum for circular Schwarzschild orbits
    double E_circ(double r) const;
    double L_circ(double r) const;
    double Omega(double r) const;

    // Build the Page-Thorne flux LUT
    void build_flux_lut();

    // Look up flux at radius r
    double flux(double r) const;

    // Compute redshift factor g = (p_μ u^μ)_emit / (p_μ u^μ)_obs
    double redshift(double r_cross, const Vec4& p_cross, double observer_r) const;
};

} // namespace grrt

#endif
```

- [ ] **Step 2: Create `src/accretion_disk.cpp`**

```cpp
#include "grrt/scene/accretion_disk.h"
#include <cmath>
#include <algorithm>
#include <numbers>

namespace grrt {

AccretionDisk::AccretionDisk(double mass, double r_inner, double r_outer,
                             double peak_temperature, int flux_lut_size)
    : mass_(mass),
      r_inner_(r_inner > 0.0 ? r_inner : 6.0 * mass),  // Default: ISCO = 6M
      r_outer_(r_outer),
      peak_temperature_(peak_temperature),
      flux_lut_size_(flux_lut_size) {
    build_flux_lut();
}

double AccretionDisk::E_circ(double r) const {
    double M = mass_;
    return (1.0 - 2.0 * M / r) / std::sqrt(1.0 - 3.0 * M / r);
}

double AccretionDisk::L_circ(double r) const {
    double M = mass_;
    return std::sqrt(M * r) / std::sqrt(1.0 - 3.0 * M / r);
}

double AccretionDisk::Omega(double r) const {
    return std::sqrt(mass_ / (r * r * r));
}

void AccretionDisk::build_flux_lut() {
    flux_r_min_ = r_inner_;
    flux_r_max_ = r_outer_;
    flux_lut_.resize(flux_lut_size_);

    double E_isco = E_circ(r_inner_);
    double L_isco = L_circ(r_inner_);

    // Precompute the integral I(r) and flux F(r) at each LUT point
    // I(r) = ∫_{r_isco}^{r} [(E(r') - E_isco) L'(r') - (L(r') - L_isco) E'(r')] dr'
    // Use composite trapezoidal rule for the cumulative integral

    flux_max_ = 0.0;
    double I_cumulative = 0.0;
    constexpr double fd_eps = 1e-6;

    double prev_integrand = 0.0; // At r_isco, integrand is 0

    for (int i = 0; i < flux_lut_size_; ++i) {
        double r = r_inner_ + (r_outer_ - r_inner_) * i / (flux_lut_size_ - 1);

        if (i == 0) {
            // At r_isco: I = 0, F = 0
            flux_lut_[i] = 0.0;
            continue;
        }

        // Derivatives by central finite differences
        double E_prime = (E_circ(r + fd_eps) - E_circ(r - fd_eps)) / (2.0 * fd_eps);
        double L_prime = (L_circ(r + fd_eps) - L_circ(r - fd_eps)) / (2.0 * fd_eps);

        // Integrand at this r
        double integrand = (E_circ(r) - E_isco) * L_prime - (L_circ(r) - L_isco) * E_prime;

        // Trapezoidal step
        double dr = (r_outer_ - r_inner_) / (flux_lut_size_ - 1);
        I_cumulative += 0.5 * (prev_integrand + integrand) * dr;
        prev_integrand = integrand;

        // Flux: F(r) = (3M / (8π r³)) × 1/(E - ΩL) × (-dΩ/dr) × I(r)
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

    // Observer: static at r_obs
    // (p_μ u^μ)_obs = p_t × u^t_obs = p_t / √(1 - 2M/r_obs)
    double u_t_obs = 1.0 / std::sqrt(1.0 - 2.0 * M / observer_r);
    double pu_obs = p[0] * u_t_obs;  // p_t × u^t

    // Emitter: circular orbit at r_cross
    // u^t_emit = 1/√(1 - 3M/r), u^φ_emit = Ω × u^t
    double u_t_emit = 1.0 / std::sqrt(1.0 - 3.0 * M / r_cross);
    double u_phi_emit = Omega(r_cross) * u_t_emit;
    double pu_emit = p[0] * u_t_emit + p[3] * u_phi_emit;  // p_t u^t + p_φ u^φ

    // g = (p_μ u^μ)_emit / (p_μ u^μ)_obs
    // Both are negative, ratio is positive
    if (std::abs(pu_obs) < 1e-30) return 1.0;
    return pu_emit / pu_obs;
}

Vec3 AccretionDisk::emission(double r_cross, const Vec4& p_cross,
                             double observer_r, const SpectrumLUT& spectrum) const {
    double T = temperature(r_cross);
    if (T <= 0.0) return {};

    double g = redshift(r_cross, p_cross, observer_r);

    // Observed temperature shifted by redshift
    double T_obs = g * T;
    if (T_obs < 100.0) return {};  // Too cold to see

    // Get color from spectrum LUT (includes luminosity)
    Vec3 color = spectrum.temperature_to_color(T_obs);

    // Scale by g³ (relativistic beaming)
    double g3 = g * g * g;
    return color * g3;
}

} // namespace grrt
```

- [ ] **Step 3: Add `src/accretion_disk.cpp` to CMakeLists.txt**

- [ ] **Step 4: Build to verify**

Run: `build.bat`
Expected: Compiles and links.

- [ ] **Step 5: Commit**

```bash
git add include/grrt/scene/accretion_disk.h src/accretion_disk.cpp CMakeLists.txt
git commit -m "feat: add accretion disk with Page-Thorne profile and redshift"
```

---

## Chunk 3: Celestial Sphere

### Task 5: Celestial Sphere

**Files:**
- Create: `include/grrt/scene/celestial_sphere.h`
- Create: `src/celestial_sphere.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Create `celestial_sphere.h`**

```cpp
#ifndef GRRT_CELESTIAL_SPHERE_H
#define GRRT_CELESTIAL_SPHERE_H

#include "grrt/math/vec3.h"
#include "grrt/math/vec4.h"
#include <vector>

namespace grrt {

class CelestialSphere {
public:
    explicit CelestialSphere(int num_stars = 5000, unsigned int seed = 42);

    // Sample the sky at the given escaped ray position
    // Uses (θ, φ) from position coordinates at large r
    Vec3 sample(const Vec4& position) const;

private:
    struct Star {
        double theta;
        double phi;
        double brightness;
    };

    std::vector<Star> stars_;
    double angular_tolerance_ = 0.01; // radians

    // Spatial grid for fast lookup
    static constexpr int grid_theta = 180;
    static constexpr int grid_phi = 360;
    std::vector<std::vector<int>> grid_; // grid[cell] -> list of star indices

    int grid_index(int t_bin, int p_bin) const;
    void build_grid();
};

} // namespace grrt

#endif
```

- [ ] **Step 2: Create `src/celestial_sphere.cpp`**

```cpp
#include "grrt/scene/celestial_sphere.h"
#include <cmath>
#include <algorithm>
#include <numbers>
#include <random>

namespace grrt {

CelestialSphere::CelestialSphere(int num_stars, unsigned int seed)
    : grid_(grid_theta * grid_phi) {

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    stars_.reserve(num_stars);
    for (int i = 0; i < num_stars; ++i) {
        Star s;
        // Uniform on sphere
        s.theta = std::acos(1.0 - 2.0 * uniform(rng));
        s.phi = 2.0 * std::numbers::pi * uniform(rng);

        // Power-law brightness: many dim, few bright
        double u = uniform(rng);
        s.brightness = 0.1 * std::pow(std::max(u, 0.001), -2.5);
        s.brightness = std::min(s.brightness, 50.0); // Clamp extreme outliers

        stars_.push_back(s);
    }

    build_grid();
}

int CelestialSphere::grid_index(int t_bin, int p_bin) const {
    t_bin = std::clamp(t_bin, 0, grid_theta - 1);
    p_bin = ((p_bin % grid_phi) + grid_phi) % grid_phi; // Wrap φ
    return t_bin * grid_phi + p_bin;
}

void CelestialSphere::build_grid() {
    for (int i = 0; i < static_cast<int>(stars_.size()); ++i) {
        int t_bin = static_cast<int>(stars_[i].theta * grid_theta / std::numbers::pi);
        int p_bin = static_cast<int>((stars_[i].phi + std::numbers::pi) * grid_phi / (2.0 * std::numbers::pi));
        t_bin = std::clamp(t_bin, 0, grid_theta - 1);
        p_bin = std::clamp(p_bin, 0, grid_phi - 1);
        grid_[grid_index(t_bin, p_bin)].push_back(i);
    }
}

Vec3 CelestialSphere::sample(const Vec4& position) const {
    double theta = position[2];
    double phi = std::fmod(position[3], 2.0 * std::numbers::pi);
    if (phi > std::numbers::pi) phi -= 2.0 * std::numbers::pi;
    if (phi < -std::numbers::pi) phi += 2.0 * std::numbers::pi;

    int t_bin = static_cast<int>(theta * grid_theta / std::numbers::pi);
    int p_bin = static_cast<int>((phi + std::numbers::pi) * grid_phi / (2.0 * std::numbers::pi));

    // Check this bin and neighbors
    for (int dt = -1; dt <= 1; ++dt) {
        for (int dp = -1; dp <= 1; ++dp) {
            int idx = grid_index(t_bin + dt, p_bin + dp);
            for (int si : grid_[idx]) {
                const Star& s = stars_[si];
                // Angular distance (approximate for small angles)
                double dtheta = theta - s.theta;
                double dphi = phi - s.phi;
                // Handle φ wrapping
                if (dphi > std::numbers::pi) dphi -= 2.0 * std::numbers::pi;
                if (dphi < -std::numbers::pi) dphi += 2.0 * std::numbers::pi;
                double sin_t = std::sin(theta);
                double ang_dist2 = dtheta * dtheta + dphi * dphi * sin_t * sin_t;

                if (ang_dist2 < angular_tolerance_ * angular_tolerance_) {
                    double b = s.brightness;
                    return {{b, b, b}};
                }
            }
        }
    }

    return {}; // Black background
}

} // namespace grrt
```

- [ ] **Step 3: Add `src/celestial_sphere.cpp` to CMakeLists.txt**

- [ ] **Step 4: Build to verify**

- [ ] **Step 5: Commit**

```bash
git add include/grrt/scene/celestial_sphere.h src/celestial_sphere.cpp CMakeLists.txt
git commit -m "feat: add celestial sphere with procedural star field"
```

---

## Chunk 4: Tracer + Renderer + API Wiring

### Task 6: Update Geodesic Tracer

**Files:**
- Modify: `include/grrt/geodesic/geodesic_tracer.h`
- Modify: `src/geodesic_tracer.cpp`

- [ ] **Step 1: Update `geodesic_tracer.h`**

Replace entire file:

```cpp
#ifndef GRRT_GEODESIC_TRACER_H
#define GRRT_GEODESIC_TRACER_H

#include "grrt/geodesic/integrator.h"
#include "grrt/math/vec3.h"

namespace grrt {

// Forward declarations
class AccretionDisk;
class SpectrumLUT;

enum class RayTermination {
    Horizon,
    Escaped,
    MaxSteps
};

struct TraceResult {
    RayTermination termination;
    Vec3 accumulated_color;  // Sum of disk crossing emissions (linear HDR)
    Vec4 final_position;     // Position at termination
    Vec4 final_momentum;     // Momentum at termination
};

class GeodesicTracer {
public:
    GeodesicTracer(const Metric& metric, const Integrator& integrator,
                   double observer_r, int max_steps = 10000, double r_escape = 1000.0);

    TraceResult trace(GeodesicState state,
                      const AccretionDisk* disk,
                      const SpectrumLUT* spectrum) const;

private:
    const Metric& metric_;
    const Integrator& integrator_;
    double observer_r_;
    int max_steps_;
    double r_escape_;
    double horizon_epsilon_ = 0.01;
};

} // namespace grrt

#endif
```

- [ ] **Step 2: Update `src/geodesic_tracer.cpp`**

Replace entire file:

```cpp
#include "grrt/geodesic/geodesic_tracer.h"
#include "grrt/scene/accretion_disk.h"
#include "grrt/color/spectrum.h"
#include <cmath>
#include <numbers>

namespace grrt {

GeodesicTracer::GeodesicTracer(const Metric& metric, const Integrator& integrator,
                               double observer_r, int max_steps, double r_escape)
    : metric_(metric), integrator_(integrator),
      observer_r_(observer_r), max_steps_(max_steps), r_escape_(r_escape) {}

TraceResult GeodesicTracer::trace(GeodesicState state,
                                  const AccretionDisk* disk,
                                  const SpectrumLUT* spectrum) const {
    const double r_horizon = metric_.horizon_radius();
    const double half_pi = std::numbers::pi / 2.0;
    Vec3 color;

    GeodesicState prev = state;

    for (int step = 0; step < max_steps_; ++step) {
        const double r = state.position[1];

        // Check termination
        if (r < r_horizon + horizon_epsilon_) {
            return {RayTermination::Horizon, color, state.position, state.momentum};
        }
        if (r > r_escape_) {
            return {RayTermination::Escaped, color, state.position, state.momentum};
        }

        // Save previous state for disk crossing detection
        prev = state;

        // Step
        const double dlambda = 0.005 * r;
        state = integrator_.step(metric_, state, dlambda);

        // Check for disk crossing (θ crosses π/2)
        if (disk && spectrum) {
            double theta_prev = prev.position[2];
            double theta_new = state.position[2];

            if ((theta_prev - half_pi) * (theta_new - half_pi) < 0.0) {
                // Interpolate to find crossing point
                double frac = (half_pi - theta_prev) / (theta_new - theta_prev);

                double r_cross = prev.position[1] + frac * (state.position[1] - prev.position[1]);
                // Interpolate momentum at crossing
                Vec4 p_cross;
                for (int mu = 0; mu < 4; ++mu) {
                    p_cross[mu] = prev.momentum[mu] + frac * (state.momentum[mu] - prev.momentum[mu]);
                }

                // Check disk bounds
                if (r_cross >= disk->r_inner() && r_cross <= disk->r_outer()) {
                    color += disk->emission(r_cross, p_cross, observer_r_, *spectrum);
                }
            }
        }
    }

    return {RayTermination::MaxSteps, color, state.position, state.momentum};
}

} // namespace grrt
```

- [ ] **Step 3: Build to verify**

Run: `build.bat`
Expected: Compile errors — `Renderer` still uses old `trace()` signature. That's OK, we fix it next.
If it errors, just verify the tracer files themselves compile (the linker will fail on renderer).

### Task 7: Update Renderer

**Files:**
- Modify: `include/grrt/render/renderer.h`
- Modify: `src/renderer.cpp`

- [ ] **Step 1: Update `renderer.h`**

Replace entire file:

```cpp
#ifndef GRRT_RENDERER_H
#define GRRT_RENDERER_H

#include "grrt/camera/camera.h"
#include "grrt/geodesic/geodesic_tracer.h"
#include "grrt/render/tonemapper.h"

namespace grrt {

// Forward declarations
class AccretionDisk;
class CelestialSphere;
class SpectrumLUT;

class Renderer {
public:
    Renderer(const Camera& camera, const GeodesicTracer& tracer,
             const AccretionDisk* disk, const CelestialSphere* sphere,
             const SpectrumLUT* spectrum, const ToneMapper& tonemapper);

    void render(float* framebuffer, int width, int height) const;

private:
    const Camera& camera_;
    const GeodesicTracer& tracer_;
    const AccretionDisk* disk_;
    const CelestialSphere* sphere_;
    const SpectrumLUT* spectrum_;
    const ToneMapper& tonemapper_;
};

} // namespace grrt

#endif
```

- [ ] **Step 2: Update `src/renderer.cpp`**

Replace entire file:

```cpp
#include "grrt/render/renderer.h"
#include "grrt/scene/accretion_disk.h"
#include "grrt/scene/celestial_sphere.h"
#include "grrt/color/spectrum.h"

namespace grrt {

Renderer::Renderer(const Camera& camera, const GeodesicTracer& tracer,
                   const AccretionDisk* disk, const CelestialSphere* sphere,
                   const SpectrumLUT* spectrum, const ToneMapper& tonemapper)
    : camera_(camera), tracer_(tracer), disk_(disk), sphere_(sphere),
      spectrum_(spectrum), tonemapper_(tonemapper) {}

void Renderer::render(float* framebuffer, int width, int height) const {
    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            GeodesicState state = camera_.ray_for_pixel(i, j);
            TraceResult result = tracer_.trace(state, disk_, spectrum_);

            Vec3 color = result.accumulated_color;

            // Add star field for escaped rays
            if (result.termination == RayTermination::Escaped && sphere_) {
                color += sphere_->sample(result.final_position);
            }

            // Tone map
            color = tonemapper_.apply(color);

            // Write to framebuffer
            const int idx = (j * width + i) * 4;
            framebuffer[idx + 0] = static_cast<float>(color[0]);
            framebuffer[idx + 1] = static_cast<float>(color[1]);
            framebuffer[idx + 2] = static_cast<float>(color[2]);
            framebuffer[idx + 3] = 1.0f;
        }
    }
}

} // namespace grrt
```

- [ ] **Step 3: Build to verify**

This should fail because `api.cpp` still uses the old `Renderer` constructor. That's OK — next task.

### Task 8: Update API

**Files:**
- Modify: `src/api.cpp`

- [ ] **Step 1: Update `src/api.cpp`**

Replace entire file:

```cpp
#include "grrt/api.h"
#include "grrt/spacetime/schwarzschild.h"
#include "grrt/geodesic/rk4.h"
#include "grrt/geodesic/geodesic_tracer.h"
#include "grrt/camera/camera.h"
#include "grrt/render/renderer.h"
#include "grrt/scene/accretion_disk.h"
#include "grrt/scene/celestial_sphere.h"
#include "grrt/color/spectrum.h"
#include "grrt/render/tonemapper.h"
#include <memory>
#include <print>

struct GRRTContext {
    GRRTParams params;
    std::unique_ptr<grrt::Metric> metric;
    std::unique_ptr<grrt::Integrator> integrator;
    std::unique_ptr<grrt::GeodesicTracer> tracer;
    std::unique_ptr<grrt::Camera> camera;
    std::unique_ptr<grrt::AccretionDisk> disk;
    std::unique_ptr<grrt::CelestialSphere> sphere;
    std::unique_ptr<grrt::SpectrumLUT> spectrum;
    std::unique_ptr<grrt::ToneMapper> tonemapper;
    std::unique_ptr<grrt::Renderer> renderer;
};

GRRTContext* grrt_create(const GRRTParams* params) {
    auto* ctx = new GRRTContext{};
    ctx->params = *params;

    // Defaults for zero-initialized fields
    double mass = params->mass > 0.0 ? params->mass : 1.0;
    double observer_r = params->observer_r > 0.0 ? params->observer_r : 50.0;
    double observer_theta = params->observer_theta > 0.0 ? params->observer_theta : 1.396;
    double fov = params->fov > 0.0 ? params->fov : 1.047;
    int max_steps = params->integrator_max_steps > 0 ? params->integrator_max_steps : 10000;

    // Core physics
    ctx->metric = std::make_unique<grrt::Schwarzschild>(mass);
    ctx->integrator = std::make_unique<grrt::RK4>();
    ctx->tracer = std::make_unique<grrt::GeodesicTracer>(
        *ctx->metric, *ctx->integrator, observer_r, max_steps);
    ctx->camera = std::make_unique<grrt::Camera>(
        *ctx->metric, observer_r, observer_theta, params->observer_phi,
        fov, params->width, params->height);

    // Color pipeline
    ctx->spectrum = std::make_unique<grrt::SpectrumLUT>();
    ctx->tonemapper = std::make_unique<grrt::ToneMapper>();

    // Accretion disk (optional)
    if (params->disk_enabled) {
        double disk_temp = params->disk_temperature > 0.0 ? params->disk_temperature : 1e7;
        double disk_outer = params->disk_outer > 0.0 ? params->disk_outer : 20.0;
        ctx->disk = std::make_unique<grrt::AccretionDisk>(
            mass, params->disk_inner, disk_outer, disk_temp);
    }

    // Background (optional)
    if (params->background_type == GRRT_BG_STARS) {
        ctx->sphere = std::make_unique<grrt::CelestialSphere>();
    }

    // Renderer
    ctx->renderer = std::make_unique<grrt::Renderer>(
        *ctx->camera, *ctx->tracer,
        ctx->disk.get(), ctx->sphere.get(),
        ctx->spectrum.get(), *ctx->tonemapper);

    std::println("grrt: created context ({}x{}, schwarzschild, M={}, r_obs={}, disk={}, stars={})",
                 params->width, params->height, mass, observer_r,
                 params->disk_enabled ? "on" : "off",
                 params->background_type == GRRT_BG_STARS ? "on" : "off");
    return ctx;
}

void grrt_destroy(GRRTContext* ctx) {
    delete ctx;
}

void grrt_update_params(GRRTContext* ctx, const GRRTParams* params) {
    ctx->params = *params;
}

int grrt_render(GRRTContext* ctx, float* framebuffer) {
    ctx->renderer->render(framebuffer, ctx->params.width, ctx->params.height);
    std::println("grrt: rendered {}x{} frame", ctx->params.width, ctx->params.height);
    return 0;
}

int grrt_render_tile(GRRTContext* /*ctx*/, float* /*buffer*/,
                     int /*x*/, int /*y*/, int /*tile_width*/, int /*tile_height*/) {
    return 0;
}

void grrt_cancel(GRRTContext* /*ctx*/) {}

float grrt_progress(const GRRTContext* /*ctx*/) {
    return 1.0f;
}

const char* grrt_error(const GRRTContext* /*ctx*/) {
    return nullptr;
}

int grrt_cuda_available(void) {
    return 0;
}
```

### Task 9: Update CLI and CMakeLists

**Files:**
- Modify: `cli/main.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Update `cli/main.cpp`**

Enable the disk and star field:

Change:
```cpp
    params.metric_type = GRRT_METRIC_SCHWARZSCHILD;
```
Keep as is.

Add after `params.integrator_max_steps = 10000;`:
```cpp
    params.disk_enabled = 1;
    params.disk_inner = 0.0;       // 0 = use ISCO
    params.disk_outer = 20.0;
    params.disk_temperature = 1e7; // 10 million K peak
    params.background_type = GRRT_BG_STARS;
```

- [ ] **Step 2: Ensure CMakeLists.txt has all new source files**

The final `add_library` block should be:

```cmake
add_library(grrt SHARED
    src/api.cpp
    src/schwarzschild.cpp
    src/rk4.cpp
    src/geodesic_tracer.cpp
    src/camera.cpp
    src/renderer.cpp
    src/spectrum.cpp
    src/tonemapper.cpp
    src/accretion_disk.cpp
    src/celestial_sphere.cpp
)
```

- [ ] **Step 3: Build**

Run: `build.bat`
Expected: Compiles and links with no errors.

- [ ] **Step 4: Run and validate**

Run: `build/Release/grrt-cli.exe`
Expected:
- `output.png` shows a black hole with a **glowing accretion disk**
- The approaching side (left or right depending on viewing angle) should be **brighter** (Doppler beaming)
- Disk color should range from blue-white (inner) to reddish (outer)
- Background should have **scattered stars** that are distorted near the shadow
- A thin **photon ring** may be visible at the shadow edge

Open `output.png` and visually verify.

- [ ] **Step 5: Commit**

```bash
git add include/grrt/geodesic/geodesic_tracer.h src/geodesic_tracer.cpp
git add include/grrt/render/renderer.h src/renderer.cpp
git add src/api.cpp cli/main.cpp CMakeLists.txt
git commit -m "feat: accretion disk with Page-Thorne, redshift, stars, tone mapping"
```

### Task 10: Debug and Tune (if needed)

- [ ] **Step 1: If disk is not visible**

Check that `disk_enabled = 1` is set in CLI params. Print `temperature(r)` at a few radii from `accretion_disk.cpp` to verify Page-Thorne produces nonzero values.

- [ ] **Step 2: If colors look wrong**

Test the spectrum LUT directly: print `temperature_to_color(5800)` — should be roughly white/yellowish (like the Sun). Print `temperature_to_color(3000)` — should be reddish. Print `temperature_to_color(20000)` — should be blue-white.

- [ ] **Step 3: If Doppler beaming is reversed**

The approaching side should be brighter. If it's the opposite, the redshift ratio is inverted — swap numerator and denominator in `AccretionDisk::redshift()`.

- [ ] **Step 4: If stars aren't visible**

Check that escaped rays have reasonable `(θ, φ)` values at termination. Print a few escaped ray final positions.

- [ ] **Step 5: Commit any fixes**

```bash
git add -u
git commit -m "fix: tune accretion disk rendering"
```
