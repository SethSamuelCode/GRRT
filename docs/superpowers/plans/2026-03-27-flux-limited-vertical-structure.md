# Flux-Limited Vertical Structure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hard z=3H density cutoff with a flux-limited diffusion vertical structure that produces smooth, physically motivated disk edges.

**Architecture:** The hydrostatic equilibrium solver gets a Levermore-Pomraning flux limiter that smoothly transitions radiation pressure from fully trapped (optically thick) to free-streaming (optically thin). The vertical grid extends dynamically until density is genuinely negligible. The hard `inside_volume` boundary is replaced with `z_max(r)` from the extended grid. CPU-only; CUDA left as-is.

**Tech Stack:** C++23, CMake, OpenMP (CPU parallelism)

**Spec:** `docs/superpowers/specs/2026-03-27-flux-limited-vertical-structure-design.md`

---

### Task 1: Add radiation constant and z_max infrastructure to header

Add the `a_rad` constant, the `z_max_lut_` member, accessor, and change `n_z_` default to 128.

**Files:**
- Modify: `include/grrt/math/constants.h:23` (add `a_rad`)
- Modify: `include/grrt/scene/volumetric_disk.h:96,108,128-131` (add z_max_lut_, accessor, n_z_ default)

- [ ] **Step 1: Add `a_rad` constant**

In `include/grrt/math/constants.h`, after line 25 (the `C_ff` constant), add:

```cpp
inline constexpr double a_rad      = 4.0 * sigma_SB / c_cgs;  // erg/(cm^3 K^4), radiation constant
```

- [ ] **Step 2: Add z_max_lut_ and update n_z_ in volumetric_disk.h**

In `include/grrt/scene/volumetric_disk.h`:

Change line 128:
```cpp
int n_z_ = 128;
```

After line 130 (`T_profile_lut_`), add:
```cpp
std::vector<double> z_max_lut_;       ///< atmosphere extent z_max(r) [geometric]
```

- [ ] **Step 3: Add z_max accessor and LUT accessor**

In `include/grrt/scene/volumetric_disk.h`, after line 108 (`noise_scale()` accessor), add:

```cpp
/// Atmosphere extent z_max(r) [geometric]. Interpolated from z_max_lut_.
double z_max_at(double r) const;
const std::vector<double>& z_max_lut() const { return z_max_lut_; }
```

- [ ] **Step 4: Add z_max_at() implementation stub**

In `src/volumetric_disk.cpp`, after the `scale_height()` function (line 201), add:

```cpp
double VolumetricDisk::z_max_at(double r) const {
    return interp_radial(z_max_lut_, r);
}
```

- [ ] **Step 5: Build to verify no compilation errors**

```bash
cmake --build build --config Release 2>&1 | tail -5
```

Expected: clean build. The `z_max_lut_` is empty but never accessed yet.

- [ ] **Step 6: Commit**

```
feat: Add radiation constant and z_max infrastructure for extended vertical grid
```

---

### Task 2: Implement the flux-limited hydrostatic solver

Replace the forward Euler solver with an RK4 solver using the Levermore-Pomraning flux limiter. The vertical grid extends dynamically until density drops to 1e-10 of midplane.

**Files:**
- Modify: `src/volumetric_disk.cpp:445-603` (`compute_vertical_profiles()`)

- [ ] **Step 1: Add flux limiter helper functions**

In `src/volumetric_disk.cpp`, before `compute_vertical_profiles()` (before line 445), add these static helpers:

```cpp
/// Levermore-Pomraning flux limiter λ(R).
static double lp_lambda(double R) {
    return (2.0 + R) / (6.0 + 3.0 * R + R * R);
}

/// Eddington factor f(R) = λ + λ²R² (radiation pressure coefficient).
/// Limits: f → 1/3 (optically thick), f → 1 (optically thin).
static double lp_eddington_factor(double R) {
    const double lam = lp_lambda(R);
    return lam + lam * lam * R * R;
}
```

- [ ] **Step 2: Rewrite `compute_vertical_profiles()`**

Replace the entire function body (lines 445-603) with the new flux-limited solver. The structure stays the same (loop over radial bins, iterate), but the density integration changes.

```cpp
void VolumetricDisk::compute_vertical_profiles() {
    using namespace constants;

    // Initial z_max guess: 3H (will be extended if needed)
    z_max_lut_.resize(n_r_);

    // Temporary storage for profiles (will be resized per radius if z_max changes)
    std::vector<double> rho_z(n_z_);
    std::vector<double> T_z(n_z_);
    std::vector<double> tau_z(n_z_);
    std::vector<double> E_rad_z(n_z_);
    std::vector<double> f_z(n_z_);      // Eddington factor at each z
    std::vector<double> mu_z(n_z_);     // mean molecular weight

    // Final 2D LUTs
    rho_profile_lut_.resize(n_r_ * n_z_, 0.0);
    T_profile_lut_.resize(n_r_ * n_z_, 0.0);

    for (int ri = 0; ri < n_r_; ++ri) {
        const double r = r_min_ + (r_outer_ - r_min_) * ri / (n_r_ - 1);
        const double H = H_lut_[ri];
        const double T_eff = T_eff_lut_[ri];
        const double rho_mid_val = rho_mid_lut_[ri];

        if (H <= 0.0 || T_eff <= 0.0 || rho_mid_val <= 0.0) {
            z_max_lut_[ri] = 3.0 * H;
            rho_profile_lut_[ri * n_z_] = 1.0;
            T_profile_lut_[ri * n_z_] = T_eff;
            for (int zi = 1; zi < n_z_; ++zi) {
                rho_profile_lut_[ri * n_z_ + zi] = 0.0;
                T_profile_lut_[ri * n_z_ + zi] = T_eff;
            }
            continue;
        }

        // Omega_z^2
        double Omz2 = omega_z_sq(r);
        if (r < r_isco_ || Omz2 <= 0.0) {
            Omz2 = omega_z_sq(r_isco_);
            if (Omz2 <= 0.0) Omz2 = omega_orb(r_isco_) * omega_orb(r_isco_);
        }

        // Reference CGS midplane density (for opacity lookups)
        const double kR_ref = opacity_luts_.lookup_kappa_ross(
            1e-10, std::clamp(T_eff, 3000.0, 1e8));
        const double kE_ref = opacity_luts_.lookup_kappa_es(
            1e-10, std::clamp(T_eff, 3000.0, 1e8));
        const double kappa_ref_total = std::max(kR_ref + kE_ref, 1.0);
        const double rho_cgs_ref = std::clamp(
            params_.tau_mid / (kappa_ref_total * 3.0 * H), 1e-18, 1e-6);

        // Midplane temperature from Eddington at tau = tau_mid
        const double T_mid4 = 0.75 * T_eff * T_eff * T_eff * T_eff
                             * (params_.tau_mid + 2.0 / 3.0);
        const double T_mid = std::pow(T_mid4, 0.25);

        // Dynamic z_max: start at 3H, extend up to 20H if needed
        double z_max = 3.0 * H;
        constexpr double Z_MAX_CAP = 20.0;  // in units of H

        // Iterate for self-consistency (up to 8 times)
        std::vector<double> prev_rho_z(n_z_, 1.0);
        for (int iter = 0; iter < 8; ++iter) {
            const double dz = z_max / (n_z_ - 1);

            // Initialize
            std::fill(rho_z.begin(), rho_z.end(), 1.0);
            std::fill(T_z.begin(), T_z.end(), T_mid);
            rho_z[0] = 1.0;
            T_z[0] = T_mid;

            // --- Pass 1: Compute tau(z) from current rho(z) ---
            std::fill(tau_z.begin(), tau_z.end(), 0.0);
            for (int zi = n_z_ - 2; zi >= 0; --zi) {
                const double rho_here_cgs = rho_z[zi] * rho_cgs_ref;
                const double rho_next_cgs = rho_z[zi + 1] * rho_cgs_ref;
                const double kR_h = opacity_luts_.lookup_kappa_ross(
                    std::clamp(rho_here_cgs, 1e-18, 1e-6),
                    std::clamp(T_z[zi], 3000.0, 1e8));
                const double kE_h = opacity_luts_.lookup_kappa_es(
                    std::clamp(rho_here_cgs, 1e-18, 1e-6),
                    std::clamp(T_z[zi], 3000.0, 1e8));
                const double kR_n = opacity_luts_.lookup_kappa_ross(
                    std::clamp(rho_next_cgs, 1e-18, 1e-6),
                    std::clamp(T_z[zi + 1], 3000.0, 1e8));
                const double kE_n = opacity_luts_.lookup_kappa_es(
                    std::clamp(rho_next_cgs, 1e-18, 1e-6),
                    std::clamp(T_z[zi + 1], 3000.0, 1e8));
                const double dtau = 0.5 * ((kR_h + kE_h) * rho_here_cgs
                                          + (kR_n + kE_n) * rho_next_cgs) * dz;
                tau_z[zi] = tau_z[zi + 1] + dtau;
            }

            // --- Pass 2: Compute T(z) from Eddington relation ---
            for (int zi = 0; zi < n_z_; ++zi) {
                const double T4 = 0.75 * T_eff * T_eff * T_eff * T_eff
                                * (tau_z[zi] + 2.0 / 3.0);
                T_z[zi] = std::pow(std::max(T4, 0.0), 0.25);
            }

            // --- Pass 3: Compute radiation field and flux limiter ---
            for (int zi = 0; zi < n_z_; ++zi) {
                E_rad_z[zi] = a_rad * T_z[zi] * T_z[zi] * T_z[zi] * T_z[zi];
                const double rho_cgs = rho_z[zi] * rho_cgs_ref;
                mu_z[zi] = opacity_luts_.lookup_mu(
                    std::clamp(rho_cgs, 1e-18, 1e-6),
                    std::clamp(T_z[zi], 3000.0, 1e8));
                if (mu_z[zi] <= 0.0 || !std::isfinite(mu_z[zi])) mu_z[zi] = 0.6;
            }

            // Compute Eddington factor f(z) via flux limiter
            for (int zi = 0; zi < n_z_; ++zi) {
                // dE_rad/dz via finite differences
                double dE_dz = 0.0;
                if (zi == 0) {
                    dE_dz = 0.0;  // symmetry at midplane
                } else if (zi == n_z_ - 1) {
                    dE_dz = (E_rad_z[zi] - E_rad_z[zi - 1]) / dz;  // one-sided
                } else {
                    dE_dz = (E_rad_z[zi + 1] - E_rad_z[zi - 1]) / (2.0 * dz);  // central
                }

                const double rho_cgs = rho_z[zi] * rho_cgs_ref;
                const double kR = opacity_luts_.lookup_kappa_ross(
                    std::clamp(rho_cgs, 1e-18, 1e-6),
                    std::clamp(T_z[zi], 3000.0, 1e8));
                const double denom = kR * rho_cgs * E_rad_z[zi];

                double R_param;
                if (denom < 1e-30) {
                    R_param = 1e30;  // free-streaming
                } else {
                    R_param = std::abs(dE_dz) / denom;
                }
                f_z[zi] = lp_eddington_factor(R_param);
            }

            // --- Pass 4: Integrate density via RK4 ---
            // ODE: dρ/dz = F(z, ρ) where
            // F = [-ρ·Ωz²·z - ρ·d(kT/μmp)/dz - d(f·E_rad)/dz] / (kT/μmp)
            // We precompute the d(kT/μmp)/dz and d(f·E_rad)/dz arrays.

            std::vector<double> d_cs2_dz(n_z_, 0.0);   // d(kT/μmp)/dz
            std::vector<double> d_fE_dz(n_z_, 0.0);     // d(f·E_rad)/dz

            for (int zi = 0; zi < n_z_; ++zi) {
                // d(kT/μmp)/dz
                if (zi == 0) {
                    d_cs2_dz[zi] = 0.0;
                } else if (zi == n_z_ - 1) {
                    const double cs2_here = k_B * T_z[zi] / (mu_z[zi] * m_p);
                    const double cs2_prev = k_B * T_z[zi-1] / (mu_z[zi-1] * m_p);
                    d_cs2_dz[zi] = (cs2_here - cs2_prev) / dz;
                } else {
                    const double cs2_next = k_B * T_z[zi+1] / (mu_z[zi+1] * m_p);
                    const double cs2_prev = k_B * T_z[zi-1] / (mu_z[zi-1] * m_p);
                    d_cs2_dz[zi] = (cs2_next - cs2_prev) / (2.0 * dz);
                }

                // d(f·E_rad)/dz
                if (zi == 0) {
                    d_fE_dz[zi] = 0.0;
                } else if (zi == n_z_ - 1) {
                    d_fE_dz[zi] = (f_z[zi]*E_rad_z[zi] - f_z[zi-1]*E_rad_z[zi-1]) / dz;
                } else {
                    d_fE_dz[zi] = (f_z[zi+1]*E_rad_z[zi+1] - f_z[zi-1]*E_rad_z[zi-1])
                                 / (2.0 * dz);
                }
            }

            // RK4 integration from midplane outward
            const double rho_floor = 1e-15;  // floor relative to midplane
            rho_z[0] = 1.0;
            for (int zi = 0; zi < n_z_ - 1; ++zi) {
                const double z_here = zi * dz;
                const double rho_here = rho_z[zi];

                // RHS function: dρ/dz at a given z and ρ
                // We use precomputed arrays evaluated at grid points,
                // linearly interpolating for fractional positions.
                auto rhs = [&](double z_eval, double rho_eval) -> double {
                    // Find the grid index for z_eval
                    const double z_frac = z_eval / dz;
                    const int idx = std::clamp(static_cast<int>(z_frac), 0, n_z_ - 2);
                    const double t = z_frac - idx;

                    const double cs2 = k_B * ((1.0-t)*T_z[idx] + t*T_z[idx+1])
                                     / (((1.0-t)*mu_z[idx] + t*mu_z[idx+1]) * m_p);
                    const double dcs2 = (1.0-t)*d_cs2_dz[idx] + t*d_cs2_dz[idx+1];
                    const double dfE  = (1.0-t)*d_fE_dz[idx] + t*d_fE_dz[idx+1];
                    const double Omz2_z = Omz2 * z_eval;

                    if (cs2 < 1e-30) return 0.0;

                    // Unit conversion: mixed-unit framework.
                    // z is geometric (units of M), Ωz² is geometric (1/M²).
                    // Thermodynamic quantities are CGS.
                    //
                    // The ODE is: dρ̃/dz = [-ρ̃·Ωz²·z - ρ̃·d(cs²)/dz - (1/ρ_cgs_ref)·d(f·E_rad)/dz] / cs²
                    // where ρ̃ = ρ/ρ_midplane (dimensionless), z is geometric.
                    //
                    // cs² = kT/(μmp) [cm²/s²] → divide by c² to get geometric (dimensionless)
                    // d(f·E_rad)/dz [erg/cm⁴] → divide by (ρ_cgs_ref · c²) to get
                    //   geometric units consistent with (ρ̃ · Ωz² · z) [1/M]
                    const double cs2_geom = cs2 / (c_cgs * c_cgs);
                    const double dcs2_geom = dcs2 / (c_cgs * c_cgs);
                    const double dfE_geom = dfE / (rho_cgs_ref * c_cgs * c_cgs);

                    return (-rho_eval * Omz2_z - rho_eval * dcs2_geom - dfE_geom)
                           / std::max(cs2_geom, 1e-30);
                };

                // RK4 step
                const double k1 = dz * rhs(z_here, rho_here);
                const double k2 = dz * rhs(z_here + 0.5*dz, std::max(rho_here + 0.5*k1, rho_floor));
                const double k3 = dz * rhs(z_here + 0.5*dz, std::max(rho_here + 0.5*k2, rho_floor));
                const double k4 = dz * rhs(z_here + dz, std::max(rho_here + k3, rho_floor));

                rho_z[zi + 1] = std::max(rho_here + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0,
                                          rho_floor);
            }

            // --- Check if z_max needs extending ---
            // rho_z is normalized to midplane = 1.0, so 1e-10 is relative to midplane
            if (rho_z[n_z_ - 1] > 1e-10 && z_max < Z_MAX_CAP * H) {
                z_max = std::min(z_max + H, Z_MAX_CAP * H);
                prev_rho_z = rho_z;  // update before restart to avoid stale comparison
                // This counts as one of the 8 outer iterations (counter increments via for-loop)
                continue;
            }

            // --- Convergence check ---
            double max_delta = 0.0;
            for (int zi = 0; zi < n_z_; ++zi) {
                if (prev_rho_z[zi] > rho_floor * 10.0) {
                    const double delta = std::abs(rho_z[zi] - prev_rho_z[zi]) / prev_rho_z[zi];
                    max_delta = std::max(max_delta, delta);
                }
            }
            prev_rho_z = rho_z;

            if (iter > 0 && max_delta < 0.001) break;  // converged
        }

        // Convergence warning (spec requirement)
        {
            double max_delta = 0.0;
            for (int zi = 0; zi < n_z_; ++zi) {
                if (prev_rho_z[zi] > rho_floor * 10.0) {
                    const double delta = std::abs(rho_z[zi] - prev_rho_z[zi]) / prev_rho_z[zi];
                    max_delta = std::max(max_delta, delta);
                }
            }
            if (max_delta >= 0.001) {
                std::fprintf(stderr,
                    "[VolumetricDisk] WARNING: vertical profile did not converge at r_idx=%d (max delta=%.2e)\n",
                    ri, max_delta);
            }
        }

        z_max_lut_[ri] = z_max;

        // Store profiles
        for (int zi = 0; zi < n_z_; ++zi) {
            rho_profile_lut_[ri * n_z_ + zi] = rho_z[zi];
            T_profile_lut_[ri * n_z_ + zi] = T_z[zi];
        }
    }

    std::printf("[VolumetricDisk] Vertical profiles computed. z_max range: %.2f H to %.2f H\n",
                *std::min_element(z_max_lut_.begin(), z_max_lut_.end()) / H_lut_[0],
                *std::max_element(z_max_lut_.begin(), z_max_lut_.end()) / H_lut_[n_r_/2]);
}
```

**Key differences from old code:**
- Flux limiter `f(z)` replaces `c_eff²` in the pressure calculation
- RK4 replaces forward Euler
- Dynamic z_max with 20H cap
- 128 bins instead of 64
- Convergence check with early exit
- Radial loop is deliberately serial (temporary vectors are reused across bins; ~800ms total is acceptable per spec)

- [ ] **Step 3: Build to verify compilation**

```bash
cmake --build build --config Release 2>&1 | tail -5
```

Expected: clean build.

- [ ] **Step 4: Quick smoke test — check z_max diagnostic output**

```bash
./build/Release/grrt-cli --disk-volumetric --backend cpu --width 64 --height 64 --output /dev/null 2>&1 | grep "z_max\|VolumetricDisk"
```

Expected: no crash, z_max range printed. Values should be 3H-20H depending on temperature.

- [ ] **Step 5: Commit**

```
feat: Implement flux-limited hydrostatic solver with Levermore-Pomraning limiter
```

---

### Task 3: Update inside_volume, interp_2d, density, and temperature to use z_max

Replace all references to the hard 3H boundary with z_max(r) from the new LUT.

**Files:**
- Modify: `src/volumetric_disk.cpp:152-155` (`inside_volume`)
- Modify: `src/volumetric_disk.cpp:171-193` (`interp_2d`)
- Modify: `src/volumetric_disk.cpp:204-205` (`density`)
- Modify: `src/volumetric_disk.cpp:227-228` (`temperature`)

- [ ] **Step 1: Update `inside_volume`**

Replace `src/volumetric_disk.cpp` lines 152-155:

```cpp
bool VolumetricDisk::inside_volume(double r, double z) const {
    if (r <= r_horizon_ || r > r_outer_) return false;
    const double zm = z_max_at(r);
    return std::abs(z) < zm;
}
```

- [ ] **Step 2: Rewrite `interp_2d` to use per-column z_max**

The bilinear interpolation must use each radial column's own `z_max_lut_[ri]` for z normalization, since adjacent columns may have very different atmosphere extents. Replace the entire `interp_2d` function body:

```cpp
double VolumetricDisk::interp_2d(const std::vector<double>& lut, double r, double z_abs) const {
    // Radial interpolation (unchanged)
    const double r_frac = std::clamp((r - r_min_) / (r_outer_ - r_min_) * (n_r_ - 1),
                                      0.0, static_cast<double>(n_r_ - 1));
    const int ri = std::min(static_cast<int>(r_frac), n_r_ - 2);
    const double tr = r_frac - ri;

    // Per-column z normalization: each column has its own z_max
    const double zm_lo = z_max_lut_[ri];
    const double zm_hi = z_max_lut_[std::min(ri + 1, n_r_ - 1)];

    // If z_abs is beyond both columns' extent, return 0
    if ((zm_lo <= 0.0 || z_abs >= zm_lo) && (zm_hi <= 0.0 || z_abs >= zm_hi))
        return 0.0;

    // Look up in column ri
    double val_lo = 0.0;
    if (zm_lo > 0.0 && z_abs < zm_lo) {
        const double z_frac_lo = std::clamp(z_abs / zm_lo * (n_z_ - 1), 0.0,
                                             static_cast<double>(n_z_ - 1));
        const int zi_lo = std::min(static_cast<int>(z_frac_lo), n_z_ - 2);
        const double tz_lo = z_frac_lo - zi_lo;
        val_lo = (1.0 - tz_lo) * lut[ri * n_z_ + zi_lo]
               + tz_lo * lut[ri * n_z_ + zi_lo + 1];
    }

    // Look up in column ri+1
    double val_hi = 0.0;
    const int ri1 = std::min(ri + 1, n_r_ - 1);
    if (zm_hi > 0.0 && z_abs < zm_hi) {
        const double z_frac_hi = std::clamp(z_abs / zm_hi * (n_z_ - 1), 0.0,
                                             static_cast<double>(n_z_ - 1));
        const int zi_hi = std::min(static_cast<int>(z_frac_hi), n_z_ - 2);
        const double tz_hi = z_frac_hi - zi_hi;
        val_hi = (1.0 - tz_hi) * lut[ri1 * n_z_ + zi_hi]
               + tz_hi * lut[ri1 * n_z_ + zi_hi + 1];
    }

    return (1.0 - tr) * val_lo + tr * val_hi;
}
```

This replaces the entire function, not just the z normalization lines.

- [ ] **Step 3: Verify density() and temperature() — no code changes needed**

Both `density()` and `temperature()` gate on `inside_volume()` which now uses `z_max_at(r)`. The actual lookups go through `interp_2d()` which also uses per-column `z_max`. No additional changes are needed to these functions — the boundary update propagates through `inside_volume` and `interp_2d`.

Verify by reading `density()` (~line 204) and `temperature()` (~line 227) to confirm they call `inside_volume()` and `interp_2d()` without any hardcoded `3H` references.

- [ ] **Step 4: Build and run quick test**

```bash
cmake --build build --config Release 2>&1 | tail -3
./build/Release/grrt-cli --disk-volumetric --backend cpu --width 256 --height 256 --output zmax_test 2>&1
```

Expected: clean build, renders without crash. Check the output image for a smooth disk edge.

- [ ] **Step 5: Commit**

```
feat: Replace hard 3H boundary with dynamic z_max(r) in density and temperature lookups
```

---

### Task 4: Update density normalization to use z_max

The column integral must use the new extended grid.

**Files:**
- Modify: `src/volumetric_disk.cpp:633-643` (column integral in `normalize_density`)

- [ ] **Step 1: Update the column integral**

In `src/volumetric_disk.cpp`, replace lines 633-636:

```cpp
    const double H_peak = H_lut_[peak_idx];
    const double z_max = 3.0 * H_peak;
    const double dz = z_max / (n_z_ - 1);
```

With:

```cpp
    const double z_max = z_max_lut_[peak_idx];
    const double dz = z_max / (n_z_ - 1);
```

No other changes needed — the trapezoidal integration loop already iterates over `n_z_ - 1` bins.

- [ ] **Step 2: Build and quick test**

```bash
cmake --build build --config Release 2>&1 | tail -3
./build/Release/grrt-cli --disk-volumetric --backend cpu --width 64 --height 64 --output /dev/null 2>&1 | grep rho_scale
```

Expected: `rho_scale` value may differ from before (the atmosphere adds column density, so normalization adjusts).

- [ ] **Step 3: Commit**

```
feat: Update density normalization to integrate over extended z_max grid
```

---

### Task 5: Update raymarcher entry/exit to use z_max

Replace the 6H entry trigger and exit condition with z_max(r) + H.

**Files:**
- Modify: `src/geodesic_tracer.cpp:64-87` (entry detection in `trace()`)
- Modify: `src/geodesic_tracer.cpp:199-212` (exit/step control in `raymarch_volumetric()`)

- [ ] **Step 1: Update entry detection in `trace()`**

In `src/geodesic_tracer.cpp`, replace lines 83-87:

```cpp
            const double H_new = vol_disk_->scale_height(r_new);
            const bool near_disk = (std::abs(z_new) < 6.0 * H_new
                                 || std::abs(z_prev) < 6.0 * vol_disk_->scale_height(r_prev))
                                && r_new >= vol_disk_->r_horizon()
                                && r_new <= vol_disk_->r_max();
```

With:

```cpp
            const double zm_new = vol_disk_->z_max_at(r_new);
            const double H_new = vol_disk_->scale_height(r_new);
            const bool near_disk = (std::abs(z_new) < zm_new + H_new
                                 || std::abs(z_prev) < vol_disk_->z_max_at(r_prev) + vol_disk_->scale_height(r_prev))
                                && r_new >= vol_disk_->r_horizon()
                                && r_new <= vol_disk_->r_max();
```

- [ ] **Step 2: Update exit condition in `raymarch_volumetric()`**

In `src/geodesic_tracer.cpp`, replace line 202:

```cpp
            if (been_inside && std::abs(z) > 6.0 * H) break;
```

With:

```cpp
            const double zm = vol_disk_->z_max_at(r);
            if (been_inside && std::abs(z) > zm + H) break;
```

- [ ] **Step 3: Build and render comparison**

```bash
cmake --build build --config Release 2>&1 | tail -3
./build/Release/grrt-cli --disk-volumetric --backend cpu --width 1024 --height 1024 --output edge_smooth_test 2>&1
```

Expected: disk edge should be smooth — no hard lines or shadow bands. Compare visually to the baseline image.

- [ ] **Step 4: Render with high turbulence to verify holes still work**

```bash
./build/Release/grrt-cli --disk-volumetric --backend cpu --width 512 --height 512 --disk-turbulence 1.5 --disk-noise-scale 5.0 --disk-noise-octaves 4 --output edge_smooth_holes 2>&1
```

Expected: holes visible, boundary still smooth.

- [ ] **Step 5: Commit**

```
fix: Replace hard 6H entry/exit trigger with z_max(r) + H in raymarcher
```

---

### Task 6: Add mass_solar and eddington_fraction parameters

Independent feature: derive peak_temperature from physical parameters.

**Files:**
- Modify: `include/grrt/types.h:47` (add fields)
- Modify: `cli/main.cpp:25-30` (help text)
- Modify: `cli/main.cpp:70` (defaults)
- Modify: `cli/main.cpp:120` (arg parsing)
- Modify: `src/api.cpp:71-89` (temperature resolution)

- [ ] **Step 1: Add fields to GRRTParams**

In `include/grrt/types.h`, after line 47 (`disk_noise_octaves`), add:

```c
    double mass_solar;          /* BH mass in solar masses (0 = use disk_temperature directly) */
    double eddington_fraction;  /* Accretion rate as fraction of Eddington (0 = use disk_temperature) */
```

- [ ] **Step 2: Add CLI help text**

In `cli/main.cpp`, add to the help output (after the disk-noise-octaves line):

```cpp
    std::println("  --mass-solar M         Black hole mass in solar masses (derives temperature)");
    std::println("  --eddington-fraction F  Accretion rate as Eddington fraction (derives temperature)");
```

- [ ] **Step 3: Add defaults**

In `cli/main.cpp`, after line 70 (`disk_noise_octaves = 2`), add:

```cpp
    params.mass_solar = 0.0;
    params.eddington_fraction = 0.0;
```

- [ ] **Step 4: Add arg parsing**

In `cli/main.cpp`, after the `--disk-noise-octaves` parsing block, add:

```cpp
        } else if (arg("--mass-solar")) {
            if (auto v = next()) params.mass_solar = std::atof(v);
        } else if (arg("--eddington-fraction")) {
            if (auto v = next()) params.eddington_fraction = std::atof(v);
```

- [ ] **Step 5: Add temperature resolution in api.cpp**

In `src/api.cpp`, before the volumetric disk construction (line 78), add temperature derivation:

```cpp
        // Derive peak_temperature from mass and Eddington fraction if provided
        double vol_disk_temp = params->disk_temperature > 0 ? params->disk_temperature : 1e7;
        if (params->mass_solar > 0.0 && params->eddington_fraction > 0.0
            && params->disk_temperature <= 0.0) {
            // Radiative efficiency: η = 1 - E_isco (Bardeen, Press & Teukolsky 1972)
            // E_isco is the specific orbital energy at the ISCO.
            // Re-derived here because this runs before VolumetricDisk construction.
            // The same formula exists in VolumetricDisk::E_isco().
            const double a_star = std::abs(params->spin);
            const double Z1 = 1.0 + std::cbrt(1.0 - a_star * a_star)
                                   * (std::cbrt(1.0 + a_star) + std::cbrt(1.0 - a_star));
            const double Z2 = std::sqrt(3.0 * a_star * a_star + Z1 * Z1);
            const double r_isco_M = 3.0 + Z2 - std::sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2));
            const double v_isco = 1.0 / std::sqrt(r_isco_M);  // Kepler v at ISCO
            const double E_isco = (1.0 - 2.0*v_isco*v_isco + a_star*v_isco*v_isco*v_isco)
                                / std::sqrt(1.0 - 3.0*v_isco*v_isco + 2.0*a_star*v_isco*v_isco*v_isco);
            const double eta = 1.0 - E_isco;
            vol_disk_temp = 5.0e7 * std::pow(eta, 0.25)
                          * std::pow(params->mass_solar, -0.25)
                          * std::pow(params->eddington_fraction, 0.25);
            std::printf("[GRRT] Derived T_peak = %.0f K from M=%.1f M_sun, f_Edd=%.3f, eta=%.4f\n",
                        vol_disk_temp, params->mass_solar, params->eddington_fraction, eta);
        }
```

Then update line 88 to use `vol_disk_temp`:

```cpp
            ctx->vol_disk = std::make_unique<grrt::VolumetricDisk>(
                params->mass, params->spin,
                params->disk_outer > 0 ? params->disk_outer : 30.0,
                vol_disk_temp,
                vp);
```

- [ ] **Step 6: Build and test**

```bash
cmake --build build --config Release 2>&1 | tail -3
# Test: 10 solar mass BH at 10% Eddington, Schwarzschild
./build/Release/grrt-cli --disk-volumetric --backend cpu --width 256 --height 256 --mass-solar 10 --eddington-fraction 0.1 --spin 0.0 --output mass_test_a0 2>&1 | grep "Derived"
# Expected: T_peak ~ 16,000 K

# Test: same mass, Kerr a=0.998
./build/Release/grrt-cli --disk-volumetric --backend cpu --width 256 --height 256 --mass-solar 10 --eddington-fraction 0.1 --output mass_test_a998 2>&1 | grep "Derived"
# Expected: T_peak ~ 26,000 K

# Test: manual temperature override takes precedence
./build/Release/grrt-cli --disk-volumetric --backend cpu --width 256 --height 256 --mass-solar 10 --eddington-fraction 0.1 --disk-temperature 5e6 --output mass_test_override 2>&1 | grep "Derived"
# Expected: no "Derived" line printed (manual override used)
```

- [ ] **Step 7: Commit**

```
feat: Add --mass-solar and --eddington-fraction CLI flags to derive disk temperature
```

---

### Task 7: Visual regression and final validation

Render comparison images to verify the edge artifacts are fixed and the disk looks physically correct.

**Files:**
- No code changes — testing only.

- [ ] **Step 1: Render high-res default comparison**

```bash
./build/Release/grrt-cli --disk-volumetric --backend cpu --width 1024 --height 1024 --output final_default 2>&1
```

Compare `final_default.png` to `baseline_cpu.png`. The disk edge should be smooth — no hard lines, no shadow bands, no blockiness at the bottom edge.

- [ ] **Step 2: Render with high turbulence**

```bash
./build/Release/grrt-cli --disk-volumetric --backend cpu --width 1024 --height 1024 --disk-turbulence 1.5 --disk-noise-scale 5.0 --disk-noise-octaves 4 --output final_holes 2>&1
```

Verify holes are visible and boundary is smooth.

- [ ] **Step 3: Render with mass/Eddington (cool disk)**

```bash
./build/Release/grrt-cli --disk-volumetric --backend cpu --width 512 --height 512 --mass-solar 1e8 --eddington-fraction 0.1 --output final_smbh 2>&1
```

Expected: a supermassive BH disk with T_peak ~5,000K should show color gradients (reddish outer, whiter inner) since the temperature is in the optical regime.

- [ ] **Step 4: Verify flux limiter diagnostics**

Check that the hydrostatic solver converges and z_max values are reasonable:

```bash
./build/Release/grrt-cli --disk-volumetric --backend cpu --width 64 --height 64 --output /dev/null 2>&1 | grep "VolumetricDisk"
```

Expected: z_max range printed, no convergence warnings.

- [ ] **Step 5: Verify CUDA still builds and runs (unchanged behavior)**

```bash
./build/Release/grrt-cli --disk-volumetric --backend cuda --width 256 --height 256 --output cuda_unchanged 2>&1
```

Expected: renders without crash. Visual quality matches pre-change CUDA output (hard edges still present — expected).

- [ ] **Step 6: Commit any fixups**

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Add a_rad constant and z_max infrastructure | `constants.h`, `volumetric_disk.h`, `volumetric_disk.cpp` |
| 2 | Flux-limited hydrostatic solver | `volumetric_disk.cpp` |
| 3 | Replace 3H boundary with z_max(r) | `volumetric_disk.cpp` |
| 4 | Update density normalization | `volumetric_disk.cpp` |
| 5 | Update raymarcher entry/exit | `geodesic_tracer.cpp` |
| 6 | Mass/Eddington parameters | `types.h`, `main.cpp`, `api.cpp` |
| 7 | Visual regression testing | — |
