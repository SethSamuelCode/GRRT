# Volumetric Accretion Disk Design Spec

## Goal

Replace the infinitely thin equatorial-plane accretion disk with a physically accurate volumetric disk featuring Shakura-Sunyaev vertical structure, Eddington radiative equilibrium, frequency-dependent opacity, MRI-inspired turbulent density pockets, and plunging-region dynamics inside the ISCO. Implemented on both CPU and CUDA backends with adaptive raymarching.

## Motivation

The current thin-disk model produces correct radial brightness profiles but lacks visual depth. A volumetric disk shows self-occlusion, limb color effects, turbulent structure, and the puffed-up inner edge from radiation pressure — all physically real features visible in GRMHD simulation renders.

## Non-Goals

- Monte Carlo radiative transfer (Compton scattering, photon packet tracking)
- Full MHD turbulence power spectra (simplex noise is the approximation)
- Frequency-resolved spectra beyond 3-channel RGB
- Time-dependent turbulence (static noise field per seed)

---

## 0. Units and Dimensional Conversion

The raytracer uses geometrized units (G = c = 1, M sets the length scale) for coordinates and geodesics. The opacity and thermodynamic formulas require CGS quantities. The bridge between these systems requires a physical black hole mass M_BH.

**Geometric to CGS conversion factors:**

Given M_BH (in grams), with G = 6.674e-8 cm^3/(g s^2) and c = 2.998e10 cm/s:

```
L_unit = G * M_BH / c^2          [cm per unit of r]
T_unit = G * M_BH / c^3          [seconds per unit of t]
```

**What is already in physical units:**

- T_eff(r) is in Kelvin (set by `--disk-temperature` which specifies T_peak in K)
- Wavelengths lambda_B, lambda_G, lambda_R are in cm

**What needs conversion:**

- rho_mid is computed in geometric normalization (see Section 1.3) and then scaled to g/cm^3 via:
  ```
  rho_CGS = rho_geometric * M_BH / L_unit^3
  ```
- ds (affine parameter step) converts to cm via: `ds_CGS = ds * L_unit`
- Omega_K is in 1/T_unit; c_s is in cm/s

**Practical approach:** The midplane density rho_mid is not computed from first principles in absolute CGS. Instead, it is **normalized** so that the midplane vertical optical depth at the peak-flux radius equals a configurable tau_mid (default 100). This normalization absorbs M_BH and all CGS conversion factors into a single scale factor computed once during LUT construction:

```
rho_scale = tau_mid / (kappa_ref * integral(rho_profile(z) * dz, z=-3H..3H))
```

where kappa_ref = kappa_abs(lambda_G, rho=1, T=T_peak) + kappa_es is the reference opacity at the peak-flux radius. All subsequent density evaluations multiply by rho_scale. This avoids carrying M_BH through the integrator and keeps the geometric coordinate system clean.

**Scale height** is computed in geometric units directly:

```
H(r) = c_eff(r) / Omega_K(r)     [geometric units, same as r]
```

where c_eff is the effective sound speed in geometric units (c=1):

```
c_eff^2 = k_B*T_eff(r) / (mu*m_p*c^2) + 4*sigma_SB*T_eff(r)^4 / (3*rho_CGS*c^3)
```

The division by c^2 and c^3 converts the CGS sound speed to geometric units (v/c).

---

## 1. Density Model

### 1.1 Vertical Profile

Gaussian hydrostatic equilibrium:

```
rho(r, z, phi) = rho_mid(r) * exp(-z^2 / (2*H(r)^2)) * taper(r) * (1 + delta * noise3D(r/H, z/H, phi))
```

where `z = r * cos(theta)`.

### 1.2 Scale Height with Radiation Pressure

```
H(r) = c_eff(r) / Omega_K(r)

c_eff^2 = k_B*T_eff(r) / (mu*m_p*c^2) + 4*sigma_SB*T_eff(r)^4 / (3*rho_mid_CGS(r)*c^3)
```

- First term: gas pressure (dominates outer disk)
- Second term: radiation pressure (dominates inner disk, produces visible puffing)
- mu = 0.6 (ionized H+He mean molecular weight)
- Division by c^2 and c^3 converts CGS sound speed to geometric units

Note: H(r) and rho_mid(r) are mutually dependent (H depends on rho_mid via radiation pressure, rho_mid depends on H via Sigma/(sqrt(2pi)*H)). Resolve by iterating: compute H assuming gas pressure only, then compute rho_mid, then recompute H with radiation pressure, repeat until convergence (2-3 iterations suffice). This is done once during LUT construction, not per-ray.

### 1.3 Midplane Density

From alpha-disk surface density:

```
Sigma(r) proportional to F(r) / (nu * Omega_K)
nu = alpha * c_s * H                          (Shakura-Sunyaev viscosity)
rho_mid(r) = Sigma(r) / (sqrt(2*pi) * H(r))
```

Normalized so that the midplane vertical optical depth at peak-flux radius equals tau_mid (default 100). See Section 0 for the normalization procedure.

### 1.4 ISCO Taper

```
taper(r) = 1                                      for r >= r_isco
taper(r) = exp(-(r_isco - r)^2 / (0.5*M)^2)       for r < r_isco
```

Smooth Gaussian falloff inside ISCO where gas is plunging, not orbiting.

### 1.5 Turbulent Noise

3D simplex noise evaluated at `(r/H(r), z/H(r), phi)`:

- Coordinates scaled by H so one noise cell ~ one scale height in each direction
- Uses z/H (not theta) as the vertical coordinate — this gives true 3D variation through the disk thickness, since theta is near pi/2 everywhere and would collapse the noise to ~2D
- 2 octaves: primary at scale H, secondary at H/3 with 0.5x amplitude
- Amplitude delta from CLI (default 0.4)
- Deterministic seed (default 42)
- Permutation table (512 ints) uploaded to constant memory

### 1.6 Disk Volume Bounds

Ray is inside the disk when:

```
|z| < 3*H(r)   AND   r_horizon < r <= r_outer
```

The 3-sigma cutoff captures 99.7% of the Gaussian density. The inner bound extends to the horizon (not ISCO) because the plunging region has nonzero density.

---

## 2. Temperature Model

### 2.1 Radial Effective Temperature

Unchanged Novikov-Thorne profile:

```
T_eff(r) = T_peak * (F(r) / F_max)^(1/4)
```

### 2.2 Vertical Temperature — Eddington Approximation

```
T^4(r, tau_abs) = (3/4) * T_eff(r)^4 * (tau_abs + 2/3)
```

- `tau_abs` is the absorption optical depth accumulated from the disk surface inward along the ray
- Computed during raymarching — not a precomputed function of (r,z)
- Midplane temperature: `T_mid = ((3/4)*(tau_mid + 2/3))^(1/4) * T_eff`
- Surface temperature: `T_surface ~ 0.84 * T_eff`

### 2.3 Turbulent Temperature Coupling

Dense clumps and sparse pockets adjust temperature via interpolated isothermal-adiabatic relation:

```
t_cool = (rho * c_s^2) / (sigma_SB * T^4) * tau_local
t_turb = 1 / Omega_K(r)
beta = (gamma - 1) * t_cool / (t_cool + t_turb)
T_turb = T * (rho_turb / rho_smooth)^beta
```

- gamma = 5/3 (ideal gas)
- beta -> 0 (isothermal) in hot inner disk where cooling is fast
- beta -> 2/3 (adiabatic) in cool outer disk where cooling is slow

---

## 3. Opacity Model

### 3.1 Kramers' Absorption Opacity (Frequency-Dependent)

The Kramers' free-free mass absorption opacity (cm^2/g). Free-free absorption is a two-body process (ion-electron collisions), so the mass opacity itself depends on density:

```
kappa_abs_mass = 3.7e22 * rho_CGS * T^(-7/2)     [cm^2/g, mass opacity — rho dependence is intrinsic]
```

The volumetric absorption coefficient (used internally) is then kappa_abs_mass * rho, giving rho^2 dependence as expected for bremsstrahlung.

Frequency dependence (Kramers' scales as nu^-3):

```
kappa_abs(lambda) = kappa_abs_mass * (lambda / lambda_G)^3     [cm^2/g]
```

where lambda_G = 550nm is the green reference wavelength. Longer wavelengths (red) have higher absorption; shorter wavelengths (blue) have lower absorption.

Evaluated at three wavelengths: lambda_B = 450nm, lambda_G = 550nm, lambda_R = 650nm.

### 3.2 Thomson Electron Scattering (Frequency-Independent)

```
kappa_es = 0.34 cm^2/g
```

### 3.3 Effective Opacity (Per Channel)

Rosseland effective opacity for combined absorption + scattering:

```
kappa_eff(lambda) = sqrt(kappa_abs(lambda) * (kappa_abs(lambda) + kappa_es))
```

### 3.4 Optical Depth Accumulation

Both mass opacities (cm^2/g) are multiplied by rho_CGS * ds_CGS to get optical depth increments:

```
dtau = kappa * rho_CGS * ds_CGS
```

Since rho is stored in geometric normalization and ds is in geometric units, the product `rho * ds` is converted via the rho_scale factor established in Section 0.

---

## 4. Radiative Transfer

### 4.1 Per-Step Update (Three Channels)

At each raymarch sample point, for each wavelength channel:

```
dtau_eff(lambda) = kappa_eff(lambda) * rho_CGS * ds_CGS
dtau_abs         = kappa_abs(lambda_G) * rho_CGS * ds_CGS    (green channel drives Eddington T)

tau_abs += dtau_abs
T = ((3/4) * T_eff(r)^4 * (tau_abs + 2/3))^(1/4)
Apply turbulent coupling: T_turb (Section 2.3)

For each channel lambda in {B, G, R}:
    lambda_emit = lambda / g                   (blueshift/redshift the wavelength)
    I(lambda) = I(lambda) * exp(-dtau_eff(lambda))
              + g^3 * B(lambda_emit, T_turb) * (1 - exp(-dtau_eff(lambda)))
```

**Redshift convention:** This codebase defines `g = (p_mu u^mu)_emit / (p_mu u^mu)_obs`, where g > 1 means blueshift (approaching matter) and g < 1 means redshift. This is the inverse of the Lindquist (1966) convention (`g_L = 1/g`). The covariant radiative transfer invariant `I_nu / nu^3 = const` gives `I_obs = g^3 * B(lambda_emit, T)` with `lambda_emit = lambda_obs / g`. For blueshift (g > 1), the emitter was radiating at a shorter wavelength that was then redshifted to the observer wavelength — the Planck function is evaluated at the emitter-frame wavelength to determine what intensity was emitted at the frequency that arrives at the observer.

### 4.2 Planck Function

```
B(lambda, T) = (2*h*c^2 / lambda^5) / (exp(h*c / (lambda*k_B*T)) - 1)
```

Evaluated at the three representative wavelengths. The three intensities (B, G, R) are normalized and mapped to sRGB at the end of the ray.

### 4.3 Redshift

At each sample point:

```
g = (p_mu * u^mu)_emit / (p_mu * u^mu)_obs
```

**Kerr auxiliary quantities** (used in 4-velocity formulas below):

```
Sigma = r^2 + a^2 * cos^2(theta)
Delta = r^2 - 2*M*r + a^2
```

**Observer 4-velocity:** Static observer at r_obs. Uses the Schwarzschild-static form `u^t_obs = 1/sqrt(1 - 2M/r_obs)`. This is valid for r_obs >> M where Kerr corrections are negligible (our default r_obs = 50M). This matches the existing thin-disk observer convention.

**Emitter 4-velocity — circular orbit (r >= r_isco):**
```
u^t_emit = 1 / sqrt(1 - 3M/r + 2*a*omega_K)
u^phi_emit = Omega_K * u^t_emit
u^r_emit = 0
u^theta_emit = 0
```

**Emitter 4-velocity — plunging geodesic (r < r_isco):**
```
E = E_isco,  L = L_isco     (conserved from last stable orbit)

u^t = (E*(r^2 + a^2) + 2*M*a*r*L/Sigma) / Delta
u^phi = (L/sin^2(theta) + 2*M*a*r*E/Sigma) / Delta
u^r = -sqrt(max(0, R(r))) / Sigma
u^theta = 0
```

where `R(r) = (E*(r^2+a^2) - a*L)^2 - Delta*(r^2 + (L - a*E)^2)` is the Kerr radial potential.

**Off-midplane approximation:** Both the circular and plunging 4-velocities are derived for equatorial orbits (theta = pi/2). For volumetric disk samples at z != 0, we use the equatorial velocity at the same r. This is standard practice in GRMHD post-processing — the gas velocity is dominated by the orbital motion, and vertical velocities (thermal, turbulent) are subsonic and contribute negligibly to the Doppler shift. The Sigma factor in the plunging formulas does use the actual theta of the sample point, providing the leading-order GR correction.

---

## 5. Adaptive Raymarching Algorithm

### 5.1 Entry Detection

During normal geodesic integration, at each adaptive RK4 step, check whether the ray has entered the disk volume:

```
z = r * cos(theta)
inside = (|z| < 3*H(r)) AND (r_horizon < r <= r_outer)
```

When the ray transitions from outside to inside, switch to raymarching mode.

### 5.2 Raymarch Loop

1. Set initial step size: `ds = H(r) / 8`
2. At each sample:
   - Advance geodesic state by RK4 step of size ds (light follows curved path through disk)
   - Evaluate density, temperature, opacity, redshift at new position
   - Update three-channel radiative transfer
   - Accumulate tau_abs for Eddington temperature
3. Adaptive step control based on green-channel optical depth per step:
   ```
   dtau_ref = kappa_eff(lambda_G) * rho_CGS * ds_CGS
   if dtau_ref > 0.1:  ds *= 0.5
   if dtau_ref < 0.01: ds *= 2.0
   ds = clamp(ds, H(r)/32, H(r))
   ```
4. Continue until exit condition.

### 5.3 Exit Conditions

- Ray leaves disk volume: `|z| > 3*H(r)` or `r > r_outer` or `r < r_horizon`
- Fully opaque: all three channels have `tau_eff > 10`
- Step cap: 512 steps reached (sufficient for edge-on views through the full disk chord)

### 5.4 Resume Normal Integration

After exiting the disk, resume the standard adaptive RK4 geodesic integrator. The ray may re-enter the disk (e.g., gravitational lensing bends it back through), in which case raymarching resumes with fresh tau_abs accumulation. This is physically correct: the Eddington approximation measures optical depth from the nearest disk surface, and re-entry constitutes entering from a different surface.

---

## 6. Implementation Scope

### 6.1 New CLI Parameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--disk-volumetric` | flag | off | Enable volumetric disk model |
| `--disk-turbulence` | float | 0.4 | Turbulence amplitude delta (0=smooth, 1=extreme) |
| `--disk-alpha` | float | 0.1 | Shakura-Sunyaev viscosity parameter |
| `--disk-seed` | int | 42 | Noise seed for deterministic turbulence |

Thin disk remains the default. All existing CLI flags (`--disk-temperature`, `--disk-outer`, etc.) continue to work and feed into the volumetric model where applicable.

### 6.2 Files to Create

| File | Purpose |
|---|---|
| `include/grrt/scene/volumetric_disk.h` | CPU volumetric disk class declaration |
| `src/volumetric_disk.cpp` | CPU density, scale height, temperature, plunging velocity |
| `cuda/cuda_volumetric_disk.h` | CUDA device functions for volumetric disk |
| `cuda/cuda_noise.h` | CUDA 3D simplex noise implementation |
| `include/grrt/math/noise.h` | CPU 3D simplex noise (matching CUDA version) |
| `src/noise.cpp` | CPU noise implementation |
| `include/grrt/math/constants.h` | Physical constants shared by CPU and CUDA |

### 6.3 Files to Modify

| File | Change |
|---|---|
| `include/grrt/scene/accretion_disk.h` | Add volumetric parameters to constructor/accessors |
| `src/accretion_disk.cpp` | Expose scale height LUT, rho_mid LUT for CUDA upload |
| `include/grrt/types.h` | Add volumetric params to GRRTParams |
| `src/api.cpp` | Wire volumetric params through C API |
| `include/grrt/api.h` | Add volumetric param fields to GRRTParams |
| `cli/main.cpp` | Parse new CLI flags |
| `cuda/cuda_types.h` | Add volumetric fields to RenderParams |
| `cuda/cuda_scene.h` | Replace disk crossing with raymarching |
| `cuda/cuda_render.cu` | Add constant memory for new LUTs, upload wrappers |
| `cuda/cuda_render_upload.h` | Declare new upload functions |
| `cuda/cuda_backend.cu` | Build and upload new LUTs |
| `src/geodesic_tracer.cpp` | CPU disk volume entry detection + raymarch loop |
| `src/renderer.cpp` | Pass volumetric flag through render path |

### 6.4 Constant Memory Budget

| Data | Size |
|---|---|
| Existing (color, luminosity, flux, params, star grid) | ~30 KB |
| Scale height LUT (500 doubles) | 4 KB |
| Midplane density LUT (500 doubles) | 4 KB |
| Simplex permutation table (512 ints) | 2 KB |
| New RenderParams fields | ~50 B |
| **Total** | **~40 KB** (within 64 KB limit) |

### 6.5 Physical Constants

Needed in both CPU and CUDA code (shared header `include/grrt/math/constants.h`):

| Constant | Value | Unit | Notes |
|---|---|---|---|
| k_B | 1.380649e-16 | erg/K | Boltzmann constant |
| sigma_SB | 5.670374e-5 | erg/(cm^2 s K^4) | Stefan-Boltzmann |
| m_p | 1.672622e-24 | g | Proton mass |
| c_cgs | 2.997924e10 | cm/s | Speed of light in CGS |
| h_planck | 6.626070e-27 | erg*s | Planck constant |
| G_cgs | 6.674e-8 | cm^3/(g s^2) | Gravitational constant in CGS |
| M_sun | 1.989e33 | g | Solar mass |
| mu | 0.6 | dimensionless | Mean molecular weight (ionized H+He) |
| gamma_gas | 5.0/3.0 | dimensionless | Adiabatic index (ideal gas) |
| X_hydrogen | 0.7 | dimensionless | Hydrogen mass fraction |

These are used only in the volumetric disk LUT construction and per-sample opacity evaluation. The geodesic integrator remains in pure geometric units.

### 6.6 Backward Compatibility

- `--disk-volumetric` off (default): existing thin-disk behavior, identical output
- Validation mode (`--validate`): compares CPU vs CUDA for whichever disk model is active
- All existing tests continue to pass unchanged

---

## 7. Validation

- **Thin disk regression**: renders without `--disk-volumetric` must produce identical output to current code
- **CPU/CUDA match**: volumetric renders must agree within ~1% relative error (relaxed from thin-disk tolerance due to noise evaluation FP differences)
- **Optical depth sanity**: face-on view of midplane should be optically thick (tau >> 1); edge-on should show vertical structure
- **Conservation**: Hamiltonian constraint must remain below 1e-10 during raymarching (same tolerance as CLAUDE.md project invariant — the smaller RK4 steps in raymarching should maintain or improve accuracy)
- **Spin sweep**: visual check that inner disk puffs up more with higher spin (radiation pressure from hotter inner edge)
- **Noise determinism**: same seed produces identical output across runs
