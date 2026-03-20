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
H(r) = c_eff(r) / Omega_z(r)     [geometric units, same as r]
```

where Omega_z is the Kerr vertical epicyclic frequency (see Section 1.2) and c_eff is the effective sound speed in geometric units (c=1):

```
c_eff^2 = k_B*T_eff(r) / (mu*m_p*c^2) + 4*sigma_SB*T_eff(r)^4 / (3*rho_CGS*c^3)
```

The division by c^2 and c^3 converts the CGS sound speed to geometric units (v/c).

---

## 1. Density Model

### 1.1 Vertical Profile — Numerical Hydrostatic Equilibrium

Instead of an analytical Gaussian (which assumes isothermal vertical structure), we solve the hydrostatic equilibrium ODE numerically to account for the non-isothermal Eddington temperature profile:

```
dP/dz = -rho * Omega_z(r)^2 * z
P(z) = P_gas(z) + P_rad(z) = rho*k_B*T(z)/(mu*m_p) + (4*sigma_SB/3c)*T(z)^4
```

where `Omega_z` is the **vertical epicyclic frequency**, which differs from the orbital frequency `Omega_K` in Kerr spacetime due to the oblate geometry:

```
Omega_z^2(r) = Omega_K^2 * (1 - 4*a*sqrt(M/r^3) + 3*a^2/r^2)
```

For Schwarzschild (a=0), Omega_z = Omega_K exactly (spherical symmetry). For Kerr, the prograde frame-dragging reduces the vertical restoring force, making the disk slightly thicker at given temperature. This correction is computed once per radial bin during LUT construction.

**Procedure (computed once per radial bin during LUT construction):**

1. Set midplane boundary conditions: `rho(0) = rho_mid(r)`, `T(0) = T_mid(r)` from Eddington at tau_abs = tau_mid
2. Integrate outward from z = 0 using RK4 in z with step dz = H_gas/20:
   - At each z, compute local tau_abs by integrating density inward from the current z: `tau_abs(z) = integral(kappa_abs * rho dz', z..z_surface)`
   - Compute T(z) from Eddington: `T^4(z) = (3/4)*T_eff^4*(tau_abs(z) + 2/3)`
   - Compute P(z) = P_gas + P_rad
   - Apply the ODE to get drho/dz from dP/dz and the equation of state
3. The integration is self-consistent: tau_abs depends on rho(z), which depends on T(z), which depends on tau_abs. Resolve by iterating the z-profile until the max relative change in rho(z) < 1e-3 (typically 3-5 iterations).
4. Store the resulting rho(z) profile as a 1D lookup table per radial bin: 64 z-bins covering `0 to z_max = 3*H(r)`, one-sided (the profile is symmetric about the midplane, so `rho(r, -z) = rho(r, z)`).

**Effect:** Compared to a Gaussian, the numerical profile is more centrally concentrated (radiation pressure support creates a flatter core with sharper wings). This is physically correct — the inner disk, where radiation pressure dominates, has a distinctly non-Gaussian profile.

The full density including taper and turbulence:

```
rho(r, z, phi) = rho_numerical(r, z) * taper(r) * (1 + delta * noise3D(r/H, z/H, phi))
```

where `z = r * cos(theta)` and `rho_numerical(r, z)` is the tabulated numerical solution. H(r) is still computed as in Section 1.2 for the purpose of scale-height-based quantities (noise scaling, step sizing, volume bounds).

### 1.2 Scale Height with Radiation Pressure

```
H(r) = c_eff(r) / Omega_z(r)

c_eff^2 = k_B*T_eff(r) / (mu*m_p*c^2) + 4*sigma_SB*T_eff(r)^4 / (3*rho_mid_CGS(r)*c^3)

Omega_z^2(r) = Omega_K^2 * (1 - 4*a*sqrt(M/r^3) + 3*a^2/r^2)     [Kerr vertical epicyclic frequency]
```

- First term in c_eff: gas pressure (dominates outer disk)
- Second term in c_eff: radiation pressure (dominates inner disk, produces visible puffing)
- mu = 0.6 (ionized H+He mean molecular weight)
- Division by c^2 and c^3 converts CGS sound speed to geometric units
- Omega_z < Omega_K for prograde Kerr, giving slightly thicker disks at high spin

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

### 3.1 Absorption Opacity (Frequency-Dependent)

Total absorption includes both free-free (bremsstrahlung) and bound-free (photoionization) processes. Both are two-body processes with the same temperature and density scaling but different coefficients.

**Free-free (bremsstrahlung) mass opacity:**

```
kappa_ff_mass = 3.68e22 * g_ff(nu, T) * (1 + X) * rho_CGS * T^(-7/2)     [cm^2/g]
```

where X = 0.7 (hydrogen mass fraction) and g_ff is the thermally-averaged Gaunt factor.

**Bound-free (photoionization) mass opacity:**

```
kappa_bf_mass = 4.34e25 * g_bf * Z * (1 + X) * rho_CGS * T^(-7/2)     [cm^2/g]
```

where Z = 0.02 (metal mass fraction, solar) and g_bf ≈ 1 (bound-free Gaunt factor, order unity). Bound-free dominates free-free by a factor of ~100 * Z ≈ 2 at the same conditions, but becomes negligible when the gas is fully ionized (T >> 10^5 K). We include it because the outer disk can be cool enough for partial ionization.

**Gaunt factor approximation:**

The free-free Gaunt factor provides a frequency and temperature-dependent correction:

```
g_ff(nu, T) = max(1.0, (sqrt(3)/pi) * ln(k_B * T / (h * nu)))
```

This is the Elwert (1954) / Karzas-Latter (1961) asymptotic form. For optical frequencies and typical disk temperatures: g_ff ≈ 1-5 (blue channel has lower g_ff than red, partially counteracting the nu^-3 frequency dependence).

**Combined absorption mass opacity:**

```
kappa_abs_mass(nu, rho, T) = (kappa_ff_mass + kappa_bf_mass) * g_ff(nu, T)
```

Note: g_ff is already included in kappa_ff_mass above; for kappa_bf_mass we use g_bf ≈ 1. To avoid double-counting, the combined formula is:

```
kappa_abs_mass = (3.68e22 * (1+X) * g_ff + 4.34e25 * Z * (1+X) * g_bf) * rho_CGS * T^(-7/2)
```

The volumetric absorption coefficient is then kappa_abs_mass * rho_CGS, giving rho^2 dependence as expected for collisional processes.

**Frequency dependence** (Kramers' processes scale as nu^-3):

```
kappa_abs(lambda) = kappa_abs_mass * (lambda / lambda_ref)^3     [cm^2/g]
```

where lambda_ref = 550nm (green reference). Longer wavelengths (red) have higher absorption; shorter (blue) have lower. During raymarching, opacity is evaluated at the emitter-frame wavelength lambda_emit = lambda_obs / g (Section 4.1).

Evaluated at three observer wavelengths: lambda_B = 450nm, lambda_G = 550nm, lambda_R = 650nm.

### 3.2 Thomson Electron Scattering (Frequency-Independent)

```
kappa_es = 0.34 cm^2/g
```

### 3.3 Total Extinction and Photon Destruction Probability

The radiative transfer (Section 4.1) uses total extinction (absorption + scattering) with an epsilon-weighted source function:

```
kappa_total(lambda) = kappa_abs(lambda) + kappa_es           [total extinction, cm^2/g]
epsilon(lambda)     = kappa_abs(lambda) / kappa_total(lambda) [photon destruction probability]
```

epsilon → 1 in absorption-dominated regions (outer disk, red channel) — the source function approaches pure Planck. epsilon → 0 in scattering-dominated regions (inner disk, blue channel) — emission is suppressed relative to extinction.

### 3.4 Optical Depth Accumulation

Mass opacities (cm^2/g) are multiplied by rho_CGS * ds_proper to get optical depth increments:

```
dtau_total = kappa_total * rho_CGS * ds_proper     [extinction optical depth, for transfer equation]
dtau_abs   = kappa_abs   * rho_CGS * ds_proper     [absorption optical depth, for Eddington temperature]
```

Since rho is stored in geometric normalization and ds is in geometric units, the product `rho * ds` is converted via the rho_scale factor established in Section 0.

---

## 4. Radiative Transfer

### 4.1 Covariant Invariant Radiative Transfer (Three Channels)

We use the frame-independent formulation from RAPTOR (Bronzwaer+ 2018), BHOSS (Younsi+ 2020), and iPOLE (Mościbrodzka & Gammie 2018). The Lorentz invariant `J = I_nu / nu^3` satisfies:

```
dJ/dtau = -J + S
```

where S is the invariant source function. This formulation is exact for any spacetime — all redshift, beaming, and aberration effects are automatically encoded in the nu^3 invariant. No ad-hoc g-factor powers are needed.

**Per-step procedure at each raymarch sample point:**

```
1. Compute emitter-frame frequency for each channel:
   nu_emit(lambda) = g * nu_obs(lambda)        where nu_obs = c / lambda
   lambda_emit = lambda / g

2. Evaluate opacity at emitter-frame frequency (see Section 3 for full formulas):
   kappa_abs_emit(lambda) = kappa_abs(lambda_emit, rho, T)     [Section 3.1, frequency-scaled]
   kappa_total_emit(lambda) = kappa_abs_emit(lambda) + kappa_es
   epsilon(lambda) = kappa_abs_emit(lambda) / kappa_total_emit(lambda)   [photon destruction probability]

3. Compute optical depth increments (in emitter frame):
   dtau_total(lambda) = kappa_total_emit(lambda) * rho_CGS * ds_proper
   dtau_abs(lambda)   = kappa_abs_emit(lambda) * rho_CGS * ds_proper

   where ds_proper is the proper distance along the ray in the emitter frame.
   For a comoving emitter: ds_proper = |p_mu u^mu_emit| * ds_affine * (L_unit)
   Here ds_affine is the geodesic affine parameter step (called `ds` in Section 5.2),
   not to be confused with optical wavelength lambda.

4. Accumulate absorption optical depth for Eddington temperature:
   tau_abs += dtau_abs(lambda_G)
   T = ((3/4) * T_eff(r)^4 * (tau_abs + 2/3))^(1/4)
   Apply turbulent coupling: T_turb (Section 2.3)

5. Compute invariant source function (absorption-only emission, scattering is conservative):
   For each channel lambda in {B, G, R}:
       S(lambda) = epsilon(lambda) * B(lambda_emit, T_turb) / nu_emit^3

   The epsilon factor is critical: only absorbed photons are thermally re-emitted. Scattered
   photons are redirected but not thermalized. Using S = B/nu^3 without epsilon would
   overestimate thermal emission in scattering-dominated regions (inner disk where kappa_es
   >> kappa_abs at short wavelengths).

6. Integrate the invariant transfer equation:
   For each channel lambda in {B, G, R}:
       J(lambda) = J(lambda) * exp(-dtau_total(lambda))
                 + S(lambda) * (1 - exp(-dtau_total(lambda)))

   Note: the optical depth used for both extinction and the exponential coupling is
   dtau_total (absorption + scattering), matching the source function modification by epsilon.

7. After raymarching is complete, recover observed intensity:
   For each channel:
       I_obs(lambda) = J(lambda) * nu_obs(lambda)^3
```

**Why this is more accurate than the g^3 formulation:**
- The g^3 * B(lambda_emit, T) approach is a shortcut that works for single-emission-point sources (thin disk). For volumetric transfer with absorption and emission at varying redshifts along the ray, it incorrectly mixes observer-frame and emitter-frame quantities.
- The invariant J formulation correctly handles varying g along the ray path — each emission/absorption event is computed in its local emitter frame, and the invariant J carries the accumulated result without frame confusion.
- Opacity is correctly evaluated at the emitter-frame frequency, which is automatic in this formulation since all emitter-frame quantities use nu_emit.
- The epsilon-weighted source function correctly handles the absorption/scattering partition — this is what RAPTOR and iPOLE actually implement.

**Redshift convention:** This codebase defines `g = nu_emit / nu_obs = (p_mu u^mu)_emit / (p_mu u^mu)_obs`, where g > 1 means blueshift (approaching matter) and g < 1 means redshift. This is the inverse of the Lindquist (1966) convention.

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

Derived from the metric normalization condition `g_μν u^μ u^ν = -1` with `u^μ = u^t(1, 0, 0, Ω_K)`:

```
Omega_K = sqrt(M) / (r^(3/2) + a*sqrt(M))       [Kerr prograde orbital frequency]

g_tt_eq   = -(1 - 2*M/r)                          [equatorial Kerr metric components]
g_tphi_eq = -2*M*a/r
g_phph_eq = r^2 + a^2 + 2*M*a^2/r

u^t_emit   = 1 / sqrt(-(g_tt_eq + 2*g_tphi_eq*Omega_K + g_phph_eq*Omega_K^2))
u^phi_emit = Omega_K * u^t_emit
u^r_emit   = 0
u^theta_emit = 0
```

Note: The sometimes-seen shortcut `u^t = 1/sqrt(1 - 3M/r + 2a*sqrt(M/r^3))` is the BPT72 energy normalization factor, NOT u^t. Using it for u^t introduces ~3% error at moderate spin, growing larger near ISCO.

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
3. Adaptive step control based on green-channel total optical depth per step:
   ```
   dtau_ref = (kappa_abs(lambda_G) + kappa_es) * rho_CGS * ds_proper
   if dtau_ref > 0.1:  ds *= 0.5
   if dtau_ref < 0.01: ds *= 2.0
   ds = clamp(ds, H(r)/32, H(r))
   ```
4. Continue until exit condition.

### 5.3 Exit Conditions

- Ray leaves disk volume: `|z| > 3*H(r)` or `r > r_outer` or `r < r_horizon`
- Fully opaque: all three channels have `tau_total > 10`
- Step cap: 512 steps reached (sufficient for edge-on views through the full disk chord)

### 5.4 Resume Normal Integration

After exiting the disk, resume the standard adaptive RK4 geodesic integrator. The ray may re-enter the disk (e.g., gravitational lensing bends it back through), in which case raymarching resumes with fresh `tau_abs` accumulation but the invariant `J` is **not** reset — it carries the accumulated emission/absorption from all previous disk crossings. Only `tau_abs` resets because the Eddington approximation measures optical depth from the nearest disk surface, and re-entry constitutes entering from a different surface.

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
| **Total (constant memory)** | **~40 KB** (within 64 KB limit) |

**Global memory (texture-cached):**

| Data | Size |
|---|---|
| Vertical density profile 2D LUT (500 r-bins × 64 z-bins, doubles) | ~250 KB |

The vertical profile LUT is too large for constant memory and is instead stored in global device memory with texture cache for 2D interpolation. Accessed via `tex2D` for hardware-accelerated bilinear interpolation on (r, z/H) coordinates.

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
| Z_metal | 0.02 | dimensionless | Metal mass fraction (solar) |

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
