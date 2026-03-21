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
- Omega_orb is in 1/T_unit; c_s is in cm/s

**Practical approach:** The midplane density rho_mid is not computed from first principles in absolute CGS. Instead, it is **normalized** so that the midplane vertical optical depth at the peak-flux radius equals a configurable tau_mid (default 100). This normalization absorbs M_BH and all CGS conversion factors into a single scale factor computed once during LUT construction:

```
rho_scale = tau_mid / (kappa_ref * integral(rho_profile(z) * dz, z=-3H..3H))
```

where kappa_ref = kappa_R_LUT(rho_guess, T_peak) + kappa_es_LUT(rho_guess, T_peak) is the reference total extinction opacity at the peak-flux radius (using the Rosseland mean absorption opacity for consistency with the Eddington T-τ relation), and rho_guess is an initial density estimate (iterated to self-consistency). The opacity LUTs (Sections 3.1.5, 3.2, and the Rosseland mean LUT described in Section 1.1) must be constructed before the density normalization. All subsequent density evaluations multiply by rho_scale. This avoids carrying M_BH through the integrator and keeps the geometric coordinate system clean.

**Scale height** is computed in geometric units directly:

```
H(r) = c_eff(r) / Omega_z(r)     [geometric units, same as r]
```

where Omega_z is the Kerr vertical epicyclic frequency (see Section 1.2) and c_eff is the effective sound speed in geometric units (c=1):

```
c_eff^2 = k_B*T_eff(r) / (mu(rho_mid,T_eff)*m_p*c^2) + 4*sigma_SB*T_eff(r)^4 / (3*rho_CGS*c^3)
```

The division by c^2 and c^3 converts the CGS sound speed to geometric units (v/c).

---

## 1. Density Model

### 1.1 Vertical Profile — Numerical Hydrostatic Equilibrium

Instead of an analytical Gaussian (which assumes isothermal vertical structure), we solve the hydrostatic equilibrium ODE numerically to account for the non-isothermal Eddington temperature profile:

```
dP/dz = -rho * Omega_z(r)^2 * z
P(z) = P_gas(z) + P_rad(z) = rho*k_B*T(z)/(mu(rho,T)*m_p) + (4*sigma_SB/3c)*T(z)^4
```

where `Omega_z` is the **vertical epicyclic frequency**, which differs from the orbital frequency in Kerr spacetime due to the oblate geometry:

```
Omega_orb = sqrt(M) / (r^(3/2) + a*sqrt(M))       [Kerr prograde orbital frequency]
Omega_z^2(r) = Omega_orb^2 * (1 - 4*a*sqrt(M/r^3) + 3*a^2/r^2)
```

Note: the base frequency is the **Kerr orbital frequency** Omega_orb, not the Newtonian Keplerian frequency sqrt(M/r^3). The ratio Omega_orb/sqrt(M/r^3) = r^{3/2}/(r^{3/2} + a*sqrt(M)), which is ~0.94 at r_isco for a=0.998 — using the Newtonian value would overestimate Omega_z^2 by ~12%.

For Schwarzschild (a=0), Omega_orb = sqrt(M/r^3) and Omega_z = Omega_orb exactly (spherical symmetry). For Kerr, the prograde frame-dragging reduces the vertical restoring force, making the disk slightly thicker at given temperature. This correction is computed once per radial bin during LUT construction.

**Procedure (computed once per radial bin during LUT construction):**

1. Set midplane boundary conditions: `rho(0) = rho_mid(r)`, `T(0) = T_mid(r)` from Eddington at tau_total = tau_mid
2. Integrate outward from z = 0 using RK4 in z with step dz = H_gas/20:
   - At each z, compute local tau_total by integrating total extinction inward from the current z: `tau_total(z) = integral((kappa_R(rho, T) + kappa_es_LUT(rho, T)) * rho dz', z..z_surface)`
     where kappa_R is the **Rosseland mean absorption opacity** (see below), NOT a monochromatic opacity at a single frequency. The Eddington T-τ relation is derived from frequency-integrated radiative transfer moment equations — using a monochromatic opacity (e.g., at ν_G) gives a temperature profile that depends on the arbitrary choice of reference frequency. The Rosseland mean properly weights by the temperature derivative of the Planck function:
     ```
     1/kappa_R(rho, T) = integral( (1/kappa_abs(nu, rho, T)) * dB_nu/dT dnu ) / integral( dB_nu/dT dnu )
     ```
     This is precomputed as a 2D LUT `kappa_R_LUT(rho, T)` on the same 100×100 grid as kappa_es_LUT, using the existing 3D absorption opacity LUT (Section 3.1.5) to evaluate kappa_abs at each frequency bin. The 20-bin frequency grid is coarse for a Rosseland integral; extend to 50 log-spaced bins from 10^{13} to 10^{16} Hz for the Rosseland computation only (the raymarching LUT retains 20 bins). Total additional size: 80 KB.
   - Compute T(z) from Eddington: `T^4(z) = (3/4)*T_eff^4*(tau_total(z) + 2/3)`
   - Compute P(z) = P_gas + P_rad
   - Apply the ODE to get drho/dz from dP/dz and the equation of state
3. The integration is self-consistent: tau_total depends on rho(z), which depends on T(z), which depends on tau_total. Resolve by iterating the z-profile until the max relative change in rho(z) < 1e-3 (typically 3-5 iterations).
4. Store the resulting rho(z) and T(z) profiles as 1D lookup tables per radial bin: 64 z-bins covering `0 to z_max = 3*H(r)`, one-sided (both profiles are symmetric about the midplane). The T(r, z) LUT is used during raymarching (Section 4.1, step 4) to avoid the physically incorrect alternative of computing temperature from the ray's slant optical depth.

**Effect:** Compared to a Gaussian, the numerical profile is more centrally concentrated (radiation pressure support creates a flatter core with sharper wings). This is physically correct — the inner disk, where radiation pressure dominates, has a distinctly non-Gaussian profile.

The full density including taper and turbulence:

```
rho(r, z, phi) = rho_numerical(r, z) * taper(r) * (1 + delta * noise3D(r/H, z/H, phi))
```

where `z = r * cos(theta)` and `rho_numerical(r, z)` is the tabulated numerical solution. H(r) is still computed as in Section 1.2 for the purpose of scale-height-based quantities (noise scaling, step sizing, volume bounds).

### 1.2 Scale Height with Radiation Pressure

```
H(r) = c_eff(r) / Omega_z(r)

c_eff^2 = k_B*T_eff(r) / (mu(rho_mid,T_eff)*m_p*c^2) + 4*sigma_SB*T_eff(r)^4 / (3*rho_mid_CGS(r)*c^3)

Omega_orb = sqrt(M) / (r^(3/2) + a*sqrt(M))                         [Kerr prograde orbital frequency]
Omega_z^2(r) = Omega_orb^2 * (1 - 4*a*sqrt(M/r^3) + 3*a^2/r^2)    [Kerr vertical epicyclic frequency]
```

- First term in c_eff: gas pressure (dominates outer disk)
- Second term in c_eff: radiation pressure (dominates inner disk, produces visible puffing)
- mu = mu_LUT(rho, T) is the **Saha-derived mean molecular weight** (see below), NOT the hardcoded 0.6 commonly used for fully ionized gas. The mean molecular weight is mu = rho / (n_total * m_p), where n_total = n_ions + n_atoms + n_e is the total particle number density from the Saha solver (Section 3.1.1). For fully ionized solar composition, mu ≈ 0.6. For neutral gas (T < 5000 K in the outer disk), mu ≈ 1.3. Hardcoding mu = 0.6 overestimates the particle density in cool gas by ~2×, which overestimates the gas pressure and inflates the scale height by ~45% in the outer disk.
- mu_LUT(rho, T) is stored as a 2D lookup table (same 100×100 grid as kappa_es_LUT), computed once during initialization alongside the ionization equilibrium solver. Total size: 80 KB.
- Division by c^2 and c^3 converts CGS sound speed to geometric units
- Omega_z < Omega_orb for prograde Kerr, giving slightly thicker disks at high spin

**Inside the ISCO (r < r_isco):** The vertical epicyclic frequency Omega_z^2 goes to zero (and negative) inside the ISCO — there is no restoring force for circular orbits. This causes H = c_eff/Omega_z to diverge, which is unphysical. The plunging gas retains approximately the vertical structure it had at the ISCO because the plunge time is much shorter than the vertical relaxation time. Therefore:

```
H(r < r_isco) = H(r_isco)       [frozen at ISCO value]
```

This is applied after computing H(r_isco) during LUT construction. The density profile inside the ISCO uses this frozen H together with the ISCO taper (Section 1.4).

Note: H(r) and rho_mid(r) are mutually dependent (H depends on rho_mid via radiation pressure, rho_mid depends on H via Sigma/(sqrt(2pi)*H)). Resolve by iterating: compute H assuming gas pressure only, then compute rho_mid, then recompute H with radiation pressure, repeat until convergence (2-3 iterations suffice). This is done once during LUT construction, not per-ray.

### 1.3 Midplane Density

From alpha-disk surface density:

```
Sigma(r) proportional to F(r) / (nu * Omega_orb^2)
nu = alpha * c_s * H                          (Shakura-Sunyaev viscosity)
rho_mid(r) = Sigma(r) / (sqrt(2*pi) * H(r))
```

Derivation: the viscous dissipation per unit area (both faces) is `2F = Sigma*nu*(r*dOmega/dr)^2` (from the shear stress). For the exact Kerr orbital frequency:

```
dOmega_orb/dr = -(3/2) * sqrt(M) * r^(1/2) / (r^(3/2) + a*sqrt(M))^2
```

So `(r*dOmega/dr)^2 = (9/4)*M*r^3 / (r^(3/2) + a*sqrt(M))^4`. This differs from the Keplerian approximation `(r*dOmega/dr)^2 ≈ (9/4)*Omega_K^2` by a factor of `[r^{3/2}/(r^{3/2}+a√M)]^4`, which is ~0.78 at r_isco for a=0.998 — a ~22% error in the shear rate and hence in the inferred surface density. Inverting gives `Sigma ∝ F/(nu*(r*dOmega/dr)^2)`. Note: the factor of 2 (two radiating faces) does not affect the proportionality and is absorbed by the normalization.

Normalized so that the midplane vertical optical depth at peak-flux radius equals tau_mid (default 100). See Section 0 for the normalization procedure.

### 1.4 ISCO Taper

```
taper(r) = 1                                              for r >= r_isco
taper(r) = exp(-(r_isco - r)^2 / w_taper^2)               for r < r_isco

w_taper = (r_isco - r_horizon) / 3
```

Smooth Gaussian falloff inside ISCO where gas is plunging, not orbiting. The taper width `w_taper` is scaled to one-third of the coordinate gap between ISCO and horizon. This is necessary because for high spin (a → M), the ISCO approaches the horizon: at a=0.998, r_isco ≈ 1.237M and r_horizon ≈ 1.063M, giving a gap of only 0.174M. A fixed width of 0.5M would extend far beyond the horizon and be nearly flat across the entire plunging region, defeating the purpose of the taper. Scaling to (r_isco - r_+)/3 ensures the taper is always resolved with ~3 e-folding lengths across the plunging region regardless of spin.

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

The 3H cutoff provides a conservative bound for the numerically computed vertical profile (which is more centrally concentrated than a Gaussian, so the density at 3H is negligible). The inner bound extends to the horizon (not ISCO) because the plunging region has nonzero density.

---

## 2. Temperature Model

### 2.1 Radial Effective Temperature

Novikov-Thorne profile for the orbiting region, with a frozen-temperature plunging extension:

```
T_eff(r) = T_peak * (F(r) / F_max)^(1/4)       for r >= r_isco
T_eff(r) = T_plunge                              for r < r_isco
```

where `T_plunge = T_eff(r_isco + epsilon)` is the effective temperature just outside the ISCO (practically, evaluate at the first radial LUT bin outside r_isco).

**Why this is needed:** The Novikov-Thorne flux vanishes exactly at r_isco by the zero-torque inner boundary condition: F(r_isco) = 0. Without the plunging extension, the entire plunging region would have T_eff = 0 — invisible despite having nonzero density (Section 1.4) and nonzero 4-velocity (Section 4.3).

**Physical justification:** Gas spiraling inward through the disk is heated by viscous dissipation. Just outside the ISCO, F(r) is small but nonzero — the gas has a definite temperature. At the ISCO, it detaches from circular orbit and plunges inward. During the plunge, there is no viscous heating (no shear — the gas is in free-fall), but the plunge timescale (~few M in coordinate time) is much shorter than the radiative cooling timescale (t_cool >> M for optically thick gas). So the gas approximately retains the temperature it had when it left the orbiting region.

T_plunge is modest compared to the disk peak (F drops steeply approaching ISCO), so the plunging region is dimmer than the main disk — consistent with the density taper (Section 1.4) producing a smooth, fading inner edge rather than a bright ring.

### 2.2 Vertical Temperature — Eddington Approximation (Precomputed LUT)

The Eddington T-tau relation determines the vertical temperature profile:

```
T^4(r, tau_z) = (3/4) * T_eff(r)^4 * (tau_z + 2/3)
```

where `tau_z` is the **vertical total extinction** optical depth (absorption + scattering) from the disk surface to height z — a property of the atmosphere at position (r, z), independent of viewing angle. The Eddington relation is derived from the radiative transfer moment equations using total extinction, not absorption alone — using absorption-only tau would underestimate the optical depth in scattering-dominated regions (inner disk) and produce an incorrectly steep temperature gradient. This is computed during LUT construction (Section 1.1) as part of the hydrostatic equilibrium solve and stored as a 2D lookup table `T_LUT(r, |z|)` alongside the density profile.

**Critical:** T(r, z) must NOT be computed from the ray's accumulated optical depth during raymarching. The ray's slant optical depth depends on the viewing angle (edge-on rays accumulate far more tau than face-on rays through the same point), but the temperature at a physical point is a property of the atmosphere, not the observer. Using ray tau would make the disk appear hotter at high inclinations — an unphysical artifact.

- Midplane temperature: `T_mid = ((3/4)*(tau_mid + 2/3))^(1/4) * T_eff`
- Surface temperature: `T_surface ~ 0.84 * T_eff`
- During raymarching: look up `T = T_LUT(r, |z|)` at each sample point

### 2.3 Turbulent Temperature Coupling

Dense clumps and sparse pockets adjust temperature via interpolated isothermal-adiabatic relation:

```
tau_local = (kappa_R_LUT(rho_smooth, T) + kappa_es_LUT(rho_smooth, T)) * rho_smooth_CGS * H(r)     [total extinction optical depth across one scale height]
t_cool = (rho * c_s^2) / (sigma_SB * T^4) * tau_local                      [photon diffusion time]
t_turb = 1 / Omega_orb(r)                                                    [turbulent turnover time]
beta = (gamma - 1) * t_cool / (t_cool + t_turb)
T_turb = T * (rho_turb / rho_smooth)^beta
```

- `tau_local` is the total extinction optical depth across one scale height at the smooth (unperturbed) density — this sets the photon diffusion time out of a turbulent clump (scattering redirects photons, keeping them trapped just like absorption does)
- `rho_smooth` is the density without turbulent noise; `rho_turb` includes noise
- gamma = 5/3 (ideal gas)
- beta -> 0 (isothermal) in hot inner disk where cooling is fast
- beta -> 2/3 (adiabatic) in cool outer disk where cooling is slow

---

## 3. Opacity Model

### 3.1 Absorption Opacity (Frequency-Dependent, Monochromatic)

All absorption opacities are evaluated monochromatically at each channel's frequency, not via Rosseland mean coefficients. Three physical processes contribute at optical wavelengths, each dominant in a different temperature regime:

| Process | Dominant regime | Key dependence |
|---|---|---|
| H⁻ (negative hydrogen ion) | 3,000 – 10,000 K | n(H I) · n_e · T |
| Bound-free from ions | 10,000 – 200,000 K | n(ion in bound state) · ν⁻³ |
| Free-free (bremsstrahlung) | > 200,000 K (fully ionized) | n_e · n_i · ν⁻³ · T⁻¹/² |

All three are computed from first principles and summed. To avoid solving Saha equilibrium per ray sample, the combined monochromatic opacity is precomputed as a **3D lookup table** `kappa_abs_LUT(nu, rho, T)` during initialization (see Section 3.1.5).

#### 3.1.1 Ionization Equilibrium (Saha Equation)

The number density of each ion in each ionization state is determined by the Saha equation. For element `s` transitioning from ionization stage `i` to `i+1`:

```
n_{s,i+1} * n_e / n_{s,i} = (2 / Lambda_dB^3) * (U_{s,i+1}(T) / U_{s,i}(T)) * exp(-chi_{s,i} / (k_B * T))
```

where:
- `Lambda_dB = h / sqrt(2 * pi * m_e * k_B * T)` is the thermal de Broglie wavelength [cm]
- `U_{s,i}(T)` is the partition function of species s in ionization stage i (ground-state dominated: U ≈ g_0, the ground-state statistical weight)
- `chi_{s,i}` is the ionization potential from stage i to i+1 [erg]

**Species included:**

| Element | Mass fraction | Ionization stages | chi (eV) | U (ground state) |
|---|---|---|---|---|
| H | X = 0.70 | H I → H II | 13.60 | U(I)=2, U(II)=1 |
| He | Y = 0.28 | He I → He II → He III | 24.59, 54.42 | U(I)=1, U(II)=2, U(III)=1 |
| C | 3.0e-3 | C I–V | 11.26, 24.38, 47.89, 64.49 | tabulated |
| O | 6.6e-3 | O I–V | 13.62, 35.12, 54.93, 77.41 | tabulated |
| Fe | 1.2e-3 | Fe I–V | 7.90, 16.19, 30.65, 54.80 | tabulated |

Higher ionization stages (VI+) have chi > 100 eV and are only populated at T >> 10^6 K where free-free dominates anyway. Partition function values are tabulated at initialization from standard atomic data (NIST ASD); ground-state-only approximation (U ≈ g_0) is acceptable since excited states are sparsely populated at disk temperatures.

**Procedure:** Given (rho, T), solve iteratively for n_e:
1. Initial guess: n_e = n_e,fully_ionized = rho * (1+X) / (2*m_p)
2. For each element, solve the Saha chain to get ionization fractions
3. Compute n_e from charge conservation: n_e = sum over all species of (ionization stage * number density)
4. Iterate until n_e converges to relative tolerance 1e-6 (typically 3-5 iterations)

#### 3.1.2 Free-Free Absorption (Bremsstrahlung)

```
alpha_ff(nu, rho, T) = C_ff * (n_e * n_ion_eff / nu^3) * g_ff(nu, T) * T^(-1/2) * (1 - exp(-h*nu/(k_B*T)))     [cm^{-1}]
```

where:
- `C_ff = 3.69e8` CGS (Rybicki & Lightman eq. 5.18a)
- `n_ion_eff = sum_s sum_i Z_i^2 * n_{s,i}` is the effective ion density weighted by charge squared (free-free scales as Z²)
- `n_e` is the electron density from Saha (Section 3.1.1), NOT the fully-ionized approximation
- The `(1 - exp(-h*nu/(k_B*T)))` factor is the **stimulated emission correction**

**Gaunt factor:**

```
g_ff(nu, T) = max(1.0, (sqrt(3)/pi) * ln(4 * k_B * T / (gamma_E * h * nu)))
```

where gamma_E = exp(0.5772) = 1.7811 (exponential of the Euler-Mascheroni constant). This is the classical asymptotic form (Rybicki & Lightman eq. 5.19b, Karzas & Latter 1961). The factor `4/gamma_E ≈ 2.25` in the logarithm argument is important — omitting it underestimates g_ff by ~15-40%. For optical frequencies and typical disk temperatures: g_ff ≈ 1-5.

**Stimulated emission effect:** In the Wien limit (hν >> k_BT), the correction → 1 and opacity scales as ν⁻³. In the Rayleigh-Jeans limit (hν << k_BT, typical for optical wavelengths in the hot inner disk where T ~ 10^7 K), the correction → hν/(k_BT) and opacity scales as ν⁻². This reduces the color difference between channels in the hot inner disk.

Note: n_e and n_ion_eff change with ionization state (from Saha), so the free-free opacity has a non-trivial temperature dependence beyond the explicit T^{-1/2} — partial ionization reduces n_e and n_ion_eff, reducing free-free opacity in the cool outer disk. This is automatically captured by using the Saha-derived densities.

#### 3.1.3 H⁻ Opacity (Negative Hydrogen Ion)

H⁻ is the **dominant opacity source at optical wavelengths** for 3,000 < T < 10,000 K — the regime relevant for cool outer disk regions, especially around supermassive black holes. H⁻ has a bound electron with binding energy 0.754 eV (threshold at 1.64 μm), so it absorbs all visible light.

**H⁻ number density** from Saha equilibrium (H I + e⁻ ↔ H⁻):

The standard Saha equation for ionization (H⁻ → H I + e⁻) is:

```
n(H I) * n_e / n(H⁻) = (2 / Lambda_dB^3) * (U(H I) / U(H⁻)) * exp(-chi_H⁻ / (k_B*T))
```

where the factor of 2 accounts for the two spin states of the free electron (g_e = 2). Inverting:

```
n(H⁻) / n(H I) = n_e * Lambda_dB^3 * U(H⁻) / (2 * U(H I)) * exp(chi_H⁻ / (k_B*T))
                = (n_e / T^(3/2)) * (h^3 / (2*pi*m_e*k_B)^(3/2)) * (1/2) * (g(H⁻)/g(H I)) * exp(chi_H⁻ / (k_B*T))
```

where chi_H⁻ = 0.754 eV = 1.208e-12 erg (electron affinity), g(H⁻) = 1, g(H I) = 2. The effective statistical weight ratio is g(H⁻)/(2·g(H I)) = 1/4 (the factor of 2 in the denominator is the electron spin degeneracy from inverting the Saha equation, distinct from g(H I) = 2).

**H⁻ bound-free cross-section** (Wishart 1979, refined by John 1988):

The H⁻ photodetachment cross-section is a smooth function of wavelength with threshold at lambda_0 = 1.6419 μm, peaking at ~4.0×10⁻¹⁷ cm² near 850 nm. Rather than use a polynomial fit (which varies between references and is easy to get wrong), we use the **tabulated cross-sections from Wishart (1979) Table 2**, linearly interpolated in lambda:

```
sigma_bf_Hminus(lambda):     [cm^2 per H⁻ ion]
  lambda_0 = 1.6419 μm      (threshold, sigma = 0)
  Key values at optical wavelengths:
    400 nm: 1.63e-17 cm^2
    450 nm: 2.33e-17
    500 nm: 3.02e-17
    550 nm: 3.58e-17
    600 nm: 3.96e-17
    650 nm: 4.14e-17
    700 nm: 4.09e-17
    850 nm: 3.98e-17     (peak)
```

Store as a 1D lookup table (50 wavelength bins from 200 nm to 1642 nm), interpolated during opacity LUT construction. Total size: negligible (400 bytes).

**H⁻ free-free absorption** (Bell & Berrington 1987):

```
alpha_ff_Hminus(nu, T) = n(H I) * n_e * sigma_ff_Hminus(lambda, T)     [cm^{-1}]
```

The H⁻ free-free cross-section sigma_ff_Hminus(lambda, T) [cm⁵] is tabulated from Bell & Berrington (1987) Table 1 on a (lambda, T) grid: 10 wavelength points × 12 temperature points from 1,000 to 10,080 K. Stored as a 2D lookup table and bilinearly interpolated during opacity LUT construction. Total size: negligible (~1 KB).

**Total H⁻ absorption coefficient:**

```
alpha_Hminus(nu, rho, T) = n(H⁻) * sigma_bf_Hminus(lambda) * (1 - exp(-h*nu/(k_B*T)))
                         + n(H I) * n_e * sigma_ff_Hminus(lambda, T)     [cm^{-1}]
```

The stimulated emission correction applies to bound-free but is already included in the free-free tabulation.

#### 3.1.4 Ion Bound-Free Absorption (Photoionization)

For each ion species `(s, i)` with at least one bound electron, the monochromatic bound-free cross-section above the ionization threshold is:

```
sigma_bf(nu, s, i) = sigma_0(s, i) * (nu_threshold / nu)^3 * g_bf     [cm^2 per ion]
```

where:
- `nu_threshold = chi_{s,i} / h` is the threshold frequency for ionization from the ground state
- `sigma_0` is the threshold cross-section. For hydrogenic ions (Kramers formula, R&L eq. 10.39):
  ```
  sigma_0(n, Z_eff) = 7.91e-18 * n / Z_eff^2     [cm^2]
  ```
  where n is the principal quantum number of the outermost electron and Z_eff is the effective nuclear charge seen by that electron. The full frequency-dependent Kramers cross-section scales as Z_eff^4 / (n^5 * nu^3), which gives sigma_0 = 7.91e-18 * n/Z_eff^2 at threshold where nu = Z_eff^2 * Ry / (h * n^2). For non-hydrogenic ions (Fe, etc.), use tabulated threshold cross-sections from TOPbase/Opacity Project.
- `g_bf ≈ 1` (bound-free Gaunt factor, weakly frequency-dependent)
- `sigma_bf = 0` for `nu < nu_threshold`

**Total ion bound-free:**

```
alpha_bf_ion(nu, rho, T) = sum_s sum_i n_{s,i}(rho, T) * sigma_bf(nu, s, i) * (1 - exp(-h*nu/(k_B*T)))     [cm^{-1}]
```

where `n_{s,i}` is the number density of species s in ionization stage i from Saha (Section 3.1.1). The sum runs over all species and ionization stages with `nu > nu_threshold` (i.e., the photon has enough energy to ionize from the ground state).

**Which edges matter at optical wavelengths (1.9 – 2.75 eV)?** Most ionization thresholds for the included species are in the UV/EUV (> 7 eV). At optical wavelengths, only the ν⁻³ tails of these edges contribute. The dominant contributors are:
- Fe I (chi = 7.90 eV, threshold at 157 nm): ν⁻³ tail reaches optical as (157/550)³ ~ 0.023 of threshold value
- C I (chi = 11.26 eV): weaker tail
- O I (chi = 13.62 eV): weaker tail
- H I (chi = 13.60 eV): similar to O I

These tails are individually small but collectively significant in the partial-ionization zone (10⁴ – 10⁵ K) where n_{s,i} is large for neutral/singly-ionized species.

#### 3.1.5 Combined Absorption and Opacity LUT

**Total monochromatic absorption coefficient:**

```
alpha_abs(nu, rho, T) = alpha_ff(nu, rho, T) + alpha_Hminus(nu, rho, T) + alpha_bf_ion(nu, rho, T)     [cm^{-1}]
kappa_abs(nu, rho, T) = alpha_abs / rho_CGS     [cm^2/g]
```

**Precomputed opacity LUT:** Evaluating Saha equilibrium per ray sample is too expensive for raymarching. Instead, precompute a 3D lookup table during initialization:

```
kappa_abs_LUT(nu_index, rho_index, T_index)     [cm^2/g]
```

- **20 frequency bins**: log-spaced from 10¹⁴ to 10¹⁶ Hz. This covers the optical range (4.6–6.7×10¹⁴ Hz for 450–650 nm) plus sufficient margin for Doppler-shifted emitter-frame frequencies (blueshifts up to g ≈ 2–3 near the inner disk push optical frequencies to ~2×10¹⁵ Hz)
- **100 density bins**: log-spaced from rho_min to rho_max (covering the full disk density range)
- **100 temperature bins**: log-spaced from T_min = 3,000 K to T_max = 10⁸ K
- **Total size**: 20 × 100 × 100 × 8 bytes = **1.6 MB** (global memory, texture-cached on GPU)

During raymarching, opacity is looked up via trilinear interpolation in (log nu_emit, log rho, log T). The emitter-frame frequency `nu_emit = g * nu_obs` is computed per sample point (Section 4.1), and the dense frequency grid allows direct interpolation without approximate ν⁻³ scaling corrections.

### 3.2 Thomson Electron Scattering (Frequency-Independent)

```
kappa_es(rho, T) = sigma_T * n_e(rho, T) / rho_CGS     [cm^2/g]
```

where sigma_T = 6.652e-25 cm² is the Thomson cross-section and n_e is the Saha-derived electron density (Section 3.1.1). For fully ionized solar composition, this gives the familiar kappa_es = 0.34 cm²/g. In the partially ionized outer disk (T < 10,000 K), n_e drops significantly, reducing scattering opacity — this is physically important because it changes the scattering-to-absorption ratio (epsilon) and the Eddington temperature profile.

**Precomputed:** kappa_es depends only on (rho, T) through the Saha electron density. Store as a 2D lookup table `kappa_es_LUT(rho, T)` alongside the 3D absorption opacity LUT, sharing the same density and temperature grids:

```
kappa_es_LUT(rho_index, T_index)     [cm^2/g]
- 100 density bins × 100 temperature bins (same grid as kappa_abs_LUT)
- Total size: 100 × 100 × 8 bytes = 80 KB
```

### 3.3 Total Extinction and Photon Destruction Probability

The radiative transfer (Section 4.1) uses total extinction (absorption + scattering) with an epsilon-weighted source function:

```
kappa_total(nu, rho, T) = kappa_abs(nu, rho, T) + kappa_es(rho, T)           [total extinction, cm^2/g]
epsilon(nu, rho, T)     = kappa_abs(nu, rho, T) / kappa_total(nu, rho, T)    [photon destruction probability]
```

Both kappa_abs and kappa_es are looked up from their respective precomputed LUTs (Sections 3.1.5 and 3.2).

epsilon → 1 in absorption-dominated regions (outer disk, red channel) — the source function approaches pure Planck. epsilon → 0 in scattering-dominated regions (inner disk, blue channel) — emission is suppressed relative to extinction. In the partially ionized outer disk, the drop in n_e reduces kappa_es faster than kappa_abs (which is dominated by H⁻), pushing epsilon toward 1 — the cool outer disk is absorption-dominated at all wavelengths.

### 3.4 Optical Depth Accumulation

Mass opacities (cm^2/g) are multiplied by rho_CGS * ds_proper to get optical depth increments:

```
dtau_total = kappa_total * rho_CGS * ds_proper     [extinction optical depth, for transfer equation]
```

Since rho is stored in geometric normalization and ds is in geometric units, the product `rho * ds` is converted via the rho_scale factor established in Section 0.

Note: the vertical total extinction optical depth used in the Eddington T-tau relation (Section 2.2) is computed only during **LUT construction** (Section 1.1). It is NOT accumulated during raymarching — temperature is looked up from the precomputed LUT.

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
1. Look up local thermodynamic state from precomputed LUTs:
   z = r * cos(theta)
   rho = rho_LUT(r, |z|) * taper(r) * (1 + delta * noise3D(r/H, z/H, phi))
   T   = T_LUT(r, |z|)
   Apply turbulent coupling: T_turb (Section 2.3)

2. Compute emitter-frame frequency for each channel:
   nu_emit(lambda) = g * nu_obs(lambda)        where nu_obs = c / lambda
   lambda_emit = lambda / g

3. Evaluate opacity at emitter-frame frequency via LUTs (Sections 3.1.5, 3.2):
   kappa_abs_emit(lambda) = kappa_abs_LUT(nu_emit, rho_CGS, T_turb)    [trilinear interpolation in log space]
   kappa_es_local          = kappa_es_LUT(rho_CGS, T_turb)              [bilinear interpolation in log space]
   kappa_total_emit(lambda) = kappa_abs_emit(lambda) + kappa_es_local
   epsilon(lambda) = kappa_abs_emit(lambda) / kappa_total_emit(lambda)   [photon destruction probability]

4. Compute optical depth increments (in emitter frame):
   dtau_total(lambda) = kappa_total_emit(lambda) * rho_CGS * ds_proper

   where ds_proper is the proper distance along the ray in the emitter frame.
   For a comoving emitter: ds_proper = |p_mu u^mu_emit| * ds_affine * (L_unit)
   Here ds_affine is the geodesic affine parameter step (called `ds` in Section 5.2),
   not to be confused with optical wavelength lambda.

5. Compute invariant source function (absorption-only emission, scattering is conservative):
   For each channel lambda in {B, G, R}:
       S(lambda) = epsilon(lambda) * B_nu(nu_emit, T_turb) / nu_emit^3

   where B_nu is the **frequency-form** Planck function (Section 4.2). This is critical:
   the invariant J = I_nu/nu^3, so S must use B_nu (per steradian per Hz), NOT B_lambda
   (per steradian per cm). Using B_lambda would introduce an error of order nu^2/c ~ 10^19.

   The epsilon factor is critical: only absorbed photons are thermally re-emitted. Scattered
   photons are redirected but not thermalized. Using S = B_nu/nu^3 without epsilon would
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

### 4.2 Planck Function (Frequency Form)

The invariant source function `S = ε · B_ν / ν³` requires the **frequency-form** Planck function B_ν, not the wavelength-form B_λ. Using B_λ instead of B_ν introduces an error factor of ν²/c (roughly 10¹⁹ at optical frequencies), which would make renders completely wrong.

```
B_nu(nu, T) = (2*h*nu^3 / c^2) / (exp(h*nu / (k_B*T)) - 1)     [erg / (cm^2 s Hz sr)]
```

This is evaluated at the emitter-frame frequency `nu_emit` for each channel. The relationship to the wavelength form is `B_nu = B_lambda * lambda^2 / c`, but since the invariant RT equation is formulated entirely in frequency space, B_nu is used directly — no conversion needed.

The three channel intensities (B, G, R) are normalized and mapped to sRGB at the end of the ray.

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

Derived from the metric normalization condition `g_μν u^μ u^ν = -1` with `u^μ = u^t(1, 0, 0, Ω_orb)`:

```
Omega_orb = sqrt(M) / (r^(3/2) + a*sqrt(M))       [Kerr prograde orbital frequency]

g_tt_eq   = -(1 - 2*M/r)                          [equatorial Kerr metric components]
g_tphi_eq = -2*M*a/r
g_phph_eq = r^2 + a^2 + 2*M*a^2/r

u^t_emit   = 1 / sqrt(-(g_tt_eq + 2*g_tphi_eq*Omega_orb + g_phph_eq*Omega_orb^2))
u^phi_emit = Omega_orb * u^t_emit
u^r_emit   = 0
u^theta_emit = 0
```

Note: The sometimes-seen shortcut `u^t = 1/sqrt(1 - 3M/r + 2a*sqrt(M/r^3))` is the BPT72 energy normalization factor, NOT u^t. Using it for u^t introduces ~3% error at moderate spin, growing larger near ISCO.

**Emitter 4-velocity — plunging geodesic (r < r_isco):**

Gas inside the ISCO follows timelike geodesics with conserved energy and angular momentum from the last stable orbit:

```
v_isco = sqrt(M / r_isco),  a_star = a / M

E_isco = (1 - 2*v_isco^2 + a_star*v_isco^3) / sqrt(1 - 3*v_isco^2 + 2*a_star*v_isco^3)
L_isco = sqrt(M*r_isco) * (1 - 2*a_star*v_isco^3 + a_star^2*v_isco^4) / sqrt(1 - 3*v_isco^2 + 2*a_star*v_isco^3)
```

These are the Bardeen-Press-Teukolsky (1972) specific energy and angular momentum for circular prograde orbits, evaluated at r_isco.

The contravariant 4-velocity is obtained by raising indices with the inverse Kerr metric. At the equatorial plane (using equatorial inverse metric components):

```
E = E_isco,  L = L_isco

u^t     = [E*(r^2 + a^2 + 2*M*a^2/r) - 2*M*a*L/r] / Delta
u^phi   = [L*(1 - 2*M/r) + 2*M*a*E/r] / Delta
u^r     = -sqrt(max(0, R(r))) / Sigma
u^theta = 0
```

where `Sigma = r^2 + a^2*cos^2(theta)` uses the actual theta of the sample point (not equatorial).

Derivation: `u^t = g^{tt}*(-E) + g^{t phi}*L` and `u^phi = g^{phi t}*(-E) + g^{phi phi}*L`, with equatorial inverse metric:
```
g^{tt}    = -(r^2 + a^2 + 2*M*a^2/r) / Delta
g^{t phi} = -2*M*a / (r * Delta)
g^{phi phi} = (1 - 2*M/r) / Delta
```

where `R(r) = (E*(r^2+a^2) - a*L)^2 - Delta*(r^2 + (L - a*E)^2)` is the Kerr radial potential.

**Off-midplane approximation:** The u^t and u^phi formulas use equatorial inverse metric components (derived for theta = pi/2). For volumetric disk samples at z != 0, we evaluate these at the same r, which is standard practice in GRMHD post-processing — the gas velocity is dominated by the orbital motion, and vertical velocities (thermal, turbulent) are subsonic and contribute negligibly to the Doppler shift. The u^r formula uses the full Sigma (with actual theta) rather than r^2, providing the leading-order GR correction for off-midplane points. The same off-midplane approximation applies to circular orbit u^t and u^phi above.

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
   - Look up density and temperature from LUT at current (r, |z|)
   - Evaluate opacity and redshift at new position
   - Update three-channel radiative transfer
3. Adaptive step control based on **two constraints** — optical depth resolution AND geodesic accuracy:
   ```
   dtau_ref = (kappa_abs_LUT(nu_G_emit, rho_CGS, T_turb) + kappa_es_LUT(rho_CGS, T_turb)) * rho_CGS * ds_proper

   # Optical depth constraint (same as before):
   ds_tau = ds
   if dtau_ref > 0.1:  ds_tau = ds * 0.5
   if dtau_ref < 0.01: ds_tau = ds * 2.0

   # Geodesic accuracy constraint: near the horizon, spacetime curvature is strong
   # and large steps cause the RK4 geodesic to drift off the true null path.
   # Scale step with distance to horizon (where curvature ~ 1/r^2):
   ds_geo = 0.1 * max(r - r_horizon, 0.5*M)

   ds = min(ds_tau, ds_geo)
   ds = clamp(ds, H(r)/64, H(r))
   ```

   The geodesic accuracy constraint is critical near the horizon where curvature is strongest. Without it, the RK4 integrator accumulates significant position error during raymarching (the fixed H/8 default step is far too large at r ~ r_horizon), causing rays to miss or double-count emission. The H/64 minimum (reduced from H/32) allows adequate resolution in the high-curvature inner region.
4. Continue until exit condition.

### 5.3 Exit Conditions

- Ray leaves disk volume: `|z| > 3*H(r)` or `r > r_outer` or `r < r_horizon`
- Fully opaque: all three channels have `tau_total > 10`
- Step cap: 4096 steps reached (edge-on views through the full disk chord at high inclination can require thousands of steps, especially for thick inner disks with small adaptive step sizes near the horizon; 512 is insufficient and causes premature termination with visible banding artifacts)

### 5.4 Resume Normal Integration

After exiting the disk, resume the standard adaptive RK4 geodesic integrator. The ray may re-enter the disk (e.g., gravitational lensing bends it back through), in which case raymarching resumes. The invariant `J` is **not** reset on re-entry — it carries the accumulated emission/absorption from all previous disk crossings. Temperature and density are always looked up from the precomputed (r, z) LUT, so no per-ray state needs resetting between crossings.

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
| `include/grrt/color/opacity.h` | Opacity model: Saha solver, ff/bf/H⁻ cross-sections, LUT builder |
| `src/opacity.cpp` | CPU opacity implementation and LUT construction |
| `include/grrt/color/atomic_data.h` | Ionization potentials, partition functions, threshold cross-sections |
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
| Vertical temperature profile 2D LUT (500 r-bins × 64 z-bins, doubles) | ~250 KB |
| Absorption opacity 3D LUT (20 freq × 100 rho × 100 T, doubles) | ~1.6 MB |
| Scattering opacity 2D LUT (100 rho × 100 T, doubles) | ~80 KB |
| Rosseland mean absorption opacity 2D LUT (100 rho × 100 T, doubles) | ~80 KB |
| Mean molecular weight 2D LUT (100 rho × 100 T, doubles) | ~80 KB |

All LUTs are too large for constant memory and are stored in global device memory with texture cache for interpolation. Vertical profile LUTs accessed via `tex2D` for hardware-accelerated bilinear interpolation on (r, z/H) coordinates. Opacity LUT accessed via `tex3D` for trilinear interpolation on (log nu, log rho, log T).

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
| mu_fully_ionized | 0.6 | dimensionless | Mean molecular weight reference (fully ionized H+He); actual mu from mu_LUT(rho,T) |
| gamma_gas | 5.0/3.0 | dimensionless | Adiabatic index (ideal gas) |
| X_hydrogen | 0.7 | dimensionless | Hydrogen mass fraction |
| Z_metal | 0.02 | dimensionless | Metal mass fraction (solar) |
| C_ff | 3.69e8 | CGS | Free-free monochromatic coefficient (R&L eq. 5.18a) |
| m_e | 9.10938e-28 | g | Electron mass (for de Broglie wavelength in Saha) |
| e_charge | 4.80326e-10 | esu | Elementary charge (CGS Gaussian) |
| chi_H | 2.179e-11 | erg | H I ionization potential (13.60 eV) |
| chi_Hminus | 1.208e-12 | erg | H⁻ electron affinity (0.754 eV) |
| eV_to_erg | 1.602e-12 | erg/eV | Unit conversion for ionization potentials |
| sigma_T | 6.652e-25 | cm^2 | Thomson scattering cross-section |
| gamma_E | 1.7811 | dimensionless | exp(Euler-Mascheroni), for Gaunt factor |
| Ry | 2.180e-11 | erg | Rydberg energy (13.6 eV) |

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
