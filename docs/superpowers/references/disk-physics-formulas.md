# Verified disk-physics formula reference

**Created:** 2026-06-01 · **Branch:** fix/volumetric-ring
**Purpose:** Single authoritative, **externally-verified** list of every load-bearing physics formula in the Approach-A disk redesign (`2026-06-01-disk-first-principles-vertical-structure-design.md`). Implementers and code reviewers must check formulas against this file rather than re-deriving from memory.

**Why this exists:** during Task 1, a code reviewer "corrected" the Eddington formula and **dropped a factor of `c`** (wrote `L_Edd ∝ 4πGMm_p/σ_T` instead of `4πGMm_p·c/σ_T`). The original was right; the "fix" was wrong. Re-derivation from memory is error-prone — verify here.

**Units convention:** physics formulas below are in **CGS** unless marked geometric. Geometric quantities (lengths in units of `M`, frequencies in units of `1/M`) convert to CGS via the length scale `r_g` (see §1 and the conversion traps in §19).

---

## 1. Gravitational radius (length scale)
```
r_g = G M / c²            [cm]          M = mass_solar · M_sun [g]
```
For 10 M_⊙: `r_g ≈ 1.477e6 cm ≈ 14.8 km`. Geometric length `L_geom` (units of M) → cm: `L_cm = L_geom · r_g`.
*Standard (Schwarzschild radius `r_s = 2GM/c² = 2 r_g`).*

## 2. Eddington luminosity ⚠ factor-`c` trap
```
L_Edd = 4π G M m_p c / σ_T            [erg/s]
```
**The `c` is in the numerator.** Do not drop it. Balances outward radiation force on electrons (Thomson) against inward gravity on protons.
*Verified: [Wikipedia](https://en.wikipedia.org/wiki/Eddington_luminosity), [Wolfram](https://scienceworld.wolfram.com/physics/EddingtonLuminosity.html).*

## 3. Radiative efficiency
```
η ≡ L / (Ṁ c²)                        [dimensionless]
η = 1 − E_isco                        (thin disk: rest energy minus ISCO binding energy)
```
`E_isco` = specific orbital energy at the ISCO (Kerr; Bardeen–Press–Teukolsky 1972). Schwarzschild: `E_isco = 2√2/3 ≈ 0.9428 → η ≈ 0.057`. Extremal Kerr (a→M): `E_isco → 1/√3 ≈ 0.577 → η ≈ 0.42`.
*Verified: [η ≡ L/Ṁc² (arXiv:1012.3213)](https://arxiv.org/pdf/1012.3213). Code: `VolumetricDisk::E_isco_` (volumetric_disk.cpp:56-65).*

## 4. Accretion rate from Eddington fraction
```
Ṁ = f_Edd · L_Edd / (η c²)            [g/s]
```
Here `f_Edd ≡ L / L_Edd` is the **Eddington luminosity ratio** (at `f_Edd = 1` the disk radiates at `L_Edd`). Follows directly from §3 with `L = f_Edd L_Edd`. Special case η=0.1 gives the textbook `Ṁ_Edd = 10 L_Edd/c²`.
*Verified: same sources as §3. Code: Task 3 of the foundation plan.*

## 5. Radial flux (Novikov–Thorne / Shakura–Sunyaev)
```
Newtonian (SS73):   F(r) = (3 G M Ṁ / 8π r³) · [1 − √(r_in/r)]      [erg/cm²/s]
Relativistic (NT73): F(r) = (Ṁ c² / 4π r_g² ) · f_NT(r, a)
```
`f_NT` = the relativistic flux-shape function (Novikov–Thorne 1973), implemented as the disk's flux shape (`build_flux_lut`). Anchoring by `Ṁ` (§4) makes `F(r)` absolute. The `[1 − √(r_in/r)]` factor is the zero-torque inner boundary condition.
*Verified: [Novikov–Thorne model](https://www.emergentmind.com/topics/novikov-thorne-accretion-disk-model). Code: `build_flux_lut`, `flux_at(r)`.*

## 6. Effective temperature
```
T_eff(r) = (F(r) / σ_SB)^(1/4)        [K]
```
*Standard Stefan–Boltzmann. Code: `T_eff_lut_`, volumetric_disk.cpp:446.*

## 7. Vertical hydrostatic equilibrium (thin disk)
```
dP/dz = − ρ · Ω_z²(r) · z
```
Vertical gravity is the **central-mass tidal term** `g_z = −Ω_z² z` (NOT disk self-gravity — see §16). `Ω_z` = vertical epicyclic frequency (Kerr: `omega_z_sq`). `P` = total pressure (§10).
*Standard thin-disk result ([UToledo notes](http://astro1.physics.utoledo.edu/~megeath/ph6820/lecture10_eqn.pdf)). Code: `omega_z_sq`.*

## 8. Viscous energy generation (Shakura–Sunyaev α-stress)
```
dF/dz = (3/2) · α · Ω(r) · P
```
From stress `t_rφ = α P` and Keplerian shear `r|dΩ/dr| = (3/2)Ω`: dissipation per unit volume `q⁺ = t_rφ · r|dΩ/dr| = (3/2) α Ω P`. `Ω` = orbital angular velocity (`omega_orb`).
*Verified: Shakura–Sunyaev 1973; [Spruit accretion-disk notes](https://wwwmpa.mpa-garching.mpg.de/~henk/pub/disksn.pdf). The (3/2)Ω shear factor is standard.*

## 9. Radiative diffusion (grey, Rosseland) ⚠ factor trap
```
F = − (4 a c T³) / (3 κ_R ρ) · dT/dz        ( = − (16 σ_SB T³)/(3 κ_R ρ) · dT/dz,  since a = 4σ_SB/c )
⇔  dT/dz = − 3 κ_R ρ F / (4 a c T³)
⇔  d(a T⁴)/dz = − 3 κ_R ρ F / c
```
`κ_R` = **Rosseland-mean** opacity (§14), not monochromatic. The three forms are algebraically identical.
*Verified: [Wikipedia radiative zone](https://en.wikipedia.org/wiki/Radiative_zone), stellar-structure standard (Kippenhahn).*

## 10. Equation of state (gas + radiation)
```
P = P_gas + P_rad = ρ k_B T / (μ m_p) + (1/3) a T⁴        [erg/cm³ = dyn/cm²]
```
`μ` = mean molecular weight (`lookup_mu`, from Saha). Gas-pressure-dominated when `P_gas/P_rad ≫ 1`.
*Standard.*

## 11. Radiation constant & radiation pressure
```
a = 4 σ_SB / c ≈ 7.5657e-15  [erg cm⁻³ K⁻⁴]      (radiation/Stefan constant)
P_rad = (1/3) a T⁴
```
*Standard (Rybicki & Lightman). Code: `a_rad = 4·sigma_SB/c_cgs` (constants.h:26).*

## 12. Optical depth & photosphere
```
dτ/dz = − κ_R ρ          (τ measured inward from the surface, τ = 0 at the surface)
photosphere:  τ = 2/3
```
*Verified: [Grey atmosphere — Wikipedia](https://en.wikipedia.org/wiki/Grey_atmosphere) (Eddington–Barbier, `τ = 2/3`).*

## 13. Grey-atmosphere temperature law (Eddington approximation)
```
T⁴(τ) = (3/4) T_eff⁴ · (τ + 2/3)        ⇒  T = T_eff at τ = 2/3
```
NOTE: Approach A does **not** assume this as a closure — it solves §9 for `T(z)` instead. This law is the seed/limit check and the surface boundary condition `T(τ=2/3)=T_eff`.
*Verified: [Grey atmosphere — Wikipedia](https://en.wikipedia.org/wiki/Grey_atmosphere), [St Andrews GRAY notes](http://www-star.st-andrews.ac.uk/~kw25/teaching/stars/GRAY.pdf).*

## 14. Rosseland mean opacity
```
1/κ_R = [ ∫ (1/κ_ν) (∂B_ν/∂T) dν ] / [ ∫ (∂B_ν/∂T) dν ]
```
Harmonic mean weighted by `∂B_ν/∂T` — the exact frequency-average for **total radiative flux** in the diffusion limit. Frequency is integrated out once at table-build time (finite ~50-point quadrature), so the structure solver uses one `κ_R(ρ,T)`.
*Standard (Mihalas). Code: `kappa_ross_lut`, opacity.cpp:285-299 (50-point quadrature).*

## 15. Sound speed & gas-pressure scale height
```
c_s² (gas) = k_B T / (μ m_p)                     [cm²/s²]
H ≈ c_s / Ω_z         (gas-dominated isothermal limit → Gaussian profile of width ~H)
```
*Standard. Used as the Newton seed and the analytic limiting-case test.*

## 15b. Characteristic midplane density (opacity-table sizing) — mass-adaptive
```
ρ_est = Ṁ · Ω² / (6π α c_s³)        [g/cm³]      c_s² = k_B T_peak/(μ m_p),  Ω in 1/s
```
Standard α-disk midplane density `ρ ~ Σ/2H` with `Σ = Ṁ/(3πν)`, `ν = α c_s²/Ω`, `H = c_s/Ω`. Used to **auto-size the opacity table's density range to the black-hole mass** (the disk's real density scales `∝ M^-0.6…-0.7` — Shakura-Sunyaev — so a *fixed* table range cannot span sub-stellar→supermassive). Evaluate Ω at a representative inner radius (e.g. `r_isco`, converted to CGS via `Ω_cgs = Ω_geom·c/r_g`); `μ ≈ 0.6` (the opacity table isn't built yet). Then `rho_max = ρ_est·10²`, `rho_min = ρ_est·10⁻¹⁶` (bracketing margins: radial spread above, photosphere falloff below), `n_rho = max(20, ⌈10·log₁₀(rho_max/rho_min)⌉)` (~10 bins/decade). **Verify mass-scaling:** `ρ_est ∝ M^-5/8` (from Ṁ∝M, Ω²∝M⁻¹ at r∝r_g, c_s³∝M^-3/8) — matches SS. A post-BVP guard warns if any real density falls outside the table.
*Composition of standard α-disk relations (Shakura-Sunyaev 1973; Frank, King & Raine). Note: the **density axis is forgiving** — opacity is ~power-law in ρ (κ_es∝ρ⁰, κ_ff/κ_bf∝ρ¹) → straight in log-ρ → log-bilinear interpolation is near-exact; the temperature axis (ionization/iron bumps) needs the fine resolution.*

## 16. Toomre Q (self-gravity check)
```
Q = c_s κ_epi / (π G Σ)  ≈  c_s Ω / (π G Σ)        (κ_epi ≈ Ω for near-Keplerian)
Q ≫ 1  ⇒  disk self-gravity negligible (only the central-mass tidal term in §7)
Q ≲ 1  ⇒  gravitationally unstable / fragments  (NOT the rendered inner-disk regime)
```
*Verified: [Toomre's stability criterion — Wikipedia](https://en.wikipedia.org/wiki/Toomre%27s_stability_criterion). Used as a validation warning only.*

## 17. Vertically-integrated α-disk energy closure
```
σ_SB T_eff⁴ = (3/2) α Ω ∫ P dz
```
Follows from integrating §8 over the column with flux BCs `F(midplane)=0`, `F(surface)=σ_SB T_eff⁴`. This is *why* Σ, ρ_mid, τ_mid are emergent outputs of the BVP rather than inputs (spec §3, §6).

## 18. Physical constants (CGS) — from `include/grrt/math/constants.h`
| Symbol | Value | Units |
|---|---|---|
| `G` | 6.674e-8 | cm³ g⁻¹ s⁻² |
| `c` | 2.997924e10 | cm/s |
| `M_sun` | 1.989e33 | g |
| `m_p` | 1.672622e-24 | g |
| `σ_T` (Thomson) | 6.652e-25 | cm² |
| `σ_SB` (Stefan–Boltzmann) | 5.670374e-5 | erg cm⁻² s⁻¹ K⁻⁴ |
| `a` (radiation) | 4σ_SB/c ≈ 7.566e-15 | erg cm⁻³ K⁻⁴ |
| `k_B` | 1.380649e-16 | erg/K |
| `h` (Planck) | 6.626070e-27 | erg·s |

## 19. Unit-conversion traps (geometric ↔ CGS)
The integrator works in geometric units (`G=c=1`, `M=1`); the column physics is CGS. Convert via `r_g`:
- **Length:** `z_cm = z_geom · r_g`, `dz_cm = dz_geom · r_g`.
- **Frequency / rate:** geometric time unit is `r_g/c`, so `Ω_cgs = Ω_geom · c / r_g` (same for `Ω_z`).
- **Optical depth must use CGS lengths:** `τ = ∫ κ[cm²/g] ρ[g/cm³] dz_cm[cm]`. Using geometric `dz` against CGS `κρ` is the original root-cause bug (no length scale) Approach A fixes.

## 20. Vertical-structure BVP (the Approach-A column solver)
The grey vertical-structure two-point boundary value problem, **verified against the published open-source formulation** (see credits below). Independent variable: height `z ∈ [0, z₀]` (midplane → surface); solved in practice on the **column-mass fraction** `q = 1 − Σ/Σ₀ ∈ [0,1]`. Variables `P, Q(≡F), T, Σ`; `ρ` from the EOS (§10).

⚠ **Two distinct frequencies** (the published Newtonian reference uses a single `ω_k` for both; for Kerr they SPLIT and must NOT be conflated): **`Ω`** = orbital angular velocity (drives the viscous *shear* → heating); **`Ω_z`** = vertical epicyclic frequency (the *vertical gravity* `g_z=Ω_z²z`). Newtonian: `Ω = Ω_z`. Kerr: `Ω_z² = Ω²(1 − 4a√M/r^{3/2} + 3a²/r²) ≠ Ω²`. (Code: `omega_orb` vs `omega_z`.)

**Four ODEs:**
```
dP/dz = −ρ Ω_z² z                                   hydrostatic — vertical gravity uses Ω_z (§7)
dQ/dz = α P · |r dΩ/dr|                             viscous flux generation — EXACT Kerr shear (Newtonian (3/2)Ω only when a=0) (§8)
dlnT/dlnP = ∇_rad = 3 κ_R P Q /(16 σ Ω_z² z T⁴)     radiative diffusion (§9, equivalent to dT⁴/dτ = 3F/4σ)
dΣ/dz = −2 ρ                                        column mass (factor 2 = both disc faces)
```
**Five boundary conditions** (3 at the surface + the surface-pressure condition + 1 at the midplane):
```
midplane  z=0 :  Q = 0                              flux symmetry
surface   z=z₀:  Q = σ_SB T_eff⁴                    all flux escaped
                 T = T_eff                          photosphere temperature
                 Σ = 0                              no mass above the surface
                 P = (2/3) Ω_z² z₀ / κ_R            ⚠ surface pressure from τ=2/3 (g·τ/κ, g=Ω_z²z₀)
```
**Unknowns balance:** midplane `(P₀, T₀, Σ₀)` + thickness `z₀` = 4, matched by the 4 surface conditions. **Emergent outputs:** `Σ₀` (surface density → `ρ_mid`), `z₀` (= `z_max`), `τ_mid = ∫κρ dz`, and the `ρ(z)`, `T(z)` profiles. The surface-pressure BC is the trap: it is *where `τ=2/3` enters as a constraint* and is what pins the free parameters — easy to omit.

**Credits.** This formulation follows the standard disc vertical-structure treatment of **Hubeny 1990** (ApJ 351, 632) and is verified against the open-source code of **Tavleev, Lipunova & Malanchev 2023**, "Analysis of accretion disc structure and stability using open code for vertical structure," MNRAS (DOI [10.1093/mnras/stad1881](https://doi.org/10.1093/mnras/stad1881), arXiv [2303.02184](https://arxiv.org/abs/2303.02184)). GRRT's solver uses Newton relaxation rather than their shooting/optimization, and grids in column-mass fraction, but the equations and boundary conditions are theirs.

## 21. Numerical formulation: gas-pressure state variable (radiation-pressure conditioning)
The §20 BVP is **physically** in total pressure `P` (hydrostatic §7 and viscous heating §8 both use total `P` — the standard α-model). But the **Newton solver must carry GAS pressure `P_gas` as the per-node state variable**, not total `P`. Reason: density from total pressure,
```
ρ = (P − a T⁴/3) · μ m_p/(k_B T)
```
subtracts two near-equal large numbers in the radiation-pressure-dominated regime (`P ≈ P_rad`, β ≡ P_gas/P → 0), so `ρ` becomes hypersensitive to `T` (`∂ρ/∂T` grows ∝ 1/β) and Newton's temperature rows cannot be satisfied — the solver stalls (ill-conditioning, not just precision loss). Carry `P_gas` instead:
```
ρ        = P_gas · μ m_p/(k_B T)        (no subtraction; ∂ρ/∂T = −ρ/T, well-conditioned)
P_total  = P_gas + (1/3) a T⁴           (reconstructed by ADDITION wherever the physics needs total P:
                                         the hydrostatic LHS, viscous heating, the surface-pressure BC)
```
The physics is identical — only the state representation changes. State vector: `[P_gas, Q, T, z]×N + [z₀, Σ₀]`.
*Externally verified: Tavleev/Lipunova/Malanchev 2023 (§20 credit) relate `ρ` to **gas** pressure (`ρ = μ P_gas/ℜT`) and track `P_rad = aT⁴/3` separately — never recovering `P_gas` by subtraction. Code: `rho_from_gas`, `p_total` in `disk_column_bvp.cpp`; analytic-vs-numerical Jacobian cross-check is exact (0.0) at both gas- and radiation-dominated operating points.*

**Known limit — radiation-pressure fold (β ≲ 2.5e-3).** The conditioning fix converges down to β ≈ 2.5e-3 (`T_eff ≈ 5.6e6 K` for the canonical inner disk). Below that, the **standard α(total-P) thin-disk vertical structure folds** — a genuine solution-branch turning point at the photosphere (the *hydrostatic/thickness* rows fail, not the temperature rows). This is the radiation-pressure-dominated α-disk limit the Tavleev code avoids entirely (*"when `P_rad ≳ P_gas` the solution becomes problematic; our code does not calculate such discs"*) — a **physical** limit of the `α·P_total` prescription (Lightman & Eardley 1974 thermal-viscous instability), not a numerical one. The standard remedy is the **β-prescription** (viscous stress ∝ `P_gas` instead of total `P`), which removes the fold at the cost of changing the heating law in the inner disk. *[STATUS 2026-06-08: decision pending — the canonical hot disk (`T_peak=1e7`) sits below the fold; see the BVP-wiring plan.]*

---

## Error-trap checklist (read before editing any formula)
1. **`L_Edd` has a factor `c` in the numerator** (`4πGMm_p·c/σ_T`). The most common slip.
2. **`a = 4σ_SB/c`**, and **`P_rad = (1/3)aT⁴`** — keep the `1/3` and the `4`.
3. **Radiative diffusion** uses **Rosseland** `κ_R`, and the constant is `4ac/3 = 16σ/3` — don't mix `σ` and `a` forms.
4. **Viscous heating** is `(3/2)αΩP` — the `3/2` is the Keplerian shear, easy to drop.
5. **Optical depth needs CGS lengths** (`× r_g`), never geometric `dz`.
6. **Photosphere `τ = 2/3`**, not `1`.
7. **The vertical-structure BVP needs THREE surface BCs** (§20) — `Q`, `T`, *and* the surface pressure `P=(2/3)ω²z₀/κ`. Omitting the surface-pressure condition leaves the free parameters (`z₀`, `Σ₀`) unpinned and the solver under-determined.
8. **The Newton solver carries GAS pressure `P_gas`, not total `P`** (§21). Recover total `P = P_gas + aT⁴/3` by *addition*; recovering `P_gas = P − P_rad` from a stored total is catastrophic cancellation in the radiation-dominated regime and stalls the solver. Physics still uses total `P` (hydrostatic, viscous heating, surface BC).
