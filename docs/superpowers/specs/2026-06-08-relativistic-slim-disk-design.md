# Relativistic slim-disk subsystem — design

**Date:** 2026-06-08
**Branch:** fix/volumetric-ring
**Status:** brainstorming approved — pending spec review → implementation plan
**Builds on:** the Approach-A column BVP (`2026-06-01-disk-first-principles-vertical-structure-design.md`) and its gas-pressure conditioning fix. Verified equations: `references/disk-physics-formulas.md` §22 (slim disk), §20 (vertical BVP), §21 (gas-pressure state variable). Model survey: `disk-radiation-pressure-model-options.md`.

---

## 1. Problem & motivation

The Approach-A column BVP solves the grey vertical structure with viscous heating `dQ/dz = α·P_total·|r dΩ/dr|` (stress ∝ **total** pressure — standard Shakura-Sunyaev). In the radiation-pressure-dominated inner disk (`β = P_gas/P ≲ 2.5e-3`) this has **no stable steady solution**: the Lightman-Eardley (1974) viscous/thermal instability folds the solution branch (a turning point at the photosphere), so the BVP cannot converge. The numerics are already correct (gas-pressure state variable, §21) — this is a **physics** limit of the `α(total-P)` thin-disk closure, confirmed by the open Tavleev code refusing the regime.

This matters acutely here because **GRRT renders at `f_Edd ≈ 0.9` by default** — firmly near-Eddington, where the inner disk is genuinely thick (`H/r ~ 0.1–0.2`) and **radial advection (photon trapping) is a major cooling channel**, not a small correction. A thin-disk model (even a stabilized one) would render a qualitatively wrong inner disk: too thin, no advection, missing the ~10% radial-infall Doppler shift in the brightest region.

The physically correct model for this regime is the **relativistic slim disk** (Abramowicz et al. 1988; Sądowski 2009/2011): it adds radial advection, which *both* removes the Lightman-Eardley fold (the extra cooling stabilizes the disk onto the slim-disk branch) *and* captures the puffed, advective near-Eddington inner disk. Crucially, the slim disk is a **strict superset of the thin disk** — it reduces exactly to Novikov-Thorne as `Ṁ→0` — so it covers the entire sub-to-near-Eddington range with one seamless model and no loss of thin-disk fidelity.

## 2. Goal & constraints

- **Render near-Eddington disks accurately.** Accurate to `f_Edd ≈ 1`; graceful (with a documented grey-diffusion caveat) to ~1–2. Genuinely super-Eddington (`≫1`, funnel + winds) is out of scope (future thick-disk model — refinements #9).
- **Physically derived, externally verified.** All equations verified against the primary literature and recorded in §22 before coding. No magic numbers; the only modelling knobs are the physical inputs `(M, a, f_Edd, α)` already in the model.
- **Self-consistent.** Full transonic radial solve (sonic-point regularity eigenvalue) coupled self-consistently (2D iteration) with the per-column vertical structure.
- **Reduces to thin disk.** At low `f_Edd` it must coincide with the existing Novikov-Thorne result (regression-tested).
- **Reuse the vertical BVP.** The existing grey per-column Newton BVP (with the gas-pressure fix) is the vertical solver — fed `f_adv(r)`.
- **Preserve the committed raymarch *stepping* fixes** (uniform fine stepper, side-impact exit, `mid_state`). Only the emission/redshift sampling changes (new velocity field).
- **Construction time is not a constraint** (results are cached; LUT/CSV export already half-built). Accuracy is the priority.
- **CPU path is the deliverable.** CUDA stays deferred.

## 3. Scope

**In scope:** the relativistic transonic radial slim-disk solver (Kerr, sonic-point regularity); the `f_adv` coupling into the vertical BVP; the self-consistent 2D iteration; the slim-disk velocity field (`u^t, u^r, u^φ`) feeding the raymarch redshift; replacement of `compute_radial_structure` and the inner-edge/plunging treatment within the slim-disk domain; validation (thin-disk limit, sonic regularity, literature benchmark, the near-Eddington render); the honest-CGS LUT switch deferred from the BVP-wiring plan (Task 4 there) lands as part of the integration.

**Out of scope (future, separate specs):**
- **Super-Eddington thick disk / radiation funnel** (`f_Edd ≫ 1`; winds, non-grey transport) — refinements #9.
- **Non-grey / line opacity** — still grey Rosseland here.
- **Radiation-MHD magnetic-pressure support** — the slim disk uses the hydrodynamic `α(total-P)` stress (advection provides the stability); magnetic support is a possible later refinement.
- **CUDA port.**

## 4. Architecture & data flow

```
INPUTS:  M [M_sun]   a [spin]   f_Edd   α                         ← anchor (already in VolumetricParams)
   │  → Ṁ = f_Edd·L_Edd/(η c²)
   ▼
RADIAL TRANSONIC SOLVE  (Kerr, global, eigenvalue ℓ_in at the sonic point)   ← NEW core
   four conservation laws (§22): mass, radial momentum (transonic), angular momentum, energy(+advection)
   OUTPUTS per r:  Σ(r), V(r)/v_r(r), Ω(r), T_c(r), H(r), f_adv(r)
   │
   ▼
VERTICAL STRUCTURE  (per column, r over the solved range)              ← REUSE the grey BVP (§20, §21)
   the existing Newton BVP, heating reduced by 1/(1+f_adv);  inputs (Σ, T_c, f_adv)
   OUTPUTS: ρ(z), T(z), z_max(r);  returns pressure/density moments
   │
   ▼  ⟲  2D ITERATION: feed column moments back into the radial EOS, re-solve, repeat to convergence
   ▼
LUT  (unchanged layout)  →  RAYMARCH
   ρ(r,z), T(r,z), z_max(r), ρ_mid(r)        emission samples the slim-disk u^μ (incl. v_r) for redshift;
                                             sonic point = structural inner edge (replaces ISCO edge/BPT72)
```

**Three components** (see §6, §22):
1. **Transonic radial solver** — the new subsystem. Relaxation/shooting on the radial ODEs with the sonic-point regularity condition as the eigenvalue closure (`ℓ_in`).
2. **Vertical BVP** — reused unchanged except the heating term gains the `1/(1+f_adv)` factor and inputs come from the radial solve.
3. **2D outer iteration** — alternates radial and vertical solves to self-consistency.

## 5. Goal of each piece — what's KEPT / REPLACED / NEW / RETIRED

| | Items |
|---|---|
| **KEEP** | Kerr mechanics (`omega_orb`, `omega_z_sq`, ISCO/horizon); the grey vertical Newton BVP (§20) + gas-pressure fix (§21); the warm-start radial march + homotopy bootstrap (they solve the column sequence); opacity physics; LUT storage layout; raymarch *stepping* (uniform stepper, side-impact, `mid_state`); turbulence overlay |
| **REPLACE** | `compute_radial_structure` (thin-disk `T_eff`/`H`/`ρ_mid`) → the **transonic radial solver** (emergent `Σ, V, Ω, T_c, H, f_adv`); the vertical heating term gains `1/(1+f_adv)`; the inner edge (hardcoded ISCO) → the **sonic point**; the raymarch redshift velocity (`circular_velocity`/`plunging_velocity`) → **`slim_disk_velocity(r)`** in the slim-disk domain |
| **NEW** | transonic radial ODE system + sonic-point regularity eigenvalue solve; `f_adv(r)` coupling; the 2D iteration loop; `slim_disk_velocity` (`u^t,u^r,u^φ`) |
| **RETIRE** | the BPT72 plunging taper + frozen-`H(r_isco)` *within the slim-disk domain* (the transonic flow is continuous through the sonic point); the radiation-pressure *fold* (advection removes it) |

## 6. The physics (verified — see §22 for equations)

**Radial.** Four height-integrated conservation laws in Kerr (§22 eqs 1–4): mass `Ṁ = −2πΣΔ^½V/√(1−V²)`; the **transonic** radial momentum equation; angular momentum with the vertically-integrated α-stress `W_rφ = αP`; energy `Q_vis = Q_rad + Q_adv` with `f_adv ≡ Q_adv/Q_vis`. The radial velocity `V(r)` passes through a **sonic point** `r_s` (Mach 1, `𝒟₀ = V²−Γ̃₁P/Σ = 0`); **regularity** (`𝒩=𝒟₀=0` together) is an eigenvalue condition fixing the inner specific angular momentum `ℓ_in`. `r_s` is found, not prescribed, and lies inside the ISCO.

**Vertical.** The existing grey BVP (§20), heating reduced by advection: `dℱ/dz = (3𝒟/2𝒞)·(αp/(1+f_adv))·(M/r³)^½`; vertical hydrostatic uses the Kerr `Ω_⊥²` (§22). Inputs per column: `Σ(r), T_c(r), f_adv(r)`.

**Coupling.** Self-consistent 2D: the column returns pressure/density moments → radial EOS coefficients → re-solve the radial problem → repeat to convergence.

**Kinematics / inner edge** (the chosen "slim disk owns the inner flow" option). The slim disk's four-velocity (`u^t, u^r, u^φ` from `V`, `Ω`, the Kerr metric) is the emitter velocity used in the raymarch redshift `g = (p·u)_emit/(p·u)_obs`. The structural inner edge is the sonic point; emission continues continuously through and inside the ISCO. Inside the sonic point the flow is supersonic — vertical hydrostatic equilibrium breaks down — so a thin supersonic-plunge layer continues the flow to the horizon with velocity continuity from the sonic point (the slim-disk `u^μ` there, transitioning to geodesic plunge).

**Thin-disk limit.** As `Ṁ→0`: `f_adv→0`, `V≪1`, `Ω→Ω_K`, sonic point→ISCO, `ℓ_in→ℓ_ISCO`, reducing to Novikov-Thorne (§22).

## 7. Numerical method

- **Radial transonic solve — relaxation (LOCKED).** A global Newton relaxation on the radial ODEs (`Σ, V, Ω, ℓ, T_c` vs `r`) over a grid spanning the sonic point, with the **sonic-point regularity** (`𝒩=0` AND `𝒟₀=0`) imposed as an interior boundary condition and `(ℓ_in, r_s)` carried as unknowns — Sądowski's method, and the *same Newton-relaxation framework as the vertical BVP*. **Chosen over shooting** for both robustness and accuracy: shooting's exponential sensitivity at the sonic point (where `𝒟→0`) both makes the `ℓ_in` root-find fragile *and* caps the achievable precision of the trans-sonic **inner** disk — the rendering-critical region — and it fits the 2D coupling poorly (two discretizations meeting at the interface). Relaxation is well-conditioned at the critical point, reuses our Jacobian / warm-start / homotopy infrastructure, and keeps the 2D loop homogeneous (relax radial ↔ relax vertical). **Seeding:** start from the known Novikov-Thorne thin disk at low `Ṁ` and **continue up in `Ṁ`** to the target `f_Edd` (the homotopy-continuation pattern already used to bootstrap the vertical columns). Construction time is uncapped (cached).
- **Vertical solve.** The existing Newton BVP with the `1/(1+f_adv)` heating factor; reuse the warm-start radial march + homotopy bootstrap to solve the column sequence (still needed — adjacent radii warm-start each other; the homotopy bootstraps the first column).
- **2D iteration.** Outer fixed-point loop: radial → vertical → update radial EOS moments → repeat until `Σ(r), H(r), f_adv(r)` converge (relative tolerance). Damp if needed.
- **Honest fallback policy** (carried from the BVP-wiring plan): a column/radius that genuinely cannot converge is a Promptable truncation at the edge or a Severe interior hole — no fabricated profile.

## 8. Interface changes

- **`VolumetricParams`:** unchanged inputs (`mass_solar`, `eddington_fraction`/`mdot_override`, `spin`, `alpha`). The slim disk consumes them directly. (Optional: a `slim_disk_max_iters`/`tol` for the 2D loop; a `disable_advection` debug flag to force the thin-disk limit.)
- **`VolumetricDisk` construction pipeline:** `build_opacity_luts` → `r_g`/`Ṁ` → **transonic radial solve** → **2D{radial ↔ vertical BVP}** → resample/store LUT (honest CGS + log-density, the deferred Task-4 switch) → `compute_sigma_s_phys` → validate (+Toomre Q, range guard). **Deleted:** the thin-disk `compute_radial_structure` internals (replaced); the BPT72 plunging path within the slim-disk domain.
- **New accessors:** `slim_disk_velocity(r, double& ut, double& ur, double& uphi)`; `f_adv_at(r)`, `sonic_radius()`, `sigma_at(r)` for diagnostics/tests.
- **Raymarch (`geodesic_tracer.cpp`):** the two redshift call sites (`sample_integrand`, `raymarch_volumetric_spectral`) call `slim_disk_velocity` in the slim-disk domain instead of `circular_velocity`/`plunging_velocity`. **Additive** — the old functions remain for the thin-disk/fallback path. Stepping logic untouched.
- **CLI:** unchanged (`--eddington-fraction` already exists).
- **CUDA:** deferred.

## 9. Testing & validation

**Unit — radial solver:**
- Sonic-point regularity: the converged solution passes the sonic point smoothly (`𝒩, 𝒟₀ → 0` together; no kink in `V(r)`).
- **Thin-disk limit:** at `f_Edd ≲ 0.05`, `f_adv(r)→0` and `Σ(r), Ω(r), T_eff(r)` match the existing Novikov-Thorne result to a few %.
- **Literature benchmark:** `H/r`, `f_adv(r)`, `Σ(r)` for a benchmark `(M, a, Ṁ)` match published Sądowski 2011 figures within tolerance.
- Mass/angular-momentum/energy conservation residuals below tol along the solution.

**Unit — coupling:** the 2D iteration converges (`Σ, H, f_adv` fixed point); the vertical heating carries `1/(1+f_adv)`; energy closes (`∫heating = (1−f_adv)·Q_vis` per face).

**Unit — kinematics:** `slim_disk_velocity` returns a normalized 4-velocity (`u·u = −1`), `u^r < 0` (inflow), reducing to `circular_velocity` (`u^r→0`) as `f_Edd→0`.

**Integration:**
- The `f_Edd≈0.9` disk **constructs** — converged across the orbiting region, profile non-collapsed, `Σ(r)/τ_mid(r)/ρ_mid(r)` smooth, inner disk visibly **thick** (`H/r ~ 0.1–0.2`). `test_hot_inner_disk_columns_converge` (the success signal) passes at the strict gate.
- **Edge-on render** (`--observer-theta 80 --fov 30 --eddington-fraction 0.9 …`): band-free, puffed inner disk, the inner-disk Doppler asymmetry from radial infall visible. Surface before/after to the user.
- **Thin-disk regression:** a low-`f_Edd` render matches the pre-slim-disk thin-disk render (the superset guarantee).
- Tile == full-frame (CLAUDE.md invariant). CUDA == CPU suspended.

## 10. Edge cases

- **Sonic point near/inside the horizon** at high `Ṁ` — the radial solve must place it correctly; the supersonic-plunge layer carries to `r_+`.
- **`f_adv` convention** (§22 trap #11): use `Q_rad/Q_vis = (1+f_adv)^{−1}` consistently.
- **Grey-diffusion strain** as `f_Edd→1` (`τ_eff→1` in the innermost cells): flag a Promptable warning above a threshold; the structure is approximate there.
- **2D iteration non-convergence:** bounded iterations + honest fallback (Promptable/Severe per the wiring-plan policy), never a fabricated profile.
- **Opacity-table range:** slim-disk inner densities are *lower* than thin-disk — confirm the mass-adaptive range still brackets them (post-solve guard).
- **Toomre Q:** still `≫1` for the rendered inner disk; warn if marginal.

## 11. Success criteria (definition of done)

1. Transonic radial solver converges with sonic-point regularity; conserves mass/angular-momentum/energy; reduces to Novikov-Thorne at low `f_Edd`; matches Sądowski 2011 benchmarks.
2. The `f_Edd≈0.9` disk constructs end-to-end: converged, non-collapsed, thick inner disk; the success-signal test passes at the strict gate.
3. Self-consistent 2D iteration converges; `f_adv` coupling correct; energy closes.
4. Raymarch renders the near-Eddington disk band-free with the radial-infall Doppler; thin-disk renders unchanged (superset).
5. Committed raymarch *stepping* fixes untouched.
6. All equations verified (§22) and conventions (traps #9–12) honored.
7. Construction within reason (cached); honest fallbacks, no fabricated profiles.

## 12. Suggested phasing (for the plan)

1. **Transonic radial solver, standalone** — the radial ODE system, sonic-point regularity eigenvalue, conservation checks, thin-disk limit, literature benchmark. (Largest piece; verify against §22 + Sądowski figures.)
2. **`f_adv` vertical coupling** — the `1/(1+f_adv)` heating factor in the BVP; energy-closure test; the hot column now converges with realistic `f_adv`.
3. **Self-consistent 2D iteration** — the outer loop; convergence + fixed-point test.
4. **Kinematics / inner-edge integration** — `slim_disk_velocity`; sonic point as the inner edge; supersonic-plunge layer; raymarch redshift call-site switch; `u·u=−1` and thin-disk-limit tests.
5. **Wire into `VolumetricDisk` + LUT** — replace `compute_radial_structure`; the deferred honest-CGS + log-density LUT switch; retire the in-domain BPT72 path; validation (Toomre, range guard, fallback policy).
6. **Integration sweep** — `f_Edd=0.9` construction + edge-on render, thin-disk regression, before/after to the user.

## 13. Out of scope / future

- **Super-Eddington thick disk / radiation funnel** (refinements #9) — the `f_Edd ≫ 1` regime (ULXs, TDEs): thick torus, winds, non-grey transport. A distinct, larger subsystem.
- **Radiation-MHD magnetic-pressure support** — a possible accuracy refinement over the hydrodynamic α-stress.
- **CUDA port** of the slim-disk-produced LUT.

## 14. References

- Abramowicz, Czerny, Lasota & Szuszkiewicz 1988, ApJ 332, 646 (slim disk).
- Sądowski 2009, ApJS 183, 171 ([arXiv:0906.0355]) — relativistic radial equations + sonic point.
- Sądowski, Abramowicz, Bursa, Kluźniak, Lasota & Różańska 2011, A&A 527, A17 ([arXiv:1006.4309]) — relativistic slim disk with vertical structure + `f_adv` coupling.
- Abramowicz & Fragile 2013, Living Rev. Rel. 16, 1 (DOI:10.12942/lrr-2013-1) — clean equation reference.
- Riffert & Herold 1995, ApJ 450, 508 — Kerr correction factors.
- Lightman & Eardley 1974, ApJL 187, L1 — the instability the slim disk resolves.
- Verified equations: `references/disk-physics-formulas.md` §22 (+ §20, §21, traps #9–12).
- Model survey: `disk-radiation-pressure-model-options.md`.
