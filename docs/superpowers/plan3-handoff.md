# Plan 3 handoff — wire the column BVP solver into VolumetricDisk

**Status as of this doc:** Plan 1 (foundation) and Plan 2 (standalone column BVP solver) are COMPLETE and committed. Plan 3 = the wiring. This doc is the "resume here" pointer; the authoritative detail is in the spec + plans + refinements + verified-formula reference (linked below).

## Where we are
- **Plan 1 (foundation, committed):** `r_g` length scale, `Ṁ` from `f_Edd`, mass-adaptive opacity table (`ρ_est`-derived range), `kappa_ross_with_grad`. See `plans/2026-06-01-disk-first-principles-foundation.md`.
- **Plan 2 (column BVP solver, committed):** standalone `solve_column_bvp` in `src/disk_column_bvp.cpp` / `include/grrt/scene/disk_column_bvp.h`, tested in `tests/test_column_bvp.cpp` (`test-column-bvp` target). Newton relaxation, **analytic block Jacobian** (engine, cross-checked to 2.5e-9 against the numerical one), damped line search, honest fallback. Converges + energy conservation closes to 0.02%. See `plans/2026-06-04-disk-column-bvp-solver.md`.
- **Spec (authoritative):** `specs/2026-06-01-disk-first-principles-vertical-structure-design.md` — §6 radial inputs/emergent outputs, §7 the BVP (verified), §9 LUT/log-density storage, §11 interface/construction pipeline, §15 phasing. Plan 3 = phases 4–5.
- **Verified formulas:** `references/disk-physics-formulas.md` (§20 the BVP; error-trap checklist). **Check formulas here, don't re-derive.**
- **Deferred refinements:** `disk-approach-a-refinements.md` (esp. **1b** — column convergence robustness + the radial-neighbor-seeding strategy that IS Plan 3's convergence plan).

## The solver interface Plan 3 calls (all CGS)
```cpp
grrt::ColumnInputs{ T_eff[K], shear[1/s]=|r dΩ/dr|, omega_z[1/s], alpha,
                    rho_mid_guess[g/cm^3], n_nodes(~150-200), max_iters, tol };
grrt::ColumnBVPSolution = solve_column_bvp(in, opacity_luts_);
//   .q,.z[cm],.P,.Q,.T[K],.rho[g/cm^3] (index 0=midplane, back=surface);
//   .z0[cm]=half-thickness, .Sigma0[g/cm^2], .tau_mid, .converged, .used_fallback
```

## What Plan 3 does (spec §6/§9/§11/§15 phases 4–5)
1. **Replace `VolumetricDisk::solve_column`** (the old collapsing solver) with a call to `solve_column_bvp`, per radial column `r ≥ r_isco`. Build `ColumnInputs` from the radial LUTs:
   - `T_eff` ← `T_eff_lut_[i]` (already absolute via Ṁ).
   - `omega_z` ← `sqrt(omega_z_sq(r))` converted geometric→cgs (`× c_cgs/r_g_`).
   - `shear` ← the **exact Kerr shear** `r·|dΩ/dr|` (the radial code already computes `dOmega_dr` at `volumetric_disk.cpp:566`; `shear_geom = sqrt(shear_sq)` there, convert to cgs `× c_cgs/r_g_`). NOTE: viscous heating uses orbital shear, NOT `omega_z` — they differ for Kerr (see verified-ref §20 / error-trap #?).
   - `alpha` ← `params_.alpha`; `rho_mid_guess` ← `ρ_est`-based (see foundation `rho_mid_estimate`) or the converged neighbor (next item).
2. **Radial neighbor-seeding (the convergence strategy — refinement 1b):** solve columns marching radially and seed each column's `rho_mid_guess` (and ideally a full warm-start) from its **converged neighbor**. This is the natural continuation that should converge the hot inner columns that the standalone sweep (fixed uniform seed) could not. Judge the real convergence rate here. (Add explicit homotopy/grey-T seed only if neighbor-seeding still leaves columns falling back.)
3. **Resample** `ρ(z), T(z)` from the BVP's column-mass grid onto **uniform-z** for the LUT; store `z_max_lut_[i] = z0/r_g_` (geometric), `rho_mid_lut_[i] = ρ.front()` (absolute cgs), the normalized profile, `T_profile_lut_`.
4. **Log-density encoding:** store `log(ρ_norm)` and interpolate in log (`density()`/`interp_2d` → log-interp; T stays linear). Update the read path. (CUDA deferred — separate task.)
5. **Retire** `normalize_density` (set `rho_scale_≡1`; densities absolute from the BVP), `nested_refine`, `compare_columns`. `compute_sigma_s_phys` reads real `ρ_mid`. Add Toomre-Q validation; surface `used_fallback` columns as a `ConstructionWarning` naming the radius. Post-BVP opacity-range guard (warn if any ρ outside the table).
6. **Tests:** profile not collapsed (the original bug), `Σ(r)/τ_mid(r)/ρ_mid(r)` smooth, banding metric < 0.25 + edge-on render band-free, construction time within a few minutes. Plus the convergence rate (how many columns converge with neighbor-seeding vs fall back).

## Plan 3 process
Use the `superpowers:writing-plans` skill to write the Plan-3 implementation plan against this handoff + spec §15 phases 4–5, then execute via `superpowers:subagent-driven-development` (same workflow as Plans 1–2: per-task gate, no auto-commit — hand commit messages to the user, present every reviewer recommendation with a take and wait for the user's call; subagents sonnet or opus only, NEVER haiku).

## Workflow constraints (carry forward)
- NEVER run `git commit` — hand the commit message to the user (memory `feedback_review_workflow`).
- Present every reviewer recommendation with my take and WAIT for the user's call before fixing.
- Subagents: sonnet or opus only.
