# Slim-disk vertical-BVP coupling — design (POC milestone)

**Created:** 2026-06-14 · **Branch:** fix/volumetric-ring
**Status:** Design (brainstormed, approved). Scope source: `docs/superpowers/plans/2026-06-14-slim-disk-vertical-bvp-coupling.md`. Verified formulas: `references/disk-physics-formulas.md` §20–§23.

---

## 1. Goal & scope

**Goal.** Cure the proven one-zone **closure inadequacy** that makes `f_Edd≈0.9` (a=0.9) unreachable, by coupling the existing grey vertical-structure BVP (`src/disk_column_bvp.cpp`) per radial node into the radial slim solve, so the radiative flux comes from the vertical integration `∫(dℱ/dz)dz` (**decoupled from H/r**) instead of the one-zone `Q_rad = 64σT_c⁴/(3κΣ)`.

**Why (the verified problem).** In the radiation-dominated inner disk the one-zone closure makes `Q_rad = 8 c H Ω_⊥²/κ` — independent of (T,Σ), **∝ H/r** (Wolfram-confirmed). Radiating `Q_vis ≈ F_NT` then *forces* `H/r ≈ 2–5` (a torus); a physical `H/r ≲ 0.5` sheds only ~10–25%, so the slim root **does not exist in the one-zone model**. The fix is a vertical-structure-informed flux (S11's full vertical integration), which breaks the `Q_rad ∝ H/r` tie.

**Scope = proof-of-concept (POC) milestone.** Prove a **physical** `f_Edd≈0.9` disk (H/r ≲ 0.5, gas-dominated outward, all validity gates passing, `rad`/`ang` residual groups at the merit floor) is a **root** of the coupled model and can be **landed** by the solver. The POC deliberately:
- **Includes** the full convergence machinery (the analytic sensitivity + the coupled Newton) — because *landing the branch* is the real risk (stiff radiation-dominated regime, near a fold), and a convergence failure must be unambiguously physics, not solver weakness.
- **Defers** performance engineering: column banding (refinement #2), coarse-then-fine `n_z` (Richardson, #1), and broad-parameter sweeps. "Slow but robust and correct" is the POC bar. Production performance is a follow-up once the physics is proven.

**Mass-independence.** The solver is in geometrized units (G=c=1, M sets the length scale); the grid (`r/M`), unknown count, Jacobian structure, and Newton are **identical for any black-hole mass** (sub-stellar → ultramassive). Mass enters only the thermodynamics/opacity (handled by the mass-adaptive opacity table; cool-outer-disk molecular/dust opacity for the largest BHs is a separate deferred refinement, #6). Nothing in this design depends on the mass.

**Non-goal / honest caveat.** This coupling makes the `f_Edd≈0.9` root **exist**; it does **not** by itself guarantee that root is **stable/reachable**. If, post-coupling, the solver still cannot land it, the next question is *instability* (the S-curve middle branch / the stability atlas, refinements #10), not closure. The POC's success criterion explicitly distinguishes these two outcomes (§8).

---

## 2. Architecture

The existing radial solver keeps its structure: unknowns `U_r = {Σ,V,ℓ,T_c}×N + {ℓ_in, r_s}` (length `4N+2`), the six residual groups (mass, angular momentum, transonic radial-momentum, energy, sonic regularity, outer BC). **Only the energy group's flux and the closure thickness change** — they stop being one-zone and come from a per-node vertical column.

### 2.1 Convergence engine — Nested Newton (nonlinear column elimination) via per-column Schur

The coupled system is `{ R_r(U_r, C) = 0 ; column_i converged ∀i }`, where `C = {F_i, z₀_i, η₃_i, η₄_i}` are the converged column outputs and each column depends only on its own node's `(Σ_i, T_c,i)` (+ geometric `shear, Ω_z`). This block structure — columns coupled to each other **only** through the radial system — is what makes the architecture tractable and final:

- **Nonlinear elimination of the columns.** At each radial Newton step, each column is **fully re-converged** (warm-started) → `C` is always a *physically valid* vertical structure. The radial Newton then operates on the **reduced** system `R_r(U_r, C(U_r)) = 0`, of size `4N+2` (~194).
- **The reduced (Schur) Jacobian** is `dR_r/dU_r = ∂R_r/∂U_r + (∂R_r/∂C)·(dC/dU_r)`, where `dC/dU_r` is the **analytic column-output sensitivity** (§3.3). This is the Schur complement of the full coupled Jacobian after eliminating the block-diagonal column DOFs — formed **per-column** (cheap), never as a global matrix.
- **Properties:** exact Newton (quadratic) on the reduced system; columns never seen half-converged (robust in the stiff β→0 regime); radial-sized linear solves; reuses the column solver + its LU factor. This is a standard production-grade multiphysics architecture (nonlinear elimination / nested Newton) — there is no more-robust form to migrate to (JFNK would only matter at 10⁶⁺ unknowns and would sit on this same Schur structure).

### 2.2 Data flow (per radial Newton step)

**Coupling causality (the resolved framing — `f_adv`-output correction, 2026-06-25).** The column provides the *closure relationship* `(Σ, T_c) → (F, z₀, η₃, η₄, f_adv)` that replaces the one-zone `Q_rad(Σ,T_c)` and `H(Σ,T_c)`. T_c stays a radial unknown; the radial **energy row keeps its form** `Q_vis − F − Q_adv = 0` (the column's emergent flux F replaces `64σT_c⁴/(3κΣ)`), so the energy balance still determines T_c — but now F comes from the vertical diffusion (decoupled from H/r) and `H = z₀` from vertical hydrostatic (not `c_s/Ω_⊥`). **`f_adv` is a determined OUTPUT, not an input.** The column's flux *generation* carries the advection reduction `αP·|r dΩ/dr|/(1+f_adv)` (§22 geodesic convention), and `f_adv` is **freed as a column unknown** — back-solved so the column hits `Σ` at the pinned `T_c`. The vertical structure is a two-parameter family `(T_c, f_adv) ≡ (T_c, Σ)`, so `(Σ, T_c)` *uniquely determine* `f_adv` (S11 §3.1–3.2; numerically confirmed 2026-06-25). Fixing `Σ`, `T_c` **and** `f_adv` over-determines the column (3 constraints on a 2-parameter family) — that was the source of the spurious "folds". **C4 consistency:** the column's emergent `F` and back-solved `f_adv` satisfy `Q_vis − F − Q_adv = 0` automatically when the column's vertically-integrated dissipation equals the radial `Q_vis` (`f_adv = Q_adv/F`); to be confirmed numerically when C4 is wired (no separate radial equation needed).

```
radial state (Σ_i, T_c,i) per node           (f_adv,i emerges from the column, not carried as a radial input)
   │
   ▼ for each node i (independent, warm-started):
   ┌─────────────────────────────────────────────┐
   │ (Σ,T_c)-driven column solve (C1)             │ → F_i (emergent flux), z₀_i, f_adv_i, T(z), ρ(z), converged_i
   │ vertical-moment post-integrate    (C2)       │ → η₃_i, η₄_i
   │ output sensitivity via IFT        (C3)       │ → ∂{F,z₀,η}_i/∂{Σ_i,T_c,i}   (reuse column LU)
   └─────────────────────────────────────────────┘
   │
   ▼ radial assembly (C4):
   • energy row:        Q_vis − F_i − Q_adv = 0      (F_i replaces 64σT_c⁴/3κΣ; determines T_c)
   • closure:           H_i = z₀_i                   (replaces c_s/Ω_⊥)
   • 𝒩₁ (C5):           restore (P/Σ)dlnη₃/dlnr + Ω_⊥²(η₄/η₃)dlnη₄/dlnr using η₃_i,η₄_i
   • reduced Jacobian:  ∂R_r/∂U_r + (∂R_r/∂C)(dC/dU_r)   (the Schur terms in energy/closure cols)
   │
   ▼ solve radial-sized system (4N+2), LM-damped step, feasibility line search
   updated U_r ──► repeat (columns re-converged next step)
```

---

## 3. Components (each: purpose · interface · dependencies)

### C1 — Column closure-map entry (the causality re-pose)
**Purpose.** Provide the column as the map `(Σ, T_c) → (F_emergent, z₀, f_adv, profile)` the radial energy row + closure need. The column is currently `T_eff`-driven (surface temperature is the input, `Σ0` an output); re-pose it so the radial-side state `(Σ, T_c)` are the inputs and the emergent flux `F`, the surface `T_eff`, and the advected fraction `f_adv` all float (freed as column unknowns).
**Interface.** `solve_column_coupled(Σ, T_c, shear, Ω_z, α, opacity, warm_start) → {F, z₀, f_adv, T(z), ρ(z), P(z), converged}`, where `F = σT_eff⁴` is the emergent (top-of-column) diffusion flux, `T_c = T(0)` (midplane), and `f_adv` is the back-solved advected fraction. (`f_adv` is **no longer an input**.)
**Mechanism (augmented row-swap, `f_adv` freed — 2026-06-25).** Pin two BCs vs the `T_eff`-driven form: fix `Σ` (was an output `Σ0=2∫ρdz`) and fix the midplane `T_c=T(0)` (was the surface `T_eff`); and **free two globals**: `T_eff` (→ emergent `F=σT_eff⁴`) **and** `f_adv` (the advection-reduction scalar in the heating `αP·|r dΩ/dr|/(1+f_adv)`, §22 geodesic convention). `f_adv` is back-solved so the structure carries column mass `Σ` at midplane `T_c` — it is S11's genuine *second* degree of freedom (`(T_c, Σ)`-parametrization); freeing it removes the over-determination that fixing `(Σ,T_c,f_adv)` would impose (the spurious "folds"; source + numerically verified 2026-06-25). F decouples from H/r because it is the *integrated radiative-diffusion* flux (set by the vertical T-gradient and opacity), and `z₀` decouples because it is the *hydrostatic* photosphere height — neither is the one-zone `c_s/Ω_⊥`. One differentiable Newton solve over the augmented state `[Pg,Q,T,z]×N + (z₀, T_eff, f_adv)`, reusing `node_deriv` + the analytic `∂R/∂U` (extended with `∂(dQ)/∂f_adv`) and the Ruiz-equilibrated / affine-invariant (Deuflhard) solver from the column-hardening work.
**Depends on.** `disk_column_bvp.cpp` internals (the residual + `∂R/∂U`).

### C2 — Vertical-moment post-integrator
**Purpose.** Compute the one-zone moments `η₃ = ∫E dz / ∫P dz` (E = internal energy density, gas + radiation), `η₄` (S11 Eqs 8/11) from a converged column profile, for the restored `𝒩₁` terms.
**Interface.** `column_moments(profile{P,P_gas,T,ρ,z}) → {η₃, η₄}` — a pure function.
**Depends on.** The column profile only. (Also exposes `∂{η₃,η₄}/∂profile` for C3.)

### C3 — Column-output sensitivity
**Purpose.** `dC/d{Σ,T_c}` = `∂{F, z₀, η₃, η₄}/∂{Σ, T_c}` for the reduced (Schur) Jacobian.
**Interface.** `column_sensitivity(converged column, ∂R/∂U LU) → ∂{F,z₀,η₃,η₄}/∂{Σ,T_c}`.
**Mechanism.** Implicit-function theorem: the column satisfies `R_c(U_c; p)=0` for parameters `p=(Σ, T_c)` — just **two** now, since `f_adv` moved *into* the augmented state `U_c`, so its response is captured by `∂R_c/∂U_c`. `dU_c/dp = −(∂R_c/∂U_c)⁻¹ (∂R_c/∂p)` via one back-substitution per parameter **reusing the existing LU factor**. Then `dC/dp = (∂C/∂U_c)(dU_c/dp)`, with `∂C/∂U_c` explicit (F, z₀, f_adv are state components; η from C2's profile derivative). The live radial sensitivities are `∂{F,z₀,η,f_adv}/∂{Σ,T_c}` (V, ℓ enter the column only through the geometric `shear`/`Ω_z`). **New code:** export `∂R_c/∂p` — now trivial: just the two pin rows `Σ0−Σ=0` and `T(0)−T_c=0` (each a `−1` column), no `f_adv` parameter column.
**Depends on.** C1 (the converged column + factorized `∂R/∂U`), C2 (moment derivatives).

### C4 — Nested coupled Newton driver
**Purpose.** Orchestrate §2.1: per step, refresh all columns (C1) + moments (C2) + sensitivities (C3); assemble the radial residual and the reduced Schur Jacobian; LM-damped radial solve + feasibility line search; iterate to the merit floor.
**Interface.** `solve_slim_disk_coupled(SlimDiskInputs, opacity) → SlimDiskRadial` (mirrors the existing `solve_slim_disk_radial`; selected for the coupled model).
**Depends on.** C1–C3, the existing radial residual/Jacobian assembly (energy + closure rows rerouted to the column outputs), the existing LM/line-search/validity machinery.

### C5 — `𝒩₁` η-gradient restoration
**Purpose.** Add back the S11 `𝒩₁` terms our `calN1` dropped — `(P/Σ)dlnη₃/dlnr` and `Ω_⊥²(η₄/η₃)dlnη₄/dlnr` (S11 Eqs 29/32–33) — now that η₃/η₄ exist per node (C2), re-derived in GRRT's `Ω_⊥²=Ω_K²ℋ` convention (§22 note). Also carry S11's `f_F` flux factor (Eq 45) on the emergent flux (minor; α-dependence to verify for α=0.1).
**Interface.** Inline in the radial residual/Jacobian (the `calN1` and Q_adv assembly).
**Depends on.** C2 (η₃/η₄ + their radial gradients via FD across nodes), C3 (their sensitivities for the Jacobian).

---

## 4. Error handling

- **Column non-convergence mid-radial-Newton** (undefined sensitivity). The radial step must not consume an invalid column. Handling: (1) the proposed radial step is **rejected by the line search** if any column fails to converge at the trial `(Σ,T_c)` (shrink the step) — preferred, keeps the Newton honest; (2) as a last resort, **fall back to one-zone `Q_rad` for that node** with a logged flag (the node is then mildly inconsistent but the solve proceeds) — used only to avoid a hard stall, surfaced in diagnostics. The robust per-column **warm-start chain** (neighbour + previous radial iterate) makes failures rare (a column barely moves between radial steps).
- **Feasibility.** Columns and the radial state must stay physical: Σ>0, T>0, `1+f_adv>ε` (the existing gate), V<0, `r_s<r_isco`. The existing feasibility line search is extended to also require all columns converged + bounded.
- **NT-limit reduction.** As `Ṁ→0` the coupled flux must reduce to the thin-disk emergent flux so the NT-reduction gate holds (§5). In the gas-dominated, optically-thick limit the coupled column's emergent `F` must match the Page-Thorne `D(r)` (the column flux `≈` the one-zone `64σT_c⁴/(3κΣ)` up to the `f_F≈0.94` vertical-structure factor) — verified by the gate, not assumed.
- **Honest fallback.** On non-convergence of the whole coupled solve, return `SlimDiskRadial{converged=false}` (no fabricated profile), as today.

---

## 5. Testing & gates

- **NT-reduction (must stay green).** `slim-nt-term-probe` / the coupled solver at `a=0.9, f_Edd=0.02` must still match `VolumetricDisk::compute_radial_structure` (`Q_vis/F_NT ≈ 1.1–1.2` band post-#12, flat). The BVP flux must reduce to the thin-disk flux in the gas-dominated limit.
- **FD-Jacobian cross-check (must stay green, extended).** The new reduced (Schur) Jacobian — including the column sensitivities `dC/dU_r` — must match an FD oracle (perturb radial `Σ,T_c`, re-solve the column, difference `F,z₀,η`). Extend `test-slim-jacobian` to cover the energy/closure rows' column-derived terms. This is the rigorous correctness gate for C3.
- **Column-internal cross-check (existing).** `column_jacobians_test` keeps `∂R_c/∂U_c` analytic ≡ FD; add a check that the new `∂R_c/∂p` export matches FD.
- **The target gate (the POC's definition of done).** A converged, **physical** `f_Edd≈0.9, a=0.9` disk: `H/r ≲ 0.5`, gas-dominated outward (β→~1), `f_adv ~ +0.3` inner → 0 outward, V<0, sonic inside ISCO, T_c physically determined, **all validity gates passing**, and the `rad`/`ang` residual groups driven from the one-zone O(200–300) (`slim-sadowski-residual-probe`) **down to the 1e-3 floor**. Cross-check `H/r(r)`, `T_eff(r)` shape vs S11 figures (ballpark).
- **Coupled re-run of `slim-sadowski-residual-probe`** under the coupled model: the Sądowski-shape structure should become a near-root (residual groups collapse toward the floor) — the direct confirmation the closure obstruction is removed.

---

## 6. POC boundaries (in / deferred)

**In scope (this milestone):** C1 causality inversion · C2 moments · C3 analytic sensitivity + `∂R/∂p` export · C4 nested coupled Newton · C5 `𝒩₁` restoration (+ `f_F`) · the extended FD-Jacobian + NT gates · landing one physical `f_Edd≈0.9, a=0.9` disk (small `N`, e.g. 48; `n_z` modest). Correctness and robustness only.

**Deferred (production follow-ups, not this milestone):** column banding (refinement #2, the O(n_z)→affordability work) · coarse-then-fine `n_z` (Richardson, #1) · broad `(M,a,f_Edd)` sweeps and the stability atlas (#10) · near-extremal spin (the spin-walk) · the volumetric thick-disk *rendering* integration · `f_F` α-dependence calibration.

---

## 7. Risks / open questions

- **Cost.** Columns re-converged each radial step (× N nodes × radial iters). Mitigated by warm-starting (cheap re-solves); for the POC, accept slow. If intractable even warm-started, bring banding (#2) forward — but it does not change the design, only the per-column solve cost.
- **C1 inversion robustness.** Promoting `T_eff` **and `f_adv`** to unknowns with integral BCs is stiffer than the `T_eff`-driven form; handled by Ruiz row/col equilibration + an affine-invariant (Deuflhard natural-monotonicity) Newton (column-hardening work, 2026-06-25). Crucially, **freeing `f_adv` (vs fixing it) *removes* the over-determination** that caused the worst stiffness/folds, so the augmented system is better-conditioned than the fixed-`f_adv` row-swap. Seed `T_eff`/`rho_mid` from the one-zone `Q_rad=Q⁺` inversion; a base-solver bring-up gives a consistent pair if needed.
- **η₄ transcription.** From S11 Eqs 8/11; gate it with its own moment probe (mirror #11's η₃ gate) before relying on the restored `𝒩₁` terms.
- **Stability (the §1 caveat).** If the coupled solver still won't land `f_Edd≈0.9` after C1–C5 pass their gates, the obstruction is *instability*, not closure — escalate to the stability question, do not assume a code bug.
- **`f_F` α-dependence** (quoted α=0.01; we run α=0.1) — verify before relying on Eq 45; it is a ~6% effect, not load-bearing for the POC.

---

## 8. Success criteria (the POC's definitive outcome)

The POC ends in exactly one of two informative states:
1. **A converged, physical `f_Edd≈0.9` disk** passing §5's target gate → **the closure inadequacy is cured and the root is reachable.** Proceed to productionize (banding/perf) + the render pipeline.
2. **The coupled solver, with C1–C5 gate-clean (NT + FD-Jacobian green), still cannot land `f_Edd≈0.9`** → the obstruction is no longer the closure; it is **instability/branch-reachability** (refinements #10). A different, well-defined next question — not a regression.

Either outcome is a decisive result: the POC removes the *model-inadequacy* explanation and tells us whether the render target is a convergence problem (solved) or a stability problem (the next investigation).

---
*Sources: S11 [arXiv:1006.4309] Eqs 8, 11, 12–16, 29, 32–33, 42, 45, §5/Fig 17; S09 [arXiv:0906.0355] Eqs 1, 4, 5, 6; Abramowicz et al. 1997. Code: `src/slim_disk_radial.cpp`, `src/disk_column_bvp.cpp` / `include/grrt/scene/disk_column_bvp.h`; `tools/slim_sadowski_residual_probe.cpp`. Diagnostic verdict + cross-check: `docs/superpowers/slim-disk-handoff-2026-06-14.md` §2, the scope `plans/2026-06-14-slim-disk-vertical-bvp-coupling.md`.*
