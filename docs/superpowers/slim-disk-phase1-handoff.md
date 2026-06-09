# Slim-disk Phase 1 handoff — resume here (2026-06-09)

**Purpose:** the "resume here" pointer for the relativistic slim-disk subsystem after a context compaction. Read this first, then the linked spec/plan/§22-§23.

## Where we are

Building the **relativistic transonic slim-disk subsystem** (spec `specs/2026-06-08-relativistic-slim-disk-design.md`, plan `plans/2026-06-08-relativistic-slim-disk.md`) to render near-Eddington (`f_Edd≈0.9`) disks — the radiation-pressure-dominated inner disk the thin-disk `α(total-P)` model can't solve (Lightman-Eardley fold; advection removes it). Executing **Phase 1** (standalone transonic radial solver) via subagent-driven development.

**Phase 1 progress (module `src/slim_disk_radial.{h,cpp}`, target `test-slim-disk-radial`):**
- **Task 1 (scaffold): DONE/committed** — `SlimDiskInputs`, `SlimDiskRadial`, the interface.
- **Task 2 (Kerr factors): DONE/committed** — `slim_detail::omega_k/calC/calD/calH/omega_perp2/kerr_delta/kerr_A`. `omega_perp2=Ω_K²ℋ` (geodesic, = `VolumetricDisk::omega_z_sq`). Caught/fixed a §22 error here (`Ω_⊥²` was mis-transcribed).
- **Task 3 (one-zone closure): DONE/committed** — `one_zone_closure(Σ,T_c,r,in,op) -> OneZoneState{H,rho_mid,c_s,p_mid,p_gas,p_rad,P,S,mu}`. Self-consistent H (gas+rad quadratic), `P=2 p_mid H` (integrated, trap #9), entropy. Guarded for transient iterates; radiation-dominated test passes.
- **Task 4 (radial residual): DONE/committed** — `slim_radial_residual(U,in,op,R)` (length 4N+2) + `build_thin_disk_seed`. State `U=[Σ,V,ℓ,T_c]×N + [ℓ_in,r_s]`. Rows: N mass + N angmom (algebraic) + N−1 radmom + N−1 energy (trapezoidal ODEs) + 2 outer BC + 2 sonic-point regularity. Opus-verified vs §22/§23; one critical `r_g²` dimensional bug (angmom) caught + fixed.
- **Task 5 (Newton solve): ATTEMPTED, does NOT converge — being RESTRUCTURED (see decision).** The Newton machinery is wired and good (in the working tree, **uncommitted**); it just can't engage the sonic-point eigenvalue. Diagnosis below.

## THE DECISION (next step): restructure the radial solver to the full Sądowski hybrid (option B)

**Root cause of Task-5 non-convergence (opus investigation, conclusive):** `𝒟₀ = V²−Γ̃₁(P/Σ) < 0 at EVERY grid point` — the flow is subsonic across the whole `[r_in,r_out]` domain; **the sonic point lies *inside* `r_in`, off the grid.** So our regularity rows (`𝒟₀(r_s)=0`, `𝒩₁(r_s)=0`, with `r_s` interpolated inside a fixed grid) are **unsatisfiable as posed**. `r_s` parks at the inner edge, `ℓ_in` freezes, energy can't close (`Q_vis∝ℓ−ℓ_in`). Our "interpolate state to a floating r_s buried in a fixed grid + 2 rows" approach is materially different from and more fragile than the published method.

**The proven method (Sądowski 2009 §3 "Numerical method", arXiv:0906.0355; thesis arXiv:1108.0396 Ch.4) — option B, the chosen path:**
1. **The sonic point IS the inner boundary of the relaxation domain.** Relax only `[r_S, r_out]` (`r_out` far out, Novikov-Thorne outer BC), with **`r_S` a free innermost grid node** (free-boundary problem), `ℓ_in` the explicit eigenvalue.
2. **Find `ℓ_in` by two-branch BRACKETING** (a 1-D outer root-find, NOT a Newton row): `ℓ_in` too high → solution terminates as `𝒟₀→0` before regularity; too low → never reaches the sonic point. Bisect between the two topologies to the common limit.
3. Impose `𝒟₀=0 ∧ 𝒩₁=0` AT the inner node `r_S` (no interpolation).
4. **Inward RK4 pass** from `r_S` to the horizon (the supersonic plunge) using the L'Hôpital slope `dV/dr|_{r_S}=lim 𝒩'/𝒟'`. **DEFERRABLE in Phase 1** — the relaxation on `[r_S,r_out]` gives the bright emitting structure; `r<r_S` can use the existing BPT72 plunge treatment for now.

User confirmed: do **B** (the full hybrid) — it's the correct, literature-guaranteed step. Method is validated across the full spin (to `a≈0.998`) and Eddington (`f_Edd` to ~1) range; our implementation is validated by the test suite (below).

## What carries forward vs what changes

**Carries forward (do NOT rebuild):** the verified §22/§23 residual physics; `one_zone_closure`; the Newton machinery from Task 5 — **row+column scaling** (Jacobian columns span ~33 orders: `ℓ_in`~1e34 vs `r_s`~1e2), **Levenberg-Marquardt damping** (the `(Σ,V)` mass-conservation block is near rank-deficient), **state-derived per-group scaled merit** (`slim_group_scales`/`slim_scaled_residual_norm`), the dense FD Jacobian + trust-region + line search + honest non-fabrication fallback. These are sound.

**Changes (the B restructure):**
- Grid: `[r_in,r_out]` fixed → **`[r_S,r_out]` with `r_S` the free innermost node** (node positions depend on `r_S`; numerical Jacobian handles the grid-stretch sensitivity).
- Eigenvalue: 2 interpolated-`r_s` Newton regularity rows → **bracketing outer loop on `ℓ_in`** + regularity at node 0.
- Add the inward RK4 plunge pass (or defer it + use the existing plunge for `r<r_S`).
- The seed band-aid (`T_ref 5e7, Σ×0.03`) was a symptom of the off-grid sonic point — revisit once the eigenvalue engages (the magic-number tuning should go).
- KEEP the good physics fix: `r_in`/`r_S` outside the prograde photon orbit `r_ph=2M[1+cos(⅔arccos(−a))]` (inside it `ℓ_K` diverges).

## Code state for compaction

- **Committed:** Tasks 1-4 (Kerr factors, closure, residual, seed), all formula-ref fixes (§22/§23 incl. the opus re-verification corrections), the refinements-doc updates (#3 DONE, #1b superseded).
- **Uncommitted in the working tree:** the Task-5 Newton machinery in `slim_disk_radial.cpp` + `test_converges_midmdot` in `test_slim_disk_radial.cpp` (non-converging, honest). The column-scaling/LM/merit code is valuable and carries into B — recommend a **WIP commit** ("wip(slim-disk): Newton machinery (column scaling, LM, scaled merit); sonic-point eigenvalue restructure to Sadowski hybrid pending") so it's preserved through compaction, then build B on top.

## Test-coverage requirements (the validation that proves B works across the range)
From the plan's "Test-coverage requirements" section: convergence (Task 5), continuation (Task 9), benchmark (Task 10), integration (Phase 6) must cover **spin `a=0.9` AND near-extremal `a=0.998`** (render spin), and **`f_Edd ∈ {0.9, 0.95, 1.0}`** plus a low-`f_Edd` thin-disk-limit case. Task 8 (thin-disk reduction vs Novikov-Thorne) and Task 10 (Sądowski 2011 figure benchmark) are the source-independent correctness proofs.

## Workflow constraints (carry forward, non-negotiable)
- **NEVER `git commit`** — hand the message to the user (memory `feedback_review_workflow`).
- **Present every reviewer recommendation with a take and WAIT** for the user's call before fixing.
- Subagents **sonnet or opus only, never haiku**. **Use opus for formula/physics verification** (it caught 4 errors sonnet missed in §22/§23 — `𝒟≠Δ/r²`, the garbled radial-momentum line, the `𝒜` symbol collision, the `Ω_⊥` convention) and for the dense physics tasks (residual, Newton solve).
- Two-stage review per task (spec then quality); honest fallbacks, no fabricated profiles, no magic numbers.

## Key references
- Spec: `specs/2026-06-08-relativistic-slim-disk-design.md` (§7 relaxation locked; the architecture).
- Plan: `plans/2026-06-08-relativistic-slim-disk.md` (Phase 1 Tasks 1-10; Phases 2-6 roadmap). **Task 5+ to be revised for the B hybrid.**
- Verified formulas: `references/disk-physics-formulas.md` **§22** (compact laws + convention note) and **§23** (one-zone residual terms: `𝒩₁/𝒟₀`, regularity, `Q_vis`, `Q_rad=64σT_c⁴/(3κ_RΣ)`, `Q_adv`, `f_adv=Q_adv/Q_rad`). Traps #9-12. GRRT uses the **geodesic convention** (`Ω_⊥²=Ω_K²ℋ`); do NOT import S11's `1/𝒞` forms.
- Model survey: `disk-radiation-pressure-model-options.md`.
- Method source: Sądowski 2009 (arXiv:0906.0355) §3; thesis (arXiv:1108.0396) Ch.4; Abramowicz & Fragile 2013 (Living Rev. Rel. 16,1) §6.

## Resume action
After compaction: WIP-commit the Task-5 machinery (if not done), then **implement option B** — restructure `solve_slim_disk_radial` to the Sądowski hybrid (free inner-boundary `r_S` + `ℓ_in` bracketing + regularity at node 0; defer/stub the inward RK plunge). Revise plan Task 5 accordingly. Gate: `test_converges_midmdot` converges at the strict tol for `a=0.9` AND `a=0.998`. Then Tasks 6-10 (regularity test, conservation, thin-disk limit, Ṁ-continuation to `f_Edd∈{0.9,0.95,1.0}`, Sądowski benchmark).
