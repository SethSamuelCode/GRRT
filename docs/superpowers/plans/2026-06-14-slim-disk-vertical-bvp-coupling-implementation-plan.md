# Slim-disk vertical-BVP coupling (POC) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **PROJECT COMMIT CONVENTION (read first):** The human runs every `git commit`. The "Commit" steps below give you the exact message to hand over — **do NOT run `git commit` yourself, and never `git commit --no-verify`.** Stage the listed files, present the message, and wait for the user's call (per the code-review/commit workflow memory).

**Goal:** Cure the proven one-zone closure inadequacy that makes `f_Edd≈0.9` (a=0.9) unreachable, by coupling the existing grey vertical-structure column BVP (`src/disk_column_bvp.cpp`) into each radial node so the radiated flux comes from vertical radiative diffusion (decoupled from H/r) instead of the one-zone `Q_rad = 64σT_c⁴/(3κΣ)`.

**Architecture:** Nested Newton (nonlinear column elimination) via per-column Schur complement. Each radial Newton step fully re-converges one vertical column per node (warm-started) to produce closure outputs `C = {F, z₀, η₃, η₄}`; the radial Newton operates on the reduced system `R_r(U_r, C(U_r))=0` of size `4N+2`, with the reduced (Schur) Jacobian `J_red = ∂R_r/∂U_r + (∂R_r/∂C)·(dC/dU_r)`, where `dC/dU_r` is the analytic implicit-function-theorem column sensitivity (reuses the column's own LU). No global coupled matrix is ever assembled.

**Tech Stack:** C++23, CGS thermodynamics on geometric (G=c=1, M-scaled) mechanics; existing `solve_column_bvp` (dense Newton + analytic `∂R_c/∂U_c`), existing `slim_radial_residual` / `slim_analytic_jacobian` / `relax_structure` machinery; CMake "include-the-.cpp" standalone test/probe pattern; OpenMP not required (POC = correctness, slow-but-robust).

---

## ⚠ AMENDMENT (2026-06-25): `f_adv` is a column OUTPUT, not an input

During C1 execution, source (S11 §3.1–3.2) + a numerical probe (`tools/slim_fadv_freedom_probe.cpp`) established that the vertical structure is a **two-parameter family** `(T_c, f_adv) ≡ (T_c, Σ)`. So the closure map is **`(Σ, T_c) → (F, z₀, η₃, η₄, f_adv)`** — `f_adv` is *determined by* `(Σ, T_c)`, not an independent input. Fixing all three over-determines the column (the cause of the spurious "folds"). The fix: **free `f_adv` as a column unknown** (augmented row-swap). See spec §2 + the reference §22 note (both amended 2026-06-25). **No task is reordered**; the sequence `C1→C2→C3→C5→C4→gates` holds. Per-task deltas:

- **Tasks 1, 2** (f_adv heating term; reusable LU) — DONE/committed, unaffected.
- **Task 3 (C1)** — RE-POSED: state becomes `[Pg,Q,T,z]×N + (z₀, T_eff, f_adv)`; pin `T(0)=T_c` and `Σ0=Σ`; free **both** `T_eff` and `f_adv`. Its "genuine fold" tests become *convergence* tests. The equilibration (Ruiz) + affine-invariant (Deuflhard) Newton from the column-hardening work are folded in as the now-well-conditioned solver.
- **Tasks 4, 5 (C2 moments)** — unaffected (read the converged profile).
- **Task 6 (C3 `∂R_c/∂p`)** — SIMPLIFIES: parameters `p=(Σ, T_c)` only (drop the `f_adv` column); `f_adv` is now in `U_c`, handled by `∂R_c/∂U_c`. The new `∂R_c/∂p` is just the two `−1` pin-row columns.
- **Task 7 (C3 sensitivity)** — SIMPLIFIES: `ColumnSensitivity` drops the `*_dfadv` scalars; `dC/d{Σ,T_c}` only. `C` gains `f_adv` as an output component.
- **Task 8 (C5 𝒩₁)** — unaffected.
- **Task 9 (C4 driver)** — SIMPLIFIES + two pre-C4 checks: the column is self-contained `(Σ,T_c)→…`; **remove** the "compute `f_adv` from the profile and pass it in" plumbing. (a) Add a **consistency check** that the column's emergent `F` + back-solved `f_adv` satisfy the radial `Q_vis − F − Q_adv = 0` (expected automatic: `f_adv = Q_adv/F` when column dissipation = radial `Q_vis` — verify numerically, no new radial equation). (b) **Stale-warm-start regression (C1 review followup, 2026-06-25):** C4 takes the warm-start path, which BYPASSES the 2-D `(T_eff,f_adv)` bring-up and runs the monolithic 4N+4 Newton directly from a possibly-stale `(Σ,T_c)` warm state — the basin the bring-up currently hides. Before trusting C4, add a test driving the monolithic Newton from a deliberately-stale warm start (e.g. previous node's converged `U` applied to a 30%-Σ-shifted target, `allow_continuation=false`); confirm direct convergence, then verify the continuation fallback fires + recovers when it doesn't. Watch per-node cost (a cold node pays the full 2-D bring-up + the augmented solve). The C1 Jacobian's `~3e-4` (→`6e-2` on f_adv-large pairs) inherited opacity-LUT inexactness sets the merit floor — keep it in view for C3's sensitivity accuracy (within Task 7's `1e-3` gate).
- **Tasks 10–12 (gates)** — conceptually unchanged.

---

## Ground-truth interface facts (verified against the real code, 2026-06-14)

These were read directly from the source; tasks below are written against them. Where the spec assumed something the code does differently, it is flagged **[SPEC-DELTA]**.

1. **Column state.** `solve_column_bvp(const ColumnInputs&, const OpacityLUTs&, const std::vector<double>* warm_start)` (`include/grrt/scene/disk_column_bvp.h:54`). State `U` length `4N+2`: per node `i` → `[Pg(=P_gas), Q(=flux), T, z]` at offsets `4i+{0,1,2,3}`; globals `U[4N]=z0`, `U[4N+1]=Sigma0`. `ColumnInputs` = `{T_eff, shear, omega_z, alpha, rho_mid_guess, n_nodes, max_iters, tol}` (`:11`). `ColumnBVPSolution` returns `{q,z,P(total),P_gas,Q,T,rho, z0, Sigma0, tau_mid, converged, iters, final_residual}` (`:24`).

2. **The 6 column BC rows** (`disk_column_bvp.cpp:136-142`, `column_residual`): `Q(0)=0`, `z(0)=0`, `Q(N-1)=σT_eff⁴`, `T(N-1)=T_eff`, `z(N-1)=z0`, and `P_tot(N-1)=(2/3)Ω_z² z0/κ_s`. **`T_eff` is an INPUT and `Sigma0` is an OUTPUT** (a free global unknown solved by the `dz/dq=Σ0/(2ρ)` rows). This is the causality C1 must invert: we want `Σ` fixed and `T_eff` free.

3. **Column heating currently has NO advection reduction.** `node_deriv` (`:102`) sets `d.dQ = (α·shear·P_tot)·dz_dq`. There is no `1/(1+f_adv)` factor. C1 must add it (S11 Eq 13).

4. **Column analytic Jacobian** is `analytic_jacobian` (`:229`), a dense `(4N+2)²` matrix `∂R_c/∂U_c`, factorized each Newton step by `dense_solve` (`:422`, in-place Gaussian elimination with partial pivoting — **no reusable LU factor object is currently kept**; see [SPEC-DELTA-A]). Test hooks: `column_residual_test`, `column_numerical_jacobian_test`, `column_jacobians_test` (`:60,65,70`) expose the analytic vs FD `∂R_c/∂U_c` at the build_seed state.

5. **Radial closure consumption points** (the ONLY places that change):
   - `eval_node` (`slim_disk_radial.cpp:546`) builds `NodeEval` from `one_zone_closure`; everything downstream reads `e.oz.{P,p_gas,p_mid,rho_mid,H,...}` and `beta_of(e.oz)`.
   - `Gbalance` (`:1298`) energy ODE: `Qrad = 64σT_c⁴/(3κΣ)` (`:1328`). **This is the term C4 replaces with the column's emergent F.**
   - `qadv_term_geom`/`Qadv` use `eta3_of_beta(beta_of(oz))` (`:1228,:1333`); `calN1` (`:580`) is the 𝒩₁ assembly that drops the η-gradient + Ω_⊥² terms — **C5 restores them.**
   - Closure thickness `H` is `e.oz.H` (one-zone `solve_H`); C4 reroutes it to `z₀`.
6. **Radial driver** `solve_slim_disk_radial` (`:3643`) runs an Ṁ-ladder + spin homotopy, calling `solve_single_am`/`relax_structure` (`:2640`) which Newton-relaxes with `slim_analytic_jacobian` (`:2756`, env `SLIM_FD_JAC` swaps in the FD oracle). C4 mirrors this driver for the coupled model.
7. **FD-oracle discipline** is enforced by `tests/test_slim_jacobian.cpp` (Richardson-extrapolated central-difference reference, per-column scaled 2-norm gate, `ROWS_ALL`). Column-internal oracle is `column_jacobians_test`. Both are permanent gates.
8. **`shear` and `omega_z` for a column at node i.** `omega_z = sqrt(omega_perp2(M,a,r))·(c_cgs/r_g)` (CGS; mirrors `one_zone_closure:39-41`). `shear = |r dΩ/dr|` in CGS where `Ω=Ω(ℓ)` is the local orbital frequency — computed from the same `dOmega_dr` FD that `Gbalance` already builds (`:1304-1305`), times `r_cm`. These are geometric inputs to the column (NOT radial unknowns), so column sensitivity is needed only w.r.t. `(Σ, T_c)` plus the parameter `f_adv` (chained in C4).

### [SPEC-DELTA] interface mismatches the orchestrator should note
- **[SPEC-DELTA-A] No reusable LU object.** The spec's C3 says "reuse the column's existing LU of `∂R_c/∂U_c`." The column solver factorizes-and-solves in one shot (`dense_solve` destroys `A`), so **there is no persisted factor to reuse.** Resolution adopted in this plan: C3 re-assembles `∂R_c/∂U_c` once at the converged column (cheap, O(n_z) assembly) and factorizes it once, then back-substitutes against each RHS column of `∂R_c/∂p`. This is the same math the spec intends (one factor, many back-subs); it just re-factors rather than reusing a cached factor. New helper `column_lu_factor` + `column_lu_solve` extracted from the existing `dense_solve` so both the column Newton and C3 share one implementation (DRY).
- **[SPEC-DELTA-B] C1 re-pose changes which BC is fixed, not the residual physics. ⇒ ROW-SWAP IS THE PRIMARY PATH (robustness decision, 2026-06-20).** The cleanest inversion keeps the column's 4 ODE rows + 4 of the 6 BC rows, and swaps the two `T_eff`-pinned rows: drop `T(N-1)=T_eff` (free `T_eff`), and replace the global-`Sigma0`-closing freedom by pinning `Sigma0 = Σ_radial`. Concretely (Task 3): add one input `Sigma_target`, add a midplane row `T(0) − T_c = 0` (pins midplane temperature to the radial unknown), and replace the `Sigma0` global unknown's defining row by `Sigma0 − Sigma_target = 0`; `T_eff` becomes the freed global unknown in its slot. The emergent flux is then `F = Q(N-1)` (a state component, no longer pinned). This keeps the state length `4N+2` and the analytic-Jacobian block structure intact — only 2 BC rows and their Jacobian entries change.
  **WHY THE ROW-SWAP, NOT THE SECANT WRAPPER:** the convergence engine's robustness (nested Newton) depends on the C3 sensitivity being ANALYTIC and exact. A secant root-find on `T_eff` (to hit `Σ_target` around the unmodified solver) is NOT cleanly differentiable — it forces FD-through-the-secant (the noisy ceiling this solver fought) AND nests a third iteration level (cost). The row-swap makes the column ONE differentiable Newton solve, so C3's IFT sensitivity is exact and C6's `∂R_c/∂p` export is straightforward (the row-swap is required for `∂R_c/∂p` regardless). **Do the row-swap in Task 3.** Keep the secant wrapper ONLY as a bring-up fallback if the row-swapped column proves hard to converge near β→0 (a converged secant solution can seed the row-swapped Newton). Tasks 3 and 6 should both assume the row-swapped, differentiable column.
- **[SPEC-DELTA-C] η₄ has no transcribed formula in-tree.** `references/disk-physics-formulas.md` §23 gives η₃ explicitly but only *names* η₄ ("vertical weight function", S11 Eqs 8/11). Task 4 (C2) must transcribe η₄ from S11 and gate it with its own moment probe before C5 relies on it. The plan implements the moment integrals generically and isolates the η₄ definition to one function so the transcription is a single, gated edit.

---

## File-structure map

| File | Create/Modify | Responsibility |
|---|---|---|
| `include/grrt/scene/disk_column_coupled.h` | **Create** | Public interface for the coupled-column layer: `ColumnCoupledInputs` (adds `Sigma_target`, `Tc`, `f_adv` to the column inputs), `ColumnClosure` (`{F, z0, eta3, eta4, converged}` + the converged `ColumnBVPSolution`), `ColumnSensitivity` (`dC/d{Σ,Tc}` 4×2 + `dC/df_adv` 4×1), and the C1/C2/C3 entry points: `solve_column_coupled`, `column_moments`, `column_sensitivity`. Test hooks for the FD cross-checks. |
| `src/disk_column_coupled.cpp` | **Create** | C1 (`solve_column_coupled`: the Σ+T_c-driven, f_adv-reduced column re-pose), C2 (`column_moments`: η₃=∫E/∫P and η₄), C3 (`column_sensitivity`: IFT `dU_c/dp` via the shared LU + `∂C/∂U_c`). Houses the new `∂R_c/∂p` export and the extracted `column_lu_factor`/`column_lu_solve`. |
| `src/disk_column_bvp.cpp` | **Modify** | (a) Add `1/(1+f_adv)` to `node_deriv`'s `dQ` (gated by a new `f_adv` member, default 0 ⇒ no behaviour change for existing callers). (b) Extract `dense_solve` into reusable `column_lu_factor`/`column_lu_solve` (no behaviour change). (c) Expose `column_residual` + `analytic_jacobian` to the coupled TU via a thin internal header include (the test pattern already #includes this .cpp; the coupled .cpp will #include it the same way — no symbol export needed). |
| `src/slim_disk_coupled.cpp` | **Create** | C4 (`solve_slim_disk_coupled`: the nested coupled Newton driver) + the coupled residual/Jacobian assembly that reroutes the energy `Qrad→F`, closure `H→z₀`, and 𝒩₁ η-terms (C5) to the column outputs, forming `J_red` per-column. Mirrors `relax_structure`/`solve_single_am` but with the Schur terms. |
| `include/grrt/scene/slim_disk_coupled.h` | **Create** | Declares `solve_slim_disk_coupled(const SlimDiskInputs&, const OpacityLUTs&) → SlimDiskRadial`. |
| `tests/test_column_coupled.cpp` | **Create** | Unit + FD cross-check gates for C1 (re-pose round-trips to the same root), C2 (moments vs direct quadrature; η₃ one-zone limit → `3−1.5β`), C3 (`dC/dp` analytic vs perturb-resolve-column FD oracle). #includes the coupled+column+opacity .cpp directly (probe pattern). |
| `tests/test_slim_jacobian.cpp` | **Modify** | Extend with a coupled-Schur cross-check: the reduced Jacobian's energy/closure/𝒩₁ rows (the column-derived terms) vs an FD oracle that perturbs radial `Σ,T_c`, re-solves the column, and differences `F,z₀,η`. Keeps the existing one-zone gates green. |
| `tools/slim_eta4_moment_probe.cpp` | **Create** | η₄ transcription gate (mirrors the η₃ gate discipline): evaluate `column_moments` on analytic test columns and confirm η₄ matches its S11 definition + the one-zone reduction. |
| `tools/slim_coupled_target_probe.cpp` | **Create** | The POC target gate: run `solve_slim_disk_coupled` at `a=0.9, f_Edd=0.9` and report H/r, β(r), f_adv(r), V<0, sonic-inside-ISCO, validity gates, and the `rad`/`ang` residual groups. Also a coupled re-run of the Sądowski-shape seed residual. |
| `CMakeLists.txt` | **Modify** | Add `src/disk_column_coupled.cpp` and `src/slim_disk_coupled.cpp` to the `grrt` library sources; register `test-column-coupled`, `slim-eta4-moment-probe`, `slim-coupled-target-probe` with the `include/ ${CMAKE_BINARY_DIR}/include third_party/` include-dirs pattern; extend the existing `test-slim-jacobian` target (no new target). |

**Decision: new files, not additions to `slim_disk_radial.cpp`.** That file is already 4281 lines. The coupling is a distinct subsystem (column re-pose + moments + sensitivity + a new driver) with its own gates, so it lives in `src/slim_disk_coupled.cpp` + `src/disk_column_coupled.cpp`. The only edits to existing source are the minimal, behaviour-preserving hooks in `disk_column_bvp.cpp` (the `f_adv` factor + the LU extraction). The coupled driver #includes `slim_disk_radial.cpp`'s machinery the same way the probes do (it reuses `eval_node`, `node_mech`, `script_A`, the LM/line-search, group scales) — see Task 9.

---

## Build & run reference (used by every "run" step)

```bash
# Configure once:
cmake -B build -G "Visual Studio 17 2022"
# Build a target:
cmake --build build --config Release --target <target>
# Run:
build/Release/<target>.exe
```

Targets introduced/used: `test-column-coupled`, `test-slim-jacobian`, `slim-eta4-moment-probe`, `slim-nt-term-probe`, `slim-sadowski-residual-probe`, `slim-coupled-target-probe`.

---

## Task 1 (C1): Add the `f_adv` heating reduction to the column ODE

**Files:**
- Modify: `include/grrt/scene/disk_column_bvp.h:11-20` (add `f_adv` to `ColumnInputs`)
- Modify: `src/disk_column_bvp.cpp:87-105` (`node_deriv`), `:124-125` (residual call sites), `:253` (`node_jac`), `:229` (analytic `dQ` partials)
- Test: `tests/test_column_bvp.cpp` (new case)

The S11 Eq 13 advection-corrected generation, re-derived in GRRT's geodesic convention (§22 note — use the exact-Kerr-shear column-BVP heating `αP·|r dΩ/dr|`, NOT S11's `(3𝒟/2𝒞)(M/r³)^½`): the per-face flux generation becomes `dℱ/dz = α·P_tot·|r dΩ/dr| / (1+f_adv)`. `f_adv` is a radial input (default 0 ⇒ existing behaviour).

- [ ] **Step 1: Write the failing test**

In `tests/test_column_bvp.cpp`, add (and call from `main`):
```cpp
static void test_fadv_reduces_heating() {
    std::printf("\n=== f_adv reduces column heating flux ===\n");
    auto lut = grrt::build_opacity_luts(1e-12, 1e6, 3000.0, 1e8);
    grrt::ColumnInputs base{};
    base.T_eff = 3e5; base.shear = 2e3; base.omega_z = 2e3;
    base.alpha = 0.1; base.rho_mid_guess = 1.0; base.n_nodes = 64;
    base.max_iters = 200; base.tol = 1e-8;
    grrt::ColumnInputs hot = base; hot.f_adv = 0.0;
    grrt::ColumnInputs adv = base; adv.f_adv = 0.5;   // 1/(1+0.5)=2/3 the generation
    auto s0 = grrt::solve_column_bvp(hot, lut);
    auto s1 = grrt::solve_column_bvp(adv, lut);
    if (!s0.converged || !s1.converged) { std::printf("  FAIL: a column did not converge\n"); failures++; return; }
    // With less generation, the surface flux Q(top) for the SAME geometry/Σ-guess is lower.
    const double F0 = s0.Q.back(), F1 = s1.Q.back();
    std::printf("  F(f_adv=0)=%.4e  F(f_adv=0.5)=%.4e  ratio=%.4f (expect ~0.667 at fixed structure)\n",
                F0, F1, F1 / F0);
    // Loose gate: f_adv MUST reduce the heating-limited flux (strict equality of 2/3 only
    // holds at frozen ρ; here Σ0 re-solves, so require a clear monotone reduction).
    if (!(F1 < F0)) { std::printf("  FAIL: f_adv did not reduce flux\n"); failures++; }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build build --config Release --target test-column-bvp && build/Release/test-column-bvp.exe`
Expected: FAIL — `ColumnInputs` has no `f_adv` member (compile error).

- [ ] **Step 3: Minimal implementation**

In `disk_column_bvp.h`, add to `ColumnInputs` (after `alpha`):
```cpp
    double f_adv = 0.0;  ///< radial advected fraction Q_adv/Q_rad; reduces flux generation by 1/(1+f_adv) (S11 Eq 13). Default 0 = thin/no reduction.
```
In `disk_column_bvp.cpp`, thread `f_adv` into `node_deriv` and `node_jac`:
- Change `node_deriv`'s signature to take `double f_adv` and set
  `d.dQ = ( alpha * shear * Ptot / (1.0 + f_adv) ) * dz_dq;`
- At its two call sites in `column_residual` (`:124-125`) pass `in.f_adv`.
- In `analytic_jacobian`'s `node_jac` (`:253`), divide every `dQ_*` partial (`dQ_dP, dQ_dT, dQ_dS`) by `(1.0 + in.f_adv)` (the factor is constant in the state, so it scales each partial identically).

- [ ] **Step 4: Run to verify it passes**

Run: `build/Release/test-column-bvp.exe`
Expected: the new case PASS; all pre-existing cases still PASS.

- [ ] **Step 5: Verify the column Jacobian gate still passes (f_adv default + nonzero)**

Add a one-line assertion path to the existing `column_jacobians_test`-driven check (or run `test-column-bvp`'s Jacobian case) with `f_adv=0.5` to confirm analytic `∂R_c/∂U_c` still matches FD after the divide. Expected: cross-check residual `< 1e-3` (the existing tolerance).

- [ ] **Step 6: Commit** (hand message to human)

Stage: `include/grrt/scene/disk_column_bvp.h src/disk_column_bvp.cpp tests/test_column_bvp.cpp`
Message:
```
feat(slim-column): add S11 Eq 13 f_adv heating reduction to the column BVP

dℱ/dz = αP|r dΩ/dr|/(1+f_adv) (GRRT geodesic convention, §22 note).
Default f_adv=0 preserves existing thin-column behaviour; analytic
∂R_c/∂U_c divides the dQ partials by the same constant factor (gate green).
```

---

## Task 2 (C1): Extract a reusable LU (factor + solve) from `dense_solve`

**Files:**
- Modify: `src/disk_column_bvp.cpp:422-440` (`dense_solve` → `column_lu_factor` + `column_lu_solve`, keep `dense_solve` as a thin wrapper)
- Test: `tests/test_column_bvp.cpp` (new case)

[SPEC-DELTA-A]: C3 needs to factor `∂R_c/∂U_c` once and back-substitute many RHS columns. Provide that primitive now, reusing the proven elimination.

- [ ] **Step 1: Write the failing test**

In `tests/test_column_bvp.cpp`:
```cpp
static void test_lu_multi_rhs() {
    std::printf("\n=== column LU: factor once, solve two RHS ===\n");
    // 3x3 well-conditioned system; A x1 = b1, A x2 = b2 via one factorization.
    std::vector<double> A = {4,1,0, 1,3,1, 0,1,2};
    std::vector<int> piv;
    std::vector<double> LU = A;
    bool ok = grrt::column_lu_factor(LU, piv, 3);
    std::vector<double> b1 = {1,2,3}, b2 = {0,1,0};
    grrt::column_lu_solve(LU, piv, b1, 3);
    grrt::column_lu_solve(LU, piv, b2, 3);
    // Verify A*x ≈ b for both.
    auto resid = [&](const std::vector<double>& x, const std::vector<double>& b){
        double m=0; for(int r=0;r<3;++r){double s=0;for(int c=0;c<3;++c)s+=A[r*3+c]*x[c]; m=std::max(m,std::abs(s-b[r]));} return m; };
    double r1 = resid(b1, {1,2,3}), r2 = resid(b2, {0,1,0});
    std::printf("  ok=%d resid1=%.2e resid2=%.2e\n", (int)ok, r1, r2);
    if (!ok || r1>1e-12 || r2>1e-12) { std::printf("  FAIL\n"); failures++; }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build build --config Release --target test-column-bvp && build/Release/test-column-bvp.exe`
Expected: FAIL — `column_lu_factor`/`column_lu_solve` undefined.

- [ ] **Step 3: Minimal implementation**

In `disk_column_bvp.h` (after `solve_column_bvp` decls), declare:
```cpp
/// Dense LU with partial pivoting, split into reusable factor + solve so one
/// factorization (O(n³)) serves many RHS back-substitutions (O(n²) each) — the
/// IFT column-sensitivity needs ∂R_c/∂p solved against the same ∂R_c/∂U_c factor.
GRRT_EXPORT bool column_lu_factor(std::vector<double>& A, std::vector<int>& piv, int n);
GRRT_EXPORT void column_lu_solve(const std::vector<double>& LU, const std::vector<int>& piv,
                                 std::vector<double>& b, int n);
```
In `disk_column_bvp.cpp`, implement `column_lu_factor` as the elimination loop of the current `dense_solve` (`:423-435`) but **store** the pivot per column in `piv` and leave the multipliers in the strict-lower triangle (standard in-place LU); `column_lu_solve` applies the recorded row swaps to `b`, does the forward substitution with the stored multipliers, then the existing back-substitution (`:438`). Re-implement the existing `dense_solve` as: `column_lu_factor` then `column_lu_solve` (so the column Newton path is unchanged). Keep both `GRRT_EXPORT` namespaced in `grrt`.

- [ ] **Step 4: Run to verify it passes**

Run: `build/Release/test-column-bvp.exe`
Expected: new case PASS; the column solve cases (which now route through the refactor) still PASS — this confirms no behaviour change.

- [ ] **Step 5: Commit** (hand message to human)

Stage: `include/grrt/scene/disk_column_bvp.h src/disk_column_bvp.cpp tests/test_column_bvp.cpp`
Message:
```
refactor(slim-column): split dense_solve into column_lu_factor/solve

Reusable factor-once/solve-many primitive for the IFT column sensitivity.
dense_solve is now factor+solve; the column Newton path is byte-identical.
```

---

## Task 3 (C1): `solve_column_coupled` — the Σ+T_c-driven re-pose

**Files:**
- Create: `include/grrt/scene/disk_column_coupled.h`
- Create: `src/disk_column_coupled.cpp`
- Modify: `CMakeLists.txt` (add `src/disk_column_coupled.cpp` to the `grrt` sources list at `:32`-area; create `test-column-coupled` target)
- Test: `tests/test_column_coupled.cpp` (created here)

[SPEC-DELTA-B]: re-pose so the radial `(Σ, T_c)` are inputs and `T_eff`/`F` float. Mechanism: pin midplane `T(0)=T_c` and `Sigma0=Σ_target`; free `T_eff` (the global slot). The implementation reuses the column's residual/Jacobian and overrides exactly the two affected BC rows + the `T_eff` unknown, then drives the same Newton.

**Concrete row swap** (against `column_residual` `:136-142`):
- Keep ODE rows 0..4N-5, and BC rows `Q(0)=0`, `z(0)=0`, `Q(N-1)=σT_eff⁴`, `z(N-1)=z0`, `P_tot(N-1)=(2/3)Ω_z²z0/κ_s` **unchanged**.
- **Replace** `T(N-1) − T_eff = 0` (surface-T pin) with `T(0) − T_c = 0` (midplane-T pin to the radial input).
- **Replace** the role of the `Sigma0` global: add `Sigma0 − Σ_target = 0` as the closing global row, and promote `T_eff` to the free global unknown that the `Q(N-1)=σT_eff⁴` row now determines.

Because the state length is unchanged (`4N+2`: we reuse `U[4N+1]`'s slot to carry `T_eff` instead of `Sigma0`, and keep `Sigma0` fixed = `Σ_target`), the simplest robust implementation for the POC is an **outer fixed-point on `T_eff`** wrapping the unmodified `T_eff`-driven `solve_column_bvp`: find `T_eff` such that the converged column's `Sigma0(T_eff) = Σ_target` (a smooth, monotone 1-D root — higher `T_eff` ⇒ more heating ⇒ different `Σ0`). This avoids touching `column_residual`'s row layout while delivering the identical coupled root. The 1-D solve is a damped secant on `g(T_eff)=Sigma0(T_eff)−Σ_target`, warm-started from the previous radial iterate's `T_eff`.

> **Implementation note for the worker:** the secant wrapper is the POC choice (robust, reuses everything). It does add an inner iteration per column. If column cost proves intractable (Task 9 timing), the genuine row-swap (pin `T(0)`, free `T_eff` as a global) is the drop-in replacement — read `column_residual:136-142` and `analytic_jacobian:374-394` and swap the two rows there. Both yield the same root; keep the secant unless profiling forces the swap.

- [ ] **Step 1: Write the failing test**

Create `tests/test_column_coupled.cpp`:
```cpp
#define GRRT_EXPORT
#define GRRT_BUILDING_DLL 1
#include "../src/opacity.cpp"
#include "../src/disk_column_bvp.cpp"
#include "../src/disk_column_coupled.cpp"
#include <cstdio>
#include <cmath>
using namespace grrt;
int failures = 0;

static void test_coupled_repose_roundtrip() {
    std::printf("\n=== C1: Sigma+Tc-driven column recovers Teff-driven root ===\n");
    auto lut = build_opacity_luts(1e-12, 1e6, 3000.0, 1e8);
    // 1) Solve a Teff-driven column; read its Sigma0 and midplane Tc.
    ColumnInputs ref{}; ref.T_eff=3e5; ref.shear=2e3; ref.omega_z=2e3;
    ref.alpha=0.1; ref.rho_mid_guess=1.0; ref.n_nodes=96; ref.max_iters=300; ref.tol=1e-8;
    auto s = solve_column_bvp(ref, lut);
    if (!s.converged) { std::printf("  FAIL: reference column did not converge\n"); failures++; return; }
    const double Sigma_target = s.Sigma0, Tc_mid = s.T.front();
    // 2) Drive the coupled column with (Sigma_target, Tc_mid, f_adv=0) and expect F≈σTeff⁴.
    ColumnCoupledInputs ci{}; ci.Sigma_target=Sigma_target; ci.Tc=Tc_mid; ci.f_adv=0.0;
    ci.shear=2e3; ci.omega_z=2e3; ci.alpha=0.1; ci.rho_mid_guess=1.0;
    ci.n_nodes=96; ci.max_iters=300; ci.tol=1e-8;
    ColumnClosure c = solve_column_coupled(ci, lut, nullptr);
    const double F_expect = grrt::constants::sigma_SB*std::pow(ref.T_eff,4);
    const double relF = std::abs(c.F - F_expect)/F_expect;
    const double relz = std::abs(c.z0 - s.z0)/s.z0;
    std::printf("  conv=%d F=%.4e (expect %.4e rel=%.2e)  z0=%.4e (ref %.4e rel=%.2e)\n",
                c.converged, c.F, F_expect, relF, c.z0, s.z0, relz);
    if (!c.converged || relF>1e-3 || relz>1e-3) { std::printf("  FAIL\n"); failures++; }
}
int main(){ test_coupled_repose_roundtrip(); std::printf("\n## %d failure(s) ##\n", failures); return failures?1:0; }
```

- [ ] **Step 2: Run to verify it fails**

Add the target to `CMakeLists.txt`:
```cmake
add_executable(test-column-coupled tests/test_column_coupled.cpp)
target_include_directories(test-column-coupled PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)
```
Run: `cmake -B build -G "Visual Studio 17 2022" && cmake --build build --config Release --target test-column-coupled`
Expected: FAIL — `disk_column_coupled.h`/`.cpp` and `ColumnCoupledInputs`/`ColumnClosure`/`solve_column_coupled` do not exist.

- [ ] **Step 3: Minimal implementation**

Create `include/grrt/scene/disk_column_coupled.h`:
```cpp
#ifndef GRRT_DISK_COLUMN_COUPLED_H
#define GRRT_DISK_COLUMN_COUPLED_H
#include "grrt/scene/disk_column_bvp.h"
#include "grrt/color/opacity.h"
#include "grrt_export.h"
#include <vector>
namespace grrt {

/// Inputs for the Σ+T_c-driven (radially-coupled) column. The radial node fixes
/// Σ and the midplane T_c; the emergent flux F = σT_eff⁴ floats. f_adv reduces
/// the flux generation (S11 Eq 13). Geometric inputs shear/omega_z come from the node.
struct ColumnCoupledInputs {
    double Sigma_target;  ///< Σ [g/cm²] (radial unknown)
    double Tc;            ///< midplane T_c [K] (radial unknown)
    double f_adv;         ///< advected fraction (radial input)
    double shear, omega_z, alpha, rho_mid_guess;
    int n_nodes = 96; int max_iters = 300; double tol = 1e-8;
    double Teff_guess = 0.0;  ///< warm-start for the T_eff secant (0 ⇒ derive from Tc)
};

/// Closure outputs the radial energy/closure/𝒩₁ rows consume.
struct ColumnClosure {
    double F = 0.0;     ///< emergent (top-of-column) flux σT_eff⁴ [erg/cm²/s]
    double z0 = 0.0;    ///< photosphere half-thickness [cm] (→ H)
    double eta3 = 0.0;  ///< ∫E dz / ∫P dz
    double eta4 = 0.0;  ///< S11 Eq 8/11 vertical weight (Task 4)
    double T_eff = 0.0; ///< the converged surface temperature (warm-start carrier)
    bool converged = false;
    ColumnBVPSolution sol;  ///< the converged profile (for C2/C3)
};

/// C1: solve the column with (Σ, T_c, f_adv) fixed and T_eff/F free.
GRRT_EXPORT ColumnClosure solve_column_coupled(const ColumnCoupledInputs& in,
                                               const OpacityLUTs& op,
                                               const std::vector<double>* warm_start);
} // namespace grrt
#endif
```
Create `src/disk_column_coupled.cpp` implementing `solve_column_coupled` as the secant wrapper:
- Build a `ColumnInputs` from the coupled inputs (copy `shear,omega_z,alpha,rho_mid_guess,n_nodes,max_iters,tol,f_adv`).
- Define `g(T_eff)`: set `cin.T_eff=T_eff`, call `solve_column_bvp(cin,op,warm_start)`; if not converged return a large sentinel; else return `sol.Sigma0 − Sigma_target`.
- Seed `T_eff` from `in.Teff_guess` if >0 else from a one-zone inversion: `T_eff0 = Tc * (some factor)` — concretely seed `T_eff0 = Tc` and a second secant point `T_eff1 = 0.9·Tc` (the surface is cooler than the midplane). Damped secant ≤40 iters to `|g|/Σ_target < 1e-9`.
- On success fill `ColumnClosure{ F=sol.Q.back(), z0=sol.z0, T_eff=T_eff*, sol, converged=true }`; `eta3/eta4` left 0 here (filled by C2 in Task 4/5). Return `{converged=false}` on failure (honest, no fabricated profile — mirrors the column contract).

- [ ] **Step 4: Run to verify it passes**

Run: `cmake --build build --config Release --target test-column-coupled && build/Release/test-column-coupled.exe`
Expected: PASS — `F` matches `σT_eff⁴` and `z0` matches the reference column to `<1e-3`.

- [ ] **Step 5: Commit** (hand message to human)

Stage: `include/grrt/scene/disk_column_coupled.h src/disk_column_coupled.cpp tests/test_column_coupled.cpp CMakeLists.txt`
Message:
```
feat(slim-coupled): C1 Σ+T_c-driven column re-pose (solve_column_coupled)

T_eff floats; the radial Σ and midplane T_c are fixed via a damped secant on
Sigma0(T_eff)=Σ_target wrapping the existing T_eff-driven column solver.
Emergent F=σT_eff⁴ decoupled from H/r; z0 is the hydrostatic photosphere.
```

---

## Task 4 (C2): `column_moments` — η₃ = ∫E dz / ∫P dz (and the η₄ slot)

**Files:**
- Modify: `include/grrt/scene/disk_column_coupled.h` (declare `column_moments`)
- Modify: `src/disk_column_coupled.cpp` (implement; call it inside `solve_column_coupled` to fill `eta3`/`eta4`)
- Test: `tests/test_column_coupled.cpp` (new case)

Energy moment (Wolfram ✅): `η₃ = ∫E dz / ∫P dz`, with `E = (3/2)p_gas + 3 p_rad` and `P = p_gas + p_rad` (total). Trapezoidal in `z` over the converged profile. One-zone reduction check: a column with constant β gives `η₃ → 3 − 1.5β`. η₄ is transcribed in Task 5; this task implements η₄'s integral *scaffold* returning a documented placeholder gated to the one-zone limit only.

- [ ] **Step 1: Write the failing test**

In `tests/test_column_coupled.cpp` add:
```cpp
static void test_moments_eta3_onezone_limit() {
    std::printf("\n=== C2: eta3 = ∫E/∫P reduces to 3-1.5β for constant β ===\n");
    // Synthetic profile: constant β via P_gas = β·P, P_rad=(1-β)·P, with P,ρ,z Gaussian-ish.
    const int N=128; const double beta=0.4;
    ColumnBVPSolution s; s.z.resize(N); s.P.resize(N); s.P_gas.resize(N); s.T.resize(N); s.rho.resize(N);
    for (int i=0;i<N;++i){ double q=double(i)/(N-1); double zc=q; double P=std::exp(-zc*zc);
        s.z[i]=zc; s.P[i]=P; s.P_gas[i]=beta*P; s.T[i]=1e5; s.rho[i]=P; }
    double eta3=0, eta4=0; grrt::column_moments(s, eta3, eta4);
    const double expect = 3.0 - 1.5*beta;
    std::printf("  eta3=%.6f expect=%.6f rel=%.2e\n", eta3, expect, std::abs(eta3-expect)/expect);
    if (std::abs(eta3-expect)/expect > 1e-6) { std::printf("  FAIL\n"); failures++; }
}
```
(Register the call in `main`.)

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build build --config Release --target test-column-coupled && build/Release/test-column-coupled.exe`
Expected: FAIL — `column_moments` undefined.

- [ ] **Step 3: Minimal implementation**

Declare in the header:
```cpp
/// C2: vertical moments from a converged profile. η₃=∫E dz/∫P dz with
/// E=(3/2)P_gas+3P_rad, P=total; η₄ per S11 Eq 8/11 (Task 5). Pure function.
GRRT_EXPORT void column_moments(const ColumnBVPSolution& s, double& eta3, double& eta4);
```
Implement in `disk_column_coupled.cpp`:
```cpp
void column_moments(const ColumnBVPSolution& s, double& eta3, double& eta4) {
    using namespace constants;
    const int N = (int)s.z.size();
    double intE=0, intP=0;
    for (int i=0;i+1<N;++i){
        const double dz = s.z[i+1]-s.z[i];
        auto E=[&](int k){ const double Prad=(a_rad/3.0)*std::pow(s.T[k],4);
                           return 1.5*s.P_gas[k] + 3.0*Prad; };       // E=(3/2)P_gas+3P_rad
        auto P=[&](int k){ return s.P[k]; };                          // total pressure
        intE += 0.5*(E(i)+E(i+1))*dz;
        intP += 0.5*(P(i)+P(i+1))*dz;
    }
    eta3 = (intP>0.0) ? intE/intP : 0.0;
    eta4 = 0.0;   // Task 5 transcribes S11 Eq 8/11; gated by slim-eta4-moment-probe.
}
```
Call `column_moments(c.sol, c.eta3, c.eta4)` at the end of `solve_column_coupled` (success path).

- [ ] **Step 4: Run to verify it passes**

Run: `build/Release/test-column-coupled.exe`
Expected: PASS — `eta3` matches `3−1.5β` to `<1e-6`.

- [ ] **Step 5: Commit** (hand message to human)

Stage: `include/grrt/scene/disk_column_coupled.h src/disk_column_coupled.cpp tests/test_column_coupled.cpp`
Message:
```
feat(slim-coupled): C2 vertical moment η₃=∫E dz/∫P dz (column_moments)

E=(3/2)P_gas+3P_rad, P=total; trapezoidal in z. One-zone limit → 3−1.5β
(gated). η₄ slot stubbed for Task 5's S11 Eq 8/11 transcription.
```

---

## Task 5 (C2): Transcribe + gate η₄ (S11 Eqs 8/11)

**Files:**
- Modify: `src/disk_column_coupled.cpp` (`column_moments` η₄ branch)
- Create: `tools/slim_eta4_moment_probe.cpp`
- Modify: `CMakeLists.txt` (register `slim-eta4-moment-probe`)

[SPEC-DELTA-C — RESOLVED]: η₄ is the **density second moment about the midplane** (S11; verified vs the primary source 2026-06-14):
```
η₄ ≡ (1/Σ) ∫₀^h ρ z² dz                         [Σ = 2∫₀^h ρ dz = Sigma0]
```
It is purely a density moment — the `Ω_⊥²` is NOT inside the integral (it multiplies η₄ in the 𝒩₁ term, C5). Physical check (use as a sanity gate): `Ω_⊥²·η₄ = 2 ×` the specific vertical gravitational PE, since the vertical PE per area is `∫ρ·½Ω_⊥²z² dz = ½Ω_⊥²Σ·η₄` — matching S11's "vertical motion of gravitational potential energy" description of the `Ω_⊥²η₄` term. No `Ω_⊥²=Ω_K²ℋ`-vs-`(M/r³)(ℋ/𝒞)` convention issue arises (η₄ has no Ω in it). One-zone reductions for the gate: a uniform-ρ column of half-thickness `h` gives `η₄ = h²/3` (`Σ=2ρh`, `∫ρz²=ρh³/3`); a Gaussian `ρ∝e^{−z²/2H²}` gives `η₄ = H²`.

- [ ] **Step 1: Write the failing gate (the probe)**

Create `tools/slim_eta4_moment_probe.cpp` (include-the-.cpp pattern): build 2-3 analytic columns (one gas-dominated constant-β, one radiation-dominated) and assert `η₄` equals the value computed by an independent reference quadrature transcribed straight from S11 Eq 11 (write the reference inline so the probe is self-checking), plus the documented one-zone reduction (S11's one-zone η₄ value at β=1 and β=0). Print PASS/FAIL and a nonzero exit on failure.

- [ ] **Step 2: Run to verify it fails**

Add to `CMakeLists.txt`:
```cmake
add_executable(slim-eta4-moment-probe tools/slim_eta4_moment_probe.cpp)
target_include_directories(slim-eta4-moment-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)
```
Run: `cmake --build build --config Release --target slim-eta4-moment-probe && build/Release/slim-eta4-moment-probe.exe`
Expected: FAIL — `column_moments` returns `eta4=0`, ≠ the S11 reference.

- [ ] **Step 3: Minimal implementation**

Replace the `eta4 = 0.0;` line in `column_moments` with the resolved S11 density second moment `η₄ = (1/Σ)∫₀^h ρz²dz` (trapezoidal over the converged profile's `z[]`, `rho[]`; `Σ = s.Sigma0`):
```cpp
// η₄ ≡ (1/Σ) ∫₀^h ρ z² dz  (S11 density second moment about the midplane; verified 2026-06-14).
double m2 = 0.0;
for (size_t i = 0; i + 1 < s.z.size(); ++i) {
    const double dz = s.z[i+1] - s.z[i];
    m2 += 0.5 * (s.rho[i]*s.z[i]*s.z[i] + s.rho[i+1]*s.z[i+1]*s.z[i+1]) * dz;  // ∫ρz²dz, one face
}
eta4 = (2.0 * m2) / std::max(s.Sigma0, 1e-300);   // ×2: both faces, matching Σ0=2∫ρdz convention
```
(Match the both-faces convention to whatever `Σ0=2∫₀^h ρdz` the column uses, so the `1/Σ` normalization is consistent — verify against the probe's one-zone reductions `h²/3` uniform, `H²` Gaussian.)

- [ ] **Step 4: Run to verify it passes**

Run: `build/Release/slim-eta4-moment-probe.exe`
Expected: PASS — η₄ matches the S11 reference + the one-zone reductions.

- [ ] **Step 5: Commit** (hand message to human)

Stage: `src/disk_column_coupled.cpp tools/slim_eta4_moment_probe.cpp CMakeLists.txt`
Message:
```
feat(slim-coupled): transcribe + gate η₄ (S11 Eqs 8/11)

Vertically-integrated η₄ in GRRT's Ω_⊥²=Ω_K²ℋ convention (NOT S11's
(M/r³)(ℋ/𝒞)). slim-eta4-moment-probe is the transcription gate (one-zone
reductions + S11 reference quadrature).
```

---

## Task 6 (C3): Export `∂R_c/∂p` (the column residual's Σ/T_c/f_adv sensitivity)

**Files:**
- Modify: `include/grrt/scene/disk_column_coupled.h` (declare `column_dRc_dp`)
- Modify: `src/disk_column_coupled.cpp` (implement)
- Test: `tests/test_column_coupled.cpp` (new case)

IFT (Wolfram ✅): `dU_c/dp = −(∂R_c/∂U_c)⁻¹ (∂R_c/∂p)`. The new export is `∂R_c/∂p`, where `p=(Σ_target, T_c, f_adv)` for the **coupled** residual. Since Task 3's POC C1 is the secant wrapper around the `T_eff`-driven `column_residual`, the cleanest exact `∂R_c/∂p` is built against the **re-posed residual** whose unknowns are `U_c=[Pg,Q,T,z]×N+[z0,T_eff]` and whose closing rows are `T(0)−T_c=0` and `Sigma0(U)−Σ_target=0`. Concretely:
- `∂R_c/∂Σ_target`: only the `Sigma0−Σ_target` row depends on `Σ_target` → a column that is `−1` in that row, `0` elsewhere.
- `∂R_c/∂T_c`: only the `T(0)−T_c` row → `−1` in that row, `0` elsewhere.
- `∂R_c/∂f_adv`: the ODE `dQ` rows (`:129`) carry `1/(1+f_adv)`; `∂(dQ/dq)/∂f_adv = −(α·shear·P_tot)·dz_dq/(1+f_adv)²` at each node → fill the `Q`-difference rows analytically (the same `node_deriv` terms, differentiated by the scalar factor).

> **Worker note:** Task 3 ships the secant POC, which does NOT expose the re-posed residual `U_c`. For C3 you need the re-posed `column_residual`/`analytic_jacobian` over `U_c=[...,z0,T_eff]`. Implement the row-swapped residual+Jacobian here (it is the genuine C1 form from [SPEC-DELTA-B]) as `coupled_column_residual`/`coupled_column_jacobian` in `disk_column_coupled.cpp`, reusing `node_deriv`/`node_jac` from `disk_column_bvp.cpp` (which the coupled .cpp already #includes via the test/probe pattern). `solve_column_coupled` can then optionally use this Newton directly; keep the secant as the fallback. This is the place the deeper C1 change lands if profiling demanded it — do it here because C3 needs it anyway.

- [ ] **Step 1: Write the failing test**

In `tests/test_column_coupled.cpp`: build a converged coupled column; assemble `∂R_c/∂p` analytically; cross-check each of the 3 parameter columns against a central FD of `coupled_column_residual` over `(Σ_target, T_c, f_adv)`:
```cpp
static void test_dRc_dp_vs_fd() {
    std::printf("\n=== C3a: ∂R_c/∂p analytic vs FD ===\n");
    // ... build converged coupled column U_c at (Σ,Tc,f_adv) ...
    // For each p in {Σ_target,Tc,f_adv}: analytic col vs (R(U_c;p+h)−R(U_c;p−h))/2h.
    // Gate: per-column 2-norm rel < 1e-7 for Σ_target,Tc (exact -1 rows) and < 1e-4 for f_adv.
}
```

- [ ] **Step 2: Run to verify it fails** — undefined `column_dRc_dp`/`coupled_column_residual`. Expected FAIL.

- [ ] **Step 3: Minimal implementation** — implement `coupled_column_residual`, `coupled_column_jacobian`, and `column_dRc_dp(U_c, in, op, std::vector<double>& dRdp /* n×3 */)` per the formulas above.

- [ ] **Step 4: Run to verify it passes** — Expected PASS (Σ_target/T_c columns match to round-off; f_adv column `<1e-4`).

- [ ] **Step 5: Commit** (hand message to human)

Stage: `include/grrt/scene/disk_column_coupled.h src/disk_column_coupled.cpp tests/test_column_coupled.cpp`
Message:
```
feat(slim-coupled): C3a export ∂R_c/∂p (column residual param-sensitivity)

Re-posed coupled column residual over U_c=[...,z0,T_eff]; ∂R_c/∂{Σ,T_c} are the
−1 closing rows, ∂R_c/∂f_adv differentiates the dQ generation. Analytic-vs-FD
gate green (the IFT input for dC/dp).
```

---

## Task 7 (C3): `column_sensitivity` — `dC/d{Σ,T_c}` via IFT, FD-cross-checked

**Files:**
- Modify: `include/grrt/scene/disk_column_coupled.h` (declare `ColumnSensitivity`, `column_sensitivity`)
- Modify: `src/disk_column_coupled.cpp` (implement)
- Test: `tests/test_column_coupled.cpp` (the C3 oracle — perturb-resolve-column FD)

`dU_c/dp = −(∂R_c/∂U_c)⁻¹(∂R_c/∂p)` (reuse one LU factor via Task 2's `column_lu_factor`/`column_lu_solve`), then `dC/dp = (∂C/∂U_c)(dU_c/dp)` where `C={F,z0,η3,η4}` and:
- `∂F/∂U_c`: `F=Q(N-1)` ⇒ `∂F/∂Q(N-1)=1`, else 0.
- `∂z0/∂U_c`: `z0` is a state global ⇒ `∂z0/∂(z0 slot)=1`, else 0.
- `∂η3/∂U_c`, `∂η4/∂U_c`: differentiate `column_moments`'s trapezoidal integrals w.r.t. the profile `(P_gas,T,z)` entries (analytic; the integrands are explicit in `P_gas`, `T` via `P_rad`, and `z` via `dz`).

The permanent oracle (mirrors `test-slim-jacobian`/`column_jacobians_test`): perturb radial `Σ` and `T_c`, **re-solve the whole coupled column**, and difference `F,z0,η3,η4`.

- [ ] **Step 1: Write the failing test** — `test_dC_dp_vs_resolve_oracle`:
```cpp
static void test_dC_dp_vs_resolve_oracle() {
    std::printf("\n=== C3b: dC/d{Σ,Tc} analytic (IFT) vs perturb-resolve oracle ===\n");
    // base coupled solve at (Σ,Tc,f_adv); analytic ColumnSensitivity sens.
    // For p in {Σ,Tc}: re-solve coupled column at p±h (warm-started), difference F,z0,η3,η4.
    // Gate: each of dF,dz0,dη3,dη4 rel < 1e-3 (re-solve FD noise floor).
}
```

- [ ] **Step 2: Run to verify it fails** — `column_sensitivity`/`ColumnSensitivity` undefined. Expected FAIL.

- [ ] **Step 3: Minimal implementation** — declare:
```cpp
struct ColumnSensitivity { double dF[2], dz0[2], deta3[2], deta4[2]; double dF_dfadv, dz0_dfadv, deta3_dfadv, deta4_dfadv; };
GRRT_EXPORT ColumnSensitivity column_sensitivity(const ColumnClosure& c,
                                                 const ColumnCoupledInputs& in,
                                                 const OpacityLUTs& op);
```
Implement: assemble `coupled_column_jacobian` at `c.sol`'s `U_c`, factor with `column_lu_factor`; get `∂R_c/∂p` (Task 6); solve `dU_c/dp = −LU⁻¹ ∂R_c/∂p` for the 3 RHS; apply `∂C/∂U_c` to get the 4×3 `dC/dp`. Pack `[Σ,T_c]` into the `[2]` arrays and `f_adv` into the scalars.

- [ ] **Step 4: Run to verify it passes** — Expected PASS (all four outputs' `dC/d{Σ,T_c}` match the re-solve oracle `<1e-3`).

- [ ] **Step 5: Commit** (hand message to human)

Stage: `include/grrt/scene/disk_column_coupled.h src/disk_column_coupled.cpp tests/test_column_coupled.cpp`
Message:
```
feat(slim-coupled): C3b dC/d{Σ,T_c,f_adv} via IFT (column_sensitivity)

dU_c/dp=−(∂R_c/∂U_c)⁻¹∂R_c/∂p (one LU, three back-subs) then dC/dp=(∂C/∂U_c)dU_c/dp.
Permanent oracle: perturb radial Σ,T_c → re-solve column → difference F,z0,η3,η4.
```

---

## Task 8 (C5): Restore the dropped 𝒩₁ η-gradient terms

**Files:**
- Modify: `src/slim_disk_coupled.cpp` (the coupled `calN1` assembly — added here, NOT in `slim_disk_radial.cpp`)
- Test: `tests/test_slim_jacobian.cpp` (the coupled-Schur cross-check covers it) + a focused unit assertion

Restored 𝒩₁ terms (S11 Eqs 29/32-33, source-verified, §23 lines 234-236): add `(P/Σ)·dlnη₃/dlnr + Ω_⊥²·(η₄/η₃)·dlnη₄/dlnr` to the coupled `calN1`, re-derived in GRRT's `Ω_⊥²=Ω_K²ℋ` convention. `η₃,η₄` and their radial gradients now exist per node (C2; gradients via FD across nodes, like the existing `dln` helper at `:1213`). The one-zone `calN1` (`slim_disk_radial.cpp:580`) drops both (documented "η-grad & Ω_⊥² drop").

> **Note:** C5 lands in the **coupled** assembly (`slim_disk_coupled.cpp`), not the one-zone `slim_disk_radial.cpp`, because the η-gradients are only meaningful once the columns supply state-dependent η₃(r),η₄(r). The one-zone path keeps its (correct, constant-η ⇒ zero-gradient) form.

- [ ] **Step 1: Write the failing test**

In the coupled residual unit test (add to `test_slim_jacobian.cpp`'s coupled section, Task 9/10, or a focused case here): construct two adjacent coupled nodes with hand-set η₃,η₄ varying in r, and assert the coupled `calN1` includes the analytic `(P/Σ)dlnη₃/dlnr + Ω_⊥²(η₄/η₃)dlnη₄/dlnr` contribution (compare to a by-hand value). Expected FAIL before C5 is added (the term is absent).

- [ ] **Step 2: Run to verify it fails** — Expected FAIL (term missing / function undefined).

- [ ] **Step 3: Minimal implementation**

In `slim_disk_coupled.cpp`'s `calN1_coupled`, after the existing `A_term + Qadv_geom + press_term`, add:
```cpp
// C5: restored S11 Eq 29/32-33 η-gradient terms (GRRT Ω_⊥²=Ω_K²ℋ convention).
const double dlneta3 = dln(eta3_i, eta3_j, r_i, r_j);   // FD across nodes
const double dlneta4 = dln(eta4_i, eta4_j, r_i, r_j);
const double Omega_perp2_geom = omega_perp2(in.mass, in.spin, e.r);   // [1/M²], geometric
const double eta_term = e.P_over_Sigma_geom * dlneta3
                      + Omega_perp2_geom * (eta4_i / eta3_i) * dlneta4;   // both dimensionless
return A_term + Qadv_geom + press_term + eta_term;
```
(Use the per-node η₃,η₄ from the column closures stored by C4; `Omega_perp2_geom` is dimensionless-consistent with the other 𝒩₁ terms since `P_over_Sigma_geom` is already `/c²` and the §23 form uses the geometric Ω_⊥².)

- [ ] **Step 4: Run to verify it passes** — Expected PASS (calN1 includes the restored terms; the value matches the by-hand reference).

- [ ] **Step 5: Commit** (hand message to human)

Stage: `src/slim_disk_coupled.cpp tests/test_slim_jacobian.cpp`
Message:
```
feat(slim-coupled): C5 restore 𝒩₁ η-gradient terms (S11 Eq 29/32-33)

(P/Σ)dlnη₃/dlnr + Ω_⊥²(η₄/η₃)dlnη₄/dlnr in GRRT's Ω_⊥²=Ω_K²ℋ convention,
using per-node η₃,η₄ from the columns (C2). One-zone path keeps its zero-gradient form.
```

---

## Task 9 (C4): The nested coupled Newton driver `solve_slim_disk_coupled`

**Files:**
- Create: `include/grrt/scene/slim_disk_coupled.h`
- Create: `src/slim_disk_coupled.cpp` (the driver + coupled residual/Jacobian; C5 from Task 8 already lives here)
- Modify: `CMakeLists.txt` (add `src/slim_disk_coupled.cpp` to `grrt` sources)
- Test: `tests/test_column_coupled.cpp` (a "links + returns honest fallback on impossible input" smoke) and `test_slim_jacobian.cpp` (the Schur gate, Task 10)

C4 orchestrates §2.1: per radial Newton step, for each node (warm-started) run `solve_column_coupled` (C1) + `column_moments` (C2, already inside C1) + `column_sensitivity` (C3); reroute the energy row (`Qrad → F`), closure (`H → z₀`), and 𝒩₁ η-terms (C5) to the column outputs; form the reduced Schur Jacobian `J_red = ∂R_r/∂U_r + (∂R_r/∂C)(dC/dU_r)` per-column; LM-damped radial solve + feasibility line search (extended to require all columns converged). Mirror `relax_structure` (`:2640`) and `solve_single_am` (`:3254`).

**Reuse strategy (the probe pattern):** `slim_disk_coupled.cpp` `#include "../src/slim_disk_radial.cpp"` is NOT possible inside the library (duplicate symbols). Instead, the driver reuses the radial machinery by calling the **public** `slim_radial_residual` for the non-column rows and re-implementing only the column-derived row overrides + the Schur Jacobian terms locally. The per-node mechanics it needs (`node_mech`, `script_A`, `omega_from_ell`, `eval_node`, group scales) are reached by promoting the handful actually required to `slim_detail` (already partly there: `omega_from_ell`, `ell_kepler`, `isco_prograde`, `omega_perp2` are exported). **Worker action:** if a needed helper (`eval_node`, `node_mech`, `script_A`, `slim_group_scales`, `relax_structure`'s LM loop) is in the anonymous namespace, either (a) promote it to `slim_detail` in the header (preferred for `node_mech`/`script_A` — small, pure), or (b) duplicate the minimal LM/line-search loop in the coupled driver (it is ~60 lines). Choose (a) for pure helpers and (b) for the driver loop to avoid coupling the two drivers. Justify the choice inline.

**Coupling causality (data flow), per radial Newton step:**
1. Read `U_r` → per node `(Σ_i, T_c,i)`; compute `f_adv,i` from the current profile (the same `slim_fadv_ok` path at `:2510-2529`, but returning the value).
2. For each node: `ColumnClosure c_i = solve_column_coupled({Σ_i,T_c,i,f_adv,i, shear_i,Ω_z,i,α,...}, op, warm_i)`; if any `!converged`, **reject the step in the line search** (shrink); persistent failure → fall back to one-zone `Q_rad` for that node with a logged flag (spec §4 graceful degrade).
3. `ColumnSensitivity s_i = column_sensitivity(c_i, ...)`.
4. Assemble the coupled residual: start from `slim_radial_residual`, then **overwrite** the energy rows' `Qrad` with `c_i.F`, the closure `H` with `c_i.z0`, and the 𝒩₁ with `calN1_coupled` (C5). (Implement a `slim_coupled_residual` that mirrors `Gbalance` but substitutes `F` for the `64σT_c⁴/(3κΣ)` term and uses `c_i.{eta3,eta4,z0}`.)
5. Assemble `J_red`: `∂R_r/∂U_r` (the analytic radial Jacobian's structure, but with the energy/closure/𝒩₁ rows' `∂/∂{Σ,T_c}` replaced by `(∂R_r/∂C)·s_i` — the Schur term `A − B D⁻¹ C` formed per-column). Cross-checked in Task 10 against FD.
6. LM-damped Newton step; feasibility line search (Σ>0, T>0, 1+f_adv>ε, V<0, r_s<r_isco, all columns converged).

- [ ] **Step 1: Write the failing test** (smoke: links + honest fallback)

In `tests/test_column_coupled.cpp`:
```cpp
static void test_coupled_driver_links_and_fallback() {
    std::printf("\n=== C4: solve_slim_disk_coupled links + honest fallback ===\n");
    auto lut = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);
    SlimDiskInputs in{}; in.mass=1; in.spin=0.9; in.alpha=0.1; in.r_g=1.48e6;
    in.r_out=50; in.n_nodes=16; in.max_iters=20; in.budget_wall_seconds=30;
    in.mdot = 0.0;  // impossible/degenerate → must return converged=false, no crash
    SlimDiskRadial out = solve_slim_disk_coupled(in, lut);
    std::printf("  converged=%d (expect honest result, no crash)\n", out.converged);
    // No assertion on converged here (mdot=0 is degenerate); the gate is "returns without UB".
}
```

- [ ] **Step 2: Run to verify it fails** — `solve_slim_disk_coupled`/`slim_disk_coupled.h` undefined. Expected FAIL (compile).

- [ ] **Step 3: Minimal implementation** — create the header + driver per the data flow above. Reuse `SlimDiskInputs`/`SlimDiskRadial`. Honor the `SolveBudget` (mirror `:3653`). Return `SlimDiskRadial{converged=false}` on non-convergence (no fabricated profile).

- [ ] **Step 4: Run to verify it passes** — Expected PASS (returns cleanly).

- [ ] **Step 5: Commit** (hand message to human)

Stage: `include/grrt/scene/slim_disk_coupled.h src/slim_disk_coupled.cpp tests/test_column_coupled.cpp CMakeLists.txt`
Message:
```
feat(slim-coupled): C4 nested coupled Newton driver (solve_slim_disk_coupled)

Per radial step: warm-started column solve per node (C1/C2/C3), energy Qrad→F,
closure H→z0, 𝒩₁ η-terms (C5), reduced Schur Jacobian per-column, LM + feasibility
line search (rejects steps with any non-converged column). Honest fallback on failure.
```

---

## Task 10 (Gate): Extend `test-slim-jacobian` to the coupled Schur rows

**Files:**
- Modify: `tests/test_slim_jacobian.cpp` (new coupled section + a `coupled` operating point)

The reduced Schur Jacobian's energy/closure/𝒩₁ rows (the column-derived terms `dC/dU_r`) MUST match an FD oracle that perturbs radial `Σ,T_c`, re-solves the column, and differences the assembled coupled residual. This is the rigorous correctness gate for C3+C4 (mirrors the existing Richardson-FD discipline). The existing one-zone gates stay green.

- [ ] **Step 1: Write the failing test**

Add `run_coupled_point(op, "coupled a=0.9 f_Edd=0.20", 0.9, 0.20, N)`: assemble `J_red` from the coupled driver's Jacobian routine at the seed state; compute the FD reference by perturbing each radial `Σ[i],T_c[i]` column, calling the **coupled** residual (which re-solves the columns), and Richardson-differencing. Gate the energy/closure/𝒩₁ rows' column-derived entries at per-column scaled 2-norm `< 1e-3` (the re-solve-FD floor is looser than the one-zone `1e-6`).

- [ ] **Step 2: Run to verify it fails** — before `J_red`'s Schur terms are exposed/correct. Expected FAIL.

- [ ] **Step 3: Minimal implementation** — expose the coupled Jacobian assembly (a `slim_coupled_jacobian` callable from the test via the include-the-.cpp pattern: the test `#include`s `src/slim_disk_coupled.cpp` + `src/disk_column_coupled.cpp` + `src/disk_column_bvp.cpp` + `src/opacity.cpp`). Wire the FD oracle.

- [ ] **Step 4: Run to verify it passes** — Expected PASS (Schur rows match the re-solve FD `<1e-3`); the three original one-zone points still PASS (0 failures total).

- [ ] **Step 5: Commit** (hand message to human)

Stage: `tests/test_slim_jacobian.cpp`
Message:
```
test(slim-coupled): extend test-slim-jacobian to the reduced Schur rows

FD oracle perturbs radial Σ,T_c → re-solves columns → differences the coupled
residual; gates the energy/closure/𝒩₁ column-derived Jacobian terms. One-zone
gates unchanged (0 failures).
```

---

## Task 11 (Gate): NT-reduction stays green under the coupled model

**Files:**
- Modify: `tools/slim_nt_term_probe.cpp` (add a coupled-flux branch) OR add a coupled assertion to `slim-coupled-target-probe` (Task 12). Prefer extending the NT probe.

The coupled emergent `F` must reduce to the thin-disk flux in the gas-dominated, optically-thick limit so the NT band holds: `F → 64σT_c⁴/(3κΣ)·f_F` with `f_F≈0.94`. At `a=0.9, f_Edd=0.02` the coupled `Q_vis/F_NT` must stay in the post-#12 band `≈1.1–1.2`, flat.

- [ ] **Step 1: Write the failing assertion** — in `slim_nt_term_probe.cpp`, add a branch that builds a coupled column at the NT state's `(Σ,T_c,f_adv≈0)` and asserts `|F_column / (64σT_c⁴/(3κΣ)) − 0.94| < 0.06` across the NT radii, and that `Q_vis/F_column` stays `1.1–1.2`. Expected FAIL until the coupled column is wired into the probe.

- [ ] **Step 2: Run to verify it fails** — Expected FAIL.

- [ ] **Step 3: Minimal implementation** — wire `solve_column_coupled` into the probe at the NT state; print the band.

- [ ] **Step 4: Run to verify it passes** — Expected PASS (`f_F≈0.94±0.06`; band `1.1–1.2`). **If the band drifts, STOP** — this is the spec's NT gate and a drift is a physics regression, not a tolerance to loosen.

- [ ] **Step 5: Commit** (hand message to human)

Stage: `tools/slim_nt_term_probe.cpp`
Message:
```
test(slim-coupled): NT-reduction gate — coupled F reduces to thin-disk flux

Coupled emergent F = 64σT_c⁴/(3κΣ)·f_F (f_F≈0.94) in the gas-dominated limit;
Q_vis/F stays in the 1.1–1.2 band at a=0.9,f_Edd=0.02 (post-#12, flat).
```

---

## Task 12 (Target gate): land a physical `f_Edd≈0.9, a=0.9` disk + coupled Sądowski re-run

**Files:**
- Create: `tools/slim_coupled_target_probe.cpp`
- Modify: `CMakeLists.txt` (register `slim-coupled-target-probe`)

The POC's definition of done (spec §5/§8). Run `solve_slim_disk_coupled` at `a=0.9, f_Edd=0.9` (N=48, modest `n_z`) and verify: H/r ≲ 0.5; gas-dominated outward (β→~1); f_adv ~ +0.3 inner → 0 outward; V<0; sonic inside ISCO; T_c physically determined; all validity gates passing; the `rad`/`ang` residual groups driven from O(200–300) (the one-zone `slim-sadowski-residual-probe` values) down toward the 1e-3 floor. Also re-run the Sądowski-shape seed residual under the coupled residual and confirm the groups collapse toward the floor.

- [ ] **Step 1: Write the gate (the probe)**

Create `tools/slim_coupled_target_probe.cpp` (include-the-.cpp pattern). It (a) solves the coupled disk at `a=0.9, f_Edd=0.9`; (b) prints H/r(r), β(r), f_adv(r), V(r), r_sonic vs r_isco, the validity-gate booleans, and the final `rad`/`ang`/`merit` group magnitudes; (c) feeds the Sądowski-shape seed (the same one `slim-sadowski-residual-probe` uses) to `slim_coupled_residual` and prints the group magnitudes. Exit nonzero if any hard gate fails (H/r>0.5 anywhere, any β never reaching gas-dominated outward, V≥0, r_sonic≥r_isco, or `rad`/`ang` not collapsed below a documented threshold).

- [ ] **Step 2: Run to verify it (initially) reports the gap**

Add to `CMakeLists.txt`:
```cmake
add_executable(slim-coupled-target-probe tools/slim_coupled_target_probe.cpp)
target_include_directories(slim-coupled-target-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)
```
Run: `cmake --build build --config Release --target slim-coupled-target-probe && build/Release/slim-coupled-target-probe.exe`
Expected (first run): may FAIL if the branch doesn't yet land — that is the POC's decisive measurement, not a code bug. Record the residual-group magnitudes.

- [ ] **Step 3: Drive to the gate**

This is the integration step, not new physics: if the coupled solve lands the physical `f_Edd≈0.9` disk with the groups at the floor → **outcome 1 (closure cured, root reachable).** If C1–C5 gates are all green (Tasks 10/11 PASS) but the solve still won't land it → **outcome 2 (instability/branch-reachability, refinements #10), NOT a regression** — record this verdict in the probe output. Do NOT loosen the NT or FD-Jacobian gates to force a pass (per `superpowers:systematic-debugging` / the spec §8 honest-outcome discipline).

- [ ] **Step 4: Run to confirm the recorded outcome**

Run: `build/Release/slim-coupled-target-probe.exe`
Expected: a decisive, recorded result — either the physical disk passes all hard gates (outcome 1), or the gate-clean-but-unlanded verdict is printed (outcome 2).

- [ ] **Step 5: Commit** (hand message to human)

Stage: `tools/slim_coupled_target_probe.cpp CMakeLists.txt`
Message:
```
test(slim-coupled): POC target gate — physical f_Edd≈0.9 a=0.9 disk + Sądowski re-run

Reports H/r, β, f_adv, V, sonic-vs-ISCO, validity gates, rad/ang group collapse;
coupled re-run of the Sądowski-shape seed. Decisive POC outcome (cured-and-reachable
vs gate-clean-but-instability), per spec §8 — gates are NOT loosened to force a pass.
```

---

## Self-review (run against the spec)

**Spec coverage:** C1 → Tasks 1,3 (f_adv reduction + Σ/T_c re-pose). C2 → Tasks 4,5 (η₃ + η₄ gated). C3 → Tasks 2,6,7 (LU primitive + ∂R_c/∂p + IFT dC/dp with the perturb-resolve oracle). C5 → Task 8 (𝒩₁ restoration). C4 → Task 9 (nested driver). Gates → Task 10 (extended FD-Jacobian Schur), Task 11 (NT band 1.1–1.2 + f_F≈0.94), Task 12 (target f_Edd≈0.9 + coupled Sądowski re-run). Error handling (spec §4: column non-convergence rejection + one-zone fallback flag, feasibility line search) → Task 9 data-flow steps 2/6. Mass-independence (spec §1) → inherited (geometric grid/Jacobian untouched). Honest fallback → Tasks 3,9,12.

**Task order matches the spec:** C1 → C2 → C3 (with FD oracle) → C5 → C4 → coupled gates. ✓

**Placeholder scan:** the one genuine in-situ item (η₄, S11 Eq 8/11) is isolated to Task 5 with the exact equations to read and its own gate — flagged as [SPEC-DELTA-C], not a vague TODO. The C1 re-pose has both the POC secant (Task 3) and the genuine row-swap (Task 6) spelled out. No "add error handling"/"TBD".

**Type consistency:** `ColumnCoupledInputs`/`ColumnClosure`/`ColumnSensitivity`/`solve_column_coupled`/`column_moments`/`column_sensitivity`/`column_dRc_dp`/`column_lu_factor`/`column_lu_solve`/`solve_slim_disk_coupled` are used with identical signatures across all tasks. `ColumnInputs.f_adv` added in Task 1 is consumed in Tasks 3/6.

---

*Sources: design spec `docs/superpowers/specs/2026-06-14-slim-disk-vertical-bvp-coupling-design.md`; scope `docs/superpowers/plans/2026-06-14-slim-disk-vertical-bvp-coupling.md`; formulas `docs/superpowers/references/disk-physics-formulas.md` §20–§23; code `src/disk_column_bvp.cpp`, `src/slim_disk_radial.cpp`, `include/grrt/scene/{disk_column_bvp,slim_disk_radial}.h`, `tests/{test_column_bvp,test_slim_jacobian}.cpp`, `CMakeLists.txt`. S11 [arXiv:1006.4309] Eqs 8,11,13,23,29,32-33,42,45.*
